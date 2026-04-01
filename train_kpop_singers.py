"""
K-pop singer recognition using SpeechBrain ECAPA-TDNN embeddings + a small softmax head.

- Expects isolated vocals under:
    root/
      groupName/
        memberName/
          train/
            Isolated_Vocals/
               *.wav (or nested folders)

- Resamples 22_050 Hz -> 16_000 Hz (ECAPA requirement).
- Builds a tiny linear classifier head over ECAPA embeddings for: [members ... , silence]
- Trains/evaluates on GPU, saves & reloads.

Tested on: Python 3.9, CUDA 12.x with torch cu124 wheels.
"""

import os, argparse, random, glob, math
import numpy as np
from typing import List
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchaudio
from datasets.new_datasets import BinaryVocalDataset
from model.helper_functions import FastEmbeddingDataset
from model.heads import MultiMemberBinaryHead
from model.encoders import MuQEncoderWrapper

from tqdm import tqdm
from muq import MuQ
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pickle, hashlib
from collections import OrderedDict

class MuQEmbeddingCache:
    def __init__(self, cacheDir: str, maxOpenSongs: int = 200):
        self.cacheDir = cacheDir
        os.makedirs(self.cacheDir, exist_ok=True)
        self._openSongCaches = OrderedDict()
        self.maxOpenSongs = maxOpenSongs

    def _makeConfigTag(
        self,
        *,
        srOut: int,
        ctxFrac: float,
        chunkSec: float,
        pcaTag: str = "pca256",
        encoderTag: str = "muq-large-msd-iter",
    ) -> str:
        return f"sr{srOut}_ctx{ctxFrac}_chunk{chunkSec}_{pcaTag}_{encoderTag}"

    def _songPath(
        self,
        *,
        songId: str,
        srOut: int,
        ctxFrac: float,
        chunkSec: float,
        pcaTag: str = "pca256",
        encoderTag: str = "muq-large-msd-iter",
    ) -> str:
        cfg = self._makeConfigTag(
            srOut=srOut,
            ctxFrac=ctxFrac,
            chunkSec=chunkSec,
            pcaTag=pcaTag,
            encoderTag=encoderTag,
        )
        safeSong = songId.replace(os.sep, "_")
        return os.path.join(self.cacheDir, f"{safeSong}__{cfg}.npy")

    def hasSong(
        self,
        *,
        songId: str,
        srOut: int,
        ctxFrac: float,
        chunkSec: float,
        pcaTag: str = "pca256",
        encoderTag: str = "muq-large-msd-iter",
    ) -> bool:
        path = self._songPath(
            songId=songId,
            srOut=srOut,
            ctxFrac=ctxFrac,
            chunkSec=chunkSec,
            pcaTag=pcaTag,
            encoderTag=encoderTag,
        )
        return os.path.exists(path)

    def loadSong(
        self,
        *,
        songId: str,
        srOut: int,
        ctxFrac: float,
        chunkSec: float,
        pcaTag: str = "pca256",
        encoderTag: str = "muq-large-msd-iter",
    ):
        cacheKey = (songId, srOut, ctxFrac, chunkSec, pcaTag, encoderTag)
        if cacheKey in self._openSongCaches:
            arr = self._openSongCaches.pop(cacheKey)
            self._openSongCaches[cacheKey] = arr
            return arr

        path = self._songPath(
            songId=songId,
            srOut=srOut,
            ctxFrac=ctxFrac,
            chunkSec=chunkSec,
            pcaTag=pcaTag,
            encoderTag=encoderTag,
        )
        arr = np.load(path)
        self._openSongCaches[cacheKey] = arr

        if len(self._openSongCaches) > self.maxOpenSongs:
            self._openSongCaches.popitem(last=False)

        return arr

    def saveSong(
        self,
        *,
        songId: str,
        embMatrix: np.ndarray,
        srOut: int,
        ctxFrac: float,
        chunkSec: float,
        pcaTag: str = "pca256",
        encoderTag: str = "muq-large-msd-iter",
    ):
        path = self._songPath(
            songId=songId,
            srOut=srOut,
            ctxFrac=ctxFrac,
            chunkSec=chunkSec,
            pcaTag=pcaTag,
            encoderTag=encoderTag,
        )
        np.save(path, embMatrix.astype(np.float32))

def makeEmbeddingKey(
    *,
    songId: str,
    centerChunk: int,
    srOut: int,
    ctxFrac: float,
    chunkSec: float,
    pcaTag: str = "pca256",
    encoderTag: str = "muq-large-msd-iter",
):
    raw = f"{songId}|{centerChunk}|sr={srOut}|ctx={ctxFrac}|chunk={chunkSec}|{pcaTag}|{encoderTag}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()

# ---------------------------
# CLI args
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Path that contains <group>/<member>/train/Isolated_Vocals")
    ap.add_argument("--group", type=str, required=True, help="Group folder name under root")
    ap.add_argument("--sr_out", type=int, default=24000, help="ECAPA target sample rate")
    ap.add_argument("--chunk-sec", type=float, default=1.0, help="Chunk length in seconds for training")
    ap.add_argument("--short-chunk-sec", type=float, default=0.4, help="Chunk length in seconds for refinement")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-dir", type=str, default="./checkpoints")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval_thr", type=float, default=0.7)
    ap.add_argument("--skip-stage1",  action="store_true", help="Checks if stage 1 can be skipped")
    
    # Variance diagnostics
    ap.add_argument("--run-var-tests", action="store_true",
                    help="Run ANOVA/LDA + variance diagnostics on MuQ embeddings before training")
    ap.add_argument("--var-per-song-sec", type=float, default=10.0,
                    help="Max seconds per (member, song) for variance diagnostics sampling")
    return ap.parse_args()

# ---------------------------
# Small utilities
# ---------------------------
def list_wavs(root: str) -> List[str]:
    exts = ("*.wav", "*.flac", "*.mp3", "*.m4a")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, "**", e), recursive=True)
    return files

def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    
def _load_audio_mono(path: str):
    wav, sr = torchaudio.load(path)  # (C, T)
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # stereo -> mono
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, sr

def _resample_and_save(in_path: str, out_path: str, sr_out: int):
    wav, sr_in = _load_audio_mono(in_path)
    
    if sr_in != sr_out:
        resampler = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out)
        wav = resampler(wav)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, wav, sample_rate=sr_out, encoding="PCM_S", bits_per_sample=24)
    
def buildTrainingCacheVocalsOnly(groupDir: str, srOut: int, numWorkers: int = 8) -> str:
    """
    Creates/uses: <groupDir>/training_cache/sr_<srOut>/
    Resamples ONLY *_vocals.wav (no leading/backing required).
    """

    groupDir = str(groupDir)
    cacheDir = os.path.join(groupDir, "training_cache", f"sr_{srOut}")
    os.makedirs(cacheDir, exist_ok=True)

    pattern = "*_vocals.wav"
    inFiles = list(Path(groupDir).glob(pattern))

    if not inFiles:
        print(f"[cache] No *_vocals.wav found in {groupDir}")
        return cacheDir

    work = []
    for inPath in inFiles:
        outPath = os.path.join(cacheDir, inPath.name)
        if os.path.exists(outPath):
            continue
        work.append((str(inPath), outPath))

    print(f"[cache] Cache dir: {cacheDir}")
    print(f"[cache] Found {len(inFiles)} vocals. Need to create {len(work)} cached wavs.")

    if not work:
        return cacheDir

    def _do_one(inPath: str, outPath: str):
        _resample_and_save(inPath, outPath, srOut)
        return os.path.basename(outPath)

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, int(numWorkers))) as ex:
        futures = [ex.submit(_do_one, inP, outP) for inP, outP in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Caching (vocals)", unit="file"):
            try:
                fut.result()
            except Exception as e:
                failures += 1
                print(f"[cache] Failed: {e}")

    if failures:
        print(f"[cache] Done with {failures} failures.")
    else:
        print("[cache] Done. All vocals cached successfully.")

    return cacheDir

def buildTrainingCacheRequireTriplet(groupDir: str, srOut: int, numWorkers: int = 8) -> str:
    """
    Creates/uses: <groupDir>/training_cache/sr_<srOut>/
    Resamples only songs that have:
        *_vocals.wav
        *_leading_vocals.wav
        *_backing_vocals.wav
    """

    groupDir = str(groupDir)
    cacheDir = os.path.join(groupDir, "training_cache", f"sr_{srOut}")
    os.makedirs(cacheDir, exist_ok=True)

    vocals = {p.stem.replace("_vocals", ""): p for p in Path(groupDir).glob("*_vocals.wav")}
    leading = {p.stem.replace("_leading_vocals", ""): p for p in Path(groupDir).glob("*_leading_vocals.wav")}
    backing = {p.stem.replace("_backing_vocals", ""): p for p in Path(groupDir).glob("*_backing_vocals.wav")}

    commonSongs = sorted(set(vocals.keys()) & set(leading.keys()) & set(backing.keys()))

    if not commonSongs:
        print("[cache] No complete (vocals+leading+backing) triplets found.")
        return cacheDir

    work = []

    for song in commonSongs:
        for stemType, stemDict, suffix in [
            ("vocals", vocals, "_vocals.wav"),
            ("leading", leading, "_leading_vocals.wav"),
            ("backing", backing, "_backing_vocals.wav"),
        ]:
            inPath = stemDict[song]
            outPath = os.path.join(cacheDir, inPath.name)
            if os.path.exists(outPath):
                continue
            work.append((str(inPath), outPath))

    print(f"[cache] Cache dir: {cacheDir}")
    print(f"[cache] Found {len(commonSongs)} complete triplets.")
    print(f"[cache] Need to create {len(work)} cached wavs.")

    if not work:
        return cacheDir

    def _do_one(inPath: str, outPath: str):
        _resample_and_save(inPath, outPath, srOut)
        return os.path.basename(outPath)

    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, int(numWorkers))) as ex:
        futures = [ex.submit(_do_one, inP, outP) for inP, outP in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Caching (triplets)", unit="file"):
            try:
                fut.result()
            except Exception as e:
                failures += 1
                print(f"[cache] Failed: {e}")

    if failures:
        print(f"[cache] Done with {failures} failures.")
    else:
        print("[cache] Done. All triplets cached successfully.")

    return cacheDir
    
class FocusCentersDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, centers):
        self.base = base_ds
        self.centers = centers  # list of dicts: {"song":.., "kind":.., "center_frame":..}

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        c = self.centers[idx]
        return self.base.get_item_for_center(c["song"], c["kind"], c["center_frame"])
    
class MiningViewDataset(torch.utils.data.Dataset):
    """
    Wraps an existing KpopVocalDataset and returns:
      - the original training tuple (seg, labels..., stem_weight)
      - plus metadata needed for hard mining
    This avoids changing the training pipeline.
    """
    def __init__(self, base_ds: BinaryVocalDataset):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # original training tuple
        item = self.base[idx]

        # metadata comes from base_samples (NOT from item)
        (song_name,
         stem_kind,
         center_frame,
         start_frame,
         start_sample_out,
         label_vec,
         importance_vec,
         harmony_vec,
         harmony_wts,
         adlib_vec,
         adlib_wts,
         stem_weight) = self.base.base_samples[idx]

        return item, song_name, stem_kind, int(center_frame)


def binarize_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """(B, C) logits -> (B, C) {0,1} via sigmoid threshold."""
    probs = logits.detach().sigmoid().to("cpu")
    return (probs >= thr).to(torch.float32)

def multilabel_micro_f1(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-8):
    """
    Micro F1 across all labels and samples.
    logits: (B, C), targets: (B, C) in {0,1}
    """
    preds = binarize_logits(logits, thr) # (B, C)
    t_cpu = targets.detach().to("cpu")
    tp = (preds * t_cpu).sum().item()
    fp = (preds * (1 - t_cpu)).sum().item()
    fn = ((1 - preds) * t_cpu).sum().item()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1, precision, recall

def extract_center_context(wavs: torch.Tensor, ctx_frac: float = 0.25) -> torch.Tensor:
    """
    wavs: (B, 1, T) or (B, T)
    Returns a center subsegment of length ~ctx_frac * T, shape (B, 1, T_ctx)
    """
    if wavs.ndim == 2:
        wavs = wavs.unsqueeze(1) # Forces (B, 1, T)
    B, C, T = wavs.shape
    ctx_len = max(1, int(T * ctx_frac))
    
    mid = T // 2
    half = ctx_len // 2
    start = max(0, mid - half)
    end = start + ctx_len
    if end > T:
        end = T
        start = max(0, T - ctx_len)
    
    ctx = wavs[:, :, start:end] # (B, 1, ctx_len)
    return ctx

def subset_accuracy(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    """
    Exact-set match accuracy: 1 only if all labels match for a sample.
    """
    preds = binarize_logits(logits, thr)
    t_cpu = targets.detach().to("cpu")
    eq = (preds == t_cpu).all(dim=1)  # (B,)
    return eq.float().mean().item()

def update_sanity_stats(stats: dict, logits: torch.Tensor, targets: torch.Tensor, thr: float, sample_cap: int = 200_000):
    """
    Streaming sanity stats:
      - avg_pred_pos vs avg_true_pos (over batch)
      - logit magnitude stats: mean_abs / p95_abs (approx) / max_abs
    """
    with torch.no_grad():
        # --- pos counts ---
        pred = (torch.sigmoid(logits) > thr).float()
        stats["pred_pos_sum"] += pred.sum(dim=1).sum().item()
        stats["true_pos_sum"] += targets.sum(dim=1).sum().item()
        stats["n_samples"] += logits.size(0)

        # --- logit magnitudes ---
        abs_logits = logits.detach().abs().flatten()
        stats["abs_sum"] += abs_logits.sum().item()
        stats["abs_count"] += abs_logits.numel()
        stats["abs_max"] = max(stats["abs_max"], abs_logits.max().item())

        # approximate p95 using a capped sample buffer
        if stats["abs_sample"] is not None:
            # take a small slice each batch to avoid big overhead
            take = min(2048, abs_logits.numel())
            if take > 0:
                samp = (
                    abs_logits[
                        torch.randint(0, abs_logits.numel(), (take,), device=abs_logits.device)
                    ]
                    .to(dtype=torch.float32, device="cpu")
                )
                stats["abs_sample"].append(samp)
                # cap memory
                if len(stats["abs_sample"]) * 2048 > sample_cap:
                    stats["abs_sample"] = stats["abs_sample"][-(sample_cap // 2048):]

def finalize_sanity_stats(stats: dict):
    avg_pred_pos = stats["pred_pos_sum"] / max(1, stats["n_samples"])
    avg_true_pos = stats["true_pos_sum"] / max(1, stats["n_samples"])

    mean_abs = stats["abs_sum"] / max(1, stats["abs_count"])
    max_abs = stats["abs_max"]

    if stats["abs_sample"] and len(stats["abs_sample"]) > 0:
        sample = torch.cat(stats["abs_sample"], dim=0)
        p95_abs = torch.quantile(sample, 0.95).item()
    else:
        p95_abs = float("nan")

    return {
        "avg_pred_pos": avg_pred_pos,
        "avg_true_pos": avg_true_pos,
        "mean_abs": mean_abs,
        "p95_abs": p95_abs,
        "max_abs": max_abs,
    }           
    
# ---------------------------
# Training / evaluation
# ---------------------------
def train_epoch(
    *,
    base_ds,
    encoder,
    head,
    trainLoadersByMember,
    device,
    optimizer,
    thr=0.5,
    use_amp=True,
):
    encoder.eval()
    head.train()

    sanity = {
        "pred_pos_sum": 0.0,
        "true_pos_sum": 0.0,
        "n_samples": 0,
        "abs_sum": 0.0,
        "abs_count": 0,
        "abs_max": 0.0,
        "abs_sample": [],
    }

    scaler = torch.amp.GradScaler(device=device, enabled=(use_amp and device.type == "cuda"))
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    member_names = base_ds.group_members
    memberToIdx = {m: i for i, m in enumerate(member_names)}

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    total_loss = 0.0
    total_count = 0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    amp_ctx = torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda"))

    for memberName in member_names:
        memberIdx = memberToIdx[memberName]
        loader = trainLoadersByMember[memberName]
        for step, batchExamples in enumerate(tqdm(loader, desc=f"Train({memberName})", leave=False)):
            if torch.is_tensor(batchExamples):
                batchExamples = batchExamples.tolist()

            emb, y, w = batchExamples
            emb = emb.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)
            w = w.to(device, non_blocking=True).view(-1)
            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                logits = head(emb, memberIdx=memberIdx)  # (B,)
                probs = torch.sigmoid(logits)
                pred = (probs > thr).to(y.dtype)

                # sanity
                abs_logits = logits.detach().abs().flatten()
                sanity["pred_pos_sum"] += float(pred.sum().item())
                sanity["true_pos_sum"] += float(y.detach().sum().item())
                sanity["n_samples"] += int(y.numel())
                sanity["abs_sum"] += float(abs_logits.sum().item())
                sanity["abs_count"] += int(abs_logits.numel())
                sanity["abs_max"] = max(sanity["abs_max"], float(abs_logits.max().item()))

                if len(sanity["abs_sample"]) < 64:
                    take = min(8, abs_logits.numel())
                    sanity["abs_sample"].append(abs_logits[:take].detach().cpu())

                loss_per = bce(logits, y.float())
                loss = (loss_per * w).mean()
                loss = loss + 1e-4 * (logits.pow(2).mean())  

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()

            tp = int(((pred == 1) & (y == 1)).sum().item())
            fp = int(((pred == 1) & (y == 0)).sum().item())
            fn = int(((pred == 0) & (y == 1)).sum().item())

            bsz = int(y.numel())
            total_loss += float(loss.item()) * bsz
            total_count += bsz
            total_tp += tp
            total_fp += fp
            total_fn += fn

    avg_loss = total_loss / max(1, total_count)

    precision = total_tp / max(1, (total_tp + total_fp))
    recall = total_tp / max(1, (total_tp + total_fn))
    micro_f1 = 0.0
    if (precision + recall) > 0:
        micro_f1 = 2.0 * precision * recall / (precision + recall)

    if len(sanity["abs_sample"]) > 0:
        sample = torch.cat(sanity["abs_sample"], dim=0)
        p95_abs = torch.quantile(sample, 0.95).item()
    else:
        p95_abs = float("nan")

    sanity_out = {
        "avg_pred_pos": sanity["pred_pos_sum"] / max(1, sanity["n_samples"]),
        "avg_true_pos": sanity["true_pos_sum"] / max(1, sanity["n_samples"]),
        "mean_abs": sanity["abs_sum"] / max(1, sanity["abs_count"]),
        "p95_abs": p95_abs,
        "max_abs": sanity["abs_max"],
    }

    return avg_loss, {
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
        "sanity": sanity_out,
    }
    
@torch.no_grad()
def eval_epoch(
    *,
    base_ds,
    encoder,
    head,
    valLoadersByMember,
    device,
    thr=0.5,
    use_amp=True,
):
    encoder.eval()
    head.eval()

    sanity = {
        "pred_pos_sum": 0.0,
        "true_pos_sum": 0.0,
        "n_samples": 0,
        "abs_sum": 0.0,
        "abs_count": 0,
        "abs_max": 0.0,
        "abs_sample_tensors": [],
    }

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    amp_ctx = torch.autocast(
        device_type=device.type,
        enabled=(use_amp and device.type == "cuda")
    )

    member_names = base_ds.group_members
    memberToIdx = {m: i for i, m in enumerate(member_names)}

    total_loss = 0.0
    total_count = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for memberName in member_names:
        memberIdx = memberToIdx[memberName]
        memberLoader = valLoadersByMember[memberName]

        # per-member accumulators
        member_loss_sum = 0.0
        member_count = 0
        member_tp = 0
        member_fp = 0
        member_fn = 0

        for step, batchExamples in enumerate(
            tqdm(memberLoader, desc=f"Eval({memberName})", leave=False)
        ):
            emb, y, w = batchExamples
            emb = emb.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).view(-1)
            w = w.to(device, non_blocking=True).view(-1)

            with amp_ctx:
                logits = head(emb, memberIdx=memberIdx)

                probs = torch.sigmoid(logits)
                pred = (probs > thr).to(y.dtype)

                abs_logits = logits.detach().abs().flatten()
                sanity["pred_pos_sum"] += float(pred.sum().item())
                sanity["true_pos_sum"] += float(y.sum().item())
                sanity["n_samples"] += int(y.numel())
                sanity["abs_sum"] += float(abs_logits.sum().item())
                sanity["abs_count"] += int(abs_logits.numel())
                sanity["abs_max"] = max(sanity["abs_max"], float(abs_logits.max().item()))

                if len(sanity["abs_sample_tensors"]) < 64:
                    take = min(8, abs_logits.numel())
                    sanity["abs_sample_tensors"].append(abs_logits[:take].detach().cpu())

                loss_per = bce(logits, y.float())
                loss = (loss_per * w).mean()
                loss = loss + 1e-4 * (logits.pow(2).mean())

            if step % 100 == 0:
                tqdm.write(
                    f"Eval step {step}: member={memberName}, "
                    f"loss={loss.item():.4f}, batch_size={y.size(0)}"
                )

            tp = int(((pred == 1) & (y == 1)).sum().item())
            fp = int(((pred == 1) & (y == 0)).sum().item())
            fn = int(((pred == 0) & (y == 1)).sum().item())

            bsz = int(y.numel())
            member_loss_sum += float(loss.item()) * bsz
            member_count += bsz
            member_tp += tp
            member_fp += fp
            member_fn += fn

        total_loss += member_loss_sum
        total_count += member_count
        total_tp += member_tp
        total_fp += member_fp
        total_fn += member_fn

    avg_loss = total_loss / max(1, total_count)

    precision = total_tp / max(1, (total_tp + total_fp))
    recall = total_tp / max(1, (total_tp + total_fn))
    micro_f1 = 0.0
    if (precision + recall) > 0:
        micro_f1 = 2.0 * precision * recall / (precision + recall)

    if len(sanity["abs_sample_tensors"]) > 0:
        sample = torch.cat(sanity["abs_sample_tensors"], dim=0)
        p95_abs = torch.quantile(sample, 0.95).item()
    else:
        p95_abs = float("nan")

    sanity_out = {
        "avg_pred_pos": sanity["pred_pos_sum"] / max(1, sanity["n_samples"]),
        "avg_true_pos": sanity["true_pos_sum"] / max(1, sanity["n_samples"]),
        "mean_abs": sanity["abs_sum"] / max(1, sanity["abs_count"]),
        "p95_abs": p95_abs,
        "max_abs": sanity["abs_max"],
    }

    return avg_loss, {
        "micro_f1": micro_f1,
        "precision": precision,
        "recall": recall,
        "sanity": sanity_out,
    }
    
@torch.no_grad()
def hard_miner(encoder, head, loader, device, thr=0.7, pain_threshold=2.0, max_hard=50000):
    """
    Runs the Phase-1 model over a loader and returns hard centers:
      [{"song":..., "kind":..., "center_frame":..., "pain":...}, ...]

    loader must yield:
      ( (seg, y_main, w_main, y_harm, w_harm, y_ad, w_ad, stem_weight),
        song_name, stem_kind, center_frame )
    """
    encoder.eval()
    head.eval()
    
    hard = []
    
    for (train_tuple, song, kind, center_frame) in tqdm(loader, desc="Mining hard windows", leave=False):
        (wavs,
         y_main, w_main,
         y_harm, w_harm,
         y_ad,   w_ad,
         stem_weight) = train_tuple
        
        wavs = wavs.to(device, non_blocking=True)
        y_main = y_main.to(device, non_blocking=True).float()
        
        # (B,1,T)->(B,T) for encoder
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs_encoder = wavs.squeeze(1)
        elif wavs.ndim == 2:
            wavs_encoder = wavs
        else:
            raise ValueError(f"Unexpected wavs shape: {wavs.shape}")
        
        # Encode (same as train/eval)
        emb_main_b1, emb_ctx_b1 = encoder.encode_batch(wavs_encoder, ctx_frac=0.25)
        emb_main = emb_main_b1.squeeze(1)
        emb_ctx  = emb_ctx_b1.squeeze(1)
        emb_fused = torch.cat([emb_main, emb_ctx], dim=1)
        
        out = head(emb_fused, emb_ctx)
        logits = out["main"]                      # (B, C)
        probs = torch.sigmoid(logits)

        predicted_present = probs > thr
        true_present = (y_main == 1)
        
        false_positive_count = (predicted_present & ~true_present).sum(dim=1).float()
        false_negative_count = ((~predicted_present) & true_present).sum(dim=1).float()
        pred_count = predicted_present.sum(dim=1).float()
        true_count = y_main.sum(dim=1).float()
        
        # pos_conf = mean prob on true positives (0 if none)
        pos_conf = torch.zeros_like(true_count)
        has_pos = true_count > 0
        if has_pos.any():
            pos_conf[has_pos] = (probs[has_pos] * y_main[has_pos]).sum(dim=1) / true_count[has_pos].clamp(min=1.0)

        # Pain score tuned for your failure mode: FP spam matters most
        pain = 2.0 * false_positive_count + 1.5 * false_negative_count + 0.5 * (pred_count - true_count).clamp(min=0) + 0.5 * (1.0 - pos_conf)
        
        is_hard = (pain >= pain_threshold) | (false_positive_count >= 2) | (false_negative_count >= 1) | (pos_conf < 0.55)
        
        idxs = torch.where(is_hard)[0].tolist()
        for i in idxs:
            hard.append({
                "song": song[i] if isinstance(song, (list, tuple)) else song,
                "kind": kind[i] if isinstance(kind, (list, tuple)) else kind,
                "center_frame": int(center_frame[i] if isinstance(center_frame, (list, tuple, torch.Tensor)) else center_frame),
                "pain": float(pain[i].item()),
            })
            
            if len(hard) >= max_hard:
                break
        
    hard.sort(key=lambda d: d["pain"], reverse=True)
    return hard

def check_run_stage1(
    *, model, ckpt_path, skip_stage1, device
):
    """
    Either runs Stage 1 training or loads a pretrained Stage 1 head
    depending on --skip-stage-1.
    """
    if skip_stage1:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"--skip-stage-1 was set but checkpoint not found: {ckpt_path}"
            )

        print(f"[Stage1] Skipping training. Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return False
    
    return True

def splitBySong(dataset, valSongCount=5, seed=123):
    # dataset.samples items begin with (song_name, kind, ...)
    songs = sorted({s[0] for s in dataset.samples})
    if valSongCount >= len(songs):
        raise ValueError(f"valSongCount={valSongCount} must be < num_songs={len(songs)}")

    rng = random.Random(seed)
    valSongs = set(rng.sample(songs, valSongCount))

    trainIdx, valIdx = [], []
    for i, sample in enumerate(dataset.samples):
        songName = sample[0]
        (valIdx if songName in valSongs else trainIdx).append(i)

    return trainIdx, valIdx, sorted(valSongs)

def _l2Normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)

@torch.no_grad()
def _encodeWavsToEmbeddings(
    *,
    encoder,
    wavBatch: torch.Tensor,   # (B,T)
    ctxFrac: float,
) -> torch.Tensor:
    """
    Returns a single fused embedding per wav: (B, D)
    using your common "main+ctx then normalize" logic.
    """
    embMain, embCtx = encoder.encode_batch(wavBatch, ctx_frac=ctxFrac)  # (B,D),(B,D)
    emb = _l2Normalize(embMain) + _l2Normalize(embCtx)
    emb = _l2Normalize(emb)
    return emb

def _sampleCentersPerSongForMember(
    *,
    ds,                 # BinaryVocalDataset
    songId: str,
    memberName: str,
    maxCenters: int,    # ~ seconds, since 1 center ~= 1s window
    requireSolo: bool = True,
    rng: np.random.Generator,
) -> list[int]:
    """
    Pick up to maxCenters center-chunk indices for (song, member) from "clean" regions.

    Clean region = member active AND not transition.
    If requireSolo, also exclude overlap chunks.
    """
    memberMask = ds.metadata.memberMask[songId][memberName]          
    transition = ds.metadata.transitionMask[songId]                  
    overlap = ds.metadata.overlapMask[songId]
    
    cand = memberMask & (~transition)
    if requireSolo:
        cand = cand & (~overlap)

    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return []

    # Randomly choose up to maxCenters without replacement
    if idx.size <= maxCenters:
        chosen = idx
    else:
        chosen = rng.choice(idx, size=maxCenters, replace=False)

    return [int(c) for c in chosen]

def _fisherPairScores(
    E: np.ndarray,          # (N,D)
    y: np.ndarray,          # (N,)
    members: list[str],
    useFullCov: bool = False,
    covReg: float = 1e-3,
):
    """
    Returns list of (score, memberA, memberB) sorted ascending (hardest pairs first).

    score = (muA-muB)^T inv(S_pooled) (muA-muB)
    - If useFullCov=False, uses diagonal pooled covariance (stable, fast).
    - If useFullCov=True, uses full pooled covariance with ridge regularization.
    """
    C = len(members)
    D = E.shape[1]

    mus = []
    ns = []
    covDiags = []
    covsFull = []

    for ci in range(C):
        Xc = E[y == ci]
        ns.append(Xc.shape[0])
        mu = Xc.mean(axis=0) if Xc.shape[0] else np.zeros(D)
        mus.append(mu)

        if Xc.shape[0] >= 2:
            # diag var
            covDiags.append(Xc.var(axis=0, ddof=1))
            if useFullCov:
                covsFull.append(np.cov(Xc, rowvar=False, ddof=1))
        else:
            covDiags.append(np.ones(D))
            if useFullCov:
                covsFull.append(np.eye(D))

    mus = np.stack(mus, axis=0)          # (C,D)
    ns = np.array(ns, dtype=np.float64)  # (C,)

    # pooled covariance
    if not useFullCov:
        # pooled diag: sum_c (n_c-1) * var_c / (N-C)
        denom = max(1.0, float(E.shape[0] - C))
        pooledDiag = np.zeros(D, dtype=np.float64)
        for ci in range(C):
            w = max(0.0, ns[ci] - 1.0)
            pooledDiag += w * covDiags[ci]
        pooledDiag = pooledDiag / denom
        pooledDiag = pooledDiag + covReg  # avoid /0

        def mahal2(diff):
            return float(np.sum((diff * diff) / pooledDiag))

    else:
        denom = max(1.0, float(E.shape[0] - C))
        pooled = np.zeros((D, D), dtype=np.float64)
        for ci in range(C):
            w = max(0.0, ns[ci] - 1.0)
            pooled += w * covsFull[ci]
        pooled = pooled / denom

        # ridge regularize
        ridge = covReg * (np.trace(pooled) / max(1, D))
        pooled = pooled + ridge * np.eye(D)

        invPooled = np.linalg.inv(pooled)

        def mahal2(diff):
            return float(diff.T @ invPooled @ diff)

    pairs = []
    for i in range(C):
        for j in range(i + 1, C):
            diff = mus[i] - mus[j]
            score = mahal2(diff)
            pairs.append((score, members[i], members[j]))

    pairs.sort(key=lambda t: t[0])  # smallest = hardest
    return pairs


def _pcaDimsForExplainedVariance(E: np.ndarray, target: float = 0.95):
    """
    Returns (k, explained_ratio_k, eigvals_sorted)
    using covariance eigenvalues of centered E.
    """
    X = E - E.mean(axis=0, keepdims=True)   # center
    # Cov = (X^T X) / (N-1)
    N = X.shape[0]
    denom = max(1, N - 1)
    cov = (X.T @ X) / denom                 # (D,D)

    # covariance is symmetric -> use eigh (more stable)
    eigvals = np.linalg.eigvalsh(cov)       # ascending
    eigvals = np.maximum(eigvals, 0.0)
    eigvals = eigvals[::-1]                 # descending

    total = float(eigvals.sum()) + 1e-12
    cumsum = np.cumsum(eigvals) / total

    k = int(np.searchsorted(cumsum, target) + 1)
    k = min(k, E.shape[1])
    return k, float(cumsum[k - 1]), eigvals

@torch.no_grad()
def runMuqVarianceTests(
    *,
    ds,
    encoder,
    device: torch.device,
    ctxFrac: float = 0.25,
    perSongSec: float = 10.0,
    seed: int = 1337,
    requireSolo: bool = True,
    batchSize: int = 32,
    topkAnova: int = 30,
):
    """
    Runs:
      - intra-member variance (trace of covariance)
      - ANOVA F-score per embedding dimension
      - LDA projection (C-1 dims) for separability inspection

    Sampling:
      For each (song, member), take up to perSongSec windows (1s each).
      If fewer exist, use what's available. No duplication.
    """
    rng = np.random.default_rng(seed)

    # Determine how many 1s windows per song cap
    maxCenters = max(1, int(round(perSongSec / ds.context_sec)))

    songs = list(ds.training_songs.keys())
    songs.sort()

    members = list(ds.group_members)
    memberToIdx = {m: i for i, m in enumerate(members)}  # faster than members.index()

    # Collect embeddings + labels
    X_chunks = []
    y_chunks = []
    meta = []  # (songId, memberName, centerChunk)

    # -------------------------
    # Stage 1: Collect windows
    # -------------------------
    totalPairs = len(songs) * max(1, len(members))
    print(f"[VarTests] Collecting windows: songs={len(songs)} members={len(members)} "
          f"(cap {maxCenters} windows per (member,song)), \nTotal pairs: {[totalPairs]}")

    songBar = tqdm(songs, desc="Collecting (songs)", unit="song")
    for songId in songBar:
        # optional: show how many we’ve collected so far
        songBar.set_postfix_str(f"clips={len(X_chunks)}")

        for memberName in members:
            centers = _sampleCentersPerSongForMember(
                ds=ds,
                songId=songId,
                memberName=memberName,
                maxCenters=maxCenters,
                requireSolo=requireSolo,
                rng=rng,
            )
            for c in centers:
                # This can be slow the first time due to cache miss / resample / disk IO
                wav = ds._load_window(songId, c)  # (1, T)
                X_chunks.append(wav.squeeze(0))   # (T,)
                y_chunks.append(memberToIdx[memberName])
                meta.append((songId, memberName, c))

    if len(X_chunks) < 10:
        print("[VarTests] Not enough samples collected. Check labels/solo filters.")
        return

    # -------------------------
    # Stage 2: Encode with MuQ
    # -------------------------
    encoder.clearPca()
    X = torch.stack(X_chunks, dim=0).to(device)  # (N,T)
    y = np.array(y_chunks, dtype=np.int64)
    N = X.size(0)

    print(f"[VarTests] Encoding MuQ embeddings: N={N} batchSize={batchSize} ctxFrac={ctxFrac}")

    embs = []
    numBatches = (N + batchSize - 1) // batchSize
    for b in tqdm(range(numBatches), desc="Encoding (MuQ)", unit="batch"):
        i = b * batchSize
        wavBatch = X[i:i+batchSize]
        emb = _encodeWavsToEmbeddings(encoder=encoder, wavBatch=wavBatch, ctxFrac=ctxFrac)  # (B,D)
        embs.append(emb.detach().cpu())

    E = torch.cat(embs, dim=0).numpy()  # (N,D)
    D = E.shape[1]
    C = len(members)

    print(f"\n[VarTests] Samples: N={N} | emb dim D={D} | members C={C} | soloOnly={requireSolo}")
    print(f"[VarTests] per-song cap: {maxCenters} windows (~{perSongSec:.1f}s) per (member,song); songs used: {len(songs)}")

    pcaMean, pcaW = fitPca(E, k=256)
    print(f"[VarTests] PCA256 ready: mean shape={pcaMean.shape}, W shape={pcaW.shape}")

    # You can save these to disk and load later
    np.savez("muq_pca_256.npz", mean=pcaMean, W=pcaW)

    # -------------------------
    # (A) Intra-member variance
    # -------------------------
    print("\n[VarTests] Intra-member spread (cov trace ~ total variance):")
    memberMeans = {}
    memberTrace = {}
    memberCounts = {}

    for ci, m in enumerate(members):
        Xm = E[y == ci]
        memberCounts[m] = Xm.shape[0]
        if Xm.shape[0] < 2:
            memberMeans[m] = Xm.mean(axis=0) if Xm.shape[0] else np.zeros(D)
            memberTrace[m] = float("nan")
            continue
        mu = Xm.mean(axis=0)
        memberMeans[m] = mu
        var = Xm.var(axis=0, ddof=1)
        memberTrace[m] = float(var.sum())

    for m in sorted(
        members,
        key=lambda mm: (-(memberTrace.get(mm, -1e9) if not math.isnan(memberTrace.get(mm, float("nan"))) else -1e9))
    ):
        print(f"  {m:>12s}: n={memberCounts[m]:5d}  traceVar={memberTrace[m]:10.4f}")

    print("\n[VarTests] Pairwise Fisher separation (larger = more separated):")
    # Diagonal-pooled Fisher is the best default (stable + fast)
    fisherPairs = _fisherPairScores(
        E=E,
        y=y,
        members=members,
        useFullCov=False,   # set True if you want full covariance (slower)
        covReg=1e-3,
    )

    # Remove Gang Vocal pairs
    filteredPairs = [
        (score, a, b)
        for score, a, b in fisherPairs
        if "Gang Vocal" not in (a, b)
    ]

    # Sort hardest → easiest (ascending Fisher = hardest first)
    filteredPairs.sort(key=lambda x: x[0])

    print("\n[VarTests] Member-vs-member Fisher ranking (hardest → easiest):")

    for rank, (score, a, b) in enumerate(filteredPairs, start=1):
        print(f"  #{rank:02d} {a} vs {b}  fisher={score:.4f}")

    # -------------------------
    # (B) ANOVA F-score per dim
    # -------------------------
    print("\n[VarTests] Computing ANOVA F-scores per embedding dim...")
    muAll = E.mean(axis=0)  # (D,)
    ssBetween = np.zeros(D, dtype=np.float64)
    ssWithin = np.zeros(D, dtype=np.float64)

    for ci in tqdm(range(C), desc="ANOVA (classes)", unit="class"):
        Xc = E[y == ci]
        if Xc.shape[0] == 0:
            continue
        muc = Xc.mean(axis=0)
        ssBetween += Xc.shape[0] * (muc - muAll) ** 2
        ssWithin += ((Xc - muc) ** 2).sum(axis=0)

    dfBetween = max(1, C - 1)
    dfWithin = max(1, N - C)
    msBetween = ssBetween / dfBetween
    msWithin = ssWithin / dfWithin
    F = msBetween / (msWithin + 1e-12)

    topIdx = np.argsort(-F)[:min(topkAnova, D)]
    print(f"\n[VarTests] ANOVA top-{len(topIdx)} embedding dimensions by F-score:")
    for rank, d in enumerate(topIdx, start=1):
        print(f"  #{rank:02d} dim={int(d):4d}  F={float(F[d]):.3f}")

    # -------------------------
    # (C) LDA (supervised proj)
    # -------------------------
    print("\n[VarTests] Building scatter matrices for LDA (this can take a bit)...")
    Sw = np.zeros((D, D), dtype=np.float64)
    Sb = np.zeros((D, D), dtype=np.float64)

    for ci in tqdm(range(C), desc="LDA (scatter)", unit="class"):
        Xc = E[y == ci]
        if Xc.shape[0] < 2:
            continue
        muc = Xc.mean(axis=0)
        Xcz = Xc - muc
        Sw += Xcz.T @ Xcz
        diff = (muc - muAll).reshape(D, 1)
        Sb += Xc.shape[0] * (diff @ diff.T)

    reg = 1e-3 * (np.trace(Sw) / max(1, D))
    SwReg = Sw + reg * np.eye(D)

    print("[VarTests] Solving generalized eigenproblem (can be slow on CPU)...")
    try:
        A = np.linalg.solve(SwReg, Sb)
        eigVals, eigVecs = np.linalg.eig(A)
        eigVals = np.real(eigVals)
        eigVecs = np.real(eigVecs)
        order = np.argsort(-eigVals)
        eigVals = eigVals[order]
        eigVecs = eigVecs[:, order]
    except np.linalg.LinAlgError:
        print("\n[VarTests] LDA failed (singular matrix). Increase samples or reduce D (PCA first).")
        return

    ldaDim = min(C - 1, D)
    W = eigVecs[:, :ldaDim]
    Z = E @ W

    print(f"\n[VarTests] LDA: using {ldaDim} components (max C-1). Top eigenvalues:")
    for i in range(min(10, ldaDim)):
        print(f"  LDA{i+1}: lambda={float(eigVals[i]):.4f}")

    Zmus = []
    for ci in range(C):
        Zmus.append(Z[y == ci].mean(axis=0))
    Zmus = np.stack(Zmus, axis=0)

    print("\n[VarTests] PCA explained-variance diagnostic:")
    try:
        k95, ratio95, eigvals = _pcaDimsForExplainedVariance(E, target=0.95)
        k90, r90, _ = _pcaDimsForExplainedVariance(E, target=0.90)
        k99, r99, _ = _pcaDimsForExplainedVariance(E, target=0.99)
        print(f"  Keep 90% variance => k={k90} dims (explained={r90*100:.2f}%)")
        print(f"  Keep 95% variance => k={k95} dims (explained={ratio95*100:.2f}%)")
        print(f"  Keep 99% variance => k={k99} dims (explained={r99*100:.2f}%)")
    except np.linalg.LinAlgError:
        print("  PCA failed (LinAlgError). If this happens, reduce sample count or do randomized PCA.")
        print("[VarTests] Done.\n")
   
def fitPca(E: np.ndarray, k: int = 256):
    """
    E: (N,D) numpy
    returns (mean, W) where:
      mean: (D,)
      W: (D,k) projection matrix
    """
    mean = E.mean(axis=0)
    X = E - mean
    N = X.shape[0]
    cov = (X.T @ X) / max(1, N - 1)

    # symmetric eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(cov)  # eigvecs: (D,D) columns
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]

    W = eigvecs[:, :k]  # (D,k)
    return mean.astype(np.float32), W.astype(np.float32)
    
class IndexDataset(torch.utils.data.Dataset):
    """
    Wraps any dataset and returns just an index integer.
    This is perfect for round-robin member training where we call base_ds.getItemForMember(...)
    """
    def __init__(self, ds):
        self.ds = ds
    def __len__(self):
        return len(self.ds)
    def __getitem__(self, i):
        return i

# Main
def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high') # Small speed bump on Ampere
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    group_dir = os.path.join(args.root, "training_data", args.group)
    json_dir = os.path.join(args.root, "saved_labels", args.group)
    cache_dir = buildTrainingCacheVocalsOnly(group_dir, srOut=args.sr_out)
    
    # Dataset & split
    ds_phase1 = BinaryVocalDataset(json_dir=json_dir, cache_dir=cache_dir, sr_out=args.sr_out)
    
    print(f"Number to train: {int(len(ds_phase1) * 0.2)} (20% of full dataset)")
    
    valSongs, tierInfo = ds_phase1.pickValidationSongsStratified(
        valEasy=2, valMed=2, valHard=1, domFracSoloThreshold=0.95
    )
    print(f"Selected {len(valSongs)} validation songs: {valSongs}")

    valSongNames = {t[0] for t in valSongs}
    allSongNames = set(ds_phase1.training_songs.keys())
    trainSongNames = allSongNames - valSongNames
    print(f"train song names: {trainSongNames}")

    # Build fixed stage-1 examples once
    trainExamplesByMember = ds_phase1.buildStage1ExamplesByMember(
        allowedSongs=trainSongNames,
        totalExamplesPerMember=4000,
        negOtherFrac=0.70,
        seed=1337,
        maxWorkers=min(8, os.cpu_count() or 1),
    )

    valExamplesByMember = ds_phase1.buildStage1ExamplesByMember(
        allowedSongs=valSongNames,
        totalExamplesPerMember=1000,
        negOtherFrac=0.70,
        seed=1338,
        maxWorkers=min(8, os.cpu_count() or 1),
    )
    trainLoadersByMember = {}
    valLoadersByMember = {}
    
    embedding_cache_dir = os.path.join(group_dir, "training_cache", f"sr_{args.sr_out}", "embedding_cache")
    embeddingCache = MuQEmbeddingCache(embedding_cache_dir)

    for memberName in ds_phase1.group_members:
        trainDsMember = FastEmbeddingDataset(
            examples=trainExamplesByMember[memberName],
            cache_dir=embedding_cache_dir,
            sr_out=args.sr_out,
            ctx_frac=0.25,
            chunk_sec=args.chunk_sec
        )
        valDsMember = FastEmbeddingDataset(
            examples=valExamplesByMember[memberName],
            cache_dir=embedding_cache_dir,
            sr_out=args.sr_out,
            ctx_frac=0.25,
            chunk_sec=args.chunk_sec
        )

        trainLoadersByMember[memberName] = DataLoader(
            trainDsMember,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers, # This now handles the disk reads in parallel
            pin_memory=True,              # Keeps memory prepared for fast GPU transfer
            drop_last=True,
        )

        valLoadersByMember[memberName] = DataLoader(
            valDsMember,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    print("Validation songs:", tierInfo["valSongs"])
    print("Tier sizes:", {k: len(v) for k, v in tierInfo.items() if k in ["easy","med","hard"]})
    
    # Load MuQ ECAPA encoder (pretraiened)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq.to(device).eval()
    
    encoder = MuQEncoderWrapper(
        muq_model=muq,
        pooling="mean",
        debug=False
    ).to(device)
    
    if getattr(args, "run_var_tests", False):
        runMuqVarianceTests(
            ds=ds_phase1,
            encoder=encoder,
            device=device,
            ctxFrac=0.25,
            perSongSec=args.var_per_song_sec,
            seed=args.seed,
            requireSolo=True,     # exclude overlap + transitions
            batchSize=32,
            topkAnova=30,
        )
        
    pca = np.load("muq_pca_256.npz")
    mean = pca["mean"]
    W = pca["W"]

    print("mean:", mean.shape, "W:", W.shape)  # should be (1024,) and (1024,256)
    encoder.setPca(pca["mean"], pca["W"])
    
    # Dummy audio to infer embedding dim
    dummy = torch.zeros(1, int(args.chunk_sec * args.sr_out), device=device)  # (B,T)

    # Ensure (B, T)
    if dummy.ndim == 3 and dummy.size(1) == 1:
        dummy = dummy.squeeze(1)

    with torch.no_grad():
        emb_main, emb_ctx = encoder.encode_batch(dummy, ctx_frac=0.25)  # (1,D), (1,D)
        emb_dim = emb_main.shape[-1]  # == emb_ctx.shape[-1]

    # Members (use ds_phase1.group_members as you requested)
    member_names = ds_phase1.group_members
    num_members = len(member_names)

    # Head: N binary classifiers in one module
    head = MultiMemberBinaryHead(
        embDim=emb_dim,
        numMembers=num_members,
        hidden=64,
        dropout=0.4
    ).to(device)
        
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.3, # multiply LR by 0.3 when plateau
        patience=2, # wait 2 epochs with no improvement
    )
    
    print("Pre-encoding all songs into cache...")
    for song_id in ds_phase1.training_songs.keys():
        if not embeddingCache.hasSong(songId=song_id, srOut=args.sr_out, ctxFrac=0.25, chunkSec=args.chunk_sec):
            # This builds and saves the .npy file automatically
            ds_phase1._buildSongEmbeddingMatrix(
                songId=song_id,
                encoder=encoder,
                device=device,
                contextSec=args.chunk_sec
            )
            
    print("Pre-encoding complete.")
    
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.group}_muq_head.pt")
    
    eval_thr = getattr(args, "eval_thr", 0.7)
    # Checks if you can skip stage 1
    run_stage1 = check_run_stage1(model=head, ckpt_path=ckpt_path, skip_stage1=args.skip_stage1, device=device)
    
    if run_stage1:
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")

            tr_loss, tr_metrics = train_epoch(
                base_ds=ds_phase1,
                encoder=encoder,
                head=head,
                trainLoadersByMember=trainLoadersByMember,
                device=device,
                optimizer=optimizer,
                thr=eval_thr,
            )

            va_loss, va_metrics = eval_epoch(
                base_ds=ds_phase1,
                encoder=encoder,
                head=head,
                valLoadersByMember=valLoadersByMember,
                device=device,
                thr=eval_thr,
            )

            # --- sanity printing: use your existing finalize_sanity_stats format ---
            tr_s = tr_metrics["sanity"]
            va_s = va_metrics["sanity"]

            print(
                f"Sanity Train: avg_pred_pos={tr_s['avg_pred_pos']:.2f} vs avg_true_pos={tr_s['avg_true_pos']:.2f} | "
                f"logit| mean={tr_s['mean_abs']:.2f} p95={tr_s['p95_abs']:.2f} max={tr_s['max_abs']:.2f}"
            )
            print(
                f"Sanity Val:   avg_pred_pos={va_s['avg_pred_pos']:.2f} vs avg_true_pos={va_s['avg_true_pos']:.2f} | "
                f"logit| mean={va_s['mean_abs']:.2f} p95={va_s['p95_abs']:.2f} max={va_s['max_abs']:.2f}"
            )

            old_lr = optimizer.param_groups[0]["lr"]
            scheduler.step(va_metrics["micro_f1"])
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr != old_lr:
                print(f"[LR Scheduler] Reducing LR: {old_lr} → {new_lr}")

            print(
                f"Train - loss: {tr_loss:.4f} | micro-F1: {tr_metrics['micro_f1']:.4f} "
                f"(P {tr_metrics['precision']:.3f}, R {tr_metrics['recall']:.3f})"
            )
            print(
                f"Val   - loss: {va_loss:.4f} | micro-F1: {va_metrics['micro_f1']:.4f} "
                f"(P {va_metrics['precision']:.3f}, R {va_metrics['recall']:.3f})"
            )

            if va_metrics["micro_f1"] > best_acc:
                best_acc = va_metrics["micro_f1"]
                torch.save(
                    {
                        "state_dict": head.state_dict(),
                        "members": ds_phase1.group_members,
                        "emb_dim": emb_dim,
                        "sr": args.sr_out,
                        "chunk_sec": args.chunk_sec,
                        "group": args.group,
                        "eval_thr": eval_thr,
                    },
                    ckpt_path,
                )
                print(f"✅ Saved best head to: {ckpt_path} (microF1={best_acc:.4f})")

        print("\nDone. Best val microF1:", best_acc)
    else:
        print("⚠️  WARNING: Stage 1 skipped — using pretrained head")
        
    # mine_ds = MiningViewDataset(ds_phase1)
    # mine_loader = DataLoader(mine_ds, batch_size=args.batch_size, shuffle=False,
    #                      num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # # 2) mine hard centers using the trained phase 1 model
    # hard_centers = hard_miner(
    #     encoder=fused_encoder,
    #     head=head,
    #     loader=mine_loader,   # based on the 2s dataset
    #     thr=eval_thr,
    #     device=device,
    # )
    # print("Hard centers mined:", len(hard_centers))
    
    # # Build once
    # ds_phase2_base = KpopVocalDataset(group_dir, sr_out=args.sr_out, context_seconds=0.4, group_name=args.group, audio_dir=cache_dir, is_phase2=True)
    
    # # Build first focused loader from inital mind centers from Phase 1
    # focus_ds = FocusCentersDataset(ds_phase2_base, hard_centers)
    # focus_loader = DataLoader(
    #     focus_ds,
    #     batch_size=args.batch_size,
    #     shuffle=True,                 # IMPORTANT: shuffle hard examples
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )
    
    # # Keep same to compare aacross epochs
    # val2_ds = KpopVocalDataset(group_dir, sr_out=args.sr_out, context_seconds=args.short_chunk_sec,
    #                        group_name=args.group, audio_dir=cache_dir, is_phase2=True)

    # n_val2 = max(1, int(len(val2_ds) * args.val_split))
    # n_train2 = len(val2_ds) - n_val2
    # _, val2_subset = random_split(val2_ds, [n_train2, n_val2])

    # val2_loader = DataLoader(
    #     val2_subset, batch_size=args.batch_size, shuffle=False,
    #     num_workers=args.num_workers, pin_memory=True, drop_last=False
    # )
    
    # head2 = PresenceHead(emb_dim_fused=fused_dim, num_members=num_members).to(device)
    # head2.load_state_dict(head.state_dict())  # start from phase1 solution
    # opt2 = torch.optim.AdamW(head2.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)

    # phase2_epochs = max(1, args.epochs // 2)  # e.g., 5 if phase1 was 10
    # best2 = -1.0
    
    # prev_p95_abs = None
    # rising_p95_streak = 0
    
    # REMINE_EVERY = 1000
    # TOP_HARD = 15000 # Cap size to keep training tight
    # PAIN_KEEP = 2.0 # Min pain threshold
    # THR2 = 0.5 # Phase-2 threshold for mining and eval
    
    # for epoch in range(1, phase2_epochs + 1):
    #     print(f"\n[Phase2] Epoch {epoch}/{phase2_epochs}")
    #     tr2_loss, tr2_metrics = train_epoch(fused_encoder, head2, focus_loader, device, opt2, thr=THR2, is_phase2=True)
    #     va2_loss, va2_metrics = eval_epoch(fused_encoder, head2, val2_loader, device, thr=THR2)

    #     print(f"[Phase2] Train loss {tr2_loss:.4f} microF1 {tr2_metrics['micro_f1']:.4f}")
    #     print(f"[Phase2] Val   loss {va2_loss:.4f} microF1 {va2_metrics['micro_f1']:.4f}")

    #     # === SCREAMING GUARDRAILS ===
    #     # Use the sanity stats you already compute in train_epoch/eval_epoch (mean_abs, p95_abs, max_abs, avg_pred_pos, avg_true_pos)
    #     # Assuming train_epoch returns them in tr2_metrics["sanity"] (if not, I’ll show where to add it)
    #     if "sanity" in tr2_metrics:
    #         s = tr2_metrics["sanity"]
    #         mean_abs = float(s.get("mean_abs", 0.0))
    #         p95_abs  = float(s.get("p95_abs", 0.0))
    #         max_abs  = float(s.get("max_abs", 0.0))
    #         avg_pred = float(s.get("avg_pred_pos", 0.0))
    #         avg_true = float(s.get("avg_true_pos", 1e-6))

    #         # p95 trend check (relapse)
    #         if prev_p95_abs is not None and p95_abs > prev_p95_abs + 0.5:
    #             rising_p95_streak += 1
    #         else:
    #             rising_p95_streak = 0
    #         prev_p95_abs = p95_abs

    #         # "Insane" definitions: these catch your old 75+ p95 meltdown early
    #         if (p95_abs >= 20.0) or (max_abs >= 60.0) or (rising_p95_streak >= 2):
    #             print(
    #                 f"\n\n🚨🚨🚨 LOGIT MELTDOWN DETECTED 🚨🚨🚨\n"
    #                 f"p95_abs={p95_abs:.2f} mean_abs={mean_abs:.2f} max_abs={max_abs:.2f} "
    #                 f"(rising_streak={rising_p95_streak})\n"
    #                 f"This usually means the model is saturating and will start spamming positives.\n"
    #             )

    #         # Prediction-rate sanity: stop if it starts screaming "everyone is singing"
    #         ratio = avg_pred / max(avg_true, 1e-6)
    #         if ratio >= 2.4:
    #             print(
    #                 f"\n\n🚨🚨🚨 FP-SPAM RELAPSE 🚨🚨🚨\n"
    #                 f"avg_pred_pos={avg_pred:.2f} avg_true_pos={avg_true:.2f} ratio={ratio:.2f}\n"
    #                 f"The model is predicting far too many singers per window.\n"
    #             )

    #         # Also catch the opposite: model becomes too timid
    #         if ratio <= 0.55:
    #             print(
    #                 f"\n⚠️  Phase2 looks overly timid: avg_pred_pos={avg_pred:.2f} avg_true_pos={avg_true:.2f} ratio={ratio:.2f}\n"
    #                 f"Not necessarily fatal, but it can mean threshold is too high or negatives too strong.\n"
    #             )

    #     # Save best
    #     if va2_metrics["micro_f1"] > best2:
    #         best2 = va2_metrics["micro_f1"]
    #         torch.save({
    #             "state_dict": head2.state_dict(),
    #             "classes": ds_phase1.class_map.idx_to_name,
    #             "emb_dim": fused_dim,
    #             "sr": args.sr_out,
    #             "chunk_sec": 0.4,
    #             "group": args.group,
    #             "eval_thr": THR2,
    #             "note": "phase2_refinement"
    #         }, os.path.join(args.save_dir, f"{args.group}_muq_head_phase2.pt"))
    #         print(f"✅ Saved best Phase2 head (microF1={best2:.4f})")
        
    #     # === RE-MINE EVERY 3 EPOCHS ===
    #     if (epoch % REMINE_EVERY) == 0:
    #         print("\n[Phase2] Re-mining hard centers...")
    #         # Mine on the *0.4s base dataset* using the current head2
    #         mine_ds = MiningViewDataset(ds_phase2_base)  # wrapper provides (train_tuple, song, kind, center_frame)
    #         mine_loader = DataLoader(
    #             mine_ds, batch_size=args.batch_size, shuffle=False,
    #             num_workers=args.num_workers, pin_memory=True, drop_last=False
    #         )

    #         new_hard = hard_miner(
    #             encoder=fused_encoder,
    #             head=head2,
    #             loader=mine_loader,
    #             device=device,
    #             thr=THR2,
    #             pain_threshold=PAIN_KEEP,
    #             max_hard=TOP_HARD * 3,  # allow extra before filtering/dedup
    #         )

    #         # Dedup and keep top N
    #         seen = set()
    #         dedup = []
    #         for c in new_hard:
    #             key = (c["song"], c["kind"], int(c["center_frame"]))
    #             if key in seen:
    #                 continue
    #             seen.add(key)
    #             dedup.append(c)
    #             if len(dedup) >= TOP_HARD:
    #                 break

    #         hard_centers = dedup
    #         print(f"[Phase2] Hard centers refreshed: {len(hard_centers)}")

    #         # Rebuild focus loader with the refreshed set
    #         focus_ds = FocusCentersDataset(ds_phase2_base, hard_centers)
    #         focus_loader = DataLoader(
    #             focus_ds, batch_size=args.batch_size, shuffle=True,
    #             num_workers=args.num_workers, pin_memory=True, drop_last=True
    #         )
    
if __name__ == "__main__":
    main()