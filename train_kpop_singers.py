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

import os, argparse,  random, glob
from typing import List

import torch
from torch.utils.data import DataLoader, Subset

import torchaudio
from model.datasets import KpopVocalDataset
from model.heads import PresenceHead
from model.encoders import MuQEncoderWrapper, FusedEncoder, FusedEncoderWithECAPA

from tqdm import tqdm
from muq import MuQ
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------
# CLI args
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Path that contains <group>/<member>/train/Isolated_Vocals")
    ap.add_argument("--group", type=str, required=True, help="Group folder name under root")
    ap.add_argument("--sr_in", type=int, default=44100, help="Expected input sample rate of your files")
    ap.add_argument("--sr_out", type=int, default=24000, help="ECAPA target sample rate")
    ap.add_argument("--chunk-sec", type=float, default=2.0, help="Chunk length in seconds for training")
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
    
def buildTrainingCache(groupDir: str, srOut: int, numWorkers: int = 8) -> str:
    """
    Creates/uses: <groupDir>/training_cache/sr_<srOut>/
    Resamples vocals/leading/backing wavs into the cache folder.

    - Multithreaded resample+save
    - tqdm progress bar (total = #songs/files to cache)
    - Skips any file that already exists in cacheDir

    Returns the cache directory path.
    """
    groupDir = str(groupDir)
    cacheDir = os.path.join(groupDir, "training_cache", f"sr_{srOut}")
    os.makedirs(cacheDir, exist_ok=True)

    # Only these patterns are used by _find_audio_triplet() in this script
    patterns = ["*_vocals.wav", "*_leading_vocals.wav", "*_backing_vocals.wav"]

    # Gather candidate input files
    inFiles = []
    for pat in patterns:
        inFiles.extend(Path(groupDir).glob(pat))

    if not inFiles:
        print(f"[cache] No matching WAV stems found in {groupDir} (patterns={patterns}).")
        return cacheDir

    # Build a work list that excludes already-cached outputs
    work = []
    for inPath in inFiles:
        outPath = os.path.join(cacheDir, inPath.name)
        if os.path.exists(outPath):
            continue
        work.append((str(inPath), outPath))

    print(f"[cache] Cache dir: {cacheDir}")
    print(f"[cache] Found {len(inFiles)} wavs. Need to create {len(work)} cached wavs (skipping {len(inFiles) - len(work)}).")

    if not work:
        return cacheDir

    # Worker wrapper so exceptions don't kill the whole pool
    def _do_one(inPath: str, outPath: str):
        _resample_and_save(inPath, outPath, srOut)
        return os.path.basename(outPath)

    # Threaded execution + tqdm progress
    # NOTE: torchaudio resample is CPU-heavy; threads help mainly if you're I/O bound.
    # If you find it CPU-bound, switch to ProcessPoolExecutor (but then _resample_and_save must be picklable).
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, int(numWorkers))) as ex:
        futures = [ex.submit(_do_one, inP, outP) for inP, outP in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Caching (resample+save)", unit="file"):
            try:
                fut.result()
            except Exception as e:
                failures += 1
                # You can print less frequently if you want
                print(f"[cache] Failed: {e}")

    if failures:
        print(f"[cache] Done with {failures} failures. (See logs above.)")
    else:
        print("[cache] Done. All cached successfully.")

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
    def __init__(self, base_ds: KpopVocalDataset):
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
def train_epoch(encoder, head, loader, device, optimizer, 
                thr=0.5, use_amp=True,
                ctx_frac: float = 0.25, is_phase2: bool = False
                ):
    """
    encoder: ECAPA model (SpeechBrain)
    head:   multi-task head taking fused embedding -> dict of logits
    loader: yields (wavs, y_main, w_main, y_harm, w_harm, y_ad, w_ad)
    """
    encoder.eval() # Extract embeddings under no_grad by default
    head.train()
    sanity = {
        "pred_pos_sum": 0.0, "true_pos_sum": 0.0, "n_samples": 0,
        "abs_sum": 0.0, "abs_count": 0, "abs_max": 0.0,
        "abs_sample": [],
    }
    
    scaler = torch.amp.GradScaler(device=device, enabled=(use_amp and device.type == "cuda"))
    
    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    for batch in tqdm(loader, desc="Train", leave=False):
        wavs, y_main, w_main, stem_weight = batch
        
        wavs = wavs.to(device, non_blocking=True)  # (B,1,T)
        y_main = y_main.to(device, non_blocking=True)
        w_main = w_main.to(device, non_blocking=True)
        stem_weight = stem_weight.to(device, non_blocking=True)
        
        wavs_encoder = wavs.squeeze(1) if (wavs.ndim == 3 and wavs.size(1) == 1) else wavs
        
        amp_ctx = torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda"))
        
        with torch.no_grad():
            emb_fused = encoder.encode_batch(wavs_encoder, ctx_frac=ctx_frac)  # (B,Df)
        
        optimizer.zero_grad(set_to_none=True)
        
        with amp_ctx:
            logits_main = head(emb_fused)  # (B, C)
            update_sanity_stats(sanity, logits_main, y_main, thr=thr)

            # weighted BCE per class
            loss_w = bce(logits_main, y_main) * w_main  # (B, C)

            # normalize by weight mass per sample (NOT by constant C)
            denom = w_main.sum(dim=1).clamp(min=1e-6)   # (B,)
            loss_per_sample = loss_w.sum(dim=1) / denom # (B,)

            # apply stem weight (broadcast-safe)
            loss = (loss_per_sample * stem_weight).mean()

            # small logit regularizer
            loss = loss + 1e-4 * logits_main.pow(2).mean()
            
        # Total loss with task weights
        loss_w = bce(logits_main, y_main) * w_main
        
        # Mean over classes
        loss_main_per_sample = loss_w.sum(dim=1) / (w_main.size(1) + 1e-6)
        # Apply stemWEight
        loss_main_per_sample = loss_main_per_sample * stem_weight

        loss = loss_main_per_sample.mean() + 1e-4 * logits_main.pow(2).mean()
 
        if is_phase2:
            probs = torch.sigmoid(logits_main)
            # probs shape: [B, num_members]
            kPred = probs[:, :head.num_members].sum(dim=1)         # expected positives per window
            kMax = 3.0

            # only penalize if we exceed target by a margin (hinge)
            margin = 0.4                      # allow some slack
            excess = torch.relu(kPred - (kMax + margin))

            # weight it: stronger on clean windows, weaker on ambiguous ones
            # cleanScore is per-window in your dataset code; if you don't have per-sample,
            # just use a scalar based on batch average of your clean/ambiguous categories.
            lambdaCard = 0.02                 # start small; this is usually enough once neg_cap is fixed
            cardLoss = (excess ** 2).mean()

            loss = loss + (lambdaCard * cardLoss)
        
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        # --- metrics --- 
        bsz = wavs.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        f1, prec, rec = multilabel_micro_f1(logits_main.detach(), y_main, thr=thr)
        total_f1 += f1 * bsz
        total_prec += prec * bsz
        total_rec += rec * bsz
        total_subset_acc += subset_accuracy(logits_main.detach(), y_main, thr=thr) * bsz
    
    avg_loss = total_loss / max(1, total_count)
    avg_f1   = total_f1 / max(1, total_count)
    avg_p    = total_prec / max(1, total_count)
    avg_r    = total_rec / max(1, total_count)
    avg_subset = total_subset_acc / max(1, total_count)
    sanity_out = finalize_sanity_stats(sanity)
    return avg_loss, {"micro_f1": avg_f1, "precision": avg_p, "recall": avg_r, "subset_acc": avg_subset, "sanity": sanity_out}

@torch.no_grad()
def eval_epoch(encoder, head, loader, device, thr=0.5, use_amp=True, ctx_frac: float = 0.25):
    """
    encoder: MuQEncoderWrapper (frozen), encode_batch(wavs, ctx_frac) -> (emb_main, emb_ctx)
             where emb_main/emb_ctx are either:
               - OLD style: (B,1,D)  (then we squeeze)
               - NEW style: (B,D)    (then we don't)
    head: PresenceHead, forward(emb_fused) -> logits_main (B, num_members+1)
    loader yields: (wavs, y_main, w_main, stem_weight)
    """
    encoder.eval()
    head.eval()

    sanity = {
        "pred_pos_sum": 0.0, "true_pos_sum": 0.0, "n_samples": 0,
        "abs_sum": 0.0, "abs_count": 0, "abs_max": 0.0,
        "abs_sample": [],
    }

    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    last_logits = None
    last_y = None

    # helper: squeeze (B,1,D)->(B,D) but leave (B,D) unchanged
    def _squeeze_b1d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.size(1) == 1:
            return x.squeeze(1)
        return x

    # helper: ensure wavs fed to encoder are (B,T)
    def _prepare_wavs(wavs: torch.Tensor) -> torch.Tensor:
        if wavs.ndim == 3 and wavs.size(1) == 1:
            return wavs.squeeze(1)
        if wavs.ndim == 2:
            return wavs
        raise ValueError(f"Unexpected wavs shape: {tuple(wavs.shape)}")

    for batch in tqdm(loader, desc="Eval", leave=False):
        wavs, y_main, w_main, stem_weight = batch

        wavs = wavs.to(device, non_blocking=True)
        y_main = y_main.to(device, non_blocking=True)
        w_main = w_main.to(device, non_blocking=True)
        stem_weight = stem_weight.to(device, non_blocking=True)

        wavs_encoder = _prepare_wavs(wavs)

        # --- encode ---
        enc_out = encoder.encode_batch(wavs_encoder, ctx_frac=ctx_frac)

        # NEW path: a single embedding tensor
        if isinstance(enc_out, torch.Tensor):
            emb_fused = _squeeze_b1d(enc_out)

        else:
            raise TypeError(
                f"encode_batch returned unexpected type/shape: {type(enc_out)}"
            )

        # --- forward + loss ---
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            logits = head(emb_fused)  # (B,C)

            update_sanity_stats(sanity, logits, y_main, thr=thr)

            # weighted BCE per class
            loss_w = bce(logits, y_main) * w_main  # (B,C)

            # normalize by weight mass per sample
            loss_per_sample = loss_w.sum(dim=1) / w_main.sum(dim=1).clamp(min=1e-6)  # (B,)

            # apply stem weight
            loss = (loss_per_sample * stem_weight).mean()

        bsz = wavs.size(0)
        total_loss += float(loss.item()) * bsz
        total_count += bsz

        f1, prec, rec = multilabel_micro_f1(logits, y_main, thr=thr)
        total_f1 += f1 * bsz
        total_prec += prec * bsz
        total_rec += rec * bsz
        total_subset_acc += subset_accuracy(logits, y_main, thr=thr) * bsz

        last_logits = logits
        last_y = y_main

    # Optional debug from last batch
    if last_logits is not None and last_y is not None:
        pred_pos = (torch.sigmoid(last_logits) > thr).float().sum(dim=1).mean().item()
        true_pos = last_y.sum(dim=1).mean().item()
        print(f"[DEBUG] avg_pred_pos={pred_pos:.2f} avg_true_pos={true_pos:.2f}")

    avg_loss = total_loss / max(1, total_count)
    avg_f1 = total_f1 / max(1, total_count)
    avg_p = total_prec / max(1, total_count)
    avg_r = total_rec / max(1, total_count)
    avg_subset = total_subset_acc / max(1, total_count)

    sanity_out = finalize_sanity_stats(sanity)
    return avg_loss, {
        "micro_f1": avg_f1,
        "precision": avg_p,
        "recall": avg_r,
        "subset_acc": avg_subset,
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

# Main
def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high') # Small speed bump on Ampere
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    group_dir = os.path.join(args.root, "training_data", args.group)
    cache_dir = buildTrainingCache(group_dir, srOut=args.sr_out)
    
    # Dataset & split
    ds_phase1 = KpopVocalDataset(group_dir, sr_out=args.sr_out, context_seconds=args.chunk_sec, group_name=args.group, audio_dir=cache_dir)
    
    print("Classes:", ds_phase1.class_map.idx_to_name)
    print("len(full_ds) =", len(ds_phase1))
    
    n_val = max(1, int(len(ds_phase1) * args.val_split))
    n_train = len(ds_phase1) - n_val
    print("train/val =", n_train, n_val)
    
    valSongs, tierInfo = ds_phase1.pickValidationSongsStratified(
        valEasy=2, valMed=2, valHard=1, domFracSoloThreshold=0.95
    )

    trainIdx, valIdx = [], []
    for i, sample in enumerate(ds_phase1.samples):
        songName = sample[0]
        if songName in valSongs:
            valIdx.append(i)
        else:
            trainIdx.append(i)

    train_ds = Subset(ds_phase1, trainIdx)
    val_ds   = Subset(ds_phase1, valIdx)

    print("Validation songs:", tierInfo["valSongs"])
    print("Tier sizes:", {k: len(v) for k, v in tierInfo.items() if k in ["easy","med","hard"]})
    
    # Load MuQ ECAPA encoder (pretraiened)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq.to(device).eval()
    
    encoder = MuQEncoderWrapper(
        muq_model=muq,
        pooling="topk",
        debug=False
    ).to(device)
    fused_encoder = FusedEncoder(encoder).to(device)
    
    dummy = torch.zeros(1, int(args.chunk_sec * args.sr_out), device=device)

    # Ensure (B, T)
    if dummy.ndim == 3 and dummy.size(1) == 1:
        dummy = dummy.squeeze(1)

    with torch.no_grad():
        fused = fused_encoder.encode_batch(dummy, ctx_frac=0.2)  # (1, D_fused)
        fused_dim = fused.shape[1]  
        
    num_members = ds_phase1.num_members
        
    head = PresenceHead(
        emb_dim_fused=fused_dim,
        num_members=num_members,
        hidden=256,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.3,   # multiply LR by 0.3 when plateau
        patience=2,   # wait 1 epoch with no improvement
    )
    
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.group}_muq_head.pt")
    
    eval_thr = getattr(args, "eval_thr", 0.7)
    # Checks if you can skip stage 1
    run_stage1 = check_run_stage1(model=head, ckpt_path=ckpt_path, skip_stage1=args.skip_stage1, device=device)
    
    if run_stage1:
        # Train phase 1
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            tr_loss, tr_metrics = train_epoch(fused_encoder, head, train_loader, device, optimizer, thr=eval_thr)
            va_loss, va_metrics = eval_epoch(fused_encoder, head, val_loader, device, thr=eval_thr)
            tr_s = tr_metrics["sanity"]
            va_s = va_metrics["sanity"]

            print(f"Sanity Train: avg_pred_pos={tr_s['avg_pred_pos']:.2f} vs avg_true_pos={tr_s['avg_true_pos']:.2f} | "
                f"logit| mean={tr_s['mean_abs']:.2f} p95={tr_s['p95_abs']:.2f} max={tr_s['max_abs']:.2f}")
            print(f"Sanity Val:   avg_pred_pos={va_s['avg_pred_pos']:.2f} vs avg_true_pos={va_s['avg_true_pos']:.2f} | "
                f"logit| mean={va_s['mean_abs']:.2f} p95={va_s['p95_abs']:.2f} max={va_s['max_abs']:.2f}")
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(va_metrics["micro_f1"])
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != old_lr:
                print(f"[LR Scheduler] Reducing LR: {old_lr} → {new_lr}")
        
            print(
                f"Train - loss: {tr_loss:.4f} | micro-F1: {tr_metrics['micro_f1']:.4f} "
                f"(P {tr_metrics['precision']:.3f}, R {tr_metrics['recall']:.3f}) | subset-acc: {tr_metrics['subset_acc']:.4f}"
            )
            print(
                f"Val   - loss: {va_loss:.4f} | micro-F1: {va_metrics['micro_f1']:.4f} "
                f"(P {va_metrics['precision']:.3f}, R {va_metrics['recall']:.3f}) | subset-acc: {va_metrics['subset_acc']:.4f}"
            )

            if va_metrics["micro_f1"] > best_acc:
                best_acc = va_metrics["micro_f1"]
                torch.save({
                    "state_dict": head.state_dict(),
                    "classes": ds_phase1.class_map.idx_to_name,
                    "emb_dim_fused": fused_dim,
                    "sr": args.sr_out,
                    "chunk_sec": args.chunk_sec,
                    "group": args.group,
                    "eval_thr": eval_thr,
                }, ckpt_path)
                print(f"✅ Saved best head to: {ckpt_path} (acc={best_acc:.4f})")
            
        print("\nDone. Best val acc:", best_acc)
    else:
        print("⚠️  WARNING: Stage 1 skipped — using pretrained head")
    
    mine_ds = MiningViewDataset(ds_phase1)
    mine_loader = DataLoader(mine_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # 2) mine hard centers using the trained phase 1 model
    hard_centers = hard_miner(
        encoder=fused_encoder,
        head=head,
        loader=mine_loader,   # based on the 2s dataset
        thr=eval_thr,
        device=device,
    )
    print("Hard centers mined:", len(hard_centers))
    
    # Build once
    ds_phase2_base = KpopVocalDataset(group_dir, sr_out=args.sr_out, context_seconds=0.4, group_name=args.group, audio_dir=cache_dir, is_phase2=True)
    
    # Build first focused loader from inital mind centers from Phase 1
    focus_ds = FocusCentersDataset(ds_phase2_base, hard_centers)
    focus_loader = DataLoader(
        focus_ds,
        batch_size=args.batch_size,
        shuffle=True,                 # IMPORTANT: shuffle hard examples
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Keep same to compare aacross epochs
    val2_ds = KpopVocalDataset(group_dir, sr_out=args.sr_out, context_seconds=args.short_chunk_sec,
                           group_name=args.group, audio_dir=cache_dir, is_phase2=True)

    n_val2 = max(1, int(len(val2_ds) * args.val_split))
    n_train2 = len(val2_ds) - n_val2
    _, val2_subset = random_split(val2_ds, [n_train2, n_val2])

    val2_loader = DataLoader(
        val2_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    
    head2 = PresenceHead(emb_dim_fused=fused_dim, num_members=num_members).to(device)
    head2.load_state_dict(head.state_dict())  # start from phase1 solution
    opt2 = torch.optim.AdamW(head2.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)

    phase2_epochs = max(1, args.epochs // 2)  # e.g., 5 if phase1 was 10
    best2 = -1.0
    
    prev_p95_abs = None
    rising_p95_streak = 0
    
    REMINE_EVERY = 1000
    TOP_HARD = 15000 # Cap size to keep training tight
    PAIN_KEEP = 2.0 # Min pain threshold
    THR2 = 0.5 # Phase-2 threshold for mining and eval
    
    for epoch in range(1, phase2_epochs + 1):
        print(f"\n[Phase2] Epoch {epoch}/{phase2_epochs}")
        tr2_loss, tr2_metrics = train_epoch(fused_encoder, head2, focus_loader, device, opt2, thr=THR2, is_phase2=True)
        va2_loss, va2_metrics = eval_epoch(fused_encoder, head2, val2_loader, device, thr=THR2)

        print(f"[Phase2] Train loss {tr2_loss:.4f} microF1 {tr2_metrics['micro_f1']:.4f}")
        print(f"[Phase2] Val   loss {va2_loss:.4f} microF1 {va2_metrics['micro_f1']:.4f}")

        # === SCREAMING GUARDRAILS ===
        # Use the sanity stats you already compute in train_epoch/eval_epoch (mean_abs, p95_abs, max_abs, avg_pred_pos, avg_true_pos)
        # Assuming train_epoch returns them in tr2_metrics["sanity"] (if not, I’ll show where to add it)
        if "sanity" in tr2_metrics:
            s = tr2_metrics["sanity"]
            mean_abs = float(s.get("mean_abs", 0.0))
            p95_abs  = float(s.get("p95_abs", 0.0))
            max_abs  = float(s.get("max_abs", 0.0))
            avg_pred = float(s.get("avg_pred_pos", 0.0))
            avg_true = float(s.get("avg_true_pos", 1e-6))

            # p95 trend check (relapse)
            if prev_p95_abs is not None and p95_abs > prev_p95_abs + 0.5:
                rising_p95_streak += 1
            else:
                rising_p95_streak = 0
            prev_p95_abs = p95_abs

            # "Insane" definitions: these catch your old 75+ p95 meltdown early
            if (p95_abs >= 20.0) or (max_abs >= 60.0) or (rising_p95_streak >= 2):
                print(
                    f"\n\n🚨🚨🚨 LOGIT MELTDOWN DETECTED 🚨🚨🚨\n"
                    f"p95_abs={p95_abs:.2f} mean_abs={mean_abs:.2f} max_abs={max_abs:.2f} "
                    f"(rising_streak={rising_p95_streak})\n"
                    f"This usually means the model is saturating and will start spamming positives.\n"
                )

            # Prediction-rate sanity: stop if it starts screaming "everyone is singing"
            ratio = avg_pred / max(avg_true, 1e-6)
            if ratio >= 2.4:
                print(
                    f"\n\n🚨🚨🚨 FP-SPAM RELAPSE 🚨🚨🚨\n"
                    f"avg_pred_pos={avg_pred:.2f} avg_true_pos={avg_true:.2f} ratio={ratio:.2f}\n"
                    f"The model is predicting far too many singers per window.\n"
                )

            # Also catch the opposite: model becomes too timid
            if ratio <= 0.55:
                print(
                    f"\n⚠️  Phase2 looks overly timid: avg_pred_pos={avg_pred:.2f} avg_true_pos={avg_true:.2f} ratio={ratio:.2f}\n"
                    f"Not necessarily fatal, but it can mean threshold is too high or negatives too strong.\n"
                )

        # Save best
        if va2_metrics["micro_f1"] > best2:
            best2 = va2_metrics["micro_f1"]
            torch.save({
                "state_dict": head2.state_dict(),
                "classes": ds_phase1.class_map.idx_to_name,
                "emb_dim": fused_dim,
                "sr": args.sr_out,
                "chunk_sec": 0.4,
                "group": args.group,
                "eval_thr": THR2,
                "note": "phase2_refinement"
            }, os.path.join(args.save_dir, f"{args.group}_muq_head_phase2.pt"))
            print(f"✅ Saved best Phase2 head (microF1={best2:.4f})")
        
        # === RE-MINE EVERY 3 EPOCHS ===
        if (epoch % REMINE_EVERY) == 0:
            print("\n[Phase2] Re-mining hard centers...")
            # Mine on the *0.4s base dataset* using the current head2
            mine_ds = MiningViewDataset(ds_phase2_base)  # wrapper provides (train_tuple, song, kind, center_frame)
            mine_loader = DataLoader(
                mine_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False
            )

            new_hard = hard_miner(
                encoder=fused_encoder,
                head=head2,
                loader=mine_loader,
                device=device,
                thr=THR2,
                pain_threshold=PAIN_KEEP,
                max_hard=TOP_HARD * 3,  # allow extra before filtering/dedup
            )

            # Dedup and keep top N
            seen = set()
            dedup = []
            for c in new_hard:
                key = (c["song"], c["kind"], int(c["center_frame"]))
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(c)
                if len(dedup) >= TOP_HARD:
                    break

            hard_centers = dedup
            print(f"[Phase2] Hard centers refreshed: {len(hard_centers)}")

            # Rebuild focus loader with the refreshed set
            focus_ds = FocusCentersDataset(ds_phase2_base, hard_centers)
            focus_loader = DataLoader(
                focus_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
    
if __name__ == "__main__":
    main()