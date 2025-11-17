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

import os, argparse,  random, glob, json
from dataclasses import dataclass
from typing import List, Tuple, Dict 

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
import numpy as np
from torchaudio.transforms import Resample

from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# CLI args
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Path that contains <group>/<member>/train/Isolated_Vocals")
    ap.add_argument("--group", type=str, required=True, help="Group folder name under root")
    ap.add_argument("--sr_in", type=int, default=22050, help="Expected input sample rate of your files")
    ap.add_argument("--sr_ecapa", type=int, default=16000, help="ECAPA target sample rate")
    ap.add_argument("--chunk-sec", type=float, default=2.0, help="Chunk length in seconds for training")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-dir", type=str, default="./checkpoints")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval_thr", type=float, default=0.5)
    ap.add_argument("--freeze-ecapa", action="store_true", help="Freeze ECAPA encoder parameters")
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
    
@dataclass
class ClassMap:
    idx_to_name: List[str]
    name_to_idx: Dict[str, int]
    
# ---------------------------
# Dataset: loads files, makes fixed-length waveform chunks, includes "silence"
# ---------------------------
class KpopVocalDataset(Dataset):
    """
    Builds 2s (chunk_sec) training examples from full-song vocals + frame label JSON.

    Directory layout (per group):
        ./training_data/<group>/
            <SongName>_vocals.wav / .mp3 / .flac / .m4a
            <SongName>_frame_labels.json

    JSON format (per song):
        {
            "group": "IVE",
            "song": "Hypnosis",
            "members": [...],              # length = num_members (no silence)
            "chunkDurationMs": 40,
            "numChunks": T,
            "presence": [ [0/1,...], ... ] # shape [T, num_members]
            ... (isAdlib, lead, isRepeat, etc. – ignored for now)
        }

    Classes used for training:
        [members..., "silence"]
        => label size = num_members + 1, last index is silence.
    """
    def __init__(self, group_dir: str, sr_out: int,
                chunk_sec: float, num_workers: int, min_song_sec: float = 4.0,
                pitch_prob: float = 0.3, pitch_semitones_range: Tuple[float, float] = (-2.0, 2.0),
                window_hop_ratio: float = 0.5, presence_thresh=0.4,
                alpha_lead=1.0, alpha_adlib = 0.5, min_weight = 0.2, max_weight = 2.0):
        super().__init__()
        self.group_dir = group_dir
        self.sr_out = sr_out
        self.chunk_sec = chunk_sec
        self.chunk_len = int(round(chunk_sec * sr_out))
        self.min_song_sec = min_song_sec

        self.pitch_semitones_range = pitch_semitones_range
        self.window_hop_ratio = window_hop_ratio
        
        self.presence_thresh = presence_thresh
        self.alpha_lead = alpha_lead
        self.alpha_adlib = alpha_adlib
        self.min_weight = min_weight
        self.max_weight = max_weight

        # ----------------------------
        # 1) Discover all JSON label files
        # ----------------------------
        json_pattern = os.path.join(group_dir, "*_frame_labels.json")
        json_files = sorted(glob.glob(json_pattern))
        if not json_files:
            raise RuntimeError(f"No *_frame_labels.json files found under {group_dir}")
        
        # ----------------------------
        # 2) Load first JSON to define members/classes
        # ----------------------------
        with open(json_files[0], "r", encoding="utf-8") as f:
            meta0 = json.load(f)
        members = meta0["members"]
        self.chunk_duration_ms = meta0["chunkDurationMs"]  # should be 40
        self.num_members = len(members)
        
        self.classes = members + ["silence"]
        self.silence_idx = len(self.classes) - 1
        self.class_map = ClassMap(
            idx_to_name=self.classes,
            name_to_idx={name: i for i, name in enumerate(self.classes)}
        )
        
        # ----------------------------
        # 3) Build index of (audio_path, start_sample_out, label_vec)
        # ----------------------------
        self.samples: List[Tuple[str, int, np.ndarray]] = []
        frame_ms = float(self.chunk_duration_ms)
        frames_per_window = int(round(self.chunk_sec * 1000.0 / frame_ms))  # e.g. 2.0s / 40ms = 50
        if frames_per_window <= 0:
            raise ValueError("frames_per_window computed as <=0, check chunk_sec and chunkDurationMs")
        
        hop_frames = max(1, int(round(frames_per_window * self.window_hop_ratio)))

        print(f"[KpopFrameDataset] Frames/window={frames_per_window}, hop_frames={hop_frames}")
        print(f"[KpopFrameDataset] Members={members} (+ 'silence')")
        
        for jpath in json_files:
            with open(jpath, "r", encoding="utf-8") as f:
                meta = json.load(f)

            song_name = meta["song"]
            members_j = meta["members"]
            # Basic sanity check: same member ordering across songs
            if members_j != members:
                raise ValueError(
                    f"Members mismatch in {jpath}: {members_j} vs {members}"
                )
            
            num_chunks = meta["numChunks"]
            presence = np.asarray(meta["presence"], dtype=np.int32)  # shape: [T, num_members]
            if presence.shape[0] != num_chunks or presence.shape[1] != self.num_members:
                raise ValueError(f"presence shape mismatch in {jpath}")
            lead_arr = np.asarray(meta["lead"], dtype=np.int32)      # [T, C]
            adlib_arr = np.asarray(meta["isAdlib"], dtype=np.int32)  # [T, C]
            
            song_dur_sec = num_chunks * frame_ms / 1000.0
            if song_dur_sec < self.min_song_sec or num_chunks < frames_per_window:
                print(f"[KpopFrameDataset] Skipping {song_name}: too short ({song_dur_sec:.2f}s)")
                continue
            
            # Find the corresponding vocals audio file
            audio_path = self._find_vocals_audio(song_name)
            if audio_path is None:
                print(f"[KpopFrameDataset] No audio found for song {song_name}, skipping.")
                continue
            
            # Slide window over frame indices
            max_start_frame = num_chunks - frames_per_window
            for start_frame in range(0, max_start_frame + 1, hop_frames):
                end_frame = start_frame + frames_per_window
                
                window_presence = presence[start_frame:end_frame]  # [F, C]
                window_lead = lead_arr[start_frame: end_frame]
                window_adlib = adlib_arr[start_frame:end_frame]
                
                # Fractions over this window (0..1)
                frames_active = window_presence.mean(axis=0) # (C,)
                lead_frac = window_lead.mean(axis=0) # (C,)
                adlib_frac = window_adlib.mean(axis=0) # (C,)
                
                # --- multi-hot label_vec ---
                label_vec = np.zeros(len(self.classes), dtype=np.float32)
                importance_vec = np.ones(len(self.classes), dtype=np.float32)
                
                total_active_frames = float(frames_active.sum())
                
                # -------- Case 1: true silence (no one sings at any 40ms frame) --------
                if total_active_frames == 0.0:
                    label_vec[self.silence_idx] = 1.0
                    importance_vec[:] = 0.0
                    importance_vec[self.silence_idx] = 1.0
                    
                # --- Case 2: at least someone clearly active ---
                # -------- Case 2: at least one singer is active in this 2s window --------
                else:
                    # You can tune this: how many frames needed for a singer to "count"?
                    # If you want "any appearance counts", set min_frames_active = 1
                    # If you want >= 3 frames (~120 ms) use that instead.
                    # Here we tie it to presence_thresh as a FRACTION of the window:
                    min_frames_active = max(
                        1,
                        int(round(self.presence_thresh * frames_per_window))
                    )
                    # e.g. if presence_thresh = 0.05 and frames_per_window=50, then >=3 frames

                    active_any = (frames_active >= min_frames_active).astype(np.float32)

                    # Failsafe: if nobody passes the threshold, at least mark the most active singer
                    if active_any.sum() == 0:
                        main_idx = int(frames_active.argmax())
                        active_any[main_idx] = 1.0

                    # Fill labels: singers + no silence
                    label_vec[:self.num_members] = active_any
                    label_vec[self.silence_idx] = 0.0

                    # Importance weights per singer, using your existing lead/adlib logic
                    importance_singers = (
                        1.0
                        + self.alpha_lead * lead_frac
                        - self.alpha_adlib * adlib_frac
                    )
                    importance_singers = np.clip(
                        importance_singers, self.min_weight, self.max_weight
                    )
                    importance_vec[:self.num_members] = importance_singers

                    # Silence is not important during singing windows
                    importance_vec[self.silence_idx] = 0.5
        
                # Map start_frame -> start sample index at sr_out
                start_time_sec = (start_frame * frame_ms) / 1000.0
                start_sample_out = int(round(start_time_sec * self.sr_out))

                self.samples.append((audio_path, start_sample_out, label_vec, importance_vec))

        if not self.samples:
            raise RuntimeError("No training windows built. Check your JSON/audio alignment.")

        
        total_labels = np.zeros(len(self.classes), dtype=np.float64)
        total_weights = np.zeros(len(self.classes), dtype=np.float64)

        for _, _, label_vec, importance_vec in self.samples:
            total_labels += label_vec
            total_weights += importance_vec
            
        print(f"Total labels: {total_labels / len(self.samples)}")
        print(f"Total weights: {total_weights / len(self.samples)}")
        
        n_silence = sum(label_vec[self.silence_idx] == 1.0 for _, _, label_vec, _ in self.samples)
        print("silence windows:", n_silence, "/", len(self.samples))
        
        self.base_samples = list(self.samples)
        self.augmented_samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        
        # ----------------------------
        # 4) Simple audio cache to avoid re-loading the same song
        # ----------------------------
        self._cache_path = None
        self._cache_wave = None  # (1, T_out)
        self._resamplers: Dict[int, Resample] = {}
        
        if pitch_prob > 0.0:
            self._build_pitch_augmented_subset(pitch_prob, num_workers)

        print(f"[KpopFrameDataset] total windows: {len(self.samples)}")
    
    def _find_vocals_audio(self, song_name: str) -> str:
        """
        Try to find <song>_vocals.(wav|flac|mp3|m4a) under group_dir.
        """
        exts = [".wav"]
        for ext in exts:
            cand = os.path.join(self.group_dir, f"{song_name}_vocals{ext}")
            if os.path.exists(cand):
                return cand
        return None
    
    def _get_resampler(self, sr_src: int) -> Resample:
        if sr_src not in self._resamplers:
            self._resamplers[sr_src] = Resample(sr_src, self.sr_out)
        return self._resamplers[sr_src]
    
    def _load_song_wave(self, path: str) -> torch.Tensor:
        """
        Load + mono + resample a *whole song* once, cache it.
        Returns Tensor (1, T_out).
        """
        if self._cache_path == path and self._cache_wave is not None:
            return self._cache_wave

        wav, sr_src = torchaudio.load(path)  # (C, T_src)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr_src != self.sr_out:
            resampler = self._get_resampler(sr_src)
            wav = resampler(wav)

        self._cache_path = path
        self._cache_wave = wav  # (1, T_out)
        return wav
    
    def _build_pitch_augmented_subset(self, pitch_aug_ratio: float, max_workers: int = 4):
        """
        Precompute pitch-shifted copies of a subset of base samples.

        Stores them in self.augmented_samples as (audio_chunk, label_tensor, weight_tensor).
        """
        num_base = len(self.base_samples)
        if num_base == 0:
            return

        num_aug = max(1, int(num_base * pitch_aug_ratio))
        # randomly choose which base samples to augment
        indices = random.sample(range(num_base), num_aug)

        print(f"[KpopVocalDataset] Precomputing {num_aug} pitch-augmented chunks "
              f"out of {num_base} base windows (ratio={pitch_aug_ratio:.2f})")

        def worker(base_idx: int):
            audio_path, start_sample_out, label_vec, importance_vec = self.base_samples[base_idx]

            # Load and resample this song independently (no cached state)
            wav, sr_src = torchaudio.load(audio_path)   # (C, T_src)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr_src != self.sr_out:
                resampler = Resample(sr_src, self.sr_out)
                wav = resampler(wav)                    # (1, T_out)

            end = start_sample_out + self.chunk_len
            if end > wav.shape[-1]:
                pad = end - wav.shape[-1]
                seg = torch.nn.functional.pad(wav[..., start_sample_out:], (0, pad))
            else:
                seg = wav[..., start_sample_out:end]    # (1, chunk_len)

            # Apply pitch shift ONCE here
            seg_aug = self._apply_pitch_shift(seg)
            if seg_aug is None:
                return None

            lab = torch.from_numpy(label_vec).to(torch.float32)
            wts = torch.from_numpy(importance_vec).to(torch.float32)
            return (seg_aug, lab, wts)

        results = []
        # If max_workers <= 1, just run sequentially (handy for debugging)
        if max_workers is None or max_workers <= 1:
            for idx in tqdm(indices, desc="Precomputing pitch augs (1 worker)"):
                out = worker(idx)
                if out is not None:
                    results.append(out)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(worker, idx) for idx in indices]
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Precomputing pitch augs"):
                    out = fut.result()
                    if out is not None:
                        results.append(out)

        self.augmented_samples.extend(results)
        print(f"[KpopVocalDataset] Done. Augmented samples: {len(self.augmented_samples)}")
    
    def _apply_pitch_shift(self, seg: torch.Tensor) -> torch.Tensor:
        """
        Apply a random pitch shift to seg (1, T). Always attempts to shift;
        if anything fails, returns seg unchanged.
        """
        n_steps = random.uniform(-2.0, 2.0)  # e.g. ±2 semitones
        try:
            if hasattr(torchaudio.functional, "pitch_shift"):
                seg2 = torchaudio.functional.pitch_shift(seg, self.sr_out, n_steps=n_steps)
            else:
                effects = [
                    ["pitch", f"{n_steps * 100.0}"],
                    ["rate", f"{self.sr_out}"],
                ]
                seg2, _ = torchaudio.sox_effects.apply_effects_tensor(seg, self.sr_out, effects)

            # Ensure exact length chunk_len
            if seg2.shape[-1] >= self.chunk_len:
                return seg2[..., : self.chunk_len]
            else:
                pad = self.chunk_len - seg2.shape[-1]
                return torch.nn.functional.pad(seg2, (0, pad))
        except Exception:
            return seg
        
    # ---------- Dataset API ----------
    def __len__(self) -> int:
        # base windows + precomputed augmented windows
        return len(self.base_samples) + len(self.augmented_samples)

    def __getitem__(self, idx: int):
        base_count = len(self.base_samples)

        # ---- Case 1: base sample (no pitch shift) ----
        if idx < base_count:
            audio_path, start_sample_out, label_vec, importance_vec = self.base_samples[idx]

            wav = self._load_song_wave(audio_path)  # (1, T_out)
            end = start_sample_out + self.chunk_len
            if end > wav.shape[-1]:
                pad = end - wav.shape[-1]
                seg = torch.nn.functional.pad(wav[..., start_sample_out:], (0, pad))
            else:
                seg = wav[..., start_sample_out:end]  # (1, chunk_len)

            labels = torch.from_numpy(label_vec).to(torch.float32)
            weights = torch.from_numpy(importance_vec).to(torch.float32)
            return seg, labels, weights

        # ---- Case 2: augmented sample (precomputed pitch-shifted) ----
        aug_idx = idx - base_count
        seg_aug, lab, wts = self.augmented_samples[aug_idx]
        # Return copies so we don't accidentally mutate the cached tensors
        return seg_aug.clone(), lab.clone(), wts.clone()

# ---------------------------
# Model: ECAPA encoder (frozen or trainable) + linear head
# ---------------------------
class MultiLabelHead(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(emb_dim, num_classes)
    
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)
  
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

def subset_accuracy(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    """
    Exact-set match accuracy: 1 only if all labels match for a sample.
    """
    preds = binarize_logits(logits, thr)
    t_cpu = targets.detach().to("cpu")
    eq = (preds == t_cpu).all(dim=1)  # (B,)
    return eq.float().mean().item()
    
    
# ---------------------------
# Training / evaluation
# ---------------------------
def train_epoch(encoder, head, loader, device, optimizer, thr=0.5, use_amp=True):
    encoder.eval() # Extract embeddings under no_grad by default
    head.train()
    
    scaler = torch.amp.GradScaler(device=device, enabled=(use_amp and device.type == "cuda"))
    
    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0

    # minor speedups on CUDA
    # torch.backends.cudnn.benchmark = True
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    for wavs, labels, weights in tqdm(loader, desc="Train", leave=False):
        wavs = wavs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        weights = weights.to(device, non_blocking=True)
        
        if wavs.ndim == 3:
            wavs = wavs.squeeze(1)
        elif wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)
        
        with torch.no_grad():
             # SpeechBrain ECAPA expects (B, T) or (B, 1, T) tensors; encode_batch handles both.
             # print(f"[DEBUG] wavs.shape = {wavs.shape}, dtype={wavs.dtype}, device={wavs.device}")
             emb = encoder.encode_batch(wavs).squeeze(1) # (B, D)
        
        optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda"))
        
        with amp_ctx:
            logits = head(emb) # (B, C)
            loss_raw = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )  # (B, C)
            loss = (loss_raw * weights).mean()
        
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        # --- metrics ---
        bsz = wavs.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        f1, prec, rec = multilabel_micro_f1(logits.detach(), labels, thr=thr)
        total_f1 += f1 * bsz
        total_prec += prec * bsz
        total_rec += rec * bsz
        total_subset_acc += subset_accuracy(logits.detach(), labels, thr=thr) * bsz
    
    avg_loss = total_loss / max(1, total_count)
    avg_f1   = total_f1 / max(1, total_count)
    avg_p    = total_prec / max(1, total_count)
    avg_r    = total_rec / max(1, total_count)
    avg_subset = total_subset_acc / max(1, total_count)
    return avg_loss, {"micro_f1": avg_f1, "precision": avg_p, "recall": avg_r, "subset_acc": avg_subset}

@torch.no_grad()
def eval_epoch(encoder, head, loader, device, thr=0.5, use_amp=True):
    encoder.eval()
    head.eval()
    
    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0
    
    # torch.backends.cudnn.benchmark = True
    
    for wavs, labels, weights in tqdm(loader, desc="Eval", leave=False):
        wavs = wavs.to(device)
        labels = labels.to(device)
        weights = weights.to(device)
        
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs = wavs.squeeze(1)

        emb = encoder.encode_batch(wavs).squeeze(1)
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda")):
            logits = head(emb)
            loss_raw = F.binary_cross_entropy_with_logits(
                logits, labels, reduction="none"
            )  # (B, C)
            loss = (loss_raw * weights).mean()
        
        bsz = wavs.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        f1, prec, rec = multilabel_micro_f1(logits, labels, thr=thr)
        total_f1 += f1 * bsz
        total_prec += prec * bsz
        total_rec += rec * bsz
        total_subset_acc += subset_accuracy(logits, labels, thr=thr) * bsz
    
    avg_loss = total_loss / max(1, total_count)
    avg_f1   = total_f1 / max(1, total_count)
    avg_p    = total_prec / max(1, total_count)
    avg_r    = total_rec / max(1, total_count)
    avg_subset = total_subset_acc / max(1, total_count)
    return avg_loss, {"micro_f1": avg_f1, "precision": avg_p, "recall": avg_r, "subset_acc": avg_subset}

# Main
def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high') # Small speed bump on Ampere
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    group_dir = os.path.join(args.root, "training_data", args.group)
    members = sorted([d for d in os.listdir(group_dir)
                      if os.path.isdir(os.path.join(group_dir, d)) and d.lower() not in ["harmonies", "images"]])
    
    print("Members detected:", members)
    
    # Dataset & split
    full_ds = KpopVocalDataset(group_dir, args.sr_ecapa, args.chunk_sec, num_workers=args.num_workers, pitch_prob=0.0)
    
    print("Classes:", full_ds.class_map.idx_to_name)
    print("len(full_ds) =", len(full_ds))
    
    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    print("train/val =", n_train, n_val)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # Load SpeachBrain ECAPA encoder (pretraiened)
    
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if device.type == "cuda" else "cpu"}
    )
    
    if args.freeze_ecapa:
        for p in encoder.parameters():
            p.requires_grad = False
            
    # Probe embedding dim with dummy forward
    dummy = torch.zeros(
        1, int(args.chunk_sec * args.sr_ecapa),
        dtype=torch.float32, device=device   
    )
    with torch.no_grad():
        emb_dim = encoder.encode_batch(dummy).squeeze(1).shape[-1]
        
    head = MultiLabelHead(emb_dim=emb_dim, num_classes=len(full_ds.class_map.idx_to_name)).to(device)
    
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.group}_ecapa_head.pt")
    
    eval_thr = getattr(args, "eval_thr", 0.5)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_metrics = train_epoch(encoder, head, train_loader, device, optimizer, thr=eval_thr)
        va_loss, va_metrics = eval_epoch(encoder, head, val_loader, device, thr=eval_thr)
        
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
                "classes": full_ds.class_map.idx_to_name,
                "emb_dim": emb_dim,
                "sr": args.sr_ecapa,
                "chunk_sec": args.chunk_sec,
                "group": args.group,
                "eval_thr": eval_thr,
            }, ckpt_path)
            print(f"✅ Saved best head to: {ckpt_path} (acc={best_acc:.4f})")
        
    print("\nDone. Best val acc:", best_acc)
    
if __name__ == "__main__":
    main()