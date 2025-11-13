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

import os, argparse, math, random, glob
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
import numpy as np
from torchaudio.transforms import Resample

from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier
from contextlib import nullcontext

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
    Creates training examples as waveform chunks (16 kHz), with labels per member + silence.

    For each member file:
    - Load audio
    - Convert to mono, resample to 16 kHz
    - Split into non-overlapping chunks of chunk_sec
    Additionally:
    - Generates synthetic 'silence' examples as zero tensors (same length)
        (You can replace this with real silence/gang-vocal negatives later.)
    """
    def __init__(self, group_dir: str, members: List[str], sr_in: int, sr_out: int,
                 chunk_sec: float, add_silence_ratio: float = 0.15,
                 p_synth_overlap: float = 0.25, snr_db_range: Tuple[float, float] = (-5.0, 5.0),
                 p_pitch: float = 0.30, pitch_semitones_range: Tuple[float, float] = (-2.0, 2.0)):
        super().__init__()
        self.num_classes = len(members)
        self.sr_in = sr_in
        self.sr_out = sr_out
        self.chunk_len = int(chunk_sec * sr_out)
        
        self.items: List[Tuple[str, int]] = []

        for m in members:
            wav_dir = os.path.join(group_dir, m)
            files = list_wavs(wav_dir)
            for f in files:
                self.items.append((f, m))
        self.add_silence_ratio = add_silence_ratio
        self.p_synth_overlap = p_synth_overlap
        self.snr_db_range = snr_db_range
        self.p_pitch = p_pitch
        self.pitch_semitones_range = pitch_semitones_range
    
        self.members = members
        
        self.class_to_index = {name: i for i, name in enumerate(self.members)}
        self._members_lower = {m.lower(): m for m in self.members}
        
        self._norm2canon = {m.lower(): m for m in members}
        self._silence_idx = self.members.index("silence")
        
        self.pitch_aug_per_member = 4
        self.mix_aug_per_member = 4
        self.cache_fp16 = True
        
        # Lazily reuse a resampler object per source sample rate
        self._cached_sr = None
        self._cached_resampler = None
        
        base_files_with_members = self._collect_base_files(group_dir)
        
        # Pre-index ALL full chunks using metadata (fast; no audio load here)
        # (filepath, start_sample_out, label_idx)
        self.base_samples: List[Tuple[str, int, np.ndarray]] = self._index_full_chunks(base_files_with_members)  
        
        # Indices for SOLO examples (exactly one member
        # 3) map from member -> indices of SOLO base samples (exactly one active member, not silence)
        self.solo_indices = self._collect_solo_indices_by_member(self.base_samples)
        
        self.augmented_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.build_cache(seed=0)
        
        # Diagnostics
        print(f"[KpopVocalDataset] members={members}")
        print(f"[KpopVocalDataset] chunk_len={self.chunk_len} @ {self.sr_out} Hz")
        print(f"[KpopVocalDataset] base_chunks={len(self.base_samples)}")
        print(f"[KpopVocalDataset] aug_chunks={len(self.augmented_samples)} "
              f"(pitch_per_member={self.pitch_aug_per_member}, mix_per_member={self.mix_aug_per_member}, fp16={self.cache_fp16})")
    
    # -----------------------------
    # Public cache API
    # -----------------------------
    def build_cache(self, seed: Optional[int] = None) -> None:
        """(Re)build the augmentation cache deterministically by seed."""
        if seed is not None:
            random.seed(None)
            np.random.seed(seed)
            torch.manual_seed(seed)
            
        self.augmented_samples = []
        self._add_snr_mixed_overlaps()
    
    # -----------------------------
    # Dataset protocol
    # -----------------------------   
    def __len__(self):
        # We return number of files; chunking happens on-the-fly per __getitem__
        # to avoid storing all chunks in memory.
        return len(self.base_samples)
    
    def _collect_base_files(self, group_dir: str) -> List[Tuple[str, List[str]]]:
        """
        Returns a list of (filepath, active_members) pairs.
        Includes solo files under <group>/<member>/... and harmony files under <group>/harmonies/.
        """
        results: List[Tuple[str, List[str]]] = []

        # SOLO: each member's folder (and optional train/Isolated_Vocals)
        for m in self.members:
            mdir = os.path.join(group_dir, m)
            paths = []
            for pat in ("**/*.wav", "**/*.flac", "**/*.mp3", "**/*.m4a"):
                paths += glob.glob(os.path.join(mdir, pat), recursive=True)
                paths += glob.glob(os.path.join(mdir, "train", "Isolated_Vocals", pat), recursive=True)
            for p in sorted(set(paths)):
                results.append((p, [m]))  # enforce list type

        # HARMONIES
        hdir = os.path.join(group_dir, "harmonies")
        if os.path.isdir(hdir):
            hpaths = []
            for pat in ("**/*.wav", "**/*.flac", "**/*.mp3", "**/*.m4a"):
                hpaths += glob.glob(os.path.join(hdir, pat), recursive=True)
            for p in sorted(set(hpaths)):
                names = self._parse_harmony_members(os.path.basename(p))
                names = [n for n in names if n.lower() != "silence"] 
                if names:
                    results.append((p, names))

        return results
    
    def _index_full_chunks(self, files_with_members: List[Tuple[str, List[str]]]) -> List[Tuple[str, int, np.ndarray]]:
        """
        For each file, compute how many full chunks fit and create labeled indices.
        Returns a list of (path, start_out, label_vector_np).
        """
        samples: List[Tuple[str, int, np.ndarray]] = []

        for path, active_members in files_with_members:
            # duration @ source
            try:
                info = torchaudio.info(path)
                sr_src = info.sample_rate
                n_src = info.num_frames
                if n_src <= 0 or sr_src <= 0:
                    continue
                dur_sec = n_src / float(sr_src)
            except Exception:
                continue

            # skip short files (< 4 s)
            if dur_sec < 4.0:
                continue

            # duration @ target sr
            n_out = int(math.floor(dur_sec * self.sr_out))
            n_chunks = n_out // self.chunk_len
            if n_chunks == 0:
                continue

            # build multi-hot label
            if isinstance(active_members, str):
                active_members = [active_members]
            label_vec = np.zeros(self.num_classes, dtype=np.float32)
            for name in active_members:
                if name in self.class_to_index:
                    label_vec[self.class_to_index[name]] = 1.0

            for k in range(n_chunks):
                start_out = k * self.chunk_len
                samples.append((path, start_out, label_vec.copy()))

        return samples
    
    def _collect_solo_indices_by_member(
        self,
        base_samples: List[Tuple[str, int, np.ndarray]],
    ) -> Dict[str, List[int]]:
        """
        Map each member name to a list of indices in base_samples that are SOLO examples for that member.
        """
        mapping: Dict[str, List[int]] = {m: [] for m in self.members if m.lower() != "silence"}
        for i, (_, _, label_np) in enumerate(base_samples):
            if label_np[self._silence_idx] == 0.0 and label_np[:-1].sum() == 1.0:
                m_idx = int(label_np[:-1].argmax())
                m_name = self.members[m_idx]
                mapping[m_name].append(i)
        return mapping
    
    def _get_resampler(self, sr_src: int):
        if self._cached_sr != sr_src:
            self._cached_sr = sr_src
            self._cached_resampler = Resample(sr_src, self.sr_out)
        return self._cached_resampler
         
    def _load_chunk(self, path: str, start_out: int) -> torch.Tensor:
        # Load audio, mono, resample to sr_out, slice exact [start_out : start_out + chunk_len]
        wav, sr_src = torchaudio.load(path)  # (C, T_src)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)  # (1, T_src)

        if sr_src != self.sr_out:
            resampler = self._get_resampler(sr_src)
            wav = resampler(wav)  # (1, T_out)

        wav = wav.squeeze(0)  # (T_out,)
        end_out = start_out + self.chunk_len
        # Guard in case resampling rounded slightly short:
        if end_out > wav.numel():
            # Drop tail if shorter than chunk (design choice: enforced by pre-index; this is rare)
            end_out = wav.numel()
        seg = wav[start_out:end_out]
        # seg should be exactly chunk_len, but in rare rounding cases it can be short: skip here by padding? No—drop.
        if seg.numel() < self.chunk_len:
            # This should not happen because we pre-indexed by n_out//chunk_len,
            # but due to resampling rounding it can. In that rare case, return None and let caller resample another idx.
            return None
        return seg.unsqueeze(0)  # (1, chunk_len)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Serve precomputed augmentations first (fast path)
        if index < len(self.augmented_samples):
            audio, label = self.augmented_samples[index]
            # Store as fp16, serve as fp32 for model safety
            if audio.dtype == torch.float16:
                audio = audio.float()
            return audio, label

        # Otherwise serve a base sample without heavy aug
        base_index = index - len(self.augmented_samples)
        path, start_out, label_np = self.base_samples[base_index]

        audio = self._load_chunk(path, start_out)
        if audio is None:
            audio = self._load_chunk(path, 0)
        if audio is None:
            audio = torch.zeros(1, self.chunk_len)
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            label[self._silence_idx] = 1.0
            return audio, label

        label = torch.from_numpy(label_np).to(torch.float32)

        # lightweight synthetic silence (kept on-the-fly; cheap)
        if self.add_silence_ratio > 0.0 and random.random() < self.add_silence_ratio:
            audio = torch.zeros_like(audio)
            label = torch.zeros(self.num_classes, dtype=torch.float32)
            label[self._silence_idx] = 1.0

        return audio, label
    
    def _rand_uniform(self, a: float, b: float) -> float:
        return a + (b - a) * random.random()
    
    def _add_snr_mixed_overlaps(self) -> None:
        """
        Add either pitch-shifted or SNR-mixed SOLO chunks (balanced per member).
        For each selected base sample:
            - With 50% chance: just apply pitch shift (solo augment)
            - With 50% chance: mix with another random SOLO chunk at random SNR
        """
        all_solo_indices = [idx for lst in self.solo_indices.values() for idx in lst]
        if not all_solo_indices:
            return

        for member in self.members:
            base_indices = self.solo_indices.get(member, [])
            if not base_indices:
                continue

            count = min(self.mix_aug_per_member, len(base_indices))
            selected = random.sample(base_indices, count) if count < len(base_indices) else base_indices

            for base_idx in selected:
                path1, start1, label1_np = self.base_samples[base_idx]
                audio1 = self._load_chunk(path1, start1)
                if audio1 is None:
                    audio1 = self._load_chunk(path1, 0)
                if audio1 is None:
                    print(f"Double check code. Something is wrong with audio1: {audio1}")
                    continue  # skip if still invalid
                label1 = torch.from_numpy(label1_np).to(torch.float32)

                # --- choose augmentation type ---
                if random.random() < 0.5:
                    # (A) pitch shift only
                    audio1 = self._apply_pitch_shift(audio1)
                    augmented = self._maybe_to_fp16(audio1)
                    self.augmented_samples.append((augmented, label1))
                    continue

                # (B) mix with another SOLO chunk at random SNR
                partner_idx = random.choice(all_solo_indices)
                path2, start2, label2_np = self.base_samples[partner_idx]
                audio2 = self._load_chunk(path2, start2)
                if audio2 is None:
                    self._load_chunk(path2, 0)
                if audio2 is None:
                    continue

                # optional pitch shift on the partner (small chance)
                if random.random() < self.p_pitch:
                    audio2 = self._apply_pitch_shift(audio2)

                label2 = torch.from_numpy(label2_np).to(torch.float32)
                snr_db = self._rand_uniform(*self.snr_db_range)
                mixed = self._mix_at_snr(audio1, audio2, snr_db)

                # combine labels (union, silence off)
                label = torch.maximum(label1, label2)
                if label[:-1].sum() > 0:
                    label[self._silence_idx] = 0.0

                mixed = self._maybe_to_fp16(mixed)
                self.augmented_samples.append((mixed, label))
                    
    def _sample_random_solo(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (seg, lab) for a random SOLO sample, already loaded and pitch-aug’d."""
        if not self.solo_indices:
            return None, None
        j = random.choice(self.solo_indices)
        path2, start2, lab2_np = self.base_samples[j]
        seg2 = self._load_chunk(path2, start2)
        if seg2 is None:
            seg2 = self._load_chunk(path2, 0)
        if seg2 is None:
            return None, None
        #optional pitch
        seg2 = self._pitch_shift(seg2)
        lab2 = torch.from_numpy(lab2_np).to(torch.float32)
        return seg2, lab2
    
    def _apply_pitch_shift(self, audio: torch.Tensor) -> torch.Tensor:
        """Optionally apply pitch shift; always preserve length T."""
        if self.p_pitch <= 0.0 or random.random() >= self.p_pitch:
            return audio
        n_semitones = self._rand_uniform(*self.pitch_semitones_range)
        try:
            if hasattr(torchaudio.functional, "pitch_shift"):
                shifted = torchaudio.functional.pitch_shift(audio, self.sr_out, n_steps=n_semitones)
            else:
                cents = n_semitones * 100.0
                effects = [["pitch", f"{cents}"], ["rate", f"{self.sr_out}"]]
                shifted, _ = torchaudio.sox_effects.apply_effects_tensor(audio, self.sr_out, effects)
            # length guard
            if shifted.shape[-1] >= audio.shape[-1]:
                return shifted[..., : audio.shape[-1]]
            pad = audio.shape[-1] - shifted.shape[-1]
            return torch.nn.functional.pad(shifted, (0, pad))
        except Exception:
            return audio
    
    def _mix_at_snr(self, a: torch.Tensor, b: torch.Tensor, snr_db: float) -> torch.Tensor:
        """
        a, b: (1, T) float waveforms in [-1,1] ideally.
        Scales b to the target SNR relative to a, then returns clipped mix.
        """
        eps = 1e-8
        pa = (a.pow(2).mean().sqrt() + eps)  # RMS
        pb = (b.pow(2).mean().sqrt() + eps)
        # target: SNR = 20*log10(pa / (g*pb)) => g = pa / (pb * 10^(SNR/20))
        g = pa / (pb * (10.0 ** (snr_db / 20.0)))
        mix = a + g * b
        return torch.clamp(mix, -1.0, 1.0)
    
    def _maybe_to_fp16(self, audio: torch.Tensor) -> torch.Tensor:
        return audio.half() if self.cache_fp16 else audio.float()

    def _rand_uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * random.random()

    def _parse_harmony_members(self, basename: str) -> List[str]:
        """
        From 'Liz-Yujin_harmony_training_vocals.wav' -> ['Liz','Yujin'] (only if in members list).
        Case-insensitive; trims around '-' and ignores non-members.
        """
        stem = os.path.splitext(basename)[0]
        lead = stem.split('_', 1)[0]
        candidates = [p.strip() for p in lead.split('-') if p.strip()]
        found = []
        for cand in candidates:
            key = cand.lower()
            if key in self._members_lower:
                found.append(self._norm2canon[key])
        seen = set()
        uniq = []
        for m in found:
            if m not in seen:
                uniq.append(m)
                seen.add(m)
        return uniq

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
def train_epoch(encoder, head, loader, device, optimizer, criterion, thr=0.5, use_amp=True):
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
    
    for wavs, labels in tqdm(loader, desc="Train", leave=False):
        wavs = wavs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
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
            loss = criterion(logits, labels) # BCEWithLogitsLoss (mean)
        
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
def eval_epoch(encoder, head, loader, device, criterion, thr=0.5, use_amp=True):
    encoder.eval()
    head.eval()
    
    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0
    
    # torch.backends.cudnn.benchmark = True
    
    for wavs, labels in tqdm(loader, desc="Eval", leave=False):
        wavs = wavs.to(device)
        labels = labels.to(device)
        
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs = wavs.squeeze(1)

        emb = encoder.encode_batch(wavs).squeeze(1)
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda")):
            logits = head(emb)
            loss = criterion(logits, labels)
        
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
    
    group_dir = os.path.join(args.root, args.group)
    members = sorted([d for d in os.listdir(group_dir)
                      if os.path.isdir(os.path.join(group_dir, d)) and d.lower() not in ["harmonies", "images"]])
    
    print("Members detected:", members)
    
    # Dataset & split
    full_ds = KpopVocalDataset(group_dir, members, args.sr_in, args.sr_ecapa, args.chunk_sec)
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
        
    head = MultiLabelHead(emb_dim=emb_dim, num_classes=len(members)).to(device)
    
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.group}_ecapa_head.pt")
    
    eval_thr = getattr(args, "eval_thr", 0.5)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_metrics = train_epoch(encoder, head, train_loader, device, optimizer, criterion, thr=eval_thr)
        va_loss, va_metrics = eval_epoch(encoder, head, val_loader, device, criterion, thr=eval_thr)
        
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
                "classes": full_ds.members,
                "emb_dim": emb_dim,
                "sr": args.sr_ecapa,
                "chunk_sec": args.chunk_sec,
                "group": args.group,
                "eval_thr": eval_thr,
            }, ckpt_path)
            print(f"✅ Saved best head to: {ckpt_path} (acc={best_acc:.4f})")
        
    print("\nDone. Best val acc:", best_acc)
    print("Class order:", full_ds.members)
    
    # Example reload and run a file
    load = torch.load(ckpt_path, map_location=device)
    head2 = MultiLabelHead(load["emb_dim"], len(load["classes"])).to(device)
    head2.load_state_dict(load["state_dict"])
    head2.eval()
    
    wavs, labels = next(iter(val_loader))
    wavs = wavs.to(device, non_blocking=True)
    if wavs.ndim == 3 and wavs.size(1) == 1:
        wavs = wavs.squeeze(1)
    with torch.no_grad():
        emb = encoder.encode_batch(wavs).squeeze(1)
        logits = head2(emb)
        probs = logits.sigmoid()  # (B, C)
        preds = (probs >= load.get("eval_thr", 0.5)).int()

    classes = load["classes"]
    for i in range(min(8, preds.size(0))):
        active = [classes[j] for j in torch.where(preds[i] == 1)[0].tolist()]
        print(f"Sample {i}: {active}  | probs={probs[i].tolist()}")
    
if __name__ == "__main__":
    main()