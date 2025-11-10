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
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
from torchaudio.transforms import Resample

from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier

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
                 chunk_sec: float, add_silence_ratio: float = 0.15):
        super().__init__()
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
        
        self.classes = members
        self.class_map = ClassMap(
            idx_to_name=self.classes,
            name_to_idx={n: i for i, n in enumerate(self.classes)}
        )
        
        # Collect audio files for each member (supports both plain folder and train/Isolated_Vocals/)
        files_per_member: List[Tuple[str, str]] = []   # (filepath, member)
        for m in members:
            mdir = os.path.join(group_dir, m)
            # search recursively in both possible locations
            cand = []
            for pat in ("**/*.wav", "**/*.flac", "**/*.mp3", "**/*.m4a"):
                cand += glob.glob(os.path.join(mdir, pat), recursive=True)
                cand += glob.glob(os.path.join(mdir, "train", "Isolated_Vocals", pat), recursive=True)
            for f in sorted(set(cand)):
                files_per_member.append((f, m))

        # Pre-index ALL full chunks using metadata (fast; no audio load here)
        self.samples: List[Tuple[str, int, int]] = []  # (filepath, start_sample_out, label_idx)
        for path, m in files_per_member:
            try:
                info = torchaudio.info(path)
                sr_src = info.sample_rate
                n_src = info.num_frames
                if n_src <= 0 or sr_src <= 0:
                    continue
                # duration @ target sr_out
                dur_sec = n_src / float(sr_src)
                n_out = int(math.floor(dur_sec * self.sr_out))
            except Exception:
                continue  # unreadable file -> skip

            # How many FULL chunks fit? (drop tail shorter than chunk_len)
            n_chunks = n_out // self.chunk_len
            if n_chunks == 0:
                # Entire file is shorter than one full chunk: skip it (as requested)
                continue

            label = self.class_map.name_to_idx[m]
            for k in range(n_chunks):
                start_out = k * self.chunk_len
                self.samples.append((path, start_out, label))

        # Lazily reuse a resampler object per source sample rate
        self._cached_sr = None
        self._cached_resampler = None

        # Diagnostics
        print(f"[KpopVocalDataset] members={self.classes}")
        print(f"[KpopVocalDataset] chunk_len={self.chunk_len} samples @ {self.sr_out} Hz")
        print(f"[KpopVocalDataset] files={len(files_per_member)}, indexed_chunks={len(self.samples)}")
    
    def __len__(self):
        # We return number of files; chunking happens on-the-fly per __getitem__
        # to avoid storing all chunks in memory.
        return len(self.samples)
    
    def _get_resampler(self, sr_src: int):
        if self._cached_sr != sr_src:
            self._cached_sr = sr_src
            self._cached_resampler = Resample(sr_src, self.sr_out)
        return self._cached_resampler
         
    def _load_slice(self, path: str, start_out: int) -> torch.Tensor:
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
    
    def __getitem__(self, idx: int):
        path, start_out, label = self.samples[idx]

        seg = self._load_slice(path, start_out)
        if seg is None:
            # Extremely rare rounding case: resample a nearby chunk deterministically
            # (or you could raise; here we just fall back to the first chunk of the file)
            seg = self._load_slice(path, 0)
            if seg is None:
                # If still None, return a strict zero segment labeled as 'silence' if class exists,
                # otherwise as the original label (last-resort safety)
                z = torch.zeros(1, self.chunk_len)
                if "silence" in {c.lower() for c in self.classes}:
                    label = self.class_map.name_to_idx[next(c for c in self.classes if c.lower()=="silence")]
                return z, label

        # Optional: synthetic silence injection
        if self.add_silence_ratio > 0.0 and random.random() < self.add_silence_ratio:
            seg = torch.zeros_like(seg)
            if "silence" in {c.lower() for c in self.classes}:
                label = self.class_map.name_to_idx[next(c for c in self.classes if c.lower()=="silence")]

        return seg, label

# ---------------------------
# Model: ECAPA encoder (frozen or trainable) + linear head
# ---------------------------
class EcapaHead(nn.Module):
    def __init__(self, emb_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(emb_dim, num_classes)
    
    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.classifier(emb)
    
# ---------------------------
# Training / evaluation
# ---------------------------
def train_epoch(encoder, head, loader, device, optimizer, criterion):
    encoder.eval() # Extract embeddings under no_grad by default
    head.train()
    
    total, correct, running = 0, 0, 0.0
    
    for wavs, labels in tqdm(loader, desc="Train", leave=False):
        wavs = wavs.to(device)
        labels = labels.to(device)
        
        if wavs.ndim == 3:
            wavs = wavs.squeeze(1)
        elif wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)
        
        with torch.no_grad():
             # SpeechBrain ECAPA expects (B, T) or (B, 1, T) tensors; encode_batch handles both.
             # print(f"[DEBUG] wavs.shape = {wavs.shape}, dtype={wavs.dtype}, device={wavs.device}")
             emb = encoder.encode_batch(wavs).squeeze(1)
        
        logits = head(emb)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        running += loss.item() * wavs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += wavs.size(0)
    
    return running / total, correct / total

@torch.no_grad()
def eval_epoch(encoder, head, loader, device, criterion):
    encoder.eval()
    head.eval()
    total, correct, running = 0, 0, 0.0
    for wavs, labels in tqdm(loader, desc="Eval", leave=False):
        wavs = wavs.to(device)
        labels = labels.to(device)
        
        if wavs.ndim == 3:
            wavs = wavs.squeeze(1)
        elif wavs.ndim == 1:
            wavs = wavs.unsqueeze(0)
            
        emb = encoder.encode_batch(wavs).squeeze(1) # B, D)
        logits = head(emb)
        loss = criterion(logits, labels)
        
        running += loss.item() * wavs.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += wavs.size(0)
    
    return running / total, correct / total

# Main
def main():
    args = parse_args()
    set_seed(args.seed)
    
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
        
    head = EcapaHead(emb_dim=emb_dim, num_classes=len(full_ds.class_map.idx_to_name)).to(device)
    
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.group}_ecapa_head.pt")
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_epoch(encoder, head, train_loader, device, optimizer, criterion)
        va_loss, va_acc = eval_epoch(encoder, head, val_loader, device, criterion)
        
        print(f"Train   - loss: {tr_loss:.4f} | acc: {tr_acc:.4f}")
        print(f"Val     - loss: {va_loss:.4f} | acc: {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "state_dict": head.state_dict(),
                "classes": full_ds.class_map.idx_to_name,
                "emb_dim": emb_dim,
                "sr": args.sr_ecapa,
                "chunk_sec": args.chunk_sec,
                "group": args.group,
            }, ckpt_path)
            print(f"✅ Saved best head to: {ckpt_path} (acc={best_acc:.4f})")
        
    print("\nDone. Best val acc:", best_acc)
    print("Class order:", full_ds.class_map.idx_to_name)
    
    # Example reload and run a file
    load = torch.load(ckpt_path, map_location=device)
    head2 = EcapaHead(load["emb_dim"], len(load["classes"])).to(device)
    head2.load_state_dict(load["state_dict"])
    head2.eval()
    
    wavs, labels = next(iter(val_loader))
    wavs = wavs.to(device)
    if wavs.ndim == 3:
        wavs = wavs.squeeze(1)
    elif wavs.ndim == 1:
        wavs = wavs.unsqueeze(0)
    with torch.no_grad():
        emb = encoder.encode_batch(wavs).squeeze(1)
        
        logits = head2(emb)
        probs = torch.softmax(logits, dim=-1)
        pred = probs.argmax(dim=-1)
    print("Pred classes:", [load["classes"][i] for i in pred[:8].tolist()])
    
if __name__ == "__main__":
    main()