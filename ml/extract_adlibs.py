import argparse
import glob
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

CHUNK_SECONDS = 0.040  # 40ms per chunk
# Pick a UVR karaoke model name you have installed/downloaded in UVR.
# Common picks in the UVR ecosystem include "MDX Karaoke" style models. :contentReference[oaicite:4]{index=4}
UVR_MODEL_NAME = "MDX_KARAOKE"

def song_name_from_vocals_path(wav_path: str) -> str:
    base = Path(wav_path).stem
    return base[:-7] if base.endswith("_vocals") else base

def chunk_to_sample(chunk_idx: int, sr: int) -> int:
    return int(round(chunk_idx * CHUNK_SECONDS * sr))

def load_labels(label_path: str):
    with open(label_path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def extract_adlib_segments_from_audio(audio, sr, labels):
    segs = []
    for lab in labels:
        if not isinstance(lab, list) or len(lab) < 5:
            continue
        member, startChunk, endChunk, isBacking, isAdlib = lab[:5]
        if not bool(isAdlib) or not bool(isBacking):
            continue
        s = chunk_to_sample(int(startChunk), sr)
        e = chunk_to_sample(int(endChunk), sr)
        s = max(0, min(s, len(audio)))
        e = max(0, min(e, len(audio)))
        if e > s:
            segs.append((str(member), s, e))
    return segs

def run_uvr_lead_backing(uvr_repo_dir: str, in_wav: str, out_dir: str):
    """
    This is intentionally implemented as a subprocess wrapper because UVR’s internal
    Python entrypoints can change between versions.
    You will need to adapt the exact CLI args to the UVR repo version you cloned.

    Goal: produce two files in out_dir:
      - {stem}_lead.wav
      - {stem}_backing.wav
    """
    os.makedirs(out_dir, exist_ok=True)
    stem = Path(in_wav).stem

    # --- YOU MUST ADAPT THIS PART ---
    # UVR has multiple inference paths depending on model type (MDX/VR/Demucs).
    # The key idea is: run a karaoke/lead-vocal model on the VOCALS STEM.
    # Many workflows treat the “instrumental” output as “backing vocals” in this context. :contentReference[oaicite:5]{index=5}
    #
    # If your UVR repo provides a CLI script, point to it here and set args accordingly.
    #
    # Example placeholder (NOT guaranteed to match your UVR version):
    cmd = [
        "python",
        os.path.join(uvr_repo_dir, "inference.py"),  # <-- may differ in your clone
        "--input", in_wav,
        "--output", out_dir,
        "--model", UVR_MODEL_NAME,
        "--device", "cuda",
    ]
    print("Running UVR:", " ".join(cmd))
    subprocess.check_call(cmd)

    # After UVR runs, locate outputs (adapt these patterns to your UVR output names)
    lead_candidates = sorted(glob.glob(os.path.join(out_dir, f"*{stem}*lead*.wav")))
    backing_candidates = sorted(glob.glob(os.path.join(out_dir, f"*{stem}*back*.wav"))) \
                      + sorted(glob.glob(os.path.join(out_dir, f"*{stem}*instr*.wav")))

    if not lead_candidates or not backing_candidates:
        raise RuntimeError(
            f"Could not find lead/backing outputs in {out_dir}. "
            f"Update the glob patterns to match UVR's filenames."
        )

    return lead_candidates[0], backing_candidates[0]