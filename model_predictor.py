import numpy as np
import json
import torch, torchaudio, math
import itertools
import torch.nn.functional as F
from train_kpop_singers import PresenceHead, MuQEncoderWrapper
from muq import MuQ
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.ndimage import median_filter
import os, csv

def _find_track_files(base_dir: Path, group_name: str, song_name: str) -> Dict[str, Path]:
    group_dir = base_dir / group_name
    if not group_dir.exists():
        raise FileNotFoundError(f"Group folder not found: {group_dir}")
    
    stem = song_name.strip()
    mix = group_dir / f"{stem}_vocals.wav"
    lead = group_dir / f"{stem}_leading_vocals.wav"
    back = group_dir / f"{stem}_backing_vocals.wav"
    
    missing = [p for p in [mix, lead, back] if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing track(s):\n" + "\n".join(str(p) for p in missing))
    
    return {'mix': mix, 'lead': lead, 'back': back}

def _resample_to_24k(in_path: Path, out_path: Path, sr_target: int = 24000) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wav, sr_in = torchaudio.load(str(in_path))

    # Convert to float early (safer for transforms)
    wav = wav.to(torch.float32)

    if sr_in != sr_target:
        resampler = torchaudio.transforms.Resample(sr_in, sr_target)
        wav = resampler(wav)

    # ---- Peak-safe scaling (prevents >0 dBFS overshoot after resample) ----
    peak = wav.abs().max().item()
    if peak > 0:
        target_peak = 0.99  # ~ -0.087 dBFS, safe headroom
        if peak > target_peak:
            wav = wav * (target_peak / peak)

    torchaudio.save(str(out_path), wav, sr_target)
    return out_path  

def _stable_runs(series: List[str]) -> List[Tuple[int, int, str]]:
    """
    Convert a per-frame label sequence into contiguous "stable runs".

    This groups consecutive identical labels into time segments so we can
    reason in durations instead of noisy 40ms frames.

    Example:
        series = ["Yujin", "Yujin", "Liz", "Liz", "silence"]

        returns:
        [
            (0, 2, "Yujin"),    # frames [0, 1]
            (2, 4, "Liz"),      # frames [2, 3]
            (4, 5, "silence")   # frame [4]
        ]

    Why this exists:
      - lets us remove very short prediction spikes (e.g. 1–2 frames)
      - lets us enforce minimum durations (e.g. backing must last ≥ 200ms)
      - lets us reason about "segments" instead of individual frames

    Returns:
        List of (start_idx, end_idx_exclusive, label) tuples.
        Indices follow Python slicing conventions.
    """
    if not series:
        return []
    runs = []
    s = 0
    cur = series[0]
    for i in range(1, len(series)):
        if series[i] != cur:
            runs.append((s, i, cur))
            s = i
            cur = series[i]
    runs.append((s, len(series), cur))
    return runs

def detect_uncertain_frames(
    probs_main_np: np.ndarray,
    rms_energy: np.ndarray,
    prob_thr: float = 0.45,
    energy_thr: float = 1e-4,
):
    """
    Returns bool mask [T] where model is uncertain but audio is present.
    """
    max_p = probs_main_np.max(axis=1)
    uncertain = (max_p < prob_thr) & (rms_energy > energy_thr)
    return uncertain

def load_difficult_masks(
    group_name: str,
    song_name: str,
    n_frames: int,
    labels_dir: str= "./saved_labels",
):
    """
    Returns:
      mask_lead, mask_back : bool arrays of shape (n_frames,)
    """
    mask_lead = np.zeros(n_frames, dtype=bool)
    mask_back = np.zeros(n_frames, dtype=bool)

    path = Path(labels_dir) / group_name / f"{song_name}_difficult.json"
    if not path.exists():
        return mask_lead, mask_back

    labels = json.load(open(path, "r"))
    
    for label in labels:
        start, end, isBacking, isAdlib = label
        if isBacking or isAdlib:
            mask_back[start:end] = True
        
        # Override for important adlib
        if isBacking and isAdlib:
            mask_lead[start:end] = True
            mask_back[start:end] = False
            
    return mask_lead, mask_back

def predict_song_selective(
    group_name: str,
    song_name: str,
    encoder_path: str,
    head1_path: str, # 2.0s model
    head2_path: str, # 0.4s Phase 2 Model,
    member_names,
    base_dir: str = "./training_data",
    cache_dirname: str = "prediction_cache_24k",
    hop_sec: float = 0.04,
    thr_main: float = 0.5,
) -> Dict[str, object]:
    """
    End-to-end:
      - find mix/lead/back wavs in ./training_data/{group_name}/
      - resample each to 24k into ./training_data/{group_name}/{cache_dirname}/
      - run predict_40ms on each
      - fuse into per-frame member lists

    Returns dict with:
      {
        "paths": {...},
        "series": {"mix": [...], "lead": [...], "back": [...]},
        "fused": [...],  # List[List[str]] per frame
      }
    """
    base = Path(base_dir)
    tracks = _find_track_files(base, group_name, song_name)
    
    group_dir = base / group_name
    cache_dir = group_dir / cache_dirname
    stem = song_name.strip()
    
    mix_24k  = _resample_to_24k(tracks["mix"],  cache_dir / f"{stem}_mix_24k.wav")
    lead_24k = _resample_to_24k(tracks["lead"], cache_dir / f"{stem}_lead_24k.wav")
    back_24k = _resample_to_24k(tracks["back"], cache_dir / f"{stem}_back_24k.wav")
    
    labels_mix, probs_mix, rms_energy = predict_40ms(
        encoder_path=encoder_path,
        head_path=head1_path,
        wav_path=str(mix_24k),
        output_dir=f"./predictions/{group_name}",
        win_sec=2.0,
        hop_sec=hop_sec,
        thr_main=thr_main,
        return_probs=True,
        return_rms=True,
    )
    
    n_frames = len(labels_mix)
    
    # ---- DIFFICULT REGIONS ----
    mask_lead, mask_back = load_difficult_masks(
        group_name, song_name, n_frames
    )
    
    # ---- PASS 2a: backing stem (0.4s) ----
    if mask_back.any():
        labels_back, probs_back = predict_40ms(
            encoder_path=encoder_path,
            head_path=head2_path,
            wav_path=str(back_24k),
            output_dir=f"./predictions/{group_name}",
            win_sec=0.4,
            hop_sec=hop_sec,
            thr_main=thr_main,
            frame_mask=mask_back,
            return_probs=True
        )
        probs_mix[mask_back] = np.maximum(
            probs_mix[mask_back], probs_back[mask_back]
        )
        
    # ---- PASS 2b: lead stem (0.4s) ----
    if mask_lead.any():
        labels_lead, probs_lead = predict_40ms(
            encoder_path=encoder_path,
            head_path=head2_path,
            wav_path=str(lead_24k),
            output_dir=f"./predictions/{group_name}",
            win_sec=0.4,
            hop_sec=hop_sec,
            thr_main=thr_main,
            frame_mask=mask_lead,
            return_probs=True,
        )
        probs_mix[mask_lead] = np.maximum(
            probs_mix[mask_lead], probs_lead[mask_lead]
        )

    # ---- PASS 3: uncertainty confirmation ----
    mask_uncertain = detect_uncertain_frames(probs_mix, rms_energy)
    
    if mask_uncertain.any():
        labels_conf, probs_conf = predict_40ms(
            encoder_path=encoder_path,
            head_path=head2_path,
            wav_path=str(mix_24k),
            output_dir=f"./predictions/{group_name}",
            win_sec=0.4,
            hop_sec=hop_sec,
            thr_main=thr_main,
            frame_mask=mask_uncertain,
            return_probs=True,
        )
        probs_mix[mask_uncertain] = np.maximum(
            probs_mix[mask_uncertain], probs_conf[mask_uncertain]
        )
        
    # ---- FINAL DECODE ----
    final_Y = decode_multilabel(
        probs_mix,
        per_class_thr=np.full(probs_mix.shape[1], thr_main),
    )
    
    mask_backing_final = mask_back.copy()
    mask_backing_final[mask_lead] = False

    labels_40ms = multilabel_matrix_to_labels_40ms(
        final_Y,
        member_names=member_names,
        backing_mask=mask_backing_final,
        include_gang=True
    )
    
    return {
        "paths": {
            "mix": str(tracks["mix"]),
            "lead": str(tracks["lead"]),
            "back": str(tracks["back"]),
        },
        "labels_40ms": labels_40ms,
        "probs_main": probs_mix,
    }


@torch.no_grad()
def predict_40ms(
    encoder_path: str, head_path: str, wav_path: str    ,
    sr_target=24000, win_sec=2.0, hop_sec=0.04, use_hann=True,
    output_dir=None, class_names=None, thr_main=0.5, 
    thr_harm: float = 0.45, thr_adlib: float = 0.6,
    frame_mask: Optional[np.ndarray] = None, return_probs: bool = False,
    return_rms: bool = False,
):
    """
    Returns labels_40ms: list of dicts, one per 40 ms frame:
      {
        "lead": str,                    # main singer (or "silence")
        "lead_conf": float,             # prob of chosen lead
        "lead_set": List[str],          # all main-active labels
        "lead_set_conf": List[float],   # probs for each member in lead_set
        "harmony": List[str],           # harmony singers
        "harmony_conf": List[float],
        "adlib": List[str],             # ad-lib singers
        "adlib_conf": List[float],
      }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    muq = muq = MuQ.from_pretrained(encoder_path)
    muq.eval()
    encoder = MuQEncoderWrapper(muq_model=muq, muq_sr=24000, pooling="mean", debug=False).to(device)
    encoder.eval()
    print("Loading head checkpoint:", head_path)
    
    ckpt = torch.load(head_path, map_location=device, weights_only=False)
    emb_dim = ckpt["emb_dim"] # 192
    model_classes = ckpt["classes"]
    if not class_names:
        class_names = list(model_classes)
        
    silence_idx = None
    for i, name in enumerate(class_names):
        if name.lower() == "silence":
            silence_idx = i
            break
    
    num_main = len(model_classes)
    num_members = num_main - 1 # drop silence
    member_names = class_names[:num_members]
    
    # Find gang vocal index (it will exist)
    gang_idx = None
    for i, name in enumerate(class_names):
        if name.lower() == "gang vocal":
            gang_idx = i
            break
    
    # print(f"Silence index: {silence_idx}, Gang index: {gang_idx}") 
    # Mask of "real members" with no gang or silence
    real_member_mask = np.ones(num_main, dtype=bool)
    real_member_mask[silence_idx] = False
    if gang_idx is not None:
        real_member_mask[gang_idx] = False
    
    fused_dim = emb_dim * 2
    head = PresenceHead(emb_dim_fused=fused_dim, emb_dim_ctx=emb_dim, num_members=num_members).to(device)
    head.load_state_dict(ckpt["state_dict"], strict=True)
    head.eval()

    # 1. load and resample mono
    wav, sr = torchaudio.load(wav_path)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sr_target:
        wav = torchaudio.transforms.Resample(sr, sr_target)(wav)
    x = wav.squeeze(0).to(device)
    T = x.numel()
    
    # 2. window/hop in samples
    win_len = int(round(win_sec * sr_target))
    hop_len = int(round(hop_sec * sr_target))
    
    if T < win_len:
        # Pad once to make one window
        pad = torch.zeros(win_len - T, device=device)
        x = torch.cat([x, pad], 0)
        T = x.numel()
    
    # -- Debug --
    eps = 1e-9

    rms = x.pow(2).mean().sqrt().item() + eps
    peak = x.abs().max().item() + eps

    rms_db = 20 * math.log10(rms)
    peak_db = 20 * math.log10(peak)

    print(
        "AUDIO stats:",
        f"rms={rms_db:.3f} dBFS,",
        f"peak={peak_db:.3f} dBFS"
    )

    # number of 40ms frames over the audio
    if frame_mask is not None:
        n_frames = frame_mask.shape[0]
    else:
        n_frames = math.ceil(T / hop_len)
    
    if frame_mask is not None:
        frame_mask = np.asarray(frame_mask).astype(bool)
        if frame_mask.shape[0] != n_frames:
            raise ValueError(f"frame_mask length {frame_mask.shape[0]} != n_frames {n_frames}")
    else:
        frame_mask = np.ones(n_frames, dtype=bool)
    
    # buffers: accumulate logits and coverage
    acc_main = torch.zeros(n_frames, num_main, device=device)
    acc_harm = torch.zeros(n_frames, num_members, device=device)
    acc_ad = torch.zeros(n_frames, num_members, device=device)
    cov = torch.zeros(n_frames, device=device)
    
    if use_hann:
        w = torch.hann_window(win_len, periodic=False, device=device)
    else:
        w = None
    
    # Slide windows and accumulate
    starts = range(0, T - win_len + 1, hop_len)
    batch_windows = []
    frame_starts = []
    
    def flush_batch():
        nonlocal batch_windows, frame_starts, acc_main, acc_harm, acc_ad, cov
        
        if not batch_windows:
            return

        batch = torch.stack(batch_windows, 0)
        # ECAPA wants (B, T)
        emb_main_b1, emb_ctx_b1 = encoder.encode_batch(batch, ctx_frac=0.25)
        emb_main = emb_main_b1.squeeze(1)
        emb_ctx  = emb_ctx_b1.squeeze(1)
        
        # Fuse + Head
        emb_fused = torch.cat([emb_main, emb_ctx], dim=1)
        
        out = head(emb_fused, emb_ctx)
        logits_main = out["main"] # (B, num_members+1)
        logits_harmony = out["harmony"] # (B, num_members)
        logits_adlib = out["adlib"] # (B, num_members)
        
        for i, f0 in enumerate(frame_starts):
            if f0 >= n_frames:
                continue

            # Optional: weight by Hann at the window center (a single scalar)
            if w is None:
                ww = 1.0
            else:
                center_sample = win_len // 2
                ww = float(w[center_sample].item() + 1e-8)

            acc_main[f0] += logits_main[i] * ww
            acc_harm[f0] += logits_harmony[i] * ww
            acc_ad[f0]   += logits_adlib[i] * ww
            cov[f0]      += ww
                
        batch_windows.clear()
        frame_starts.clear()
    
    # Assemble windows in small batches to speed up
    B = 64
    for s in starts:
        f0 = s // hop_len
        if f0 >= n_frames:
            continue
        if not frame_mask[f0]:
            continue
        chunk = x[s:s + win_len]
        if chunk.numel() < win_len:
            pad = torch.zeros(win_len - chunk.numel(), device=device)
            chunk = torch.cat([chunk, pad], dim=0)
        batch_windows.append(chunk)
        frame_starts.append(s // hop_len)
        if len(batch_windows) == B:
            flush_batch()
    flush_batch()
    
    # Normalize by coverage
    cov = cov.clamp_min(1e-6).unsqueeze(1) # n_frames, 1)
    logits_main_frame = acc_main / cov # (n_frames, num_main)
    logits_harm_frame = acc_harm / cov # (n_frames, num_members)
    logits_ad_frame = acc_ad / cov # (n_frames, num_members)
    
    probs_main_frame = torch.sigmoid(logits_main_frame)
    probs_harm_frame = torch.sigmoid(logits_harm_frame)
    probs_ad_frame   = torch.sigmoid(logits_ad_frame)
    
    # ---- APPLY FRAME MASK BEFORE DECODE ----
    if frame_mask is not None:
        mask_t = torch.from_numpy(frame_mask).to(device=device)
        skip = ~mask_t
        if skip.any():
            probs_main_frame[skip] = 0.0
            probs_harm_frame[skip] = 0.0
            probs_ad_frame[skip]   = 0.0
    
    mask_t = torch.from_numpy(frame_mask).to(device=device)
    lm = logits_main_frame[:, :num_members]

    lm_eval = lm[mask_t]  # only frames where model ran
    print("LOGITS stats (eval frames):",
        "mean_abs", lm_eval.abs().mean().item(),
        "p95_abs",  torch.quantile(lm_eval.abs().flatten(), 0.95).item(),
        "max_abs",  lm_eval.abs().max().item())

    # Optional: also show how many frames were evaluated
    print(f"[mask] evaluated {int(mask_t.sum().item())}/{len(frame_mask)} frames")

    # Move to CPU numpy for decode
    probs_main_np = probs_main_frame.cpu().numpy()
    probs_harm_np = probs_harm_frame.cpu().numpy()
    probs_ad_np = probs_ad_frame.cpu().numpy()
    
    # --- Decode main / harmony / ad-lib ---
    base_thr_main = np.full(probs_main_np.shape[1], thr_main, dtype=np.float32)
    decoded_main_np = decode_multilabel(probs_main_np, per_class_thr=base_thr_main, max_gap_frames=2)
    
    # harmony / adlib thresholds are per member (no silence)
    base_thr_harm = np.full(num_members, thr_harm, dtype=np.float32)
    base_thr_ad = np.full(num_members, thr_adlib, dtype=np.float32)
    
    decoded_harm_np = decode_multilabel(probs_harm_np, per_class_thr=base_thr_harm, min_on_frames=2)
    decoded_ad_np = decode_multilabel(probs_ad_np, per_class_thr=base_thr_ad, min_on_frames=2)
    
    # --- Build main_idx_np (single "main" singer per frame) ---
    multi_hot_main = decoded_main_np.astype(bool)
    main_idx_np = np.zeros(n_frames, dtype=np.int64)
    
    for t in range(n_frames):
        p = probs_main_np[t]       # (C,)
        active = multi_hot_main[t] # bool mask over classes

        chosen = None

        if active.any():
            # 1) REAL MEMBERS FIRST (exclude gang & silence)
            real_active_ids = np.where(active & real_member_mask)[0]
            if len(real_active_ids) > 0:
                chosen = int(real_active_ids[np.argmax(p[real_active_ids])])
            # 2) If no real member, but Gang Vocal is active -> main = Gang
            elif gang_idx is not None and active[gang_idx]:
                chosen = int(gang_idx)

        # 3) Fallbacks when nothing above chose a label
        if chosen is None:
            if silence_idx is not None:
                chosen = int(silence_idx)
            else:
                chosen = int(np.argmax(p))

        main_idx_np[t] = chosen
    
    
    # Reduces single-frame blips and bridges tiny silence gaps
    # main_idx_np = smooth_main_track(main_idx_np, silence_idx)
    pred_idx = torch.from_numpy(main_idx_np).to(device=device, dtype=torch.long)
    
    # ---- 4. Write predictions to .txt ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(wav_path))[0]
        csv_path = os.path.join(output_dir, f"{base}_predictions.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "start_chunk", "start_time", "end_time",
                "main_label", "main_confidence",
                "active_labels", "active_confidences",
                "adlibs", "adlib_probs",
                "probabilities_main",
            ]
            writer.writerow(header)

            multi_hot_main_np = multi_hot_main
            pred_idx_np = pred_idx.cpu().numpy()

            for i in range(n_frames):
                start_t = i * hop_sec
                end_t = start_t + hop_sec

                # === MAIN PROBS ===
                p_main = probs_main_np[i] # (C_main,)
                mh_main = multi_hot_main_np[i] # bool mask

                main_id = int(pred_idx_np[i])
                main_name = class_names[main_id]
                main_conf = float(p_main[main_id])
                
                active_main_ids = np.where(mh_main)[0].tolist()
                active_main_names = [class_names[j] for j in active_main_ids]
                active_main_confs = [float(p_main[j]) for j in active_main_ids]
                
                # === HARMONY ===
                harm_mask = decoded_harm_np[i].astype(bool)
                harm_ids = np.where(harm_mask)[0].tolist()
                harm_names = [member_names[j] for j in harm_ids]

                # collect harmony probabilities ONLY if something is active
                if len(harm_ids) > 0:
                    harm_full_probs_str = "[" + ", ".join(f"{x:.3f}" for x in probs_harm_np[i]) + "]"
                    harm_names_str = "|".join(harm_names)
                else:
                    harm_full_probs_str = "[]"
                    harm_names_str = ""
                
                # === AD-LIB ===
                ad_mask = decoded_ad_np[i].astype(bool)
                ad_ids = np.where(ad_mask)[0].tolist()
                ad_names = [member_names[j] for j in ad_ids]

                if len(ad_ids) > 0:
                    ad_probs = [float(probs_ad_np[i][j]) for j in ad_ids]
                    ad_probs_str = "[" + ", ".join(f"{c:.3f}" for c in ad_probs) + "]"
                    ad_names_str = "|".join(ad_names)
                else:
                    ad_probs_str = "[]"
                    ad_names_str = ""

                row = [
                    i,
                    round(start_t, 3),
                    round(end_t, 3),
                    main_name,
                    main_conf,
                    "|".join(active_main_names),
                    "[" + ", ".join(f"{c:.3f}" for c in active_main_confs) + "]",
                    # harm_names_str,
                    # harm_full_probs_str,
                    ad_names_str,
                    ad_probs_str,
                    "[" + ", ".join(f"{x:.3f}" for x in p_main) + "]",
                ]
                writer.writerow(row)
        print(f"✅ Saved predictions to {csv_path}")
        
    if return_rms:
        rms_energy = np.zeros(n_frames)
        for i in range(n_frames):
            a = i * hop_len
            b = min(a + hop_len, T)
            rms_energy[i] = float(x[a:b].pow(2).mean().sqrt().cpu())

    # ---- Build labels_40ms (final output) ----
    labels_40ms = []
    multi_hot_main = decoded_main_np.astype(bool)

    for t in range(n_frames):
        active_ids = np.where(multi_hot_main[t])[0].tolist()
        active_member_ids = [i for i in active_ids if real_member_mask[i]]

        labels_40ms.append({
            "main": [class_names[i] for i in active_member_ids],
            "harmony": [],
            "adlib": [],
        })

    # ---- RETURN LOGIC ----
    if return_probs or return_rms:
        out = [labels_40ms]
        if return_probs:
            out.append(probs_main_frame[:, :num_members].cpu().numpy())
        if return_rms:
            out.append(rms_energy)
        return tuple(out)

    return labels_40ms

def median_smooth_over_time(P, k=3):
    """
    Median-filter probabilities over time to reduce jitter.

    P: (T, C) array of probabilities
    window_frames: odd integer, size of the median window in frames.
                   For 40ms frames, 3 means 120ms context on each side.
    """
    return median_filter(P, size=(k, 1))

def apply_on_off_thresholds(p, on=0.60, off=0.45):
    """
    Turn a 1D probability sequence into 0/1 using two thresholds:

    - When currently OFF, we only turn ON if p >= on_thresh.
    - When currently ON, we stay ON until p <= off_thresh.

    This avoids flickering when probabilities hover near the threshold.
    """
    y = np.zeros_like(p, dtype=np.int32)
    active = False
    for i, s in enumerate(p):
        if not active and s >= on:  active = True
        if active and s <= off:     active = False
        y[i] = 1 if active else 0
    return y

def enforce_min_active_and_fill_gaps(y, min_on_frames=3, max_gap_frames=2):
    """
    Post-process a 0/1 sequence:

    - Remove very short active runs (1s) shorter than min_on_frames.
    - Fill short gaps of 0s (silence) between 1s if the gap length
      is <= max_gap_frames.

    Example:
      max_gap_frames = 2 means we allow up to 2 consecutive 0s
      between 1s and we treat them as if the class stayed active.
    """
    out = y.copy()
    T = len(y)
    for val, grp in itertools.groupby(range(T), key=lambda i: y[i]):
        idx = list(grp)
        length = len(idx)
        
        if val == 1 and length < min_on_frames:
            # Too-short active run → turn off
            out[idx] = 0
    
    # Second pass: fill small 0-gaps between 1s
    y2 = out.copy()
    for val, grp in itertools.groupby(range(T), key=lambda i: y2[i]):
        idx = list(grp)
        length = len(idx)

        if val == 0 and length <= max_gap_frames:
            # Only fill if surrounded by 1s on both sides
            left = idx[0] - 1
            right = idx[-1] + 1
            if left >= 0 and right < T and y2[left] == 1 and y2[right] == 1:
                out[idx] = 1
                
    return out

def cap_topk(P_row, k=2):
    keep = np.zeros_like(P_row, dtype=bool)
    keep[np.argsort(-P_row)[:k]] = True
    return keep

def decode_multilabel(probs, per_class_thr=None, k_smooth=3, on_add=0.02, off_sub=0.02,
                      min_on_frames=3, max_gap_frames=2, topk=2):
    """
    Decode multi-label probabilities over time into 0/1 activity:

    1. Median smooth over time with window 'smooth_window'.
    2. For each class:
         - use on/off thresholds (hysteresis) around 'per_class_thr'
         - enforce a minimum active duration (min_on_frames)
         - fill small gaps of up to 'max_gap_frames' between active regions
    3. Optionally keep at most 'topk' active classes per frame.

    Returns:
        Y: int32 array of shape (T, C) with 0/1 per frame per class.
    """
    # Smooth in time
    P = median_smooth_over_time(probs, k=k_smooth)
    T, C = P.shape
    
    # 2. thresholds per class
    base = (
        np.full(C, 0.50, np.float32)
        if per_class_thr is None
        else np.asarray(per_class_thr, np.float32)
    )
    on = np.clip(base + on_add, 0, 1)
    off = np.clip(base - off_sub, 0, 1)

    Y = np.zeros_like(P, dtype=np.int32)

    for c in range(C):
        # 2a. hysteresis on/off
        y = apply_on_off_thresholds(P[:, c], on=on[c], off=off[c])
        # 2b. enforce min active duration & fill short gaps
        y = enforce_min_active_and_fill_gaps(
            y,
            min_on_frames=min_on_frames,
            max_gap_frames=max_gap_frames,
        )
        Y[:, c] = y

    # 3. optional per-frame top-k restriction
    if topk is not None:
        for t in range(T):
            keep = cap_topk(P[t], k=topk)
            Y[t] = Y[t] * keep.astype(np.int32)

    return Y

def multilabel_matrix_to_labels_40ms(
    Y: np.ndarray,
    member_names: list,
    *,
    backing_mask: Optional[np.ndarray] = None,
    include_gang: bool = False,
):
    """
    Y: (T, C) int32 0/1 matrix. C should match len(member_names) (+optional extras if you include them).
    member_names: list like ["Gaeul","Yujin","Rei","Wonyoung","Liz","Leeseo","Gang Vocal"]
    backing_mask: (T,) bool array. True => put active labels into 'backing' instead of 'main'
    """
    T, C = Y.shape
    if len(member_names) != C:
        raise ValueError(f"member_names len {len(member_names)} != Y.shape[1] {C}")

    if backing_mask is None:
        backing_mask = np.zeros(T, dtype=bool)
    else:
        backing_mask = np.asarray(backing_mask).astype(bool)
        if backing_mask.shape[0] != T:
            raise ValueError(f"backing_mask len {backing_mask.shape[0]} != T {T}")

    out = []
    for t in range(T):
        active_ids = np.where(Y[t] > 0)[0].tolist()
        names = [member_names[i] for i in active_ids]

        # Optional: drop "Gang Vocal" from outputs if you don't want it displayed
        if not include_gang:
            names = [n for n in names if n.lower() != "gang vocal"]

        if backing_mask[t]:
            out.append({"main": [], "harmony": names})
        else:
            out.append({"main": names, "harmony": []})

    return out

def smooth_main_track(main_idx, silence_idx=None, min_singer_len=3, 
                         min_silence_len=1, bridge_silence_len=2):
    """
    Post-process the per-frame main class index sequence.

    - Remove very short singer segments (shorter than min_singer_len).
    - Optionally enforce a minimum silence length.
    - Bridge small silence gaps (<= bridge_silence_len) between the same singer.

    This operates AFTER the per-class multi-label decode, so it's a last pass
    over the single 'main' label track.
    """
    main_idx = np.asarray(main_idx, dtype=np.int64)
    T = len(main_idx)
    out = main_idx.copy()
    
    start = 0
    while start < T:
        label = out[start]
        end = start + 1
        while end < T and out[end] == label:
            end += 1
        length = end - start
        
        # Handle singer segments
        if silence_idx is not None and label != silence_idx:
            if length < min_singer_len:
                # Too short to be reliable singer segment
                prev_label = out[start - 1] if start > 0 else None
                next_label = out[end] if end < T else None
                if prev_label is not None and prev_label == next_label:
                    out[start:end] = prev_label
                elif prev_label is not None:
                    out[start:end] = prev_label
                elif next_label is not None:
                    out[start:end] = next_label
                else:
                    out[start:end] = silence_idx
                    
        # Handle silence segments
        if silence_idx is not None and label == silence_idx:
            # Optional: enforce min_silence_len
            if length < min_silence_len:
                prev_label = out[start - 1] if start > 0 else None
                next_label = out[end] if end < T else None
                if prev_label is not None and prev_label == next_label:
                    out[start:end] = prev_label
            # Bridge tiny silence gaps between same singer
            if length <= bridge_silence_len:
                prev_label = out[start - 1] if start > 0 else None
                next_label = out[end] if end < T else None
                if (
                    prev_label is not None 
                    and prev_label == next_label 
                    and prev_label != silence_idx
                ):
                    out[start:end] = prev_label
        
        start = end
    
    return out