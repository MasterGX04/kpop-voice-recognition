import numpy as np
import torch, torchaudio, math
import itertools
import torch.nn.functional as F
from train_kpop_singers import MultiTaskHead, MuQEncoderWrapper
from muq import MuQ
from scipy.ndimage import median_filter
import os, csv

@torch.no_grad()
def predict_40ms(
    encoder_path: str, head_path: str, wav_path: str    ,
    sr_target=24000, win_sec=2.0, hop_sec=0.04, use_hann=True,
    output_dir=None, class_names=None, thr_main=0.5, 
    thr_harm: float = 0.45, thr_adlib: float = 0.6
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
    
    ckpt = torch.load(head_path, map_location=device)
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
    head = MultiTaskHead(emb_dim_fused=fused_dim, emb_dim_ctx=emb_dim, num_members=num_members).to(device)
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
    
    # number of 40ms frames over the audio
    n_frames = math.ceil(T / hop_len)
    
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
        
        n_win_frames = math.ceil(win_len / hop_len)
        # Add into per-frame accumulators
        for i, f0 in enumerate(frame_starts):
            if w is None:
                weights = torch.ones(n_win_frames, device=device)
            else:
                # Approximate Hann weighting per 40 ms frame
                j = torch.arange(n_win_frames, device=device)
                centers = (j * hop_len + min(hop_len, win_len) // 2).clamp(max=win_len - 1)
                weights = w[centers] + 1e-8  # avoid zeros
                
            end = min(f0 + n_win_frames, n_frames)
            wf = end - f0
            w_slice = weights[:wf].unsqueeze(1)
            
            acc_main[f0:end] += logits_main[i].unsqueeze(0)[:wf] * w_slice
            acc_harm[f0:end] += logits_harmony[i].unsqueeze(0)[:wf] * w_slice
            acc_ad[f0:end] += logits_adlib[i].unsqueeze(0)[:wf] * w_slice
            cov[f0:end] += weights[:wf]
                
        batch_windows.clear()
        frame_starts.clear()
    
    # Assemble windows in small batches to speed up
    B = 64
    for s in starts:
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
                "harmonies", "harmony_probs",
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
                    harm_names_str,
                    harm_full_probs_str,
                    ad_names_str,
                    ad_probs_str,
                    "[" + ", ".join(f"{x:.3f}" for x in p_main) + "]",
                ]
                writer.writerow(row)
        print(f"✅ Saved predictions to {csv_path}")
        
    # Build segments
    labels_40ms = []

    for t in range(n_frames):
        main_id = int(main_idx_np[t])

        # --- Determine main name ---
        if main_id == silence_idx:
            main_name = "silence"
            main_is_valid = False
        else:
            main_name = class_names[main_id]
            main_is_valid = (main_name != "Gang Vocal")

        # --- Harmony (members only) ---
        if main_is_valid:
            harm_mask = decoded_harm_np[t].astype(bool)
            harm_ids = np.where(harm_mask)[0].tolist()
            harm_names = [member_names[j] for j in harm_ids]
        else:
            # No main singer → harmony makes no sense
            harm_names = []

        # --- Ad-lib (can exist with or without main, your choice) ---
        ad_mask = decoded_ad_np[t].astype(bool)
        ad_ids = np.where(ad_mask)[0].tolist()
        ad_names = [member_names[j] for j in ad_ids]

        labels_40ms.append({
            "main": main_name,
            "harmony": [],
            "adlib": [],
        })

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