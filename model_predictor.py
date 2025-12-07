import numpy as np
import torch, torchaudio, math
import itertools
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from train_kpop_singers import MultiTaskHead, extract_center_context
from scipy.ndimage import median_filter
import os, csv

@torch.no_grad()
def predict_40ms(
    encoder_path: str, head_path: str, wav_path: str,
    sr_target=16000, win_sec=2.0, hop_sec=0.04, use_hann=True,
    silence_idx=None, output_dir=None, class_names=None, 
    thr_main=0.6, thr_harm: float = 0.45, thr_adlib: float = 0.6
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
    encoder = EncoderClassifier.from_hparams(
        source=encoder_path,
        run_opts={"device": device.type}
    )
    
    ckpt = torch.load(head_path, map_location=device)
    base_dim = ckpt["emb_dim"] # 192
    model_classes = ckpt["classes"]
    if not class_names:
        class_names = list(model_classes)
        
    if silence_idx is None:
        silence_idx = len(model_classes) - 1
    
    num_main = len(model_classes)
    num_members = num_main - 1
    member_names = class_names[:num_members]
    
    fused_dim = base_dim * 2
    head = MultiTaskHead(fused_dim, num_members).to(device)
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
        emb_main = encoder.encode_batch(batch).squeeze(1)
        
        # --- CONTEXT EMBEDDING ---
        # 0.5 s center context per 2s window
        wavs_3d = batch.unsqueeze(1) # (B, 1, T)
        ctx_wavs = extract_center_context(wavs_3d, ctx_frac=0.25)  # (B, 1, T_ctx)
        ctx_ecapa = ctx_wavs.squeeze(1)                    
        emb_ctx = encoder.encode_batch(ctx_ecapa).squeeze(1)
        
        # fUSE + HEAD 
        emb_fused = torch.cat([emb_main, emb_ctx], dim=1)
        
        out = head(emb_fused)
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
    decoded_main_np = decode_multilabel(probs_main_np, per_class_thr=base_thr_main)
    
    # harmony / adlib thresholds are per member (no silence)
    base_thr_harm = np.full(num_members, thr_harm, dtype=np.float32)
    base_thr_ad = np.full(num_members, thr_adlib, dtype=np.float32)
    
    decoded_harm_np = decode_multilabel(probs_harm_np, per_class_thr=base_thr_harm)
    decoded_ad_np = decode_multilabel(probs_ad_np, per_class_thr=base_thr_ad)
    
    # --- Build main_idx_np (single "main" singer per frame) ---
    multi_hot_main = decoded_main_np.astype(bool)
    main_idx_np = np.zeros(n_frames, dtype=np.int64)
    
    for t in range(n_frames):
        p = probs_main_np[t] # (C, )
        active = multi_hot_main[t] # Bool mask over classes
        if active.any():
            active_ids = np.where(active)[0]
            # choose active class with highest prob
            best_local = active_ids[np.argmax(p[active_ids])]
            main_idx_np[t] = int(best_local)
        else:
            # no active class after smoothing
            if silence_idx is not None:
                main_idx_np[t] = silence_idx
            else:
                main_idx_np[t] = int(np.argmax(p))

    # Reduces single-frame blips and bridges tiny silence gaps
    main_idx_np = smooth_main_sequence(main_idx_np, silence_idx)
    pred_idx = torch.from_numpy(main_idx_np).to(device=device, dtype=torch.long)
        
    # ---- 4. Write predictions to .txt ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(wav_path))[0]
        csv_path = os.path.join(output_dir, f"{base}_predictions.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "start_time", "end_time",
                "main_label", "main_confidence",
                "active_labels", "active_confidences",
                "probabilities",
            ]
            writer.writerow(header)

            multi_hot_main_np = multi_hot_main
            pred_idx_np = pred_idx.cpu().numpy()

            for i in range(n_frames):
                start_t = i * hop_sec
                end_t = start_t + hop_sec

                p = probs_main_np[i]              # (C,)
                mh = multi_hot_main_np[i]

                main_id = int(pred_idx_np[i])
                main_names = class_names[main_id]
                main_conf = float(p[main_id])

                active_ids = np.where(mh)[0].tolist()
                active_names = [class_names[j] for j in active_ids]
                active_confs = [float(p[j]) for j in active_ids]

                row = [
                    round(start_t, 3),
                    round(end_t, 3),
                    main_names,
                    main_conf,
                    "|".join(map(str, active_names)),
                    "[" + ", ".join(f"{c:.3f}" for c in active_confs) + "]",
                    "[" + ", ".join(f"{x:.3f}" for x in p) + "]",
                ]
                writer.writerow(row)
        print(f"✅ Saved predictions to {csv_path}")
        
    # Build segments
    labels_40ms = []

    for t in range(n_frames):
        # main / lead
        main_id = int(main_idx_np[t])  

        if main_id == silence_idx:
            main_name = "" # silence → blank string
        else:
            main_name = class_names[main_id]


        # harmony (members only)
        harm_mask = decoded_harm_np[t].astype(bool)
        harm_ids = np.where(harm_mask)[0].tolist()
        harm_names = [member_names[j] for j in harm_ids]

        # ad-lib (members only)
        ad_mask = decoded_ad_np[t].astype(bool)
        ad_ids = np.where(ad_mask)[0].tolist()
        ad_names = [member_names[j] for j in ad_ids]
        
        labels_40ms.append({
            "main": main_name,
            "harmony": harm_names,
            "adlib": ad_names,
        })
        
    return labels_40ms

def smooth_probs(P, k=5):   # k odd: 5 or 7 works well for 40ms frames
    return median_filter(P, size=(k, 1))

def hysteresis_decode(p, on=0.60, off=0.45):
    y = np.zeros_like(p, dtype=np.int32)
    active = False
    for i, s in enumerate(p):
        if not active and s >= on:  active = True
        if active and s <= off:     active = False
        y[i] = 1 if active else 0
    return y

def min_duration(y, min_on=3, min_off=2):
    out = y.copy()
    for val, grp in itertools.groupby(range(len(y)), key=lambda i: y[i]):
        idx = list(grp)
        if val == 1 and len(idx) < min_on:  out[idx] = 0
        if val == 0 and len(idx) < min_off: out[idx] = 1
    return out

def cap_topk(P_row, k=2):
    keep = np.zeros_like(P_row, dtype=bool)
    keep[np.argsort(-P_row)[:k]] = True
    return keep

def decode_multilabel(probs, per_class_thr=None, k_smooth=5, on_add=0.02, off_sub=0.02,
                      min_on=3, min_off=2, topk=2):
    P = smooth_probs(probs, k=k_smooth)
    C = P.shape[1]
    base = np.full(C, 0.50, np.float32) if per_class_thr is None else np.asarray(per_class_thr, np.float32)
    on  = np.clip(base + on_add, 0, 1); off = np.clip(base - off_sub, 0, 1)

    Y = np.zeros_like(P, dtype=np.int32)
    for c in range(C):
        y = hysteresis_decode(P[:, c], on=on[c], off=off[c])
        Y[:, c] = min_duration(y, min_on=min_on, min_off=min_off)

    if topk is not None:
        for t in range(P.shape[0]):
            Y[t] = Y[t] * cap_topk(P[t], k=topk).astype(np.int32)
    return Y

def smooth_main_sequence(main_idx, silence_idx=None, min_singer_len=3, 
                         min_silence_len=1, bridge_silence_len=2):
    """
    main_idx: 1D np.array of class indices per frame (length T)
    silence_idx: which index represents silence, or None
    min_singer_len: minimum frames for a singer segment to be kept
    min_silence_len: (optional) minimum frames for a silence segment
    bridge_silence_len: if a silence run <= this and surrounded by same singer,
                        convert silence to that singer (gap-bridging).
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
                if prev_label is not None and prev_label == next_label and prev_label != silence_idx:
                    out[start:end] = prev_label
        
        start = end
    
    return out