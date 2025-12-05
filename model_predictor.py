import numpy as np
import torch, torchaudio, math
import itertools
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from train_kpop_singers import MultiTaskHead
from scipy.ndimage import median_filter
import os, csv

@torch.no_grad()
def predict_40ms(
    encoder_path: str, head_path: str, wav_path: str,
    sr_target=16000, win_sec=2.0, hop_sec=0.04, use_hann=True,
    silence_idx=None, output_dir=None, class_names=None, thr=0.6
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderClassifier.from_hparams(
        source=encoder_path,
        run_opts={"device": device.type}
    )
    ckpt = torch.load(head_path, map_location=device)
    emb_dim = ckpt["emb_dim"]
    model_classes = ckpt["classes"]
    
    if not class_names:
        class_names = list(model_classes)
    
    head = MultiTaskHead(emb_dim, len(model_classes)).to(device)
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
    n_classes = head.fc.out_features
    acc = torch.zeros(n_frames, n_classes, device=device)
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
        nonlocal batch_windows, frame_starts, acc, cov
        if not batch_windows:
            return

        batch = torch.stack(batch_windows, 0)
        # ECAPA wants (B, T)
        emb = encoder.encode_batch(batch).squeeze(1) # (B, D)
        logits = head(emb) # (B, C)
        
        n_win_frames = math.ceil(win_len / hop_len)
        # Add into per-frame accumulators
        for i, f0 in enumerate(frame_starts):
            if w is None:
                acc[f0:f0 + n_win_frames] += logits[i]
                cov[f0:f0 + n_win_frames] += 1.0
            else:
                # Approximate Hann weighting per 40 ms frame
                j = torch.arange(n_win_frames, device=device)
                centers = (j * hop_len + min(hop_len, win_len) // 2).clamp(max=win_len - 1)
                weights = w[centers] + 1e-8  # avoid zeros
                acc[f0:f0 + n_win_frames] += logits[i] * weights.unsqueeze(1)
                cov[f0:f0 + n_win_frames] += weights
                
        batch_windows.clear()
        frame_starts.clear()
    
    # Assemble windows in small batches to speed up
    B = 64
    for s in starts:
        chunk = x[s:s + win_len]
        batch_windows.append(chunk)
        frame_starts.append(s // hop_len)
        if len(batch_windows) == B:
            flush_batch()
    flush_batch()
    
    # Normalize by coverage
    cov = cov.clamp_min(1e-6).unsqueeze(1) # n_frames, 1)
    logits_frame = acc / cov # (n_frames, C)
    probs_frame = torch.sigmoid(logits_frame) 
    # Move to CPU numpy for decode
    probs_np = probs_frame.detach().cpu().numpy()
    base_thr = np.full(probs_np.shape[1], thr, dtype=np.float32)
    
    # Multi-label decode with temperol smoothing
    decoded_np = decode_multilabel(probs_np, per_class_thr=base_thr)
    
    multi_hot = torch.from_numpy(decoded_np).to(device=device, dtype=torch.bool)
    
    # Build Main label sequence using probs + multi_hot
    multi_hot_np = decoded_np.astype(bool)
    main_idx_np = np.zeros(n_frames, dtype=np.int64)
    
    for t in range(n_frames):
        p = probs_np[t] # (C, )
        active = multi_hot_np[t] # Bool mask over classes
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
        
    # ---- 4. Write predictions to .txt (new) ----
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

            probs_np = probs_frame.cpu().numpy()
            multi_hot_np = multi_hot.cpu().numpy()
            pred_idx_np = pred_idx.cpu().numpy()

            for i in range(n_frames):
                start_t = i * hop_sec
                end_t = start_t + hop_sec

                p = probs_np[i]              # (C,)
                mh = multi_hot_np[i].astype(bool)

                main_id = int(pred_idx_np[i])
                main_name = class_names[main_id] if class_names else main_id
                main_conf = float(p[main_id])

                active_ids = np.where(mh)[0].tolist()
                active_names = [class_names[j] for j in active_ids] if class_names else active_ids
                active_confs = [float(p[j]) for j in active_ids]

                row = [
                    round(start_t, 3),
                    round(end_t, 3),
                    main_name,
                    main_conf,
                    "|".join(map(str, active_names)),
                    "[" + ", ".join(f"{c:.3f}" for c in active_confs) + "]",
                    "[" + ", ".join(f"{x:.3f}" for x in p) + "]",
                ]
                writer.writerow(row)
        print(f"✅ Saved predictions to {csv_path}")
        
    # Build segments
    labels_40ms = []
    multi_hot_np = multi_hot.cpu().numpy()

    for t in range(n_frames):
        active_ids = np.where(multi_hot_np[t].astype(bool))[0].tolist()
        active_names = [class_names[j] for j in active_ids] if class_names else active_ids
        labels_40ms.append(active_names)
        
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