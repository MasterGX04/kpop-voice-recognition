import numpy as np
import torch, torchaudio, math
import torch.nn.functional as F
from speechbrain.inference.speaker import EncoderClassifier
from train_kpop_singers import EcapaHead
import os, csv

@torch.no_grad()
def predict_40ms(
    encoder_path: str, head_path: str, wav_path: str,
    sr_target=16000, win_sec=2.0, hop_sec=0.04, use_hann=True,
    min_frames=3, silence_idx=None, enter_sil=0.55, exit_sil=0.45,
    output_dir=None, class_names=[]
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = EncoderClassifier.from_hparams(
        source=encoder_path,
        run_opts={"device": device.type}
    )
    ckpt = torch.load(head_path, map_location=device)
    emb_dim = ckpt["emb_dim"]
    model_classes = ckpt["classes"]
    
    head = EcapaHead(ckpt["emb_dim"], len(ckpt["classes"])).to(device)
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
    n_classes = head.classifier.out_features
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
        # Add into per-frame accumulators
        for i, f0 in enumerate(frame_starts):
            # Window spans frames
            n_win_frames = math.ceil(win_len / hop_len)
            # weight per frame (derived from Hann over samples, approximated uniformly here)
            if w is None:
                # Uniform frame weights
                acc[f0:f0 + n_win_frames] += logits[i]
                cov[f0:f0 + n_win_frames] += 1
            else:
                # Compute per-frame weights from the Hann window
                # Sample midpoint per frame segment
                j = torch.arange(n_win_frames, device=device)
                centers = (j * hop_len + min(hop_len, win_len) // 2).clamp(max=win_len-1)
                weights = w[centers] + 1e-8  # avoid exact zero
                acc[f0:f0 + n_win_frames] += logits[i] * weights.unsqueeze(1)
                cov[f0:f0 + n_win_frames] += weights
        batch_windows.clear()
        frame_starts.clear()
    
    # Assemble windows in small batches to speed up
    B = 128
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
    probs_frame = F.softmax(logits_frame, dim=-1)
    
    # Raw 40 ms labels
    pred_idx = probs_frame.argmax(dim=-1) # (n_frames,)
    
    # Post processing: min_duration + silence
    if min_frames > 1:
        pred_idx = _enforce_min_duration(pred_idx, min_frames)
    
    if silence_idx is not None:
        pred_idx = _hysteresis_silence(pred_idx, probs_frame[:, silence_idx], silence_idx, enter_sil, exit_sil)
    
    # ---- 4. Write predictions to .txt (new) ----
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(wav_path))[0]
        csv_path = os.path.join(output_dir, f"{base}_predictions.csv")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            header = [
                "start_time", "end_time",
                "predicted_label", "confidence",
                "top2_label", "top2_confidence",
                "probabilities"
            ]
            writer.writerow(header)
            probs_np = probs_frame.cpu().numpy()
            for i in range(len(pred_idx)):
                start_t = i * hop_sec
                end_t = start_t + hop_sec
                p = probs_np[i]
                top2 = np.argsort(p)[-2:][::-1]
                pred_name = class_names[int(pred_idx[i])] if class_names else int(pred_idx[i])
                top2_name = class_names[top2[1]] if class_names else int(top2[1])
                row = [
                    round(start_t, 3),
                    round(end_t, 3),
                    pred_name,
                    float(p[int(pred_idx[i])]),
                    top2_name,
                    float(p[top2[1]]),
                    "[" + ", ".join(f"{x:.3f}" for x in p) + "]"
                ]
                writer.writerow(row)
        print(f"✅ Saved predictions to {csv_path}")
        
    # Build segments
    segments = []
    cur = int(pred_idx[0]); start = 0
    for i in range(1, len(pred_idx)):
        if int(pred_idx[i]) != cur:
            segments.append((start, i, cur))
            start = i; cur = int(pred_idx[i])
    segments.append((start, len(pred_idx), cur))

    labels_40ms = [[class_names[i]] for i in pred_idx] 
    return labels_40ms

def _enforce_min_duration(pred_idx: torch.Tensor, k: int) -> torch.Tensor:
    # Merge runs shorter than k frames into the neighbor with higher count
    y = pred_idx.clone()
    i = 0
    while i < len(y):
        j = i + 1
        while j < len(y) and y[j] == y[i]: j += 1
        if (j - i) < k:
            left = y[i-1] if i > 0 else None
            right = y[j] if j < len(y) else None
            y[i:j] = left if left is not None else right
        i = j
    return y

def _hysteresis_silence(pred_idx, p_sil, silence_idx, enter=0.55, exit=0.45):
    # Raise bar to switch into silence, lower it to exit.
    y = pred_idx.clone()
    in_sil = False
    for t in range(len(y)):
        if in_sil:
            if p_sil[t] < exit:
                in_sil = False
                # choose best non-silence class
                # (you can carry over from last non-silence instead for stability)
        else:
            if p_sil[t] > enter:
                y[t] = silence_idx
                in_sil = True
    return y