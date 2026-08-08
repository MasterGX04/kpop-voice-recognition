import os, json, argparse
import torch

from wavlm_mlp_train import WavLMTimbreMLP, get_member_names

# ==========================================
# 1. Load a trained checkpoint
# ==========================================
def load_model(group_name, models_dir="./models", groups_json="./groups.json"):
    ckpt_path = os.path.join(models_dir, f"{group_name}_wavlm_mlp.pth")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    # gru.weight_ih_l0 has shape [3 * gru_hidden_dim, input_dim] (GRU packs 3 gates together)
    input_dim = state_dict["gru.weight_ih_l0"].shape[1]
    gru_hidden_dim = state_dict["gru.weight_ih_l0"].shape[0] // 3
    num_classes = state_dict["network.8.weight"].shape[0]

    class_names = get_member_names(group_name, num_classes, json_path=groups_json)

    model = WavLMTimbreMLP(input_dim=input_dim, gru_hidden_dim=gru_hidden_dim, num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.eval()
    return model, input_dim, class_names

# ==========================================
# 2. Run the model over every cached window for a song and
#    stitch overlapping windows back into one continuous timeline
# ==========================================
def stitch_song_predictions(group_name, song_name, model, input_dim, base_dir="./audits", stride_sec=8.0, fps=50):
    song_dir = os.path.join(base_dir, group_name, "embeddings_wavlm", song_name)
    if not os.path.exists(song_dir):
        raise FileNotFoundError(f"No embeddings found at {song_dir}")

    window_files = sorted(f for f in os.listdir(song_dir) if f.endswith(".pt"))
    if not window_files:
        raise FileNotFoundError(f"No .pt window files inside {song_dir}")

    stride_frames = int(stride_sec * fps)
    num_classes = model.network[-1].out_features

    windows = []
    max_end = 0
    for fname in window_files:
        window_idx = int(fname.replace("window_", "").replace(".pt", ""))
        data = torch.load(os.path.join(song_dir, fname), map_location="cpu", weights_only=True)
        features = data["features"]  # [T, D]

        if features.shape[-1] != input_dim:
            raise ValueError(
                f"Checkpoint expects {input_dim}-dim features but {fname} has "
                f"{features.shape[-1]}-dim features. The cached embeddings for "
                f"'{group_name}' don't match the model that was trained on -- "
                f"regenerate embeddings with the matching extractor or retrain "
                f"before running inference."
            )

        start_frame = window_idx * stride_frames
        windows.append((start_frame, features))
        max_end = max(max_end, start_frame + features.shape[0])

    prob_sum = torch.zeros(max_end, num_classes)
    prob_count = torch.zeros(max_end, 1)

    with torch.no_grad():
        for start_frame, features in windows:
            T = features.shape[0]
            logits = model(features.unsqueeze(0))  # -> [T, num_classes]
            probs = torch.sigmoid(logits)
            prob_sum[start_frame:start_frame + T] += probs
            prob_count[start_frame:start_frame + T] += 1.0

    prob_count = prob_count.clamp(min=1.0)
    return prob_sum / prob_count  # [total_frames_at_50hz, num_classes]

# ==========================================
# 3. Collapse 50Hz predictions back to the 25fps chunks used in
#    saved_labels/*_labels.json, and turn them into [member, start, end, is_bg, is_adlib] runs
# ==========================================
def probs_to_label_segments(avg_probs, class_names, threshold=0.5):
    T = avg_probs.shape[0]
    T_25fps = T // 2
    # Ground truth labels were built at 25fps then np.repeat(x, 2) to reach WavLM's 50Hz.
    # Average each duplicated pair back down before thresholding.
    paired = avg_probs[: T_25fps * 2].view(T_25fps, 2, -1).mean(dim=1)
    binary = (paired > threshold).int()  # [T_25fps, num_classes]

    segments = []
    for class_idx, member in enumerate(class_names):
        col = binary[:, class_idx].tolist()
        active = False
        start = 0
        for i, val in enumerate(col):
            if val == 1 and not active:
                active, start = True, i
            elif val == 0 and active:
                active = False
                segments.append([member, start, i, False, False])
        if active:
            segments.append([member, start, T_25fps, False, False])

    segments.sort(key=lambda seg: seg[1])
    return segments

# ==========================================
# 4. Entry point: writes ./predicted_labels/{group}/{song}_labels.json
#    in the exact same format as ./saved_labels/{group}/{song}_labels.json
# ==========================================
def generate_predicted_labels(group_name, song_name, out_dir="./predicted_labels", threshold=0.5):
    model, input_dim, class_names = load_model(group_name)
    avg_probs = stitch_song_predictions(group_name, song_name, model, input_dim)
    segments = probs_to_label_segments(avg_probs, class_names, threshold=threshold)

    song_out_dir = os.path.join(out_dir, group_name)
    os.makedirs(song_out_dir, exist_ok=True)
    out_path = os.path.join(song_out_dir, f"{song_name}_labels.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(segments, f)

    print(f"Saved {len(segments)} predicted segments to {out_path}")
    return out_path

def main():
    parser = argparse.ArgumentParser(
        description="Run a trained WavLM MLP head over cached embeddings and export predictions in saved_labels format."
    )
    parser.add_argument("--group", required=True, help="Name of the group (e.g., aespa).")
    parser.add_argument("--song", required=True, help="Song name, matching the folder under audits/{group}/embeddings_wavlm/")
    parser.add_argument("--threshold", type=float, default=0.5, help="Sigmoid threshold for a class being 'active'.")
    args = parser.parse_args()

    generate_predicted_labels(args.group, args.song, threshold=args.threshold)

if __name__ == "__main__":
    main()
