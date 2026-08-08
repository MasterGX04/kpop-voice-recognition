import os, json, csv
import argparse
import torch
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. Dataset & DataLoader Definition
# ==========================================
class WavLMDataset(Dataset):
    def __init__(self, group_name, songs, base_dir="./audits"):
        self.file_paths = []
        for song in songs:
            song_dir = os.path.join(base_dir, group_name, "embeddings_wavlm", song)

            if os.path.exists(song_dir):
                for f in os.listdir(song_dir):
                    if f.endswith(".pt"):
                        # Now saving a tuple: (file_path, song_name)
                        self.file_paths.append((os.path.join(song_dir, f), song))
            else:
                print(f"Warning: No WavLM directory found for {song}")

        self.num_classes = 0
        if len(self.file_paths) > 0:
            sample_data = torch.load(self.file_paths[0][0], weights_only=True)
            self.num_classes = sample_data['labels'].shape[1]

        self.song_stats = self._compute_song_stats()

    def _compute_song_stats(self):
        """
        Computes a per-song (mean, std) over the raw WavLM feature vectors, pooling
        every frame from every window belonging to that song. Subtracting this at
        __getitem__ strips out whatever is constant across an entire song -- mix,
        mastering, backing instrumentation bleed -- so the classifier is pushed to
        key off the frame-to-frame timbre differences (i.e. who's singing) instead
        of memorizing each training song's overall production fingerprint.

        Solo tracks (bracketed names like "8[Wonyoung]") are skipped: with only one
        member ever singing, standardizing would zero out that member's own baseline
        instead of removing shared production artifacts.
        """
        song_frames = defaultdict(list)
        for file_path, song_name in self.file_paths:
            if "[" in song_name and "]" in song_name:
                continue
            data = torch.load(file_path, weights_only=True)
            song_frames[song_name].append(data['features'])

        song_stats = {}
        for song_name, feats_list in song_frames.items():
            all_feats = torch.cat(feats_list, dim=0)
            mean = all_feats.mean(dim=0, keepdim=True)
            std = all_feats.std(dim=0, keepdim=True) + 1e-8
            song_stats[song_name] = (mean, std)

        return song_stats

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path, song_name = self.file_paths[idx]
        data = torch.load(file_path, weights_only=True)
        features = data['features']

        if song_name in self.song_stats:
            mean, std = self.song_stats[song_name]
            features = (features - mean) / std

        # Return the song name alongside the tensors
        return features, data['labels'], song_name

# ==========================================
# 2. Multi-Label MLP Architecture
# ==========================================
class WavLMTimbreMLP(nn.Module):
    def __init__(self, input_dim=768, gru_hidden_dim=128, hidden_dim=256, num_classes=5, dropout=0.4):
        super().__init__()

        # A single isolated 20ms WavLM frame is a very thin sliver of audio to identify
        # a specific person's voice from. This bidirectional GRU lets every frame's
        # prediction draw on the whole window around it (both before and after) instead
        # of being classified in total isolation, before the per-frame classifier head runs.
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        gru_output_dim = gru_hidden_dim * 2

        self.network = nn.Sequential(
            nn.Linear(gru_output_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),

            # Outputs RAW logits. NO Softmax or Sigmoid here!
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input x is shape: [Batch_Size, T, input_dim]
        B, T, D = x.shape

        # Each output frame now carries context from nearby frames in both directions
        context, _ = self.gru(x)  # [B, T, gru_hidden_dim * 2]

        # Flatten to [Batch_Size * T, gru_hidden_dim * 2] for BatchNorm and Linear layers
        context_flat = context.reshape(B * T, -1)

        # Get predictions
        logits = self.network(context_flat)

        # Return shape [Batch_Size * T, num_classes]
        return logits

def get_member_names(group_name, num_classes, json_path="./groups.json"):
    """
    Reads groups.json to get the dynamic list of member names.
    Appends 'Gang Vocal' to the end.
    Falls back to C0, C1... if the file is missing or lengths don't match.
    """
    names = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if group_name in data and "members" in data[group_name]:
                # Extract names in the EXACT order they appear in the JSON
                names = [member["name"] for member in data[group_name]["members"]]
    except Exception as e:
        print(f"Warning: Could not read {json_path} - {e}")

    # Always append Gang Vocal as the final output node
    if len(names) > 0:
        names.append("Gang Vocal")

    # Safety check: If for some reason the JSON has 13 members but the .pt 
    # file only has 5 columns, fall back to C0, C1 so it doesn't crash.
    if len(names) != num_classes:
        print(f"Warning: JSON names ({len(names)}) don't match matrix columns ({num_classes}). Falling back to C-indexes.")
        names = [f"C{i}" for i in range(num_classes)]

    return names

def compute_pos_weight(dataset, num_classes):
    """
    Computes a per-class pos_weight for BCEWithLogitsLoss from the training set's
    actual label distribution: pos_weight[c] = (# negative frames) / (# positive frames).
    This offsets class imbalance (most members are silent most of the time) so the
    loss can't be minimized by just predicting everyone's low prior probability.
    """
    pos_counts = torch.zeros(num_classes)
    total_frames = 0

    for file_path, _ in dataset.file_paths:
        data = torch.load(file_path, weights_only=True)
        labels = data['labels']
        pos_counts += labels.sum(dim=0)
        total_frames += labels.shape[0]

    neg_counts = total_frames - pos_counts
    pos_weight = neg_counts / pos_counts.clamp(min=1.0)
    return pos_weight

def compute_classification_metrics(tp, fp, fn, tn):
    """
    Turns per-class TP/FP/FN/TN counts into Recall, Precision, F1, and Balanced
    Accuracy per class, plus macro-averages across classes.

    Plain thresholded accuracy is misleading here: most members are silent most
    of the time, so a model that always predicts "not singing" already scores
    ~85-90% without ever correctly identifying anyone. These metrics are built
    from the confusion counts directly, so a class the model never detects
    (Recall = 0%) shows up as exactly that, instead of being buried under a
    high score driven by the easy true negatives.
        Recall           = of the frames this member REALLY sang, what fraction did we catch?
        Precision        = of the frames we SAID this member sang, what fraction were right?
        F1               = harmonic mean of Recall and Precision (0 if the model never fires)
        Balanced Accuracy = average of Recall and Specificity (TNR) -- the imbalance-corrected
                            analogue of plain accuracy; ~50% is "no better than guessing"
                            regardless of how rare the positive class is.
    """
    recall = tp / (tp + fn).clamp(min=1e-8)
    precision = tp / (tp + fp).clamp(min=1e-8)
    specificity = tn / (tn + fp).clamp(min=1e-8)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    balanced_acc = (recall + specificity) / 2

    # Classes with zero true positives in this split can't have a meaningful
    # Recall/F1 -- report them as exactly 0 rather than a division-by-epsilon artifact.
    has_positives = (tp + fn) > 0
    recall = torch.where(has_positives, recall, torch.zeros_like(recall))
    f1 = torch.where(has_positives, f1, torch.zeros_like(f1))

    return {
        "recall": recall,
        "precision": precision,
        "specificity": specificity,
        "f1": f1,
        "balanced_acc": balanced_acc,
        "macro_f1": f1.mean().item(),
        "macro_balanced_acc": balanced_acc.mean().item(),
    }

# ==========================================
# 3. Main Training Function
# ==========================================
def train_model(group_name, train_songs, val_songs, epochs=50, batch_size=8, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if not train_songs:
        print(f"Error: No training songs found for group '{group_name}'")
        return

    train_dataset = WavLMDataset(group_name, train_songs)
    val_dataset = WavLMDataset(group_name, val_songs)
    
    num_classes = train_dataset.num_classes
    if num_classes == 0:
        print("Error: Could not determine num_classes from the .pt files.")
        return
        
    class_names = get_member_names(group_name, num_classes)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = WavLMTimbreMLP(input_dim=768    , num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Increased weight decay slightly to combat overfitting

    pos_weight = compute_pos_weight(train_dataset, num_classes)
    print("Pos weight per class (higher = rarer positive class, penalized harder when missed):")
    for name, w in zip(class_names, pos_weight.tolist()):
        print(f"   {name}: {w:.2f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    print(f"\n--- Starting Training on {device.upper()} ---")
    
    # --- SETUP CSV LOGGER ---
    csv_file_path = f"./logs/{group_name}_training_logs.csv"
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        headers = ["Epoch", "Train Loss", "Val Loss", "Flat Val Acc (misleading)", "Macro F1", "Balanced Acc"]
        headers += [f"Mem: {name}" for name in class_names]
        headers += [f"Recall: {name}" for name in class_names]
        headers += [f"F1: {name}" for name in class_names]
        headers += [f"Val Song: {s}" for s in sorted(val_songs)]
        headers += [f"Train Song: {s}" for s in sorted(train_songs)]
        writer.writerow(headers)

    os.makedirs("./models", exist_ok=True)
    model_save_path = f"./models/{group_name}_wavlm_mlp.pth"
    best_macro_f1 = -1.0
    macro_f1 = -1.0

    try:
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
        
            train_song_correct = defaultdict(float)
            train_song_total = defaultdict(float)
        
            # 1. Training Batch Loop (Indented 8 spaces inside train_model / 4 inside epoch)
            for features, labels, song_names in train_loader:
                B = features.size(0)
                features, labels = features.to(device), labels.to(device)

                min_seq_len = min(features.size(1), labels.size(1))
                features = features[:, :min_seq_len, :]
                labels = labels[:, :min_seq_len, :]

                optimizer.zero_grad()
                logits = model(features)
                labels_flat = labels.reshape(-1, num_classes)

                unreduced_loss = F.binary_cross_entropy_with_logits(
                    logits, labels_flat, reduction="none"
                )
                unreduced_loss = unreduced_loss.view(B, min_seq_len, num_classes)

                loss_weights = torch.ones_like(unreduced_loss)

                for i in range(B):
                    s_name = song_names[i]
                    if "[" in s_name and "]" in s_name:
                        solo_member = s_name.split("[")[1].split("]")[0]
                        if solo_member in class_names:
                            member_idx = class_names.index(solo_member)
                            member_mask = torch.zeros(num_classes, device=device)
                            member_mask[member_idx] = 1.0
                            loss_weights[i] = member_mask

                loss = (unreduced_loss * loss_weights).mean()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                with torch.no_grad():
                    probs = torch.sigmoid(logits)
                    preds_flat = (probs > 0.5).float()
                    preds_3d = preds_flat.view(B, min_seq_len, num_classes)

                    for i in range(B):
                        s_name = song_names[i]
                        train_song_correct[s_name] += (
                            (preds_3d[i] == labels[i]).float().sum().item()
                        )
                        train_song_total[s_name] += labels[i].numel()
                    
            # 2. Validation Phase (Un-indented back to 8 spaces - OUTSIDE train_loader loop!)
            model.eval()
            val_loss = 0.0
            correct_frames = 0
            total_frames = 0
        
            member_correct = torch.zeros(num_classes).to(device)
            member_total = 0

            tp = torch.zeros(num_classes).to(device)
            fp = torch.zeros(num_classes).to(device)
            fn = torch.zeros(num_classes).to(device)
            tn = torch.zeros(num_classes).to(device)

            val_song_correct = defaultdict(float)
            val_song_total = defaultdict(float)

            with torch.no_grad():
                for features, labels, song_names in val_loader:
                    B = features.size(0)
                    features, labels = features.to(device), labels.to(device)

                    min_seq_len = min(features.size(1), labels.size(1))
                    features = features[:, :min_seq_len, :]
                    labels = labels[:, :min_seq_len, :]

                    logits = model(features)
                    labels_flat = labels.reshape(-1, num_classes)

                    loss = criterion(logits, labels_flat)
                    val_loss += loss.item()

                    probabilities = torch.sigmoid(logits)
                    predictions_flat = (probabilities > 0.5).float()

                    correct_frames += (predictions_flat == labels_flat).float().sum().item()
                    total_frames += labels_flat.numel()
                    member_correct += (predictions_flat == labels_flat).float().sum(dim=0)
                    member_total += labels_flat.shape[0]

                    is_pred_pos = predictions_flat == 1
                    is_label_pos = labels_flat == 1
                    tp += (is_pred_pos & is_label_pos).float().sum(dim=0)
                    fp += (is_pred_pos & ~is_label_pos).float().sum(dim=0)
                    fn += (~is_pred_pos & is_label_pos).float().sum(dim=0)
                    tn += (~is_pred_pos & ~is_label_pos).float().sum(dim=0)

                    preds_3d = predictions_flat.view(B, min_seq_len, num_classes)
                    for i in range(B):
                        s_name = song_names[i]
                        val_song_correct[s_name] += (preds_3d[i] == labels[i]).float().sum().item()
                        val_song_total[s_name] += labels[i].numel()

            # 3. Summary & Printing (OUTSIDE train_loader loop)
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = (correct_frames / total_frames) * 100 if total_frames > 0 else 0.0

            member_accs = (member_correct / member_total) * 100
            member_acc_strs = [f"{name}: {acc.item():.1f}%" for name, acc in zip(class_names, member_accs)]
            member_str = " | ".join(member_acc_strs)

            metrics = compute_classification_metrics(tp, fp, fn, tn)
            recall_strs = [f"{name}: {r.item()*100:.1f}%" for name, r in zip(class_names, metrics["recall"])]
            f1_strs = [f"{name}: {v.item()*100:.1f}%" for name, v in zip(class_names, metrics["f1"])]
            macro_f1 = metrics["macro_f1"] * 100
            macro_bal_acc = metrics["macro_balanced_acc"] * 100

            print(f"Epoch {epoch+1:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | (misleading) Flat Val Acc: {val_accuracy:.2f}%")
            print(f"   -> Flat Acc per member (inflated by silence): [ {member_str} ]")
            print(f"   -> Recall (did we actually catch them singing?): [ {' | '.join(recall_strs)} ]")
            print(f"   -> F1: [ {' | '.join(f1_strs)} ] | Macro F1: {macro_f1:.2f}% | Balanced Acc: {macro_bal_acc:.2f}%")

            if macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                torch.save(model.state_dict(), model_save_path)
                print(f"   -> New best Macro F1 ({macro_f1:.2f}%). Saved checkpoint to {model_save_path}")

            # 4. CSV Logging (OUTSIDE train_loader loop)
            with open(csv_file_path, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                row = [
                    f"{epoch+1}",
                    f"{avg_train_loss:.4f}",
                    f"{avg_val_loss:.4f}",
                    f"{val_accuracy:.2f}",
                    f"{macro_f1:.2f}",
                    f"{macro_bal_acc:.2f}",
                ]
                row += [f"{acc.item():.2f}" for acc in member_accs]
                row += [f"{r.item()*100:.2f}" for r in metrics["recall"]]
                row += [f"{v.item()*100:.2f}" for v in metrics["f1"]]

                for s in sorted(val_songs):
                    s_acc = (val_song_correct[s] / val_song_total[s] * 100) if val_song_total[s] > 0 else 0.0
                    row.append(f"{s_acc:.2f}")
                
                for s in sorted(train_songs):
                    s_acc = (train_song_correct[s] / train_song_total[s] * 100) if train_song_total[s] > 0 else 0.0
                    row.append(f"{s_acc:.2f}")
                
                writer.writerow(row)
   
    except KeyboardInterrupt:
        print(f"\n\nTraining interrupted by user (Ctrl-C).")
        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            torch.save(model.state_dict(), model_save_path)
            print(f"Current Macro F1 ({macro_f1:.2f}%) is a new best. Saved checkpoint to {model_save_path}")
        elif best_macro_f1 >= 0:
            print(f"Best model (Macro F1: {best_macro_f1:.2f}%) already saved at {model_save_path}")
        else:
            print("No checkpoint was saved yet (interrupted before completing the first epoch).")
        print(f"Partial metrics written to: {csv_file_path}")
        return

    print(f"\nTraining complete. Best Macro F1: {best_macro_f1:.2f}%")
    print(f"Best model saved to: {model_save_path}")
    print(f"Full metrics written to: {csv_file_path}")
    
# ==========================================
# 4. Command Line Entry Point
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Train MLP Classifier on Cached WavLM Embeddings.")
    
    parser.add_argument("--group", type=str, required=True, help="Name of the group (e.g., aespa).")
    parser.add_argument("--val_songs", type=str, nargs="+", required=True, help="One or more song names to hold out for validation.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    args = parser.parse_args()

    # Automatically find all processed songs for this group
    base_dir = f"./audits/{args.group}/embeddings_wavlm"
    all_available_songs = []
    
    if os.path.exists(base_dir):
        # List all subdirectories (which represent the songs)
        all_available_songs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    else:
        print(f"[ERROR] Directory not found: {base_dir}")
        print("Make sure you have run the embedding processing step first.")
        return

    # Calculate train_songs by removing the validation songs
    val_songs_set = set(args.val_songs)
    train_songs = [song for song in all_available_songs if song not in val_songs_set]
    
    # Check if validation songs actually exist in the available data
    missing_val_songs = [song for song in args.val_songs if song not in all_available_songs]
    if missing_val_songs:
        print(f"[WARNING] The following validation songs were not found in {base_dir}: {missing_val_songs}")

    # Run the training loop
    train_model(
        group_name=args.group,
        train_songs=train_songs,
        val_songs=args.val_songs,
        epochs=args.epochs,
        lr=args.lr
    )

if __name__ == "__main__":
    main()