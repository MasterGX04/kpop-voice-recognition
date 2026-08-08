import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Import pre-built extraction helpers
from visualize_embeddings import process_muq_files_with_pca, process_single_emb_file

class VocalDataset(Dataset):
    """PyTorch Dataset for pooled audio chunk embeddings."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class TimbreMLP(nn.Module):
    """Multi-Layer Perceptron with Bottleneck for Vocal Identity Classification."""
    def __init__(self, input_dim, hidden_dim, bottleneck_dim, num_classes, dropout=0.3):
        super(TimbreMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(bottleneck_dim, num_classes)
        )

    def forward(self, x):
        return self.network(x)

    def extract_bottleneck(self, x):
        """Extracts 64-D feature representations prior to the final classification layer."""
        # Pass through up to second BatchNorm+ReLU block
        for i in range(7):
            x = self.network[i](x)
        return x
    
def load_and_prepare_data(group_name, val_songs, model_type='both'):
    audit_dir = f"./audits/{group_name}"
    muq_dir = os.path.join(audit_dir, "embeddings_muq")
    sb_dir = os.path.join(audit_dir, "embeddings_sb")

    base_dir = sb_dir if model_type == 'sb' else muq_dir
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Base embeddings directory not found: {base_dir}")

    all_files = [f for f in os.listdir(base_dir) if f.endswith('.npy')]
    
    # 1. Pre-process MuQ embeddings via PCA
    muq_pca_data = {}
    if model_type in ['muq', 'both']:
        print("Fitting PCA (256 components) on MuQ embeddings...")
        muq_pca_data = process_muq_files_with_pca(muq_dir, all_files, n_components=256)

    # 2. Group chunks by song and separate into Train / Validation sets
    train_song_data = {}
    val_song_data = {}
    
    val_songs_lower = [s.lower() for s in val_songs]

    for filename in all_files:
        prefix, _ = filename.rsplit('_phrase', 1)
        member_name, song_name = prefix.split('_', 1)

        # Assemble features based on model choice
        if model_type == 'muq':
            if filename not in muq_pca_data:
                continue
            pooled_vec = muq_pca_data[filename]

        elif model_type == 'sb':
            sb_path = os.path.join(sb_dir, filename)
            if not os.path.exists(sb_path):
                continue
            pooled_vec = process_single_emb_file(sb_path)

        elif model_type == 'both':
            sb_path = os.path.join(sb_dir, filename)
            if filename not in muq_pca_data or not os.path.exists(sb_path):
                continue
            muq_vec = muq_pca_data[filename]
            sb_vec = process_single_emb_file(sb_path)
            pooled_vec = np.concatenate([muq_vec, sb_vec])

        # Assign to train or validation split based on song name
        target_dict = val_song_data if song_name.lower() in val_songs_lower else train_song_data
        
        if song_name not in target_dict:
            target_dict[song_name] = []
        target_dict[song_name].append((pooled_vec, member_name))

    # 3. Apply Intra-Song Standardization independently per split
    def standardize_and_flatten(song_data_dict):
        X_list, y_list = [], []
        for song_name, chunks in song_data_dict.items():
            song_vecs = np.array([item[0] for item in chunks])
            members = [item[1] for item in chunks]

            is_solo = ("[" in song_name and "]" in song_name) or (len(set(members)) == 1)
            if not is_solo and len(song_vecs) > 1:
                song_mean = np.mean(song_vecs, axis=0, keepdims=True)
                song_std = np.std(song_vecs, axis=0, keepdims=True) + 1e-8
                song_vecs = (song_vecs - song_mean) / song_std

            X_list.append(song_vecs)
            y_list.extend(members)

        return np.vstack(X_list), np.array(y_list)

    X_train, y_train_labels = standardize_and_flatten(train_song_data)
    X_val, y_val_labels = standardize_and_flatten(val_song_data)

    # 4. Global Scaling fitted strictly on training data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # 5. Encode Labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)

    return X_train, y_train, X_val, y_val, label_encoder

def train_mlp(group_name, val_songs, model_type='wavlm', epochs=50, batch_size=32, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    X_train, y_train, X_val, y_val, label_encoder = load_and_prepare_data(
        group_name, val_songs, model_type=model_type
    )

    print(f"\nDataset Statistics:")
    print(f"  Train Samples: {len(X_train)} chunks")
    print(f"  Validation Samples: {len(X_val)} chunks (Songs: {val_songs})")
    print(f"  Feature Dimensions: {X_train.shape[1]}")
    print(f"  Classes ({len(label_encoder.classes_)}): {list(label_encoder.classes_)}")

    train_loader = DataLoader(VocalDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(VocalDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    num_classes = len(label_encoder.classes_)
    model = TimbreMLP(
        input_dim=X_train.shape[1], 
        hidden_dim=128, 
        bottleneck_dim=32, 
        num_classes=num_classes
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    print("\nStarting Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            train_correct += (outputs.argmax(1) == y_batch).sum().item()

        train_loss /= len(X_train)
        train_acc = train_correct / len(X_train)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        # --- NEW: Trackers for individual member accuracy ---
        class_correct = {i: 0 for i in range(num_classes)}
        class_total = {i: 0 for i in range(num_classes)}

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                
                preds = outputs.argmax(1)
                val_loss += loss.item() * len(y_batch)
                val_correct += (preds == y_batch).sum().item()
                
                # --- NEW: Calculate per-class stats for this batch ---
                correct_tensor = (preds == y_batch)
                for i in range(num_classes):
                    # Create a mask for where the true label is class 'i'
                    class_mask = (y_batch == i)
                    class_total[i] += class_mask.sum().item()
                    # Sum the correct predictions only for class 'i'
                    class_correct[i] += correct_tensor[class_mask].sum().item()

        val_loss /= len(X_val)
        val_acc = val_correct / len(X_val)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} - Acc: {train_acc*100:.1f}% | Val Loss: {val_loss:.4f} - Acc: {val_acc*100:.1f}%")
            
            # --- NEW: Format and print the member breakdown ---
            member_acc_strings = []
            for i in range(num_classes):
                if class_total[i] > 0: # Prevent division by zero if a member isn't in the validation set
                    m_acc = 100.0 * class_correct[i] / class_total[i]
                    member_name = label_encoder.classes_[i]
                    member_acc_strings.append(f"{member_name}: {m_acc:.1f}%")
            
            print(f"   -> Member Val Acc: [ {' | '.join(member_acc_strings)} ]")

    print("\nTraining complete.")
    
def main():
    parser = argparse.ArgumentParser(description="Train MLP Classifier on Combined Vocal Embeddings.")
    
    parser.add_argument("--group", type=str, required=True, help="Name of the group (e.g., aespa).")
    parser.add_argument("--val_songs", type=str, nargs="+", required=True, help="One or more song names to hold out for validation.")
    parser.add_argument("--model", type=str, choices=['wavlm', 'sb'], default='wavlm', help="Model embeddings to use.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")

    args = parser.parse_args()

    train_mlp(
        group_name=args.group, 
        val_songs=args.val_songs, 
        model_type=args.model, 
        epochs=args.epochs,
        lr=args.lr
    )


if __name__ == "__main__":
    main()