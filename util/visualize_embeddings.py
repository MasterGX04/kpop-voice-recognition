import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def process_muq_files_with_pca(muq_dir, file_list, n_components=256):
    """
    Loads raw MuQ embeddings, applies PCA across all frame time-steps to extract
    the top 256 variance components, and computes the pooled mean and std vectors.
    """
    raw_chunks = {}
    frame_list = []
    
    # 1. Load raw chunks and collect all frames for fitting PCA
    for filename in file_list:
        filepath = os.path.join(muq_dir, filename)
        if not os.path.exists(filepath):
            continue
            
        emb = np.load(filepath).squeeze()
        if emb.ndim == 1:
            emb = np.expand_dims(emb, axis=0)  # Ensure shape is (num_frames, 1024)
            
        raw_chunks[filename] = emb
        frame_list.append(emb)
        
    if not frame_list:
        return {}

    # Stack all frames across all chunks: shape = (total_frames, 1024)
    all_frames = np.vstack(frame_list)
    
    # Dynamic component selection in case total frames < 256
    actual_components = min(n_components, all_frames.shape[0], all_frames.shape[1])
    
    # 2. Fit PCA on the unlabelled frame pool
    pca = PCA(n_components=actual_components, random_state=42)
    pca.fit(all_frames)
    
    # 3. Transform each chunk and compute pooled mean/std
    processed_dict = {}
    for filename, emb in raw_chunks.items():
        reduced_emb = pca.transform(emb)  # Shape: (num_frames, 256)
        
        mean_vec = np.mean(reduced_emb, axis=0)
        std_vec = np.std(reduced_emb, axis=0)
        pooled_vec = np.concatenate([mean_vec, std_vec])  # Shape: (512,)
        
        processed_dict[filename] = pooled_vec
        
    return processed_dict

def process_single_emb_file(filepath):
    """Fallback helper to process non-PCA embeddings (e.g., SpeechBrain)."""
    emb = np.load(filepath).squeeze()
    if emb.ndim == 1:
        emb = np.expand_dims(emb, axis=0)
        
    mean_vec = np.mean(emb, axis=0)
    std_vec = np.std(emb, axis=0)
    return np.concatenate([mean_vec, std_vec])

def visualize_embeddings(group_name, target_song_name=None, model_type='both'):
    # 1. Setup paths - Point explicitly to our two folders
    audit_dir = f"./audits/{group_name}"
    muq_dir = os.path.join(audit_dir, "embeddings_muq")
    sb_dir = os.path.join(audit_dir, "embeddings_sb")
    json_path = f"./group_icons/{group_name}/group.json"
    
    # Decide which directory acts as our "base" for finding filenames
    base_dir = sb_dir if model_type == 'sb' else muq_dir
    
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} not found.")
        return
        
    if not os.path.exists(json_path):
        print(f"Group JSON not found at {json_path}")
        return

    # 2. Extract Color Map from JSON
    with open(json_path, 'r') as f:
        group_data = json.load(f)
        
    color_map = {
        member["name"]: member["color"] 
        for member in group_data.get("members", [])
    }
    print(f"Loaded color map: {color_map}")

    # 3. Filter Paths based on target_song_name
    # We loop over FILENAMES now instead of full paths, so we can match them across directories
    base_files = [f for f in os.listdir(base_dir) if f.endswith('.npy')]
    filtered_files = []
    
    for filename in base_files:
        prefix, _ = filename.rsplit('_phrase', 1)
        member_name, current_song_name = prefix.split('_', 1)
        
        # Filtering logic
        if target_song_name is None:
            filtered_files.append(filename)
        elif target_song_name.lower() == 'solo':
            # Check if it's a solo track (has brackets in the name)
            if '[' in current_song_name and ']' in current_song_name:
                filtered_files.append(filename)
        else:
            # Exact match (case-insensitive)
            if target_song_name.lower() == current_song_name.lower():
                filtered_files.append(filename)
                
    if not filtered_files:
        print(f"No chunks found matching the song filter: '{target_song_name}'")
        return

    # 4. Pre-process MuQ embeddings with PCA if required
    muq_pca_data = {}
    if model_type in ['muq', 'both']:
        print("Extracting top 256 PCA components from MuQ embeddings...")
        muq_pca_data = process_muq_files_with_pca(muq_dir, filtered_files, n_components=256)
        
    # 5. Load and Group Embeddings by Song
    song_data = {}
    print(f"Loading {len(filtered_files)} filtered chunks using model: {model_type.upper()}...")
    
    for filename in filtered_files:
        prefix, _ = filename.rsplit('_phrase', 1)
        member_name, song_name = prefix.split('_', 1)
        
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
            
            # Concatenate PCA-reduced MuQ vector (512-D) with SpeechBrain vector (384-D)
            pooled_vec = np.concatenate([muq_vec, sb_vec])
        
        if song_name not in song_data:
            song_data[song_name] = []
            
        song_data[song_name].append((pooled_vec, member_name))
        
    # 5. Perform Intra-Song Standardization
    print("Applying intra-song standardization (skipping solo tracks)...")
    X_list = []
    y_list = []
    
    for song_name, chunks in song_data.items():
        song_vecs = np.array([item[0] for item in chunks])
        members = [item[1] for item in chunks]
        
        # 1. Identify if it's a solo song based on your bracket naming convention
        # Alternatively, check if there's only 1 unique singer in the entire song data
        is_solo = ("[" in song_name and "]" in song_name) or (len(set(members)) == 1)
        
        if is_solo:
            # Skip normalization: Leave the raw embeddings alone so their baseline isn't zeroed out
            norm_song_vecs = song_vecs
        else:
            # Apply normalization: Group song detected, strip out the track-level production mix
            song_mean = np.mean(song_vecs, axis=0, keepdims=True)
            song_std = np.std(song_vecs, axis=0, keepdims=True) + 1e-8
            norm_song_vecs = (song_vecs - song_mean) / song_std
        
        X_list.append(norm_song_vecs)
        y_list.extend(members)
        
    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"Total chunks processed: {X.shape[0]}. Feature dimensions: {X.shape[1]}")

    # 6. Global Scaling (This safely aligns everything together at the end)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 7. Dynamic n_neighbors and UMAP
    # UMAP requires n_neighbors to be less than or equal to the number of samples.
    # We cap it at 30 for global structure, but scale it down for small filtered datasets.
    n_samples = X_scaled.shape[0]
    dynamic_n_neighbors = min(5, max(2, n_samples - 1))
    
    print(f"Running UMAP (n_neighbors={dynamic_n_neighbors})...")
    reducer = umap.UMAP(
        n_neighbors=dynamic_n_neighbors, 
        min_dist=0.1, 
        metric='cosine', 
        random_state=42
    )
    embedding_2d = reducer.fit_transform(X_scaled)

    # 8. Plotting
    plt.figure(figsize=(10, 8))
    
    unique_members = np.unique(y)
    for member in unique_members:
        idx = np.where(y == member)
        member_color = color_map.get(member, "#808080")
        
        plt.scatter(
            embedding_2d[idx, 0],
            embedding_2d[idx, 1],
            label=member,
            color=member_color,
            alpha=0.7,
            s=20,
            edgecolors='none'
        )

    plt.legend(markerscale=2)
    
    title_suffix = f" ({target_song_name.capitalize()})" if target_song_name else " (All Songs)"
    plt.title(f"UMAP Projection of {group_name.capitalize()}'s Vocal Timbres{title_suffix}")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.show()
    
def main():
    parser = argparse.ArgumentParser(
        description="Visualize K-pop vocal embeddings using MuQ, SpeechBrain, or both."
    )
    
    parser.add_argument(
        "--group", 
        type=str, 
        required=True, 
        help="Name of the K-pop group (e.g., aespa)."
    )
    
    parser.add_argument(
        "--song_name", 
        type=str, 
        default=None, 
        help="Optional: Name of the song to visualize, or 'solo' for all solo tracks. If omitted, plots all songs."
    )

    parser.add_argument(
        "--model", 
        type=str, 
        choices=['muq', 'sb', 'both'], 
        default='both', 
        help="Model embeddings to load: 'muq', 'sb' (SpeechBrain), or 'both' (default: 'both')."
    )

    args = parser.parse_args()

    print(f"Starting analysis for Group: {args.group} | Model: {args.model.upper()} | Song: {args.song_name if args.song_name else 'All Songs'}")
    
    visualize_embeddings(
        group_name=args.group, 
        target_song_name=args.song_name, 
        model_type=args.model
    )

if __name__ == "__main__":
    main()