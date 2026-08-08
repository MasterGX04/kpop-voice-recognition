import os, argparse
import glob
import json
import numpy as np
import pyloudnorm as pyln
import torch
import librosa
from phrase_extraction import extract_all_phrases
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

try:
    from speechbrain.inference.speaker import EncoderClassifier
    from speechbrain.utils.fetching import LocalStrategy
except ImportError:
    pass # Handled gracefully if the user only runs MuQ

def load_and_normalize_audio(wav_path, target_sr=24000, target_lufs=-23.0):
    # 1. Load and resample directly in RAM
    wav, sr = librosa.load(wav_path, sr=target_sr)

    # 2. Measure integrated loudness
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(wav)

    # 3. Apply transparent gain adjustment
    # Handles potential edge cases like silent files gracefully
    if not np.isinf(loudness):
        normalized_wav = pyln.normalize.loudness(
            wav, loudness, target_lufs
        )
    else:
        normalized_wav = wav

    return normalized_wav, sr

def split_long_phrase(phrase_audio, sr, target_max_sec=8.0, fade_ms=15):
    """
    Splits long continuous audio phrases into even ~6-8 second sub-chunks.
    Applies a micro-fade (15ms) to the boundaries to eliminate cut clicks/pops
    without relying on unpredictable amplitude thresholds.
    """
    total_samples = len(phrase_audio)
    max_samples = int(target_max_sec * sr)
    
    # If the phrase is already within safe limits, return as-is
    if total_samples <= max_samples:
        return [phrase_audio]
        
    # Calculate even split count so chunks are balanced (e.g., a 24s verse becomes 3 x 8s chunks)
    num_splits = int(np.ceil(total_samples / max_samples))
    chunk_size = total_samples // num_splits
    
    fade_samples = int((fade_ms / 1000.0) * sr)
    chunks = []
    
    for i in range(num_splits):
        start = i * chunk_size
        # The last chunk grabs whatever remaining audio is left at the end
        end = (i + 1) * chunk_size if i < num_splits - 1 else total_samples
        
        sub_chunk = phrase_audio[start:end].copy()
        
        # Apply a micro linear fade-in and fade-out to prevent boundary popping
        if len(sub_chunk) > (2 * fade_samples):
            fade_in = np.linspace(0.0, 1.0, fade_samples)
            fade_out = np.linspace(1.0, 0.0, fade_samples)
            
            sub_chunk[:fade_samples] *= fade_in
            sub_chunk[-fade_samples:] *= fade_out
            
        chunks.append(sub_chunk)
        
    return chunks

def process_speechbrain_embeddings(group_name, song_name, max_gap_sec=3.0, min_duration_sec=0.5, strict_clean=True, delete_old=False):
    """
    Extracts SpeechBrain ECAPA-TDNN embeddings on a phrase-by-phrase basis.
    Phrases are generated using extract_all_phrases to preserve continuous vocal context.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_opts = {"device": device}
    
    print("Loading SpeechBrain ECAPA-TDNN model...")
    try:
        classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb", 
            savedir="tmpdir",
            run_opts=run_opts,
            local_strategy=LocalStrategy.COPY
        )
    except NameError:
        print("Error: SpeechBrain is not installed. Please run 'pip install speechbrain'.")
        return
    
    audio_dir = f"./training_data/{group_name}"
    out_dir = f"./audits/{group_name}/embeddings_sb"
    os.makedirs(out_dir, exist_ok=True)
    
    wav_paths = []
    if not os.path.exists(audio_dir):
        print(f"Directory not found: {audio_dir}")
        return
        
    all_files = os.listdir(audio_dir)
    
    if song_name.lower() == 'solo':
        for f in all_files:
            if "[" in f and "]" in f and f.endswith("_vocals.wav"):
                wav_paths.append(os.path.join(audio_dir, f))
    else:
        target_file = f"{song_name}_vocals.wav"
        if target_file in all_files:
            wav_paths.append(os.path.join(audio_dir, target_file))
            
    if not wav_paths:
        print(f"No audio files found for {song_name} in {audio_dir}")
        return

    for wav_path in wav_paths:
        base_filename = os.path.basename(wav_path).replace("_vocals.wav", "")
        
        # --- NEW DELETION LOGIC ---
        if delete_old:
            # Find only files that match this specific song
            old_files = glob.glob(os.path.join(out_dir, f"*_{base_filename}_phrase*.npy"))
            if old_files:
                print(f"Cleaning up {len(old_files)} old embedding files for {base_filename}...")
                for f in old_files:
                    try:
                        os.remove(f)
                    except OSError as e:
                        print(f"Error deleting {f}: {e}")
                        
        # 1. Extract continuous phrases using turn-based phrase logic
        phrases, totals = extract_all_phrases(group_name, base_filename, max_gap_sec=max_gap_sec, strict_clean=strict_clean)
        
        if not phrases:
            print(f"No valid phrases found for {base_filename}. Skipping.")
            continue
            
        print(f"\nProcessing SpeechBrain audio for {base_filename} ({len(phrases)} base phrases total)...")
        wav, sr = load_and_normalize_audio(wav_path, target_sr=16000)
        
        # Track phrase numbering per member to maintain unique indexed filenames
        member_phrase_counts = {}
        
        for p in phrases:
            member_name = p["member"]
            start_chunk = p["start_chunk"]
            end_chunk = p["end_chunk"]
            duration_sec = p["duration_sec"]
            
            # Skip extremely short micro-phrases
            if duration_sec < min_duration_sec:
                continue
            
            start_sample = int(start_chunk * 0.04 * sr)
            end_sample = int(end_chunk * 0.04 * sr)
            
            phrase_audio = wav[start_sample:end_sample]
            if len(phrase_audio) == 0:
                continue
                
            # 2. Slice the audio if it exceeds our safe limit
            sub_chunks = split_long_phrase(phrase_audio, sr, target_max_sec=8.0, fade_ms=15)
            
            # 3. Process each sub-chunk
            for sub_chunk in sub_chunks:
                # Convert audio phrase into PyTorch tensor [batch, time]
                wav_tensor = torch.tensor(sub_chunk, dtype=torch.float32).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # SpeechBrain attentive pooling handles variable-length inputs natively
                    embeddings = classifier.encode_batch(wav_tensor)
                    embedding = embeddings.squeeze().cpu().numpy()
                    
                idx = member_phrase_counts.get(member_name, 0)
                member_phrase_counts[member_name] = idx + 1
                
                out_filename = f"{member_name}_{base_filename}_phrase{idx:03d}.npy"
                out_filepath = os.path.join(out_dir, out_filename)
                np.save(out_filepath, embedding)
            
        print(f"Successfully saved phrase embeddings for {base_filename}.")

    print(f"\nAll done! SpeechBrain phrase embeddings saved to {out_dir}")

def process_all_group_songs(group_name, max_gap_sec=3.0, min_duration_sec=0.5, strict_clean=True, model_type='speechbrain', delete_old=False):
    """
    Scans for all labeled songs for a group, verifies the audio file exists, 
    and batches them through the embedding extractor.
    """
    labels_dir = f"./saved_labels/{group_name}"
    audio_dir = f"./training_data/{group_name}"
    
    if not os.path.exists(labels_dir):
        print(f"Error: Labels directory not found at {labels_dir}")
        return
        
    # 1. Find all label JSON files for this group
    label_files = glob.glob(os.path.join(labels_dir, "*_labels.json"))
    
    if not label_files:
        print(f"No label files found in {labels_dir}")
        return
        
    valid_songs = []
    
    # 2. Extract song names and check for matching audio files
    for label_path in label_files:
        filename = os.path.basename(label_path)
        song_name = filename.replace("_labels.json", "")
        
        vocal_file_path = os.path.join(audio_dir, f"{song_name}_vocals.wav")
        
        if os.path.exists(vocal_file_path):
            valid_songs.append(song_name)
        else:
            print(f"Skipping '{song_name}': Found labels, but no audio file at {vocal_file_path}")
            
    if not valid_songs:
        print(f"No matching vocal files found for any labels in {group_name}. Aborting.")
        return
        
    print(f"\nFound {len(valid_songs)} valid songs for {group_name}. Starting batch extraction...")
    
    # 3. Call the embedding extractor for all valid songs
    for song in valid_songs:
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SONG: {song}")
        print(f"{'='*60}")
        
        if model_type == 'speechbrain':
            process_speechbrain_embeddings(
                group_name, 
                song, 
                max_gap_sec=max_gap_sec, 
                min_duration_sec=min_duration_sec, 
                strict_clean=strict_clean,
                delete_old=delete_old
            )
        elif model_type == 'wavlm':
            process_wavlm_sliding_windows(
                group_name,
                song    
            )
            
    print(f"\nBatch processing complete for all {len(valid_songs)} songs!")
    
def process_wavlm_sliding_windows(group_name, song_name, window_sec=10.0, stride_sec=8.0):
    """
    Extracts WavLM embeddings and dynamically aligned JSON label matrices 
    using a sliding window over the full song.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Setup Paths
    wav_path = f"./training_data/{group_name}/{song_name}_vocals.wav"
    json_path = f"./saved_labels/{group_name}/{song_name}_labels.json"
    groups_json_path = "./groups.json"
    out_dir = f"./audits/{group_name}/embeddings_wavlm/{song_name}/"
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(wav_path) or not os.path.exists(json_path):
        print(f"Missing audio or labels for {song_name}. Skipping.")
        return

    # 2. Dynamically Load Group Members
    member_to_idx = {}
    try:
        with open(groups_json_path, 'r', encoding='utf-8') as f:
            groups_data = json.load(f)
            if group_name in groups_data and "members" in groups_data[group_name]:
                # Map each member to an index in the exact order they appear
                for idx, member in enumerate(groups_data[group_name]["members"]):
                    member_to_idx[member["name"]] = idx
            else:
                print(f"[ERROR] Group '{group_name}' not found in {groups_json_path} or missing 'members' array.")
                return
    except Exception as e:
        print(f"[ERROR] Could not read {groups_json_path}: {e}")
        return
        
    # Always append Gang Vocal at the end
    member_to_idx["Gang Vocal"] = len(member_to_idx)
    num_classes = len(member_to_idx)
    print(f"Loaded {num_classes} classes for {group_name}: {member_to_idx}")

    # 3. Load JSON Labels
    with open(json_path, 'r') as f:
        labels_json = json.load(f)

    # 4. Load WavLM Model & Extractor
    print("Loading WavLM Base+ model...")
    processor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus")
    model = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(device)
    model.eval() # Freeze WavLM backbone

    # 5. Load Audio (Must be 16kHz for WavLM)
    print(f"Processing {song_name} audio...")
    wav, sr = librosa.load(wav_path, sr=16000)
    total_duration = len(wav) / sr

    # 6. Dynamic Helper function to build the Target Matrix
    def build_window_labels_dynamic(start_sec, end_sec, fps=25):
        start_chunk = int(start_sec * fps)
        end_chunk = int(end_sec * fps)
        total_chunks = end_chunk - start_chunk
        
        # Initialize pure zeroes with dynamic num_classes
        target_matrix = np.zeros((total_chunks, num_classes), dtype=np.float32)
        
        for entry in labels_json:
            member, lbl_start, lbl_end, is_bg, is_adlib = entry
            
            # If the label name isn't in our dynamic dictionary, ignore it
            if member not in member_to_idx:
                continue
                
            overlap_start = max(start_chunk, lbl_start)
            overlap_end = min(end_chunk, lbl_end)
            
            if overlap_start < overlap_end:
                rel_start = overlap_start - start_chunk
                rel_end = overlap_end - start_chunk
                m_idx = member_to_idx[member]
                target_matrix[rel_start:rel_end, m_idx] = 1.0
                
        # Repeat every 40ms row twice to match WavLM's 20ms (50Hz) native resolution
        target_matrix_50hz = np.repeat(target_matrix, 2, axis=0)
        return torch.tensor(target_matrix_50hz)

    # 7. Sliding Window Loop
    current_sec = 0.0
    window_idx = 0
    
    with torch.no_grad():
        while current_sec + window_sec <= total_duration:
            end_sec = current_sec + window_sec
            
            # Slice audio
            start_sample = int(current_sec * sr)
            end_sample = int(end_sec * sr)
            audio_slice = wav[start_sample:end_sample]
            
            # Extract WavLM Embedding Tensor (Shape: [1, 500, 768])
            inputs = processor(audio_slice, sampling_rate=sr, return_tensors="pt").to(device)
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state.cpu().squeeze(0) # [500, 768]
            
            # Extract corresponding Label Tensor dynamically
            label_matrix = build_window_labels_dynamic(current_sec, end_sec)
            
            # Only save windows that actually contain vocals (sum > 0)
            if label_matrix.sum() > 0:
                torch.save({
                    'features': hidden_states,
                    'labels': label_matrix
                }, os.path.join(out_dir, f"window_{window_idx:04d}.pt"))
            
            current_sec += stride_sec
            window_idx += 1

    print(f"Finished {song_name}. Saved {window_idx} synchronized embedding/label blocks.")
    
def main():
    parser = argparse.ArgumentParser(
        description="Extract audio embeddings for K-pop vocal chunks."
    )
    
    parser.add_argument(
        "--type",
        type=str,
        required=True,
        choices=['speechbrain', 'wavlm'],
        help="Which model to use for embedding extraction: 'speechbrain' or 'wavlm."
    )
    
    parser.add_argument(
        "--group", 
        type=str, 
        required=True, 
        help="Name of the K-pop group (e.g., aespa)."
    )
    
    parser.add_argument(
        "--song", 
        type=str, 
        required=True, 
        help="Name of the song (e.g., Forever) or 'solo' for all solo tracks."
    )
    
    parser.add_argument(
        "--clean", 
        action="store_true", 
        help="If set, only includes pure solo vocals. Otherwise, extracts everything."
    )
    
    parser.add_argument(
        "--delete", 
        action="store_true", 
        help="If set, deletes existing embedding files for the target song(s) before extracting new ones."
    )

    args = parser.parse_args()

    print(f"Starting pipeline for Model: {args.type.upper()} | Group: {args.group} | Song: {args.song} | Include Secondary: {args.clean}")
    
    # --- NEW LOGIC HERE ---
    if args.song.lower() == 'all':
        process_all_group_songs(
            group_name=args.group, 
            strict_clean=args.clean, 
            model_type=args.type,
            delete_old=args.delete
        )
    else:
        # Standard single-song or solo processing
        if args.type == 'speechbrain':
            process_speechbrain_embeddings(args.group, args.song, strict_clean=args.clean, delete_old=args.delete)
        elif args.type == "wavlm":
            process_wavlm_sliding_windows(args.group, args.song)

if __name__ == "__main__":
    main()