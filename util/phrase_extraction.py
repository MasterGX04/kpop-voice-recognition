import json
import os
import argparse
from pydub import AudioSegment
from collections import defaultdict

import os
import json

def extract_all_phrases(group_name, song_name, max_gap_sec=3.0, strict_clean=True):
    """
    Groups chunks into continuous phrases independently for ALL active members.
    Permits self-ad-libbing.
    If strict_clean is True, any chunk containing another member's voice 
    (main, ad-lib, or backing) acts as a hard wall and severs the phrase.
    """
    file_path = f"./saved_labels/{group_name}/{song_name}_labels.json"

    if not os.path.exists(file_path):
        print(f"[ERROR] Could not find labels at {file_path}")
        return [], {}

    with open(file_path, 'r') as f:
        labels = json.load(f)
        
    if not labels:
        return [], {}
        
    # 1. Build frame-level presence matrices
    main_presence = defaultdict(set)
    bg_presence = defaultdict(set)
    max_chunk = 0
    
    for lbl in labels:
        member, start, end, is_bg, is_adlib = lbl
        if end > max_chunk:
            max_chunk = end
            
        if member == "Cut" or member == "Gang Vocal":
            for i in range(start, end):
                bg_presence[i].add(member)
        elif is_bg or is_adlib:
            for i in range(start, end):
                bg_presence[i].add(member)
        else:
            for i in range(start, end):
                main_presence[i].add(member)

    # 2. Extract phrases independently per member
    all_turns = []
    member_totals = {}
    
    # Identify all members who sang a primary line in the track
    unique_members = {m for chunk_set in main_presence.values() for m in chunk_set}
    max_gap_chunks = int(max_gap_sec / 0.04)

    for target_member in unique_members:
        current_start = None
        current_end = None
        
        min_duration_sec = 1.0  # Set minimum phrase duration threshold in seconds

        def finalize_turn(st, en):
            if st is not None and en is not None and en > st:
                dur = (en - st) * 0.04
                # ONLY append and tally if the phrase meets or exceeds the minimum duration
                if dur >= min_duration_sec:
                    all_turns.append({
                        "member": target_member,
                        "start_chunk": st,
                        "end_chunk": en,
                        "duration_sec": dur
                    })
                    member_totals[target_member] = member_totals.get(target_member, 0.0) + dur

        # Scan the entire timeline linearly for this specific member
        for i in range(max_chunk + 1):
            is_active = target_member in main_presence[i]
            
            # Determine if this chunk is corrupted for THIS specific member
            is_corrupted = False
            if strict_clean:
                # Subtract the target_member to see if anyone else is present
                other_mains = main_presence[i] - {target_member}
                other_bgs = bg_presence[i] - {target_member}
                has_cut = "Cut" in bg_presence[i]
                has_gang = "Gang Vocal" in bg_presence[i]
                
                if other_mains or other_bgs or has_cut or has_gang:
                    is_corrupted = True
            
            # Hard wall: A corrupted chunk immediately severs the active phrase
            if is_corrupted and current_start is not None:
                finalize_turn(current_start, current_end)
                current_start = None
                current_end = None
                continue
                
            # Valid chunk processing
            if is_active and not is_corrupted:
                if current_start is None:
                    current_start = i
                    current_end = i + 1
                else:
                    # Check if the silence gap between the last end and current chunk is too large
                    if (i - current_end) > max_gap_chunks:
                        finalize_turn(current_start, current_end)
                        current_start = i
                    current_end = i + 1

        # Finalize any remaining phrase at the end of the song
        finalize_turn(current_start, current_end)
        
    # 3. Sort final output chronologically by start chunk
    all_turns.sort(key=lambda x: x["start_chunk"])
        
    return all_turns, member_totals
    
def export_member_audio(group_name, song_name, phrases):
    """
    Slices the original vocal wav file using the chunk data and 
    exports a combined mp3 for each member for auditing.
    """
    wav_path = f"./training_data/{group_name}/{song_name}_vocals.wav"
    
    if not os.path.exists(wav_path):
        print(f"[ERROR] Could not find audio at {wav_path}")
        return

    print(f"Loading {wav_path}...")
    song_audio = AudioSegment.from_wav(wav_path)
    
    # Dictionary to hold the combined audio segments for each member
    member_audio = {}

    for p in phrases:
        member = p["member"]
        
        # Convert chunks to milliseconds (1 chunk = 40 ms)
        start_ms = p["start_chunk"] * 40
        end_ms = p["end_chunk"] * 40
        
        # Slice the audio
        audio_slice = song_audio[start_ms:end_ms]
        
        # Append to the member's existing audio, or start a new track
        if member in member_audio:
            member_audio[member] += audio_slice
        else:
            member_audio[member] = audio_slice

    # Ensure the output directory exists
    output_dir = f"./audits/split_parts/{group_name}/{song_name}"
    os.makedirs(output_dir, exist_ok=True)

    # Export each member's combined parts to an mp3
    for member, audio_data in member_audio.items():
        output_path = os.path.join(output_dir, f"{song_name}_{member}.mp3")
        print(f"Exporting {member}'s parts to {output_path}...")
        audio_data.export(output_path, format="mp3")
        
    print("Done exporting audit files.")
    
def main():
    parser = argparse.ArgumentParser(description="Split labeled vocals into member-specific continuous audio files.")
    parser.add_argument("group_name", type=str, help="Name of the K-pop group (e.g., aespa)")
    parser.add_argument("song_name", type=str, help="Name of the song (e.g., Forever)")
    parser.add_argument("strict_clean", type=bool, default=False, help="Does it have to be only one member of audio?")

    args = parser.parse_args()

    print(f"--- Processing {args.group_name} - {args.song_name} ---")
    
    # 1. Extract the turns based on the labels
    phrases, totals = extract_all_phrases(args.group_name, args.song_name, 3.0, args.strict_clean)
    
    if not phrases:
        print("No valid phrases found. Exiting.")
        return

    print("\nTotal audio captured per member:")
    for member, time in totals.items():
        print(f"  {member}: {time:.2f} seconds")
        
    # 2. Slice and export the audio for auditing
    print("\nSlicing audio...")
    export_member_audio(args.group_name, args.song_name, phrases)


if __name__ == "__main__":
    main()