import glob
import os
import re

def rename_ive_wavs(directory="."):
    # Pattern to match files starting with '1' and ending in .wav
    pattern = os.path.join(directory, "1*.wav")
    
    for file_path in glob.glob(pattern):
        filename = os.path.basename(file_path)
        
        # Case 1: 1_{songName}_(Vocals).wav  --> {songName}_vocals.wav
        m1 = re.match(r"1_(.+)_\(Vocals\)\.wav", filename)
        if m1:
            song = m1.group(1)
            new_name = f"{song}_vocals.wav"
        
        # Case 2: 1_{songName}_(Vocals_lead_only).wav --> {songName}_leading_vocals.wav
        m2 = re.match(r"1_(.+)_\(Vocals_lead_only\)\.wav", filename)
        if m2:
            song = m2.group(1)
            new_name = f"{song}_leading_vocals.wav"
        
        # Case 3: 1_{songName}_(Vocals_backing_only).wav --> {songName}_backing_vocals.wav
        m3 = re.match(r"1_(.+)_\(Vocals_backing_only\)\.wav", filename)
        if m3:
            song = m3.group(1)
            new_name = f"{song}_backing_vocals.wav"
        
        # If none of the patterns matched, skip
        if not (m1 or m2 or m3):
            continue
        
        # Full path for renaming
        new_path = os.path.join(directory, new_name)
        
        print(f"Renaming: {filename}  →  {new_name}")
        os.rename(file_path, new_path)

# Example usage:
rename_ive_wavs()