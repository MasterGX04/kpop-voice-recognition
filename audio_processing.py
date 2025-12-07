import os, sys, glob
import json
from pydub import AudioSegment
import librosa
from concurrent.futures import ThreadPoolExecutor  
import numpy as np
from tqdm import tqdm
import pickle
import queue
import threading, collections
import traceback

CHUNK_DURATION = 40
CHUNK_DURATION_200MS = 200  # 200ms chunk duration

def convertToWav(inputMp3Path, outputWavPath):
    audio = AudioSegment.from_mp3(inputMp3Path)
    monoAudio = audio.set_channels(1).set_frame_rate(22050)
    monoAudio.export(outputWavPath, format="wav")
    
def combineMemberVocals(jsonFiles, vocalsOnlySongs, selectedGroup, members):
    """
    Build frame-wise labels for each vocals-only song.

    NOTE: Semantics (per 40 ms chunk, per member):
      - presence[c]   = 1 if member c is singing anything (lead, harmony, adlib, gang)
      - lead[c]       = 1 if member c is the perceptual main voice in that chunk
      - isRepeat[c]   = 1 if member c is backing/harmony in that chunk
      - isAdlib[c]    = 1 if member c is doing an ad-lib line in that chunk

    'isRepeat' in the JSON is now interpreted as "backing vocal?" at segment level.
    'Gang Vocal' is treated as a special member for non-attributable vocals.
    """
    base_dir = f"./training_data/{selectedGroup}"
    os.makedirs(base_dir, exist_ok=True)
    
    # add Silence and Gang Vocal into members  
    if len(members) > 1:
        memberList = members + ["Silence"] + ["Gang Vocal"]
        

    # Map song title → JSON file
    jsonFileMap = {
        os.path.splitext(os.path.basename(f))[0].replace("_labels", ""): f
        for f in jsonFiles
    }

    # Class indices
    member_to_idx = {name: i for i, name in enumerate(memberList)}
    
    silence_name = None
    silence_idx = None
    for m in memberList:
        if m.lower() == "silence":
            silence_name = m
            silence_idx = member_to_idx[m]
            break
    
    # Gang vocal name
    gang_name = None
    for m in memberList:
        if m.lower() == "gang vocal":
            gang_name = m
            break
        
    for vocalsFile in vocalsOnlySongs:
        songTitle = (
            os.path.basename(vocalsFile)
            .replace("_vocals.mp3", "")
        )
        
        mp3_path = os.path.join(base_dir, f"{songTitle}_vocals.mp3")
        wav_path = os.path.join(base_dir, f"{songTitle}_vocals.wav")
        
        if not os.path.isfile(wav_path):
            print(f"[Audio] No WAV file found for {songTitle}. Creating {wav_path}...")
            try:
                convertToWav(mp3_path, wav_path)
                print(f"[Audio]   ✔ Converted {vocalsFile} → {wav_path} @ 22050 Hz")
            except Exception as e:
                print(f"[Audio]   ❌ Failed to convert {vocalsFile} to WAV: {e}")
                continue  # skip this song if conversion fails
        
        print(f"[build_frame_labels] Processing {songTitle}")
        jsonFilePath = jsonFileMap.get(songTitle)

        if not jsonFilePath:
            print(f"  Warning: No matching JSON file found for {songTitle}. Skipping.")
            continue

        with open(jsonFilePath, 'r', encoding="utf-8") as file:
            labels = json.load(file)

        # --- Build per-chunk maps from your (member, startChunk, endChunk, isRepeat, isAdlib) labels ---
        activationMap = collections.defaultdict(list)  # chunkIdx -> {memberName}
        maxChunkIdx = 0

        for label in labels:
            # JSON: [memberName, startChunk, endChunk, isBacking, isAdlib]
            memberName, startChunk, endChunk, isBacking, isAdlib = label
            isBacking = bool(isBacking)
            isAdlib = bool(isAdlib)
            
            for chunkIdx in range(startChunk, endChunk):
                activationMap[chunkIdx].append(
                    (memberName, startChunk, isBacking, isAdlib)
                )
                if chunkIdx > maxChunkIdx:
                    maxChunkIdx = chunkIdx

        if maxChunkIdx == 0 and not activationMap:
            print(f"  Warning: No labeled chunks for {songTitle}. Skipping.")
            continue

        num_chunks = maxChunkIdx + 1
        C = len(memberList)

        # Allocate label arrays: shape (num_chunks, C)
        presence = np.zeros((num_chunks, C), dtype=np.int32)
        lead = np.zeros((num_chunks, C), dtype=np.int32)
        isBacking_arr = np.zeros((num_chunks, C), dtype=np.int32)
        isAdlib_arr  = np.zeros((num_chunks, C), dtype=np.int32)
        
        # Helper: choose lead per frame according hierarchy
        def choose_lead(valid_segments):
            """
            valid_segments: list of (name, idx, startChunk, isBacking, isAdlib)
            Returns: lead_idx or None
            """
            if not valid_segments:
                return None

            # Split into gang vs non-gang
            non_gang = []
            gang_segs = []
            for name, idx, startC, isBacking, isAdlib in valid_segments:
                if gang_name is not None and name == gang_name:
                    gang_segs.append((name, idx, startC, isBacking, isAdlib))
                else:
                    non_gang.append((name, idx, startC, isBacking, isAdlib))
            
            # If any non-gang exists, we never let gang be lead
            if non_gang:
                # Priority A: non-gang, not ad-lib, not backing
                candA = [
                    seg for seg in non_gang
                    if not seg[3] and (not seg[4]) # not backing or adlib     
                ]
                
                if candA:
                    return max(candA, key=lambda s: s[2])[1]  # idx
                
                # Priority B: non-gang, not ad-lib
                candB = [seg for seg in non_gang if not seg[4]]  # not adlib
                if candB:
                    return max(candB, key=lambda s: s[2])[1]
                
                # Priority C: non-gang ad-lib (no main vocals present)
                candC = [seg for seg in non_gang if seg[4]]  # adlib only
                if candC:
                    return max(candC, key=lambda s: s[2])[1]
                
                # Fallback: any non-gang
                return max(non_gang, key=lambda s: s[2])[1]

            # Only gang vocals
            if gang_segs:
                # Allow gang vocal to be lead if it's alone
                return max(gang_segs, key=lambda s: s[2])[1]

            return None

        # --- Fill label arrays ---
        for chunkIdx in tqdm(range(num_chunks), desc=f"Chunks for {songTitle}"):
            segs = activationMap.get(chunkIdx, [])

            if len(segs) == 0:
                # Silence
                if silence_idx is not None:
                    presence[chunkIdx, silence_idx] = 1
                continue

            # Build valid segments with indices
            valid_segments = []
            for (memberName, startChunk, isBacking, isAdlib) in segs:
                if memberName not in member_to_idx:
                    # Unknown label (e.g. future groups) -> ignore
                    continue
                m_idx = member_to_idx[memberName]
                valid_segments.append(
                    (memberName, m_idx, startChunk, isBacking, isAdlib)
                )

                # Base tags: everyone active has presence
                presence[chunkIdx, m_idx] = 1
                if isAdlib:
                    isAdlib_arr[chunkIdx, m_idx] = 1
                if isBacking:
                    # segment-level backing flag
                    isBacking_arr[chunkIdx, m_idx] = 1

            if not valid_segments:
                # Nothing we know about; treat as silence if desired
                if silence_idx is not None:
                    presence[chunkIdx, silence_idx] = 1
                continue

            # Decide lead index using hierarchy + latest start
            lead_idx = choose_lead(valid_segments)
            if lead_idx is not None:
                lead[chunkIdx, lead_idx] = 1

            # Now assign harmony/backing vs ad-lib for non-leads
            for (name, m_idx, startC, isBacking, isAdlib) in valid_segments:
                if m_idx == lead_idx:
                    # Lead can still be ad-lib in solo sections; keep isAdlib_arr as set above.
                    continue

                # Non-lead:
                if isAdlib:
                    # Already tagged in isAdlib_arr; we treat this as ad-lib only
                    continue
                else:
                    # Active, non-adlib, non-lead → harmony/backing
                    isBacking_arr[chunkIdx, m_idx] = 1

        # --- Save to JSON (you can switch to npy if you prefer) ---
        out_labels_path = os.path.join(base_dir, f"{songTitle}_frame_labels.json")
        out_data = {
            "group": selectedGroup,
            "song": songTitle,
            "members": memberList,
            "silenceName": silence_name,
            "chunkDurationMs": CHUNK_DURATION,
            "numChunks": int(num_chunks),
            # store as nested lists for JSON; or you can save as .npy and keep them as arrays
            "presence": presence.tolist(),
            "lead": lead.tolist(),
            "isBacking": isBacking_arr.tolist(),
            "isAdlib": isAdlib_arr.tolist(),
        }

        with open(out_labels_path, "w", encoding="utf-8") as jf:
            json.dump(out_data, jf, indent=2, separators=(",", ":"),)

        print(f"  ✅ Saved frame labels: {out_labels_path}")
       
def segmentAndSaveAudio(audioPath: str,
                        featOut: str = '',
                        rawOut: str = '',
                        chunkMs: int = 1000,
                        sr: int = 22050,
                        nMfcc: int = 40,
                        nFft: int = 512):
    """
    Slice <audioPath> into <chunkMs>-ms windows, compute MFCC (+Δ, ΔΔ) and
    log-magnitude STFT for each window, and optionally cache results.

    Returns
    -------
    features : np.ndarray, shape (N_chunks, T, 120 + (nFft//2 + 1))
    raw      : list[np.ndarray], each of length samplesPerChunk
    """

    # 0) Fast path: load cached features (and raws) if present
    if featOut and os.path.exists(featOut):
        feats = np.load(featOut, allow_pickle=True)
        raws  = pickle.load(open(rawOut, "rb")) if rawOut and os.path.exists(rawOut) else None
        return feats, raws

    # 1) Load mono @ 22050 Hz
    y, _ = librosa.load(audioPath, sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"{audioPath} contains no samples.")
    y = librosa.util.normalize(y) # Scale to [-1, 1]

    # 2) Split into fixed-length chunks
    samplesPerChunk = int(sr * (chunkMs / 1000.0))
    pad = (-len(y)) % samplesPerChunk
    if pad:
        y = np.pad(y, (0, pad))
    rawChunks = y.reshape(-1, samplesPerChunk)  # (N, S)

    hopLen = max(1, sr // 100)

    # stftBins = nFft // 2 + 1
    featChunks = []
    
    chunkBar = tqdm(rawChunks,desc=f"Extracting features from {os.path.basename(audioPath)}",
                                 total=len(rawChunks),
                                 dynamic_ncols=True,
                                 leave=False,
                                 position=1)
    mel_fb = librosa.filters.mel(sr=sr, n_fft=nFft, n_mels=128, fmax=sr//2)
    
    for chunk in chunkBar:
        # STFT → log|S|
        # MFCC (+Δ, ΔΔ)
        #log-Mel spectrogram
        S = librosa.stft(chunk, n_fft=nFft, hop_length=hopLen, center=False)
        mag = np.abs(S)
        power = mag**2
        mel_spec = mel_fb @ power
        log_mel = librosa.power_to_db(mel_spec, ref=np.max) # Shape (128, T1)
        
        # MFCC + Δ + ΔΔ
        mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=nMfcc,
                                    n_fft=nFft, hop_length=hopLen, center=False)
        d1 = librosa.feature.delta(mfcc)
        d2 = librosa.feature.delta(mfcc, order=2)
        mfccFull = np.vstack([mfcc, d1, d2]) 
        
        # Pitch contour
        f0 = librosa.yin(chunk, fmin=80, fmax=1000, sr=sr, frame_length=nFft, hop_length=hopLen)
        
        # Handle NaNs and replace with 0 or interpolated
        f0 = np.nan_to_num(f0, nan=0.0)[None, :] 
        
        # Spectral flatness + centroid + bandwidth
        flatness = librosa.feature.spectral_flatness(y=chunk, n_fft=nFft,
                                                      hop_length=hopLen)
        centroid = librosa.feature.spectral_centroid(y=chunk, sr=sr,
                                                      n_fft=nFft, hop_length=hopLen)
        bandwidth = librosa.feature.spectral_bandwidth(y=chunk, sr=sr,
                                                       n_fft=nFft, hop_length=hopLen)
        
        extras = np.vstack([flatness, centroid, bandwidth])
        
        # Fix this LPC
        lpc_feat = extract_formants(chunk, sr=sr, order=12, n_formants=3)
        
        #Combine features: Choose common time dimension T = min(T1,T2,T3,T4) 
        T = min(log_mel.shape[1], mfccFull.shape[1], f0.shape[1], extras.shape[1])
        log_mel_t = log_mel[:, :T]
        mfccFull_t = mfccFull[:, :T]
        f0_t = f0[:, :T]
        extras_t = extras[:, :T]
        lpc_rep = lpc_feat.reshape(-1, 1).repeat(T, axis=1)
        
        combined = np.vstack([log_mel_t,
                              mfccFull_t,
                              f0_t,
                              extras_t, lpc_rep])
        
        # Time-align branches by right-padding the shorter time axis
        featChunks.append(combined.T)

    features = np.array(featChunks, dtype=np.float32)
    
    m = np.mean(features, axis=(0,1), keepdims=True)
    s = np.std(features, axis=(0,1), keepdims=True) + 1e-8
    features = (features - m) / s

    # 4) Cache
    if featOut:
        np.save(featOut, features)
    if rawOut:
        with open(rawOut, "wb") as f:
            pickle.dump(rawChunks, f)

    return features, rawChunks
    
def getSongsFromSameAlbum():
    songsFromSameAlbum = {
        'IVE': {
            'Ive_Switch': ['해야 (HEYA)', 'Accendio', 'Blue Blood', 'Summer Festa', "Blue Heart", "Hypnosis", "WOW", "My Satisfaction", "LOVE DIVE", "Ice Queen", "Baddie", "Heroine", "Supernova Love"],
            'Ive_Empathy': ['Rebel Heart', 'Flu', 'You Wanna Cry', 'ATTITUDE','Thank U', 'TKO', 'Mine', 'ELEVEN', 'Summer[Liz]', 'Wish[Yujin]', 'Payback', 'XOXZ', "Off The Record"]},
        'ITZY': {
            'Born To Be': ['Born To Be', 'Mr. Vampire']   
        },
        'BTS': {
            'Proof': ['Epiphany[Jin]', 'Euphoria[Jungkook]', "Filter[Jimin]", "Love Me Again[V]", "Persona[RM]", "Stigma[V]", "First Love[Suga]", "Disease", "Mama[J-Hope]", "달려라 방탄"],
            'Love_Yourself_Tear': ['Fake Love', 'Love Maze', 'Magic Shop']
        },
        'Fifty Fifty': {
            'Day & Night': ['Cupid', 'Skittlez']
        }
    }
    
    return songsFromSameAlbum

def loadLabels(jsonPath):
        """Load labeled data and dynamically check if a chunk is within a singer's range"""
        with open(jsonPath, "r") as file:
            labels = json.load(file)

        # ✅ Store singers in a list of (startChunk, endChunk)
        chunkRanges = []
        for entry in labels:
            singer, start, end, isRepeat, isAdLib = entry
            chunkRanges.append((singer, start, end, isRepeat, isAdLib))

        return chunkRanges 
    
def createLabelArray(totalChunks, labelRanges, memberList):
    """
    Returns a label array of shape (totalChunks, numMembers).
    Default is all zeros (representing silence).
    If one or more singers are active during a chunk, the respective indices are set.
    """
    labelArray = np.zeros((totalChunks, len(memberList)), dtype=np.float32)
    memberIndex = {name: i for i, name in enumerate(memberList)}

    for label in labelRanges:
        singer, start40, end40 = label[:3]

        if singer in memberIndex:
            idx = memberIndex[singer]

            # Convert 40ms chunk indices to 200ms indices
            start200 = start40 // 5
            end200 = end40 // 5

            # Clamp to bounds
            start200 = max(0, start200)
            end200 = min(totalChunks - 1, end200)

            labelArray[start200:end200+1, idx] = 1.0  # One-hot

    return labelArray

def extract_formants(chunk: np.ndarray, sr: int, order: int = 12, n_formants: int = 3):
    # 1) LPC coefficients
    a = librosa.lpc(chunk, order=order)
    # 2) Roots
    rts = np.roots(a)
    # 3) Keep only roots with positive imag part
    rts = [r for r in rts if np.imag(r) >= 0]
    # 4) Convert to frequencies & bandwidth
    formants = []
    for r in rts:
        ang = np.arctan2(np.imag(r), np.real(r))
        freq = ang * (sr / (2*np.pi))
        bw = -1.0 * (sr / (2*np.pi)) * np.log(np.abs(r))
        formants.append((freq, bw))
    # 5) Sort by frequency
    formants_sorted = sorted(formants, key=lambda x: x[0])
    # 6) Take top n_formants
    selected = formants_sorted[:n_formants]
    # 7) Flatten into 1-D feature vector
    feat = []
    for (f, b) in selected:
        feat.extend([f, b])
    return np.array(feat, dtype=np.float32) 

def extractVocalSegmentsByType(labels, minOverlapChunks=10):
        """
        Extracts segments from label data and classifies into:
        - Solo segments
        - Ad-lib (label[4] = True)
        - Harmony (non-adlib overlaps ≥ minOverlapChunks)
        - Transition (non-adlib overlaps < minOverlapChunks)
        """
        CHUNK_DURATION = 40
        
        # Timeline of active singers per chunk
        chunkMap = {}
        for label in labels:
            name, start, end, _, isAdLib = label
            for i in range(start, end + 1):
                chunkMap.setdefault(i, []).append((name, isAdLib))
        
        # Build labeled segments per singer
        soloSegments = []
        adlibSegments = []
        harmonySegments = []
        transitionSegments = []

        for label in labels:
            name, start, end, _, isAdLib = label
            overlap = False
            overlapLength = 0

            for i in range(start, end + 1):
                if len(chunkMap[i]) > 1:
                    overlap = True
                    overlapLength += 1

            if isAdLib:
                adlibSegments.append((name, start, end))
            elif overlap:
                if overlapLength >= minOverlapChunks:
                    harmonySegments.append((name, start, end))
                else:
                    transitionSegments.append((name, start, end))
            else:
                soloSegments.append((name, start, end))
        
        return {
            "solo": soloSegments,
            "adlib": adlibSegments,
            "harmony": harmonySegments,
            "transition": transitionSegments
        }
      
def extractAndSaveHarmoniesFromSong(selectedGroup, songName):
        """
        Extracts vocal parts from a labeled song and saves categorized segments (solo, harmony, adlib, transition)
        in both .mp3 and Mel chunk format to:
            ./training_data/{selectedGroup}/songs_to_update/{songName}/
        """
        labelPath = f"./saved_labels/{selectedGroup}/{songName}_labels.json"
        vocalsPath = f"./training_data/{selectedGroup}/{songName}_vocals.mp3"
        outDir = f"./training_data/{selectedGroup}/songs_to_update/{songName}"
        os.makedirs(outDir, exist_ok=True)

        if not os.path.exists(labelPath) or not os.path.exists(vocalsPath):
            print(f"❌ Missing label or vocals file for {songName}")
            return

        with open(labelPath, "r") as f:
            labels = json.load(f)

        vocals = AudioSegment.from_file(vocalsPath)
        segmentMap = extractVocalSegmentsByType(labels, minOverlapChunks=10)

        for typ, segments in segmentMap.items():
            for idx, (name, startChunk, endChunk) in enumerate(segments):
                startTime = startChunk * 40
                endTime = (endChunk + 1) * 40
                segment = vocals[startTime:endTime]

                # Save .mp3
                mp3Path = os.path.join(outDir, f"{typ}_{name}_{idx}.mp3")
                wavPath = mp3Path.replace(".mp3", ".wav")
                melPath = mp3Path.replace(".mp3", "_mel_200ms.npy")
                rawPath = mp3Path.replace(".mp3", "_raw_200ms.pkl")

                segment.export(mp3Path, format="mp3")
                segment.export(wavPath, format="wav")

                # Save Mel chunks
                segmentAndSaveAudio(
                    wavPath,
                    savePath=melPath,
                    rawSavePath=rawPath,
                    segmentDuration=200
                )

        print(f"Extracted harmonies and vocals from {songName} into {outDir}")
      
def estimatePitchRanges(audioChunks, sr=22050, groupSize=3, groupName='', savePath=None):
    """
    Estimate pitch ranges using chunk groups (minimum length ~2048).
    Groups every `groupSize` chunks together and assigns one pitch label to all of them.
    Uses librosa.pyin() in 15 threads.
    """
    from concurrent.futures import as_completed
    
    def timedInput(prompt, timeout=10, default='y'):
        q = queue.Queue()

        def getInput():
            try:
                userInput = input(prompt).strip().lower()
                if userInput == "":
                    q.put(default)
                else:
                    q.put(userInput)
            except:
                q.put(default)

        t = threading.Thread(target=getInput)
        t.daemon = True
        t.start()
        
        try:
            return q.get(timeout=timeout)
        except queue.Empty:
            print(f"\n⏰ No response in {timeout} seconds. Defaulting to '{default}'.")
            return default
        
    
    def checkChunkLoudness(numSamples=20):
        import random
        sampleIndices = random.sample(range(len(audioChunks)), min(len(audioChunks), numSamples))
        print(f"\n🔍 Checking loudness of {len(sampleIndices)} random chunks:")
        for i in sampleIndices:
            audio = audioChunks[i].flatten()
            rms = np.sqrt(np.mean(audio ** 2))
            peak = np.max(np.abs(audio))
            print(f"Chunk {i}: RMS={rms:.4f}, Peak={peak:.4f}")

    checkChunkLoudness()
    print("Chunk shape:", audioChunks[0].shape)
    os.makedirs(f"./{groupName}", exist_ok=True)

    if os.path.exists(savePath):
        reuse = timedInput(f"⚠️ Found cached pitch range file for {groupName}. Reuse it? (y/n): ", timeout=10).strip().lower()
        if reuse == "y":
            print("Reusing cached pitch ranges.")
            return list(np.load(savePath, allow_pickle=True))

    # Thread-safe output collector
    # Step 1: Group chunks
    numChunks = len(audioChunks)
    groupedChunks = []
    groupIndices = []
    
    for i in range(0, numChunks, groupSize):
        group = audioChunks[i:i + groupSize]
        concatenated = np.concatenate([chunk.flatten() for chunk in group if len(chunk.flatten()) > 0])
        groupedChunks.append(concatenated)
        groupIndices.append(list(range(i, i + groupSize)))
    
    print(f"\n🎼 Estimating pitch ranges using 15 threads for {len(groupedChunks)} groups...")
    
    # Estimate pitch in parallel
    groupLabels = [None] * len(groupedChunks) 

    def estimate_group(index, group):
        try:        
            audio = group.flatten()
            
            if len(audio) < 2048:
                # Skip too short groups
                print(f"Group is too short with length {len(audio)}")
                return index, "none"
            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=sr
            )
            f0_clean = f0[~np.isnan(f0)]
            if len(f0_clean) == 0:
                print("f0_clean is empty")
                return index, "none"
            avgF0 = np.mean(f0_clean)
            if avgF0 < 220:
                return index, "low"
            elif avgF0 < 440:
                return index, "mid"
            else:
                return index, "high"
        except Exception as e:
            print(f"🔥 Exception in group {index}: {e}")
            traceback.print_exc()
            return index, "error"

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {executor.submit(estimate_group, i, audio): i for i, audio in enumerate(groupedChunks)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing pitch groups"):
            idx, label = future.result()
            groupLabels[idx] = label
    
    print("All groupLabels after estimation:", groupLabels[:10])
    
    # Assign label to each chunk in the group
    pitchRanges = ["none"] * numChunks
    for groupIdx, chunkIndices in enumerate(groupIndices):
        label = groupLabels[groupIdx]
        for idx in chunkIndices:
            if idx < numChunks:  # ✅ prevent out-of-range
                pitchRanges[idx] = label

    # Save to file
    np.save(savePath, np.array(pitchRanges, dtype=object))
    print(f"💾 Saved pitch range labels to {savePath}")
    
    # Print count summary
    return pitchRanges
    
def generateUemFromRttm(rttm_path, uem_path):
    with open(rttm_path, "r") as rttm_file:
        lines = rttm_file.readlines()

    uri = None
    start_times = []
    end_times = []

    for line in lines:
        parts = line.strip().split()
        if parts[0] != "SPEAKER":
            continue
        uri = parts[1].replace(" ", "")  # clean the URI in case it's still spaced
        start = float(parts[3])
        duration = float(parts[4])
        start_times.append(start)
        end_times.append(start + duration)

    if not start_times:
        return

    min_start = min(start_times)
    max_end = max(end_times)

    with open(uem_path, "w") as uem_file:
        uem_file.write(f"{uri} 1 {min_start:.3f} {max_end:.3f}\n")
    print(f"UEM saved to {uem_path} for URI: {uri}")
        
def main():
    if len(sys.argv) < 2:
        print("Usage: python audio_processing.py <group_name> [-r] [--u]")
        sys.exit(1)

    group = sys.argv[1]
    convertToRttm = "-r" in sys.argv
    generateUem = "-u" in sys.argv

    labelDir = f"./saved_labels/{group}"
    jsonFiles = glob.glob(os.path.join(labelDir, "*labels.json"))

    if not jsonFiles:
        print(f"No label JSON files found in {labelDir}")
        sys.exit(0)

    for jsonFile in jsonFiles:
        baseName = os.path.basename(jsonFile).replace("_labels.json", "").replace(" ", "")
        rttmPath = os.path.join(labelDir, f"{baseName}.rttm")

        if convertToRttm:
            convertJsonToRttm(jsonFile, baseName, rttmPath)
            print(f"Converted {jsonFile} to {rttmPath}")

        if generateUem:
            if not os.path.exists(rttmPath):
                print(f"⚠️ RTTM not found for {baseName}, skipping UEM generation.")
                continue
            uemPath = os.path.join(labelDir, f"{baseName}.uem")
            generateUemFromRttm(rttmPath, uemPath)
            print(f"Created UEM file: {uemPath}")
            
if __name__ == "__main__":
    main()