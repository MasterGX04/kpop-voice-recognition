import os, sys, glob
import json
from pydub import AudioSegment
import librosa
from concurrent.futures import ThreadPoolExecutor  
import numpy as np
from tqdm import tqdm
import queue
import threading, collections
import traceback
import random

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
    
    # add Gang Vocal into members  
    memberList = list(members)
    if "Gang Vocal" not in memberList:
        memberList.append("Gang Vocal")
 
    # Map song title → JSON file
    jsonFileMap = {
        os.path.splitext(os.path.basename(f))[0].replace("_labels", ""): f
        for f in jsonFiles
    }

    # Class indices
    member_to_idx = {name: i for i, name in enumerate(memberList)}
    
    silence_name = "silence"
    
    # Gang vocal name
    gang_name = None
    for m in memberList:
        if m.lower() == "gang vocal":
            gang_name = m
            break
        
    for vocalsFile in vocalsOnlySongs:
        songTitle = (
            os.path.basename(vocalsFile)
            .replace("_vocals.wav", "")
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
            
            for chunkIdx in range(startChunk, endChunk + 1):
                activationMap[chunkIdx].append(
                    (memberName, startChunk, isBacking, isAdlib)
                )
                maxChunkIdx = max(maxChunkIdx, endChunk)

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
        adlibPrimary = np.zeros((num_chunks, C), dtype=np.int32)
        stemChoice = np.zeros((num_chunks, C), dtype=np.int8) # 0=vocals, 1=lead, 2=backing
        
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
                # Priority A: non-gang ad-lib that is lead present)
                candA = [seg for seg in non_gang if seg[4] and seg[3]]  # adlib primary
                if candA:
                    return max(candA, key=lambda s: s[2])[1]
                
                # Priority B: non-gang, not ad-lib, not backing
                candB = [
                    seg for seg in non_gang
                    if not seg[3] and (not seg[4]) # not backing or adlib     
                ]
                
                if candB:
                    return max(candB, key=lambda s: s[2])[1]  # idx
                
                # Priority C: non-gang, background ad-lib
                candC = [seg for seg in non_gang if seg[4] and (not seg[3])]
                if candC:
                    return max(candC, key=lambda s: s[2])[1]
                
                # Priority D: non-gang, backing vocal
                candD = [seg for seg in non_gang if seg[3] and not seg[4] ]  # not adlib
                if candD:
                    return max(candD, key=lambda s: s[2])[1]
                
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
                    
                    # primary/foreground ad-lib
                    if isBacking:
                        adlibPrimary[chunkIdx, m_idx] = 1
                        
                elif isBacking and not isAdlib:
                    # segment-level backing flag
                    isBacking_arr[chunkIdx, m_idx] = 1

            if not valid_segments:
                # Nothing we know about; treat as silence by leaving row zero
                continue

            # Decide lead index using hierarchy + latest start
            # foreground candidates (non-gang main voices)
            foreground = [
                (name, idx, startC, isBacking, isAdlib)
                for (name, idx, startC, isBacking, isAdlib) in valid_segments
                if (gang_name is None or name != gang_name) and (not isBacking) and (not isAdlib)
            ]

            if len(foreground) >= 2:
                for (_, idx, _, _, _) in foreground:
                    lead[chunkIdx, idx] = 1
            else:
                lead_idx = choose_lead(valid_segments)
                if lead_idx is not None:
                    lead[chunkIdx, lead_idx] = 1
                    
            VOCALS, LEADSTEM, BACKSTEM = 0, 1, 2
                
            has_non_gang = any(
                (gang_name is None or name != gang_name)
                for (name, _, _, _, _) in valid_segments
            )
            
            for (memberName, m_idx, startChunk, isBacking, isAdlib) in valid_segments:
                if gang_name is not None and memberName == gang_name:
                    stemChoice[chunkIdx, m_idx] = VOCALS if not has_non_gang else BACKSTEM
                else:
                    if isAdlib and isBacking:
                        stemChoice[chunkIdx, m_idx] = LEADSTEM   # your adlibPrimary override
                    elif isAdlib and (not isBacking):
                        stemChoice[chunkIdx, m_idx] = BACKSTEM   # default adlib source
                    elif isBacking and (not isAdlib):
                        stemChoice[chunkIdx, m_idx] = BACKSTEM
                    else:
                        stemChoice[chunkIdx, m_idx] = LEADSTEM
                        
        overlapRuns = buildOverlapRunsFromActivationMap(
            activationMap,
            minMembers=2,
            minRunLen=20
        )
        
        pick = pickRandomOverlapWindow(overlapRuns, maxLen=100)
        
        if pick is None:
            print(f"[debug] {songTitle}: no overlap runs found (>=2 members for >=20 chunks).")
        else:
            winStart, winEnd, membersSet = pick
            # show only the members involved in that overlap (plus Gang Vocal if you want)
            showMembers = [m for m in memberList if m in membersSet]
            print(f"\n[debug] {songTitle}: overlap window {winStart}..{winEnd} (len={winEnd-winStart+1}) members={sorted(showMembers)}")
            debugPrintWindow(
                presence, lead,
                isBacking_arr, isAdlib_arr,
                memberList, member_to_idx,
                winStart, winEnd,
                showMembers=showMembers
            )
            print()
                
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
            "adlibPrimary": adlibPrimary.tolist(),
            "stemChoice": stemChoice.tolist(),
        }

        with open(out_labels_path, "w", encoding="utf-8") as jf:
            json.dump(out_data, jf, separators=(",", ":"))

        print(f"  ✅ Saved frame labels: {out_labels_path}")


def buildOverlapRunsFromActivationMap(activationMap, minMembers=2, minRunLen=20):
    """
    activationMap: dict chunkIdx -> list of tuples (memberName, startChunk, isBacking, isAdlib)
    Returns list of runs: [(runStart, runEnd, membersSet), ...] where runEnd is inclusive.
    A run is a contiguous chunk span where >= minMembers are active.
    """
    if not activationMap:
        return []

    allChunks = sorted(activationMap.keys())

    # Helper: active member set at chunk
    def membersAtChunk(t):
        return set(seg[0] for seg in activationMap.get(t, []))

    runs = []
    runStart = None
    runMembersUnion = set()
    prev = None

    for t in allChunks:
        activeMembers = membersAtChunk(t)
        qualifies = (len(activeMembers) >= minMembers)

        contiguous = (prev is None or t == prev + 1)

        if qualifies and (runStart is None):
            runStart = t
            runMembersUnion = set(activeMembers)
        elif qualifies and runStart is not None and contiguous:
            runMembersUnion |= activeMembers
        elif qualifies and runStart is not None and (not contiguous):
            # close previous run, start new
            runEnd = prev
            if (runEnd - runStart + 1) >= minRunLen:
                runs.append((runStart, runEnd, set(runMembersUnion)))
            runStart = t
            runMembersUnion = set(activeMembers)
        elif (not qualifies) and runStart is not None:
            # close run
            runEnd = prev
            if (runEnd - runStart + 1) >= minRunLen:
                runs.append((runStart, runEnd, set(runMembersUnion)))
            runStart = None
            runMembersUnion = set()

        prev = t

    # close if ended in run
    if runStart is not None:
        runEnd = prev
        if (runEnd - runStart + 1) >= minRunLen:
            runs.append((runStart, runEnd, set(runMembersUnion)))

    return runs

def pickRandomOverlapWindow(overlapRuns, maxLen=100):
    """
    overlapRuns: [(runStart, runEnd, membersSet), ...]
    Returns (winStart, winEnd, membersSet) with winEnd inclusive, or None if no runs.
    """
    if not overlapRuns:
        return None

    runStart, runEnd, membersSet = random.choice(overlapRuns)
    runLen = runEnd - runStart + 1
    winLen = min(maxLen, runLen)

    # choose a random sub-window inside the run
    maxStart = runEnd - winLen + 1
    winStart = random.randint(runStart, maxStart)
    winEnd = winStart + winLen - 1
    return (winStart, winEnd, membersSet)

def debugPrintWindow(presence, lead, isBackingStyle, isAdlib, memberList, member_to_idx,
                     winStart, winEnd, showMembers=None):
    """
    Prints P/L/S/B/A tags for chunks winStart..winEnd (inclusive)
    P=presence, L=lead, S=secondaryRole, B=backing style, A=adlib style
    """
    if showMembers is None:
        showMembers = memberList

    header = "chunk | " + " | ".join([f"{m:>10}" for m in showMembers])
    print(header)
    print("-" * len(header))

    for t in range(winStart, winEnd + 1):
        row = []
        for m in showMembers:
            i = member_to_idx[m]
            tags = []
            if presence[t, i]: tags.append("P")
            if lead[t, i]: tags.append("L")
            if isBackingStyle[t, i]: tags.append("B")
            if isAdlib[t, i]: tags.append("A")
            row.append("".join(tags) if tags else ".")
        print(f"{t:5d} | " + " | ".join([f"{cell:>10}" for cell in row]))
      
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

def _stftMag(y, sr, nFft=1024, hopLength=256):
    S = np.abs(librosa.stft(y, n_fft=nFft, hop_length=hopLength, win_length=nFft))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=nFft)
    return S, freqs

def computeChunkVocalSilenceFlags(
    y,
    sr,
    chunkMs=40,
    nFft=1024,
    hopLength=256,
    vocalBand=(300, 4000),
    smoothWin=5,          # chunks
    minOnRun=3,           # chunks: require at least this many consecutive "on"
    minOffRun=3           # chunks: require at least this many consecutive "off"
):
    """
    Returns:
      isSilence: (T,) bool
      hasVocal:  (T,) bool
      isVoiced:  (T,) bool   # voiced (vowel/pitched) vs unvoiced (consonant/breath)

    Notes:
      - isSilence: mostly RMS + (optional) bandRatio guard.
      - hasVocal: "voice-like" score + rising-energy ON gate + hysteresis + smoothing.
      - isVoiced: pYIN voiced probability aggregated per chunk (+ optional harmonic/flatness guards).
    """
    # -------------------------
    # chunking
    # -------------------------
    chunkLen = int(sr * (chunkMs / 1000.0))
    if chunkLen <= 0:
        raise ValueError("chunkLen <= 0")

    T = int(np.ceil(len(y) / chunkLen))
    pad = T * chunkLen - len(y)
    if pad > 0:
        y = np.pad(y, (0, pad), mode="constant")

    chunks = y.reshape(T, chunkLen)

    # -------------------------
    # Feature 1: RMS (per chunk)
    # -------------------------
    rms = np.sqrt(np.mean(chunks * chunks, axis=1) + 1e-12).astype(np.float32)  # (T,)

    # -------------------------
    # STFT features (per frame -> aggregate to chunk)
    # -------------------------
    S, freqs = _stftMag(y, sr, nFft=nFft, hopLength=hopLength)  # (F, frames)
    nFrames = S.shape[1]
    frameTimes = (np.arange(nFrames) * hopLength)  # in samples
    frameChunk = np.clip(frameTimes // chunkLen, 0, T - 1).astype(int)

    # pYIN voiced prob per frame (aligned to hopLength)
    # Note: pyin can be slow, but it's the best "voiced vs unvoiced" signal you can add quickly.
    _, _, voicedProbFrame = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=nFft,
        hop_length=hopLength
    )
    voicedProbFrame = np.nan_to_num(voicedProbFrame, nan=0.0).astype(np.float32)  # (frames,)

    # vocal-band energy ratio per frame
    vLo, vHi = vocalBand
    vMask = (freqs >= vLo) & (freqs <= vHi)
    bandEnergy = np.sum(S[vMask, :]**2, axis=0) + 1e-12
    totalEnergy = np.sum(S**2, axis=0) + 1e-12
    bandRatioFrame = (bandEnergy / totalEnergy).astype(np.float32)  # (frames,)

    # spectral flatness per frame (noise-like -> high)
    flatFrame = librosa.feature.spectral_flatness(S=S)[0].astype(np.float32)  # (frames,)

    # harmonicity proxy via HPSS (framewise)
    H, P = librosa.decompose.hpss(S)
    harmE = np.sum(H**2, axis=0) + 1e-12
    percE = np.sum(P**2, axis=0) + 1e-12
    harmRatioFrame = (harmE / (harmE + percE)).astype(np.float32)  # (frames,)

    # Aggregate per chunk
    bandRatio = np.zeros(T, dtype=np.float32)
    flatness = np.zeros(T, dtype=np.float32)
    harmRatio = np.zeros(T, dtype=np.float32)
    voicedProb = np.zeros(T, dtype=np.float32)
    counts = np.zeros(T, dtype=np.int32)

    for i in tqdm(range(nFrames), desc="Aggregating STFT frames"):
        c = frameChunk[i]
        bandRatio[c] += bandRatioFrame[i]
        flatness[c] += flatFrame[i]
        harmRatio[c] += harmRatioFrame[i]
        voicedProb[c] += voicedProbFrame[i]
        counts[c] += 1

    counts = np.maximum(counts, 1)
    bandRatio /= counts
    flatness /= counts
    harmRatio /= counts
    voicedProb /= counts
    print("counts min/med/max:", counts.min(), np.median(counts), counts.max())
    print("num chunks with 0 frames:", int(np.sum(counts == 0)))

    # -------------------------
    # Feature 5: ZCR (per chunk)
    # -------------------------
    signs = np.sign(chunks)
    signs[signs == 0] = 1
    zcr = np.mean(signs[:, 1:] != signs[:, :-1], axis=1).astype(np.float32)  # (T,)

    # -------------------------
    # 1) Silence detection
    # -------------------------
    med = float(np.median(rms))
    mad = float(np.median(np.abs(rms - med)) + 1e-12)

    silenceThr = med + 0.5 * mad
    # "true silence" should be low energy; bandRatio guard helps reject weird low-energy mid-band artifacts
    isSilence = (rms < silenceThr) & (bandRatio < 0.25)

    # -------------------------
    # 2) Voiced vs unvoiced (voicing)
    # -------------------------
    # Start with voicedProb > 0.7, then optionally add harmonic/flatness guards to reduce false voiced.
    energyGuard = rms > (med + 0.1 * mad)

    isVoiced = (voicedProb > 0.7) & (~isSilence) & energyGuard
    # Optional extra robustness:
    isVoiced = isVoiced & (harmRatio > 0.55) & (flatness < 0.35)
    isVoiced = enforceMinRun(isVoiced, minOnRun=3, minOffRun=3)

    # -------------------------
    # 3) Vocal activity detection (hasVocal)
    # -------------------------
    score = (1.2 * bandRatio + 1.0 * harmRatio - 0.8 * flatness).astype(np.float32)
    score -= 0.2 * (zcr > 0.25).astype(np.float32)

    nonSilent = ~isSilence
    if np.any(nonSilent):
        sMed = float(np.median(score[nonSilent]))
        sMad = float(np.median(np.abs(score[nonSilent] - sMed)) + 1e-12)
        vocalThrOn = sMed + 0.75 * sMad
        vocalThrOff = sMed + 0.40 * sMad
    else:
        vocalThrOn, vocalThrOff = 0.5, 0.3

    # rising-energy ON gate (prevents reverb tails from constantly re-triggering)
    rmsDiff = np.diff(rms, prepend=rms[0]).astype(np.float32)
    riseThr = 0.0 + 0.05 * mad  

    rawVocal = (score > vocalThrOn) & energyGuard  & (~isSilence)

    # Hysteresis
    hasVocal = np.zeros(T, dtype=bool)
    state = False
    for t in range(T):
        if not state:
            if rawVocal[t]:
                state = True
        else:
            if (score[t] < vocalThrOff) or isSilence[t]:
                state = False
        hasVocal[t] = state

    # Smoothing
    if smoothWin and smoothWin > 1:
        k = int(smoothWin)
        padL = k // 2
        padded = np.pad(hasVocal.astype(np.int32), (padL, padL), mode="edge")
        sm = np.convolve(padded, np.ones(k, dtype=np.int32), mode="valid")
        hasVocal = sm >= (k // 2 + 1)

    hasVocal = enforceMinRun(hasVocal, minOnRun=minOnRun, minOffRun=minOffRun)

    return isSilence, hasVocal, isVoiced

def enforceMinRun(flags, minOnRun=3, minOffRun=3):
    """
    Removes short on-blips and short off-gaps.
    """
    flags = flags.astype(bool)
    n = len(flags)
    if n == 0:
        return flags

    # run-length encoding
    out = flags.copy()
    start = 0
    cur = out[0]
    for i in range(1, n + 1):
        if i == n or out[i] != cur:
            runLen = i - start
            if cur and runLen < minOnRun:
                out[start:i] = False
            if (not cur) and runLen < minOffRun:
                out[start:i] = True
            if i < n:
                start = i
                cur = out[i]
    return out
       
def _toBoolList(x: np.ndarray):
    return [bool(v) for v in x.tolist()]

def _ensureDir(path: str):
    os.makedirs(path, exist_ok=True)

def resolveVocalsPath(group: str, songName: str, useCache: bool) -> str:
    base = os.path.join(".", "training_data", group)
    if not useCache:
        return os.path.join(base, f"{songName}_vocals.wav")
    return os.path.join(base, "training_cache", "sr_24000", f"{songName}_vocals.wav")

def resolveOutputJsonPath(group: str, songName: str, useCache: bool) -> str:
    base = os.path.join(".", "training_data", group)
    if not useCache:
        return os.path.join(base, f"{songName}_vocals_40ms_activity.json")
    return os.path.join(base, "training_cache", "sr_24000", f"{songName}_vocals_40ms_activity.json")

def writeChunkActivityJson(
    inputWavPath: str,
    outputJsonPath: str,
    isSilence: np.ndarray,
    hasVocal: np.ndarray,
    isVoiced: np.ndarray,
    sr: int,
    chunkMs: int,
    nSamples: int
):
    _ensureDir(os.path.dirname(outputJsonPath))

    # Safety: ensure equal lengths
    if not (len(hasVocal) == len(isVoiced) == len(isSilence)):
        raise ValueError("Activity arrays must all have same length")

    # Identity-valid mask
    # Must have vocal presence AND be voiced (pitched)
    validMask = np.logical_and(hasVocal, isVoiced)

    chunkLenSamples = int(round(sr * (chunkMs / 1000.0)))
    numChunks = int(len(hasVocal))
    durationSec = float(nSamples) / float(sr)

    payload = {
        "inputWavPath": os.path.normpath(inputWavPath),
        "sampleRate": int(sr),
        "durationSec": durationSec,
        "chunkMs": int(chunkMs),
        "chunkLenSamples": int(chunkLenSamples),
        "numChunks": numChunks,

        # raw activity
        "isSilence": _toBoolList(isSilence),
        "hasVocal": _toBoolList(hasVocal),
        "isVoiced": _toBoolList(isVoiced),

        # final identity-valid mask
        "validIdentityMask": _toBoolList(validMask),
    }

    with open(outputJsonPath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Compute 40ms vocal-vs-silence activity flags for vocals-only wavs (native sample-rate)."
    )
    parser.add_argument("-group", required=True, type=str)
    parser.add_argument("-song_name", required=True, type=str)
    parser.add_argument(
        "-use_cache",
        action="store_true",
        help="If set, use ./training_data/{group}/training_cache/sr_24000/{song_name}_vocals.wav"
    )
    parser.add_argument("-chunk_ms", type=int, default=40)

    args = parser.parse_args()
    group = args.group
    songName = args.song_name
    useCache = bool(args.use_cache)
    chunkMs = int(args.chunk_ms)

    inputWavPath = resolveVocalsPath(group, songName, useCache)
    if not os.path.isfile(inputWavPath):
        raise FileNotFoundError(f"Could not find vocals wav at: {os.path.abspath(inputWavPath)}")

    # Native sample rate (no resampling)
    y, sr = librosa.load(inputWavPath, sr=None, mono=True)
    if y is None or len(y) == 0:
        raise ValueError(f"Loaded empty audio from: {os.path.abspath(inputWavPath)}")

    # Compute flags on a 40ms grid in *time* (chunk boundaries are derived from sr)
    isSilence, hasVocal, isVoiced = computeChunkVocalSilenceFlags(
        y=y,
        sr=sr,
        chunkMs=chunkMs,
        vocalBand=(300, 4000),
        smoothWin = 11,      # 360ms
        minOnRun = 8,      # 240ms
        minOffRun = 8
    )

    outputJsonPath = resolveOutputJsonPath(group, songName, useCache)
    writeChunkActivityJson(
        inputWavPath=inputWavPath,
        outputJsonPath=outputJsonPath,
        isSilence=isSilence,
        hasVocal=hasVocal,
        isVoiced=isVoiced,
        sr=sr,
        chunkMs=chunkMs,
        nSamples=len(y)
    )

    print(f"[OK] input sr={sr}, chunks={len(hasVocal)}")
    print(f"[OK] Wrote: {os.path.abspath(outputJsonPath)}")

if __name__ == "__main__":
    main()