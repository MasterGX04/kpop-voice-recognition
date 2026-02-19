"""
K-pop singer recognition using SpeechBrain ECAPA-TDNN embeddings + a small softmax head.

- Expects isolated vocals under:
    root/
      groupName/
        memberName/
          train/
            Isolated_Vocals/
               *.wav (or nested folders)

- Resamples 22_050 Hz -> 16_000 Hz (ECAPA requirement).
- Builds a tiny linear classifier head over ECAPA embeddings for: [members ... , silence]
- Trains/evaluates on GPU, saves & reloads.

Tested on: Python 3.9, CUDA 12.x with torch cu124 wheels.
"""

import os, argparse,  random, glob, json
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Tuple, Dict 

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
import numpy as np
from model.heads import PresenceHead
from model.encoders import MuQEncoderWrapper, FusedEncoder, FusedEncoderWithECAPA
from torchaudio.transforms import Resample

from tqdm import tqdm
from muq import MuQ
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------
# CLI args
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Path that contains <group>/<member>/train/Isolated_Vocals")
    ap.add_argument("--group", type=str, required=True, help="Group folder name under root")
    ap.add_argument("--sr_in", type=int, default=44100, help="Expected input sample rate of your files")
    ap.add_argument("--sr_out", type=int, default=24000, help="ECAPA target sample rate")
    ap.add_argument("--chunk-sec", type=float, default=2.0, help="Chunk length in seconds for training")
    ap.add_argument("--short-chunk-sec", type=float, default=0.4, help="Chunk length in seconds for refinement")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-dir", type=str, default="./checkpoints")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval_thr", type=float, default=0.7)
    ap.add_argument("--skip-stage1",  action="store_true", help="Checks if stage 1 can be skipped")
    return ap.parse_args()

# ---------------------------
# Small utilities
# ---------------------------
def list_wavs(root: str) -> List[str]:
    exts = ("*.wav", "*.flac", "*.mp3", "*.m4a")
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, "**", e), recursive=True)
    return files

def set_seed(s: int):
    random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    
def _load_audio_mono(path: str):
    wav, sr = torchaudio.load(path)  # (C, T)
    if wav.ndim == 2 and wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)  # stereo -> mono
    elif wav.ndim == 1:
        wav = wav.unsqueeze(0)
    return wav, sr

def _resample_and_save(in_path: str, out_path: str, sr_out: int):
    wav, sr_in = _load_audio_mono(in_path)
    
    if sr_in != sr_out:
        resampler = torchaudio.transforms.Resample(orig_freq=sr_in, new_freq=sr_out)
        wav = resampler(wav)
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torchaudio.save(out_path, wav, sample_rate=sr_out, encoding="PCM_S", bits_per_sample=24)
    
def buildTrainingCache(groupDir: str, srOut: int, numWorkers: int = 8) -> str:
    """
    Creates/uses: <groupDir>/training_cache/sr_<srOut>/
    Resamples vocals/leading/backing wavs into the cache folder.

    - Multithreaded resample+save
    - tqdm progress bar (total = #songs/files to cache)
    - Skips any file that already exists in cacheDir

    Returns the cache directory path.
    """
    groupDir = str(groupDir)
    cacheDir = os.path.join(groupDir, "training_cache", f"sr_{srOut}")
    os.makedirs(cacheDir, exist_ok=True)

    # Only these patterns are used by _find_audio_triplet() in this script
    patterns = ["*_vocals.wav", "*_leading_vocals.wav", "*_backing_vocals.wav"]

    # Gather candidate input files
    inFiles = []
    for pat in patterns:
        inFiles.extend(Path(groupDir).glob(pat))

    if not inFiles:
        print(f"[cache] No matching WAV stems found in {groupDir} (patterns={patterns}).")
        return cacheDir

    # Build a work list that excludes already-cached outputs
    work = []
    for inPath in inFiles:
        outPath = os.path.join(cacheDir, inPath.name)
        if os.path.exists(outPath):
            continue
        work.append((str(inPath), outPath))

    print(f"[cache] Cache dir: {cacheDir}")
    print(f"[cache] Found {len(inFiles)} wavs. Need to create {len(work)} cached wavs (skipping {len(inFiles) - len(work)}).")

    if not work:
        return cacheDir

    # Worker wrapper so exceptions don't kill the whole pool
    def _do_one(inPath: str, outPath: str):
        _resample_and_save(inPath, outPath, srOut)
        return os.path.basename(outPath)

    # Threaded execution + tqdm progress
    # NOTE: torchaudio resample is CPU-heavy; threads help mainly if you're I/O bound.
    # If you find it CPU-bound, switch to ProcessPoolExecutor (but then _resample_and_save must be picklable).
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, int(numWorkers))) as ex:
        futures = [ex.submit(_do_one, inP, outP) for inP, outP in work]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Caching (resample+save)", unit="file"):
            try:
                fut.result()
            except Exception as e:
                failures += 1
                # You can print less frequently if you want
                print(f"[cache] Failed: {e}")

    if failures:
        print(f"[cache] Done with {failures} failures. (See logs above.)")
    else:
        print("[cache] Done. All cached successfully.")

    return cacheDir
    
@dataclass
class ClassMap:
    idx_to_name: List[str]
    name_to_idx: Dict[str, int]
    
# ---------------------------
# Dataset: loads files, makes fixed-length waveform chunks, includes "silence"
# ---------------------------
class KpopVocalDataset(Dataset):
    """
    Builds 2s (chunk_sec) training examples from full-song vocals + frame label JSON.

    Directory layout (per group):
        ./training_data/<group>/
            <SongName>_vocals.wav / .mp3 / .flac / .m4a
            <SongName>_frame_labels.json

    JSON format (per song):
        {
            "group": "IVE",
            "song": "Hypnosis",
            "members": [...],              # length = num_members (no silence)
            "chunkDurationMs": 40,
            "numChunks": T,
            "presence": [ [0/1,...], ... ] # shape [T, num_members]
            ... (isAdlib, lead, isRepeat, etc. – ignored for now)
        }

    Classes used for training:
        [members..., "silence"]
        => label size = num_members + 1, last index is silence.
    """
    def __init__(self, group_dir: str, sr_out: int,
                context_seconds: float, group_name: str, audio_dir: str, is_phase2: bool = False, min_song_sec: float = 4.0,
                window_hop_ratio: float = 0.5, presence_thresh=0.4, alpha_lead=1.0, 
                alpha_adlib = 0.5, min_weight = 0.2, max_weight = 2.0, k_train: int = 2):
        super().__init__()
        self.group_dir = group_dir
        self.sr_out = sr_out # This should be 24000 HZ
        self.context_seconds = context_seconds
        self.chunk_len = int(round(context_seconds * sr_out))
        self.min_song_sec = min_song_sec
        self.is_phase2 = is_phase2
        
        self.window_hop_ratio = window_hop_ratio
        self.k_train = k_train
        
        self._song_audio = {}
        self.song_cache = {}  # song_name -> label arrays + num_chunks + frame_ms
        self.presence_thresh = presence_thresh
        self.alpha_lead = alpha_lead
        self.alpha_adlib = alpha_adlib
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.audio_dir = audio_dir

        # ----------------------------
        # 1) Discover all JSON label files
        # ----------------------------
        json_files = self._discover_json_files()
        
        # ----------------------------
        # 2) Load first JSON to define members/classes
        # ----------------------------
        members = self._init_class_map_from_first_json(json_files)
        
        # ----------------------------
        # 1b) Optional: manual harmony annotations
        #     ./saved_labels/<group>/<Song>_test_harmonies.json
        #     ./debug_harmonies/<group>/<Song>/<pair_name>.mp3
        # ---
        self.group_name = group_name
        
        # JSON with manual pairs
        harmony_label_dir = os.path.join(".", "saved_labels", group_name)
        harmony_pattern = os.path.join(harmony_label_dir, "*_test_harmonies.json")
        
        # Root where debug harmony mp3 clips live
        self.debug_harmony_root = os.path.join(".", "debug_harmonies", group_name)
        # self.manual_harmonies = self._load_manual_harmony_jsons(harmony_files)
        
        # self.synthetic_harmony_clips = self._collect_synthetic_harmony_clips()
        
        self.gang_idx = None
        for i, name in enumerate(self.classes):
            if name.lower() == "gang vocal":
                self.gang_idx = i
                break
            
        self._init_debug_counters()
        
        # ----------------------------
        # 3) Build index of (audio_path, start_sample_out, label_vec)
        # ----------------------------
        frames_per_window, hop_frames, frame_ms = self._compute_window_params()
        self.local_band_frames = 8
        
        self.samples: List[Tuple[str, int,
                         np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray]] = []
        
        for jpath in json_files:
            with open(jpath, "r", encoding="utf-8") as f:
                meta = json.load(f)

            song_name = meta["song"]
            members_j = meta["members"]
            # Basic sanity check: same member ordering across songs
            if members_j != members:
                raise ValueError(
                    f"Members mismatch in {jpath}: {members_j} vs {members}"
                )
            
            num_chunks = meta["numChunks"]
            presence = np.asarray(meta["presence"], dtype=np.int32)  # shape: [T, num_members]
            if presence.shape[0] != num_chunks or presence.shape[1] != self.num_members:
                raise ValueError(f"presence shape mismatch in {jpath}")
            lead_arr = np.asarray(meta["lead"], dtype=np.int32)      # [T, C]
            adlib_arr = np.asarray(meta["isAdlib"], dtype=np.int32)  # [T, C]
            backing_arr = np.asarray(meta["isBacking"], dtype=np.int32) # backing /harmony
            adlib_primary_arr = np.asarray(meta["adlibPrimary"], dtype=np.int32)
            stem_choice_arr = np.asarray(meta["stemChoice"], dtype=np.int8)
            
            self.song_cache[song_name] = {
                "num_chunks": num_chunks,
                "presence": presence,
                "lead": lead_arr,
                "adlib": adlib_arr,
                "backing": backing_arr,
                "adlib_primary": adlib_primary_arr,
                "stem_choice": stem_choice_arr,
            }
        
            song_dur_sec = num_chunks * frame_ms / 1000.0
            if song_dur_sec < self.min_song_sec or num_chunks < frames_per_window:
                print(f"[KpopFrameDataset] Skipping {song_name}: too short ({song_dur_sec:.2f}s)")
                continue
            
            # Find the corresponding vocals audio file
            mix_path, leading_path, backing_path = self._find_audio_triplet(song_name)
            if leading_path is None: 
                leading_path = mix_path
            if backing_path is None: 
                backing_path = mix_path
                
            if mix_path is None:
                print(f"[KpopFrameDataset] No audio found for song {song_name}, skipping.")
                continue
            
            self._song_audio[song_name] = {
                "mix": mix_path,
                "lead": leading_path if leading_path is not None else mix_path,
                "back": backing_path if backing_path is not None else mix_path,
            }
            
            # Slide window over frame indices
            half_win = frames_per_window // 2
            
            # only use frames that have a full window around them
            first_center = half_win
            last_center = num_chunks - half_win - 1
            
            # Just in case
            if last_center <= first_center:
                print(f"[KpopFrameDataset] {song_name}: not enough frames for a full window, skipping.")
                continue
            
            for center_frame in range(first_center, last_center + 1, hop_frames):
                # Window [start_frame : end_frame) is the 2s context around this 40ms frame
                start_frame = center_frame - half_win
                end_frame = start_frame + frames_per_window
                
                # Sanity check
                if start_frame < 0 or end_frame > num_chunks:
                    continue # Skip weird edge cases
                
                # 2s context labels (for importance)
                window_presence = presence[start_frame:end_frame]   # [F, C]
                window_lead = lead_arr[start_frame:end_frame]
                window_adlib = adlib_arr[start_frame:end_frame]
                window_backingStyle = backing_arr[start_frame:end_frame]
                
                lossWeightVec = np.ones(len(self.classes), dtype=np.float32)
                
                # Fractions over this window (0..1) - still useful for importance weights
                presenceCenter = presence[center_frame].astype(np.int32)          # (C,)
                centerHasVocal = presenceCenter.sum() > 0

                anyActivePerFrame = (window_presence.sum(axis=1) > 0).astype(np.float32)
                windowVocalFrac = float(anyActivePerFrame.mean())

                windowOverlapFrac = float((window_presence.sum(axis=1) > 1).astype(np.float32).mean())

                # Avoid weird ratios when windowVocalFrac is tiny
                framesActive = window_presence.mean(axis=0) # (C,)
                domIdx = int(framesActive.argmax())
                domFracAmongFrames = float(framesActive[domIdx]) # fraction of frames in window
                domFracAmongVocals = float(framesActive[domIdx] / max(windowVocalFrac, 1e-3))
  
                # ----------------------------
                # Stage 1: set targets (truth)
                # ----------------------------
                targetVec = self._build_targets_for_center(
                    centerHasVocal=centerHasVocal, 
                    presenceCenter=presenceCenter, 
                    window_presence=window_presence, 
                    windowVocalFrac=windowVocalFrac,
                )
                
                # Local band (already discussed earlier)
                lf, rf, local_presence, local_dom_frac = self._compute_local_band(
                    window_presence=window_presence,
                    center_in_window=half_win,
                    local_band_frames=self.local_band_frames,
                )
                
                # ----------------------------
                # Stage 2: set loss weights (how hard to learn)
                # ----------------------------
                # Default: don’t nuke anything to 0 unless you truly want "ignore"
                lossWeightVec, category = self._build_weights_for_center(
                    centerHasVocal=centerHasVocal,
                    targetVec=targetVec,
                    presenceCenter=presenceCenter,

                    window_presence=window_presence,
                    window_lead=window_lead,
                    window_adlib=window_adlib,
                    window_backingStyle=window_backingStyle,
                    domFracAmongVocals=domFracAmongVocals,
                    windowOverlapFrac=windowOverlapFrac,

                    local_presence=local_presence,
                    local_dom_frac=local_dom_frac,
                    lf=lf,
                    rf=rf,

                    lead_arr=window_lead,
                    backing_arr=window_backingStyle,
                    adlib_arr=window_adlib,
                )
                
                # ---------------------------
                # Debug accounting
                # ---------------------------
                # if category not in self.debug_category_counts:
                #     # should never happen, but safe-guard
                #     self.debug_category_counts[category] = 0
                #     self.debug_category_examples[category] = []

                # self.debug_category_counts[category] += 1
                # count = self.debug_category_counts[category]

                # debug_info = {
                #     "song": song_name,
                #     "center_frame": int(center_frame),
                #     "vocal_frac_window": float(vocal_frac_window),
                #     "overlap_frac_window": float(overlap_frac_window),
                #     "dominant_idx": int(dominant_idx),
                #     "dominant_frac": float(dominant_frac),
                #     "frame_label": frame_label.tolist(),
                # }

                # reservoir = self.debug_category_examples[category]
                # max_n = self._max_debug_examples

                # if len(reservoir) < max_n:
                #     # fill up until we reach max_n
                #     reservoir.append(debug_info)
                # else:
                #     # reservoir sampling: replace an existing one with decreasing probability
                #     j = random.randint(0, count - 1)
                #     if j < max_n:
                #         reservoir[j] = debug_info
                
                # ---------------------------
                # Multi-task: harmony + ad-lib targets
                # ---------------------------
                # Per-singer arrays
                harmony_vec = np.zeros(self.num_members, np.float32)
                adlib_vec = np.zeros(self.num_members, np.float32)

                harmony_wts = np.zeros(self.num_members, np.float32)
                adlib_wts = np.zeros(self.num_members, np.float32)

                # center band
                band = 1
                cf0 = max(start_frame, center_frame - band)
                cf1 = min(end_frame, center_frame + band + 1)

                center_presence = presence[cf0:cf1].max(axis=0).astype(bool)
                center_adlib = adlib_arr[cf0:cf1].max(axis=0).astype(bool)
                center_backing = backing_arr[cf0:cf1].max(axis=0).astype(bool)
                center_overlap = (presence[cf0:cf1].sum(axis=1) > 1).any()
                center_primary_adlib = adlib_primary_arr[cf0:cf1].max(axis=0).astype(bool)
                center_present_mask = center_presence[:self.num_members]

                # vectors
                adlib_vec[center_adlib] = 1.0
                pos_harm = center_backing & ~center_adlib  # harmony-ish backing that isn't adlib
                harmony_vec[pos_harm] = 1.0

                # weights = just masks
                adlib_wts[center_presence] = 1.0
                if center_overlap:
                    harmony_wts[center_presence] = 1.0
                    lossWeightVec[:self.num_members] *= 0.6
                    # and especially don't enforce negatives
                    lossWeightVec[:self.num_members][~center_present_mask] *= 0.2
                    
                adlib_wts[center_primary_adlib] = 2.0
                                
                # Map window start frame -> start sample at sr_out
                start_time_sec = (start_frame * frame_ms) / 1000.0
                start_sample_out = int(round(start_time_sec * self.sr_out))
                
                # ---------------------------
                # Choose which stem(s) to emit for this center_frame
                # ---------------------------
                VOCALS, LEAD, BACK = 0, 1, 2

                center_n_active = int(presenceCenter.sum())
                any_vocal_center = center_n_active > 0
                
                if not any_vocal_center or center_n_active <= 1:
                    emit_kinds = ["mix"]
                else:
                    # active member indices at center
                    active_idx = np.flatnonzero(presenceCenter > 0)
                    
                    # which stems are requested by stemChoice among active members
                    requested = set(int(stem_choice_arr[center_frame, i]) for i in active_idx)
                    
                    # Map requested stem ids -> kind strings
                    emit_kinds = []
                    if LEAD in requested:
                        emit_kinds.append("lead")
                    if BACK in requested:
                        emit_kinds.append("back")
                    
                    # if nobody 'requested' lead/back, fall back to mix
                    if not emit_kinds:
                        emit_kinds = ["mix"]
    
                stemWeightPerKind = 1.0 if len(emit_kinds) == 1 else 0.5
                
                for kind in emit_kinds:
                    kind_id = {"mix": VOCALS, "lead": LEAD, "back": BACK}[kind]
                    lossWeightVec2 = lossWeightVec.copy()
                    harmony_wts2 = harmony_wts
                    adlib_wts2 = adlib_wts
                    
                    # if a member is active at the center but their stemChoice doesn't match this sample's stem,
                    # reduce their contribution for this sample only.
                    if any_vocal_center:
                        active_idx = np.flatnonzero(presenceCenter[:self.num_members] > 0)
                        mism = [i for i in active_idx if int(stem_choice_arr[center_frame, i]) != kind_id]
                        # tune this; 0.25 is a good start
                        lossWeightVec2[mism] *= 0.25

                    if kind == "back" and self._song_audio[song_name]["back"] != self._song_audio[song_name]["mix"]:
                        lossWeightVec = lossWeightVec.copy()
                        harmony_wts2 = harmony_wts.copy()
                        adlib_wts2 = adlib_wts.copy()
                        lossWeightVec[:self.num_members] *= 0.85
                        harmony_wts2 *= 1.15
                        adlib_wts2 *= 1.15

                    self.samples.append(
                        (
                            song_name, # <-- store song
                            kind, # <-- store stem kind
                            center_frame,
                            start_frame,
                            start_sample_out,
                            targetVec,
                            lossWeightVec2,
                            harmony_vec,
                            harmony_wts2,
                            adlib_vec,
                            adlib_wts2,
                            stemWeightPerKind
                        )
                    )
    
        if not self.samples:
            raise RuntimeError("No training windows built. Check your JSON/audio alignment.")
        
        total_labels = np.zeros(len(self.classes), dtype=np.float64)
        total_weights = np.zeros(len(self.classes), dtype=np.float64)

        for (_, _, _, _, _, targetVec, lossWeightVec,
             _, _, _, _, _) in self.samples:
            total_labels += targetVec
            total_weights += lossWeightVec
            
        print(f"Total labels: {total_labels / len(self.samples)}")
        print(f"Total weights: {total_weights / len(self.samples)}")
        
        n_silence = sum(
            label_vec[self.silence_idx] == 1.0
            for (_, _, _, _, _, label_vec, _, _, _, _, _, _) in self.samples
        )
        print("silence windows:", n_silence, "/", len(self.samples))
        
        self.base_samples = list(self.samples)
        
        # Test for Dataset to see proportion of Data
        print("\n[KpopFrameDataset] Category counts:")
        total_windows = sum(self.debug_category_counts.values())
        for cat, cnt in self.debug_category_counts.items():
            frac = cnt / max(total_windows, 1)
            print(f"  {cat:13s}: {cnt:7d} ({frac:5.1%})")

        print("\n[KpopFrameDataset] Example windows per category:")
        for cat, examples in self.debug_category_examples.items():
            print(f"\n  Category: {cat}  (showing {len(examples)} examples)")
            for ex in examples:
                print(f"    song={ex['song']}, center_frame={ex['center_frame']}, "
                    f"vocal_frac={ex['vocal_frac_window']:.2f}, "
                    f"overlap_frac={ex['overlap_frac_window']:.2f}, "
                    f"dominant_idx={ex['dominant_idx']}, "
                    f"dominant_frac={ex['dominant_frac']:.2f}, "
                    f"frame_label={ex['frame_label']}")
        
        # ----------------------    ------
        # 4) Simple audio cache to avoid re-loading the same song
        # ----------------------------
        self._max_cached_songs = 32       # tune this (8–32 is typical)
        self._wave_cache = OrderedDict()  # path -> wav (1, T_out)
        self._resamplers: Dict[int, Resample] = {}

        print(f"[KpopFrameDataset] total windows: {len(self.samples)}")
    
    def _discover_json_files(self) -> List[str]:
        """
        Find and validate all *_frame_labels.json files under the group directory.

        Returns:
            A sorted list of JSON label file paths.
        Raises:
            RuntimeError if no label JSON files are found.
        """
        json_pattern = os.path.join(self.group_dir, "*_frame_labels.json")
        json_files = sorted(glob.glob(json_pattern))
        if not json_files:
            raise RuntimeError(f"No *_frame_labels.json files found under {self.group_dir}")
        
        return json_files
        
    def _init_class_map_from_first_json(self, json_files: List[str]) -> List[str]:
        """
        Read the first frame-label JSON to initialize dataset-wide metadata:
        - member list / class names
        - chunk duration in ms
        - number of members
        - silence index and ClassMap lookup tables

        Returns:
            The members list (no 'silence' included).
        """
        with open(json_files[0], "r", encoding="utf-8") as f:
            meta0 = json.load(f)
        members = meta0["members"]
        self.chunk_duration_ms = meta0["chunkDurationMs"]  # should be 40
        self.num_members = len(members)
        
        self.classes = members + ['silence']
        self.silence_idx = len(self.classes) - 1
        self.class_map = ClassMap(
            idx_to_name=self.classes,
            name_to_idx={name: i for i, name in enumerate(self.classes)}
        )        

        return members
    
    def _init_debug_counters(self, max_examples: int = 10) -> None:
        """
        Initialize per-category debug counters and example storage.
        Used to inspect dataset composition (clear vs semi vs ambiguous vs silence)
        without affecting training behavior.
        """
        # Debug stats: how many chunks per category, and a few examples
        self.debug_category_counts = {
            "true_silence": 0,
            "clear_vocal": 0,
            "semi_clear_vocal": 0,
            "ambiguous": 0,
        }

        # store up to N examples per category
        self.debug_category_examples = {
            "true_silence": [],
            "clear_vocal": [],
            "semi_clear_vocal": [],
            "ambiguous": [],
        }
        self._max_debug_examples = max_examples

    def _compute_window_params(self) -> Tuple[int, int, float]:
        """
        Compute windowing parameters for slicing label frames:
        - frames_per_window: number of 40ms frames in the context window
        - hop_frames: stride between centers (k_train overrides window_hop_ratio)
        - frame_ms: chunk duration in ms (usually 40)

        Returns:
            (frames_per_window, hop_frames, frame_ms)
        """
        frame_ms = float(self.chunk_duration_ms)
        frames_per_window = int(round(self.context_seconds * 1000.0 / frame_ms))  # e.g. 2.0s / 40ms = 50
        
        if frames_per_window <= 0:
            raise ValueError("frames_per_window computed as <=0, check chunk_sec and chunkDurationMs")
        
        hop_frames = max(1, int(round(frames_per_window * self.window_hop_ratio)))
        
        if self.k_train is not None:
            hop_frames = max(1, int(self.k_train))
        else:
            hop_frames = max(1, int(round(frames_per_window * self.window_hop_ratio)))

        print(f"[KpopFrameDataset] Frames/window={frames_per_window}, hop_frames={hop_frames}")
        print(f"[KpopFrameDataset] Members={self.classes}")
        
        return frames_per_window, hop_frames, frame_ms
    
    def _build_targets_for_center(self, *, centerHasVocal: bool, presenceCenter: np.ndarray,
                              window_presence: np.ndarray, windowVocalFrac: float) -> Tuple[np.ndarray, bool]:
        """
        Build the target vector for a single center frame.

        Rules:
        - If center has vocal: member presence at center is the ground truth; silence=0.
        - If center is silent: decide TRUE silence vs REST using window/local vocal fractions.
        - Optionally set special targets like 'gang vocal' based on window presence.

        Returns:
            (targetVec, trueSilenceFlag)
        """
        TAU_TRUE_SILENCE_WIN = 0.20      # window mostly silent\
        targetVec = np.zeros(len(self.classes), dtype=np.float32)
                            
        if centerHasVocal:
            # center singer(s) are the ground truth
            targetVec[:self.num_members] = presenceCenter.astype(np.float32)
            targetVec[self.silence_idx] = 0.0

            # Gang vocal: window-based tag
            if self.gang_idx is not None:
                GANG_TAU = 0.1
                presenceFrac = window_presence.mean(axis=0)
                targetVec[self.gang_idx] = 1.0 if presenceFrac[self.gang_idx] >= GANG_TAU else 0.0

        else:
            # center is silent: decide TRUE silence vs REST
            trueSilence = (windowVocalFrac <= TAU_TRUE_SILENCE_WIN)

            if trueSilence:
                targetVec[self.silence_idx] = 1.0
            else:
                # REST: not true silence, but don’t label any singer as present either
                targetVec[self.silence_idx] = 0.0
                
        return targetVec
    
    def _compute_local_band(
        self,
        window_presence: np.ndarray,
        center_in_window: int,
        local_band_frames: int,
    ) -> tuple[int, int, np.ndarray, float]:
        """
        Compute a small 'local band' around the center frame inside the current window.

        Purpose:
            - Provide lf/rf indices for slicing arrays (presence/lead/backing/adlib)
            - Compute local_presence (frames x classes)
            - Compute local_dom_frac: dominance of the top member inside the local band

        Args:
            window_presence: (T, num_classes) presence array for the current window.
            center_in_window: center index inside the window (0..T-1).
            local_band_frames: half-width of the local band in frames (e.g., 6 -> ~0.24s at 40ms).

        Returns:
            lf: left slice index (inclusive)
            rf: right slice index (exclusive)
            local_presence: window_presence[lf:rf]
            local_dom_frac: dominance fraction in the local band among members
        """
        T = window_presence.shape[0]
        lf = max(0, center_in_window - local_band_frames)
        rf = min(T, center_in_window + local_band_frames + 1)

        local_presence = window_presence[lf:rf]

        # dominance among members only
        local_presence_frac = local_presence.mean(axis=0)[:self.num_members]
        denom = float(local_presence_frac.sum()) + 1e-8
        local_dom_frac = float(local_presence_frac.max() / denom) if denom > 0 else 0.0

        return lf, rf, local_presence, local_dom_frac
    
    def _build_weights_for_center(
        self,
        *,
        centerHasVocal: bool,
        targetVec: np.ndarray,
        presenceCenter: np.ndarray,

        # window-level stats
        window_presence: np.ndarray,
        window_lead: np.ndarray,
        window_adlib: np.ndarray,
        window_backingStyle: np.ndarray,
        domFracAmongVocals: float,
        windowOverlapFrac: float,

        # local-band stats
        local_presence: np.ndarray,
        local_dom_frac: float,
        lf: int,
        rf: int,

        # arrays (window-aligned)
        lead_arr: np.ndarray,
        backing_arr: np.ndarray,
        adlib_arr: np.ndarray,
    ) -> tuple[np.ndarray, str]:
        """
        Build the per-class loss weight vector for a single center frame.

        Responsibilities:
        - Decide clear / semi-clear / ambiguous category
        - Assign singer weights based on dominance + role (lead/backing/adlib)
        - Cap negative weights to prevent multi-logit spam
        - Stabilize silence vs rest behavior

        Returns:
            lossWeightVec: np.ndarray [num_classes]
            category: str
        """
        # ----------------------------
        # Thresholds (same idea, clearer names)
        # ----------------------------
        TAU_CLEAN_DOM = 0.55
        TAU_OVERLAP_MAX = 0.30
        TAU_LOCAL_DOM = 0.65
        
        # Define thresholds
        W_LEAD = 0.9 # How much lead boosts weight
        W_BACKING = 0.6 # backing/harmony boost
        W_ADLIB = 0.3 # How strongly ad-libs DOWN-weight
        
        def clamp01(x: float) -> float:
            return max(0.0, min(1.0, float(x)))

        def lerp(a: float, b: float, t: float) -> float:
            t = clamp01(t)
            return a + t * (b - a)
        
        domScore = clamp01((domFracAmongVocals - 0.55) / (TAU_CLEAN_DOM - 0.55 + 1e-6))
        ovlScore = 1.0 - clamp01(windowOverlapFrac / (TAU_OVERLAP_MAX + 1e-6))
        cleanScore = clamp01(0.65 * domScore + 0.35 * ovlScore)
        
        def negCapFor(category: str, cleanScore: float, isPhase2: bool) -> float:
            # ranges are (messy_low, clean_high)
            # "clean_high" is how hard we punish FPs when the window is trustworthy
            # "messy_low" is how forgiving we are when the window is unreliable

            if category == "clear_vocal":
                lo, hi = (0.45, 1.00)   # even messy-ish clear_vocal should not be super cheap
            elif category == "semi_clear_vocal":
                lo, hi = (0.35, 0.95)
            elif category == "ambiguous":
                lo, hi = (0.30, 0.85)
            elif category == "ambiguous_rest":
                lo, hi = (0.30, 0.80)
            elif category == "true_silence":
                lo, hi = (0.25, 0.70)   # keep some forgiveness: silence labeling can be noisy too
            else:
                lo, hi = (0.30, 0.80)

            cap = lerp(lo, hi, cleanScore)

            # Phase2 safety floor: never let negatives be near-free again.
            if isPhase2:
                cap = max(cap, 0.30)
            else:
                cap = max(cap, 0.12)

            return cap
                
        def computeSingerWeights(presenceFrac, leadFrac, backingFrac, adlibFrac, *,
                    minW, maxW,
                    W_LEAD=0.9, W_BACKING=0.6, W_ADLIB=0.3,
                    scale=1.0):
                """
                Returns importance_singers (shape: [num_members]) using ONE consistent formula.
                scale < 1.0 makes it "less confident" without changing the logic.
                """
                numMembers = len(presenceFrac)
                w = np.full(numMembers, minW, dtype=np.float32)

                presentMask = presenceFrac > 1e-3
                if not np.any(presentMask):
                    return w

                num_active = np.sum(presentMask)
                # Base formula for present singers only
                lf = leadFrac.astype(np.float32)
                bf = backingFrac.astype(np.float32)
                af = adlibFrac.astype(np.float32)

                adlibPenalty = W_ADLIB * af
                # cancel penalty for adlibs that are still "vocal identity" (your intention)
                adlibPenalty = np.where(af > 0.3, adlibPenalty * 0.2, adlibPenalty)

                base = 1.0 + W_LEAD * lf + W_BACKING * bf - adlibPenalty

                # Don’t crash pure-adlib regions too low
                base = np.where((af > 0.5) & (lf < 0.2) & (bf < 0.2), np.maximum(base, 0.7), base)

                if num_active > 1:
                    base = base / num_active
                    
                # Confidence scaling toward 1.0 (keeps 1.0 fixed)
                scaled = 1.0 + scale * (base - 1.0)

                w[presentMask] = np.clip(scaled[presentMask], minW, maxW)
                return w
            
        num_classes = self.num_members + 1
        lossWeightVec = np.ones(num_classes, dtype=np.float32)

        # ----------------------------
        # Vocal center
        # ----------------------------
        if centerHasVocal:
            presenceFrac = window_presence.mean(axis=0)[:self.num_members]
            leadFrac = window_lead.mean(axis=0)[:self.num_members]
            adlibFrac = window_adlib.mean(axis=0)[:self.num_members]
            backFrac = window_backingStyle.mean(axis=0)[:self.num_members]

            if (domFracAmongVocals >= TAU_CLEAN_DOM) and (windowOverlapFrac <= TAU_OVERLAP_MAX):
                singerW = computeSingerWeights(
                    presenceFrac=presenceFrac,
                    leadFrac=leadFrac,
                    backingFrac=backFrac,
                    adlibFrac=adlibFrac,
                    minW=self.min_weight,
                    maxW=self.max_weight,
                    W_LEAD=W_LEAD,
                    W_BACKING=W_BACKING,
                    W_ADLIB=W_ADLIB,
                    scale=1.0,
                )
                category = "clear_vocal"
                neg_cap = negCapFor(category, cleanScore, self.is_phase2)

            elif local_dom_frac >= TAU_LOCAL_DOM:
                local_presence_frac = local_presence.mean(axis=0)[:self.num_members]
                local_lead_frac = lead_arr[lf:rf].mean(axis=0)[:self.num_members]
                local_back_frac = backing_arr[lf:rf].mean(axis=0)[:self.num_members]
                local_adlib_frac = adlib_arr[lf:rf].mean(axis=0)[:self.num_members]

                singerW = computeSingerWeights(
                    presenceFrac=local_presence_frac,
                    leadFrac=local_lead_frac,
                    backingFrac=local_back_frac,
                    adlibFrac=local_adlib_frac,
                    minW=self.min_weight,
                    maxW=self.max_weight,
                    W_LEAD=W_LEAD,
                    W_BACKING=W_BACKING,
                    W_ADLIB=W_ADLIB,
                    scale=0.65,
                )
                category = "semi_clear_vocal"
                neg_cap = negCapFor(category, cleanScore, self.is_phase2)

            else:
                singerW = np.full(self.num_members, 0.3, dtype=np.float32)
                category = "ambiguous"
                neg_cap = negCapFor(category, cleanScore, self.is_phase2)

            lossWeightVec[:self.num_members] = singerW
            lossWeightVec[self.silence_idx] = 1.0

        # ----------------------------
        # Silent center
        # ----------------------------
        else:
            trueSilence = targetVec[self.silence_idx] > 0.5

            if trueSilence:
                lossWeightVec[:self.num_members] = 0.1
                lossWeightVec[self.silence_idx] = 1.0
                category = "true_silence"
                neg_cap = negCapFor(category, cleanScore, self.is_phase2)
            else:
                lossWeightVec[:self.num_members] = 0.15
                lossWeightVec[self.silence_idx] = 0.25
                category = "ambiguous_rest"
                neg_cap = negCapFor(category, cleanScore, self.is_phase2)

        # ----------------------------
        # Negative capping (critical)
        # ----------------------------
        center_present_mask = presenceCenter[:self.num_members].astype(bool)
        not_present = ~center_present_mask

        lossWeightVec[:self.num_members][not_present] = np.minimum(
            lossWeightVec[:self.num_members][not_present],
            neg_cap,
        )

        return lossWeightVec, category
    
    def _find_vocals_audio(self, song_name: str) -> str:
        """
        Try to find <song>_vocals.(wav|flac|mp3|m4a) under group_dir.
        """
        exts = [".wav"]
        for ext in exts:
            cand = os.path.join(self.audio_dir, f"{song_name}_vocals{ext}")
            if os.path.exists(cand):
                return cand
        return None
    
    def _find_audio_triplet(self, song_name: str):
        """
        Returns (mix_path, lead_path, back_path) or (mix_path, None, None) if stems missing.
        """
        # try wav first (fastest + consistent)
        mix = os.path.join(self.audio_dir, f"{song_name}_vocals.wav")
        lead = os.path.join(self.audio_dir, f"{song_name}_leading_vocals.wav")
        back = os.path.join(self.audio_dir, f"{song_name}_backing_vocals.wav")

        if not os.path.isfile(mix):
            # fallback to your existing _find_vocals_audio if you support mp3/flac/m4a
            mix = self._find_vocals_audio(song_name)
            if mix is None:
                return None, None, None

        lead = lead if os.path.isfile(lead) else None
        back = back if os.path.isfile(back) else None
        return mix, lead, back
    
    def _collect_synthetic_harmony_clips(self):
        """
        Returns a list of dicts:
        {"path": mp3_path, "lead_idx": int, "harm_idx": int}
        Deduped by mp3 path.
        """
        seen = set()
        clips = []

        for song, segs in self.manual_harmonies.items():
            for seg in segs:
                mp3 = seg.get("debug_mp3")
                if not mp3:
                    continue

                key = (mp3, seg["lead_idx"], seg["harm_idx"])
                if key in seen:
                    continue
                seen.add(key)

                clips.append({
                    "path": mp3,
                    "lead_idx": int(seg["lead_idx"]),
                    "harm_idx": int(seg["harm_idx"]),
                })

        print(f"[KpopFrameDataset] Synthetic harmony clips: {len(clips)}")
        return clips
    
    def _get_resampler(self, sr_src: int) -> Resample:
        key = (sr_src, self.sr_out)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(sr_src, self.sr_out)
        return self._resamplers[key]
    
    def _load_song_wave(self, path: str) -> torch.Tensor:
        """
        Load + mono + resample whole song once, cache it (LRU).
        Returns (1, T_out) at self.sr_out.
        """
        if path in self._wave_cache:
            wav = self._wave_cache.pop(path)  # refresh LRU
            self._wave_cache[path] = wav
            return wav

        wav, sr_src = torchaudio.load(path)  # (C, T_src)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr_src != self.sr_out:
            resampler = self._get_resampler(sr_src)
            wav = resampler(wav)

        wav = wav.contiguous()  # good practice

        # store in cache
        self._wave_cache[path] = wav

        # evict oldest if too large
        if len(self._wave_cache) > self._max_cached_songs:
            self._wave_cache.popitem(last=False)

        return wav
    
    def _load_manual_harmony_jsons(self, harmony_files: List[str]) -> Dict[str, List[dict]]:
        """
        Parse <Song>_test_harmonies.json files and build:
            manual_harmonies[song_name] = [
                {
                    "start": int,
                    "end": int,
                    "lead_idx": int,
                    "harm_idx": int,
                    "pair_name": sffortr,
                    "debug_mp3": Optional[str],
                }, ...
            ]
        The first member in 'members' is treated as lead, second as harmony.
        """
        manual: Dict[str, List[dict]] = {}
        
        # Only real members, no "silence"
        member_to_idx = {
            name: i for i, name in enumerate(self.classes)
            if i != self.silence_idx
        }
        
        for hpath in harmony_files:
            try:
                with open(hpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[KpopFrameDataset] Failed to read harmony file {hpath}: {e}")
                continue
            
            song = data.get("song")
            if not song:
                print(f"[KpopFrameDataset] Harmony file {hpath} has no 'song' key, skipping.")
                continue
                
            song_segments: List[dict] = manual.get(song, [])
            
            for pair in data.get("pairs", []):
                members = pair.get("members", [])
                if len(members) < 2:
                    continue
                
                lead_name, harm_name = members[0], members[1]
                if lead_name not in member_to_idx or harm_name not in member_to_idx:
                    print(
                        f"[KpopFrameDataset] Harmony file {hpath}: unknown members "
                        f"{members} (skipping this pair)"
                    )
                    continue
                
                lead_idx = member_to_idx[lead_name]
                harm_idx = member_to_idx[harm_name]
                pair_name = pair.get("name", "")
                
                # Try to find debug mp3
                debug_mp3 = None
                song_debug_dir = os.path.join(self.debug_harmony_root, song)
                if os.path.isdir(song_debug_dir) and pair_name:
                    mp3_pattern = os.path.join(song_debug_dir, f"{pair_name}*.mp3")
                    mp3_matches = glob.glob(mp3_pattern)
                    if mp3_matches:
                        debug_mp3 = mp3_matches[0]
                
                # Treat every segment (A + B) as "lead=harmony[0], harmony=harmony[1]"
                # to match your description: first member = lead, second = harmony.
                for key in ("segmentsA", "segmentsB"):
                    for seg in pair.get(key, []):
                        start_chunk = int(seg["startChunk"])
                        end_chunk = int(seg["endChunk"])
                        song_segments.append(
                            {
                                "start": start_chunk,
                                "end": end_chunk,
                                "lead_idx": lead_idx,
                                "harm_idx": harm_idx,
                                "pair_name": pair_name,
                                "debug_mp3": debug_mp3,
                            }
                        )

            if song_segments:
                manual[song] = song_segments

        print(f"[KpopFrameDataset] Loaded manual harmonies for {len(manual)} songs.")
        return manual                        
                
            
    # ---------- Dataset API ----------
    def __len__(self) -> int:
        # base windows + precomputed augmented windows
        return len(self.base_samples)
    
    def get_item_for_center(self, song_name: str, kind: str, center_frame: int):
        song = self.song_cache[song_name]
        presence = song["presence"]
        lead_arr = song["lead"]
        adlib_arr = song["adlib"]
        backing_arr = song["backing"]
        adlib_primary_arr = song["adlib_primary"]

        num_chunks = presence.shape[0]
        frame_ms = float(self.chunk_duration_ms)

        frames_per_window = int(round(self.context_seconds * 1000.0 / frame_ms))
        half_win = frames_per_window // 2

        start_frame = center_frame - half_win
        end_frame = start_frame + frames_per_window
        if start_frame < 0 or end_frame > num_chunks:
            raise IndexError("center_frame too close to edges for this contextSeconds")

        window_presence = presence[start_frame:end_frame]
        window_lead = lead_arr[start_frame:end_frame]
        window_adlib = adlib_arr[start_frame:end_frame]
        window_backing = backing_arr[start_frame:end_frame]

        presenceCenter = presence[center_frame].astype(np.int32)
        centerHasVocal = presenceCenter.sum() > 0

        anyActivePerFrame = (window_presence.sum(axis=1) > 0).astype(np.float32)
        windowVocalFrac = float(anyActivePerFrame.mean())
        windowOverlapFrac = float((window_presence.sum(axis=1) > 1).astype(np.float32).mean())

        framesActive = window_presence.mean(axis=0)
        domIdx = int(framesActive.argmax())
        domFracAmongVocals = float(framesActive[domIdx] / max(windowVocalFrac, 1e-3))

        # targets
        targetVec = self._build_targets_for_center(
            centerHasVocal=centerHasVocal,
            presenceCenter=presenceCenter,
            window_presence=window_presence,
            windowVocalFrac=windowVocalFrac,
        )

        # local band
        center_in_window = half_win
        lf, rf, local_presence, local_dom_frac = self._compute_local_band(
            window_presence=window_presence,
            center_in_window=center_in_window,
            local_band_frames=self.local_band_frames,
        )

        # weights
        lossWeightVec, category = self._build_weights_for_center(
            centerHasVocal=centerHasVocal,
            targetVec=targetVec,
            presenceCenter=presenceCenter,
            window_presence=window_presence,
            window_lead=window_lead,
            window_adlib=window_adlib,
            window_backingStyle=window_backing,
            domFracAmongVocals=domFracAmongVocals,
            windowOverlapFrac=windowOverlapFrac,
            local_presence=local_presence,
            local_dom_frac=local_dom_frac,
            lf=lf,
            rf=rf,
            lead_arr=window_lead,
            backing_arr=window_backing,
            adlib_arr=window_adlib,
        )

        # audio slice (same as __getitem__)
        paths = self._song_audio[song_name]
        audio_path = paths.get(kind, paths["mix"])
        start_time_sec = (start_frame * frame_ms) / 1000.0
        start_sample_out = int(round(start_time_sec * self.sr_out))
        wav = self._load_song_wave(audio_path)
        seg = self._slice_audio(wav, start_sample_out, self.chunk_len)

        labels_main  = torch.from_numpy(targetVec).to(torch.float32)
        weights_main = torch.from_numpy(lossWeightVec).to(torch.float32)

        # Phase-2: keep harmony/adlib targets consistent (use your existing logic)
        # For now, simplest = compute them the same way you do in __init__/__getitem__.
        # If you want Phase-2 to only train main, we can zero these out safely.
        labels_harm  = torch.zeros(self.num_members, dtype=torch.float32)
        weights_harm = torch.zeros(self.num_members, dtype=torch.float32)
        labels_ad    = torch.zeros(self.num_members, dtype=torch.float32)
        weights_ad   = torch.zeros(self.num_members, dtype=torch.float32)

        stem_weight_t = torch.tensor(1.0, dtype=torch.float32)

        return (seg,
                labels_main, weights_main,
                labels_harm, weights_harm,
                labels_ad,   weights_ad,
                stem_weight_t)
    
    def _slice_audio(self, wav: torch.Tensor, start_sample: int, length: int) -> torch.Tensor:
        """
        Slice a fixed-length audio segment from a waveform, padding if needed.

        Args:
            wav: Tensor shaped (..., num_samples)
            start_sample: starting sample index
            length: number of samples to extract

        Returns:
            Tensor shaped (..., length)
        """
        end = start_sample + length

        if start_sample < 0:
            start_sample = 0

        if end > wav.shape[-1]:
            pad = end - wav.shape[-1]
            seg = torch.nn.functional.pad(
                wav[..., start_sample:], (0, pad)
            )
        else:
            seg = wav[..., start_sample:end]

        return seg
    
    def __getitem__(self, idx: int):
        (
            song_name,
            stem_kind,
            center_frame,
            start_frame,
            start_sample_out,
            label_vec,
            importance_vec,
            _harmony_vec,
            _harmony_wts,
            _adlib_vec,
            _adlib_wts,
            stem_weight,
        ) = self.base_samples[idx]

        # Resolve the actual file path for this item
        paths = self._song_audio[song_name]
        audio_path = paths.get(stem_kind, paths["mix"]) or paths["mix"]

        # Load cached waveform (1, T_out)
        wav = self._load_song_wave(audio_path)

        # Slice + pad to fixed length (1, chunk_len)
        seg = self._slice_audio(wav, start_sample_out, self.chunk_len)

        # New, simplified outputs:
        labels_main = torch.from_numpy(label_vec).to(torch.float32)          # (C_main,)
        weights_main = torch.from_numpy(importance_vec).to(torch.float32)    # (C_main,)
        stem_weight = torch.tensor(stem_weight, dtype=torch.float32)         # scalar

        # Return only what PresenceHead training needs
        return (seg, labels_main, weights_main, stem_weight)

class FocusCentersDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, centers):
        self.base = base_ds
        self.centers = centers  # list of dicts: {"song":.., "kind":.., "center_frame":..}

    def __len__(self):
        return len(self.centers)

    def __getitem__(self, idx):
        c = self.centers[idx]
        return self.base.get_item_for_center(c["song"], c["kind"], c["center_frame"])
    
class MiningViewDataset(torch.utils.data.Dataset):
    """
    Wraps an existing KpopVocalDataset and returns:
      - the original training tuple (seg, labels..., stem_weight)
      - plus metadata needed for hard mining
    This avoids changing the training pipeline.
    """
    def __init__(self, base_ds: KpopVocalDataset):
        self.base = base_ds

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # original training tuple
        item = self.base[idx]

        # metadata comes from base_samples (NOT from item)
        (song_name,
         stem_kind,
         center_frame,
         start_frame,
         start_sample_out,
         label_vec,
         importance_vec,
         harmony_vec,
         harmony_wts,
         adlib_vec,
         adlib_wts,
         stem_weight) = self.base.base_samples[idx]

        return item, song_name, stem_kind, int(center_frame)


def binarize_logits(logits: torch.Tensor, thr: float = 0.5) -> torch.Tensor:
    """(B, C) logits -> (B, C) {0,1} via sigmoid threshold."""
    probs = logits.detach().sigmoid().to("cpu")
    return (probs >= thr).to(torch.float32)

def multilabel_micro_f1(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-8):
    """
    Micro F1 across all labels and samples.
    logits: (B, C), targets: (B, C) in {0,1}
    """
    preds = binarize_logits(logits, thr) # (B, C)
    t_cpu = targets.detach().to("cpu")
    tp = (preds * t_cpu).sum().item()
    fp = (preds * (1 - t_cpu)).sum().item()
    fn = ((1 - preds) * t_cpu).sum().item()
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return f1, precision, recall

def extract_center_context(wavs: torch.Tensor, ctx_frac: float = 0.25) -> torch.Tensor:
    """
    wavs: (B, 1, T) or (B, T)
    Returns a center subsegment of length ~ctx_frac * T, shape (B, 1, T_ctx)
    """
    if wavs.ndim == 2:
        wavs = wavs.unsqueeze(1) # Forces (B, 1, T)
    B, C, T = wavs.shape
    ctx_len = max(1, int(T * ctx_frac))
    
    mid = T // 2
    half = ctx_len // 2
    start = max(0, mid - half)
    end = start + ctx_len
    if end > T:
        end = T
        start = max(0, T - ctx_len)
    
    ctx = wavs[:, :, start:end] # (B, 1, ctx_len)
    return ctx

def subset_accuracy(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5):
    """
    Exact-set match accuracy: 1 only if all labels match for a sample.
    """
    preds = binarize_logits(logits, thr)
    t_cpu = targets.detach().to("cpu")
    eq = (preds == t_cpu).all(dim=1)  # (B,)
    return eq.float().mean().item()

def update_sanity_stats(stats: dict, logits: torch.Tensor, targets: torch.Tensor, thr: float, sample_cap: int = 200_000):
    """
    Streaming sanity stats:
      - avg_pred_pos vs avg_true_pos (over batch)
      - logit magnitude stats: mean_abs / p95_abs (approx) / max_abs
    """
    with torch.no_grad():
        # --- pos counts ---
        pred = (torch.sigmoid(logits) > thr).float()
        stats["pred_pos_sum"] += pred.sum(dim=1).sum().item()
        stats["true_pos_sum"] += targets.sum(dim=1).sum().item()
        stats["n_samples"] += logits.size(0)

        # --- logit magnitudes ---
        abs_logits = logits.detach().abs().flatten()
        stats["abs_sum"] += abs_logits.sum().item()
        stats["abs_count"] += abs_logits.numel()
        stats["abs_max"] = max(stats["abs_max"], abs_logits.max().item())

        # approximate p95 using a capped sample buffer
        if stats["abs_sample"] is not None:
            # take a small slice each batch to avoid big overhead
            take = min(2048, abs_logits.numel())
            if take > 0:
                samp = (
                    abs_logits[
                        torch.randint(0, abs_logits.numel(), (take,), device=abs_logits.device)
                    ]
                    .to(dtype=torch.float32, device="cpu")
                )
                stats["abs_sample"].append(samp)
                # cap memory
                if len(stats["abs_sample"]) * 2048 > sample_cap:
                    stats["abs_sample"] = stats["abs_sample"][-(sample_cap // 2048):]

def finalize_sanity_stats(stats: dict):
    avg_pred_pos = stats["pred_pos_sum"] / max(1, stats["n_samples"])
    avg_true_pos = stats["true_pos_sum"] / max(1, stats["n_samples"])

    mean_abs = stats["abs_sum"] / max(1, stats["abs_count"])
    max_abs = stats["abs_max"]

    if stats["abs_sample"] and len(stats["abs_sample"]) > 0:
        sample = torch.cat(stats["abs_sample"], dim=0)
        p95_abs = torch.quantile(sample, 0.95).item()
    else:
        p95_abs = float("nan")

    return {
        "avg_pred_pos": avg_pred_pos,
        "avg_true_pos": avg_true_pos,
        "mean_abs": mean_abs,
        "p95_abs": p95_abs,
        "max_abs": max_abs,
    }           
    
# ---------------------------
# Training / evaluation
# ---------------------------
def train_epoch(encoder, head, loader, device, optimizer, 
                thr=0.5, use_amp=True,
                ctx_frac: float = 0.25, is_phase2: bool = False
                ):
    """
    encoder: ECAPA model (SpeechBrain)
    head:   multi-task head taking fused embedding -> dict of logits
    loader: yields (wavs, y_main, w_main, y_harm, w_harm, y_ad, w_ad)
    """
    encoder.eval() # Extract embeddings under no_grad by default
    head.train()
    sanity = {
        "pred_pos_sum": 0.0, "true_pos_sum": 0.0, "n_samples": 0,
        "abs_sum": 0.0, "abs_count": 0, "abs_max": 0.0,
        "abs_sample": [],
    }
    
    scaler = torch.amp.GradScaler(device=device, enabled=(use_amp and device.type == "cuda"))
    
    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    for batch in tqdm(loader, desc="Train", leave=False):
        wavs, y_main, w_main, stem_weight = batch
        
        wavs = wavs.to(device, non_blocking=True)  # (B,1,T)
        y_main = y_main.to(device, non_blocking=True)
        w_main = w_main.to(device, non_blocking=True)
        stem_weight = stem_weight.to(device, non_blocking=True)
        
        wavs_encoder = wavs.squeeze(1) if (wavs.ndim == 3 and wavs.size(1) == 1) else wavs
        
        amp_ctx = torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda"))
        
        with torch.no_grad():
            emb_fused = encoder.encode_batch(wavs_encoder, ctx_frac=ctx_frac)  # (B,Df)
        
        optimizer.zero_grad(set_to_none=True)
        
        with amp_ctx:
            logits_main = head(emb_fused)  # (B, C)
            update_sanity_stats(sanity, logits_main, y_main, thr=thr)

            loss_w = bce(logits_main, y_main) * w_main
            loss_per = loss_w.sum(dim=1) / w_main.sum(dim=1).clamp(min=1e-6)
            loss = (loss_per * stem_weight).mean() + 1e-4 * logits_main.pow(2).mean()
            
        # Total loss with task weights
        loss_w = bce(logits_main, y_main) * w_main
        
        # Mean over classes
        loss_main_per_sample = loss_w.sum(dim=1) / (w_main.size(1) + 1e-6)
        # Apply stemWEight
        loss_main_per_sample = loss_main_per_sample * stem_weight

        loss = loss_main_per_sample.mean() + 1e-4 * logits_main.pow(2).mean()
 
        if is_phase2:
            probs = torch.sigmoid(logits_main)
            # probs shape: [B, num_members]
            kPred = probs.sum(dim=1)          # expected positives per window
            kMax = 3.0

            # only penalize if we exceed target by a margin (hinge)
            margin = 0.4                      # allow some slack
            excess = torch.relu(kPred - (kMax + margin))

            # weight it: stronger on clean windows, weaker on ambiguous ones
            # cleanScore is per-window in your dataset code; if you don't have per-sample,
            # just use a scalar based on batch average of your clean/ambiguous categories.
            lambdaCard = 0.02                 # start small; this is usually enough once neg_cap is fixed
            cardLoss = (excess ** 2).mean()

            loss = loss + (lambdaCard * cardLoss)
        
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
    
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
            
        # --- metrics --- 
        bsz = wavs.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        f1, prec, rec = multilabel_micro_f1(logits_main.detach(), y_main, thr=thr)
        total_f1 += f1 * bsz
        total_prec += prec * bsz
        total_rec += rec * bsz
        total_subset_acc += subset_accuracy(logits_main.detach(), y_main, thr=thr) * bsz
    
    avg_loss = total_loss / max(1, total_count)
    avg_f1   = total_f1 / max(1, total_count)
    avg_p    = total_prec / max(1, total_count)
    avg_r    = total_rec / max(1, total_count)
    avg_subset = total_subset_acc / max(1, total_count)
    sanity_out = finalize_sanity_stats(sanity)
    return avg_loss, {"micro_f1": avg_f1, "precision": avg_p, "recall": avg_r, "subset_acc": avg_subset, "sanity": sanity_out}

@torch.no_grad()
def eval_epoch(encoder, head, loader, device, thr=0.5, use_amp=True, ctx_frac: float = 0.25):
    """
    encoder: MuQEncoderWrapper (frozen), encode_batch(wavs, ctx_frac) -> (emb_main, emb_ctx)
             where emb_main/emb_ctx are either:
               - OLD style: (B,1,D)  (then we squeeze)
               - NEW style: (B,D)    (then we don't)
    head: PresenceHead, forward(emb_fused) -> logits_main (B, num_members+1)
    loader yields: (wavs, y_main, w_main, stem_weight)
    """
    encoder.eval()
    head.eval()

    sanity = {
        "pred_pos_sum": 0.0, "true_pos_sum": 0.0, "n_samples": 0,
        "abs_sum": 0.0, "abs_count": 0, "abs_max": 0.0,
        "abs_sample": [],
    }

    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    last_logits = None
    last_y = None

    # helper: squeeze (B,1,D)->(B,D) but leave (B,D) unchanged
    def _squeeze_b1d(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3 and x.size(1) == 1:
            return x.squeeze(1)
        return x

    # helper: ensure wavs fed to encoder are (B,T)
    def _prepare_wavs(wavs: torch.Tensor) -> torch.Tensor:
        if wavs.ndim == 3 and wavs.size(1) == 1:
            return wavs.squeeze(1)
        if wavs.ndim == 2:
            return wavs
        raise ValueError(f"Unexpected wavs shape: {tuple(wavs.shape)}")

    for batch in tqdm(loader, desc="Eval", leave=False):
        wavs, y_main, w_main, stem_weight = batch

        wavs = wavs.to(device, non_blocking=True)
        y_main = y_main.to(device, non_blocking=True)
        w_main = w_main.to(device, non_blocking=True)
        stem_weight = stem_weight.to(device, non_blocking=True)

        wavs_encoder = _prepare_wavs(wavs)

        # --- encode ---
        enc_out = encoder.encode_batch(wavs_encoder, ctx_frac=ctx_frac)

        # NEW path: a single embedding tensor
        if isinstance(enc_out, torch.Tensor):
            emb_fused = _squeeze_b1d(enc_out)

        else:
            raise TypeError(
                f"encode_batch returned unexpected type/shape: {type(enc_out)}"
            )

        # --- forward + loss ---
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type == "cuda")):
            logits = head(emb_fused)  # (B,C)

            update_sanity_stats(sanity, logits, y_main, thr=thr)

            # weighted BCE per class
            loss_w = bce(logits, y_main) * w_main  # (B,C)

            # normalize by weight mass per sample
            loss_per_sample = loss_w.sum(dim=1) / w_main.sum(dim=1).clamp(min=1e-6)  # (B,)

            # apply stem weight
            loss = (loss_per_sample * stem_weight).mean()

        bsz = wavs.size(0)
        total_loss += float(loss.item()) * bsz
        total_count += bsz

        f1, prec, rec = multilabel_micro_f1(logits, y_main, thr=thr)
        total_f1 += f1 * bsz
        total_prec += prec * bsz
        total_rec += rec * bsz
        total_subset_acc += subset_accuracy(logits, y_main, thr=thr) * bsz

        last_logits = logits
        last_y = y_main

    # Optional debug from last batch
    if last_logits is not None and last_y is not None:
        pred_pos = (torch.sigmoid(last_logits) > thr).float().sum(dim=1).mean().item()
        true_pos = last_y.sum(dim=1).mean().item()
        print(f"[DEBUG] avg_pred_pos={pred_pos:.2f} avg_true_pos={true_pos:.2f}")

    avg_loss = total_loss / max(1, total_count)
    avg_f1 = total_f1 / max(1, total_count)
    avg_p = total_prec / max(1, total_count)
    avg_r = total_rec / max(1, total_count)
    avg_subset = total_subset_acc / max(1, total_count)

    sanity_out = finalize_sanity_stats(sanity)
    return avg_loss, {
        "micro_f1": avg_f1,
        "precision": avg_p,
        "recall": avg_r,
        "subset_acc": avg_subset,
        "sanity": sanity_out,
    }

@torch.no_grad()
def hard_miner(encoder, head, loader, device, thr=0.7, pain_threshold=2.0, max_hard=50000):
    """
    Runs the Phase-1 model over a loader and returns hard centers:
      [{"song":..., "kind":..., "center_frame":..., "pain":...}, ...]

    loader must yield:
      ( (seg, y_main, w_main, y_harm, w_harm, y_ad, w_ad, stem_weight),
        song_name, stem_kind, center_frame )
    """
    encoder.eval()
    head.eval()
    
    hard = []
    
    for (train_tuple, song, kind, center_frame) in tqdm(loader, desc="Mining hard windows", leave=False):
        (wavs,
         y_main, w_main,
         y_harm, w_harm,
         y_ad,   w_ad,
         stem_weight) = train_tuple
        
        wavs = wavs.to(device, non_blocking=True)
        y_main = y_main.to(device, non_blocking=True).float()
        
        # (B,1,T)->(B,T) for encoder
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs_encoder = wavs.squeeze(1)
        elif wavs.ndim == 2:
            wavs_encoder = wavs
        else:
            raise ValueError(f"Unexpected wavs shape: {wavs.shape}")
        
        # Encode (same as train/eval)
        emb_main_b1, emb_ctx_b1 = encoder.encode_batch(wavs_encoder, ctx_frac=0.25)
        emb_main = emb_main_b1.squeeze(1)
        emb_ctx  = emb_ctx_b1.squeeze(1)
        emb_fused = torch.cat([emb_main, emb_ctx], dim=1)
        
        out = head(emb_fused, emb_ctx)
        logits = out["main"]                      # (B, C)
        probs = torch.sigmoid(logits)

        predicted_present = probs > thr
        true_present = (y_main == 1)
        
        false_positive_count = (predicted_present & ~true_present).sum(dim=1).float()
        false_negative_count = ((~predicted_present) & true_present).sum(dim=1).float()
        pred_count = predicted_present.sum(dim=1).float()
        true_count = y_main.sum(dim=1).float()
        
        # pos_conf = mean prob on true positives (0 if none)
        pos_conf = torch.zeros_like(true_count)
        has_pos = true_count > 0
        if has_pos.any():
            pos_conf[has_pos] = (probs[has_pos] * y_main[has_pos]).sum(dim=1) / true_count[has_pos].clamp(min=1.0)

        # Pain score tuned for your failure mode: FP spam matters most
        pain = 2.0 * false_positive_count + 1.5 * false_negative_count + 0.5 * (pred_count - true_count).clamp(min=0) + 0.5 * (1.0 - pos_conf)
        
        is_hard = (pain >= pain_threshold) | (false_positive_count >= 2) | (false_negative_count >= 1) | (pos_conf < 0.55)
        
        idxs = torch.where(is_hard)[0].tolist()
        for i in idxs:
            hard.append({
                "song": song[i] if isinstance(song, (list, tuple)) else song,
                "kind": kind[i] if isinstance(kind, (list, tuple)) else kind,
                "center_frame": int(center_frame[i] if isinstance(center_frame, (list, tuple, torch.Tensor)) else center_frame),
                "pain": float(pain[i].item()),
            })
            
            if len(hard) >= max_hard:
                break
        
    hard.sort(key=lambda d: d["pain"], reverse=True)
    return hard

def check_run_stage1(
    *, model, ckpt_path, skip_stage1, device
):
    """
    Either runs Stage 1 training or loads a pretrained Stage 1 head
    depending on --skip-stage-1.
    """
    if skip_stage1:
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"--skip-stage-1 was set but checkpoint not found: {ckpt_path}"
            )

        print(f"[Stage1] Skipping training. Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        return False
    
    return True

# Main
def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high') # Small speed bump on Ampere
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    group_dir = os.path.join(args.root, "training_data", args.group)
    cache_dir = buildTrainingCache(group_dir, srOut=args.sr_out)
    
    # Dataset & split
    ds_phase1 = KpopVocalDataset(group_dir, sr_out=args.sr_out, context_seconds=args.chunk_sec, group_name=args.group, audio_dir=cache_dir)
    
    print("Classes:", ds_phase1.class_map.idx_to_name)
    print("len(full_ds) =", len(ds_phase1))
    
    n_val = max(1, int(len(ds_phase1) * args.val_split))
    n_train = len(ds_phase1) - n_val
    print("train/val =", n_train, n_val)
    train_ds, val_ds = random_split(ds_phase1, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # Load MuQ ECAPA encoder (pretraiened)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq.to(device).eval()
    
    encoder = MuQEncoderWrapper(
        muq_model=muq,
        pooling="topk",
        debug=False
    ).to(device)
    fused_encoder = FusedEncoder(encoder).to(device)
    
    dummy = torch.zeros(1, int(args.chunk_sec * args.sr_out), device=device)

    # Ensure (B, T)
    if dummy.ndim == 3 and dummy.size(1) == 1:
        dummy = dummy.squeeze(1)

    with torch.no_grad():
        fused = fused_encoder.encode_batch(dummy, ctx_frac=0.2)  # (1, D_fused)
        fused_dim = fused.shape[1]  
        
    num_members = ds_phase1.num_members
        
    head = PresenceHead(
        emb_dim_fused=fused_dim,
        num_members=num_members,
        hidden=256,
        dropout=0.2
    ).to(device)
    
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.3,   # multiply LR by 0.3 when plateau
        patience=2,   # wait 1 epoch with no improvement
    )
    
    best_acc = 0.0
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, f"{args.group}_muq_head.pt")
    
    eval_thr = getattr(args, "eval_thr", 0.7)
    # Checks if you can skip stage 1
    run_stage1 = check_run_stage1(model=head, ckpt_path=ckpt_path, skip_stage1=args.skip_stage1, device=device)
    
    if run_stage1:
        # Train phase 1
        for epoch in range(1, args.epochs + 1):
            print(f"\nEpoch {epoch}/{args.epochs}")
            tr_loss, tr_metrics = train_epoch(fused_encoder, head, train_loader, device, optimizer, thr=eval_thr)
            va_loss, va_metrics = eval_epoch(fused_encoder, head, val_loader, device, thr=eval_thr)
            tr_s = tr_metrics["sanity"]
            va_s = va_metrics["sanity"]

            print(f"Sanity Train: avg_pred_pos={tr_s['avg_pred_pos']:.2f} vs avg_true_pos={tr_s['avg_true_pos']:.2f} | "
                f"logit| mean={tr_s['mean_abs']:.2f} p95={tr_s['p95_abs']:.2f} max={tr_s['max_abs']:.2f}")
            print(f"Sanity Val:   avg_pred_pos={va_s['avg_pred_pos']:.2f} vs avg_true_pos={va_s['avg_true_pos']:.2f} | "
                f"logit| mean={va_s['mean_abs']:.2f} p95={va_s['p95_abs']:.2f} max={va_s['max_abs']:.2f}")
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(va_metrics["micro_f1"])
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != old_lr:
                print(f"[LR Scheduler] Reducing LR: {old_lr} → {new_lr}")
        
            print(
                f"Train - loss: {tr_loss:.4f} | micro-F1: {tr_metrics['micro_f1']:.4f} "
                f"(P {tr_metrics['precision']:.3f}, R {tr_metrics['recall']:.3f}) | subset-acc: {tr_metrics['subset_acc']:.4f}"
            )
            print(
                f"Val   - loss: {va_loss:.4f} | micro-F1: {va_metrics['micro_f1']:.4f} "
                f"(P {va_metrics['precision']:.3f}, R {va_metrics['recall']:.3f}) | subset-acc: {va_metrics['subset_acc']:.4f}"
            )

            if va_metrics["micro_f1"] > best_acc:
                best_acc = va_metrics["micro_f1"]
                torch.save({
                    "state_dict": head.state_dict(),
                    "classes": ds_phase1.class_map.idx_to_name,
                    "emb_dim_fused": fused_dim,
                    "sr": args.sr_out,
                    "chunk_sec": args.chunk_sec,
                    "group": args.group,
                    "eval_thr": eval_thr,
                }, ckpt_path)
                print(f"✅ Saved best head to: {ckpt_path} (acc={best_acc:.4f})")
            
        print("\nDone. Best val acc:", best_acc)
    else:
        print("⚠️  WARNING: Stage 1 skipped — using pretrained head")
    
    mine_ds = MiningViewDataset(ds_phase1)
    mine_loader = DataLoader(mine_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # 2) mine hard centers using the trained phase 1 model
    hard_centers = hard_miner(
        encoder=fused_encoder,
        head=head,
        loader=mine_loader,   # based on the 2s dataset
        thr=eval_thr,
        device=device,
    )
    print("Hard centers mined:", len(hard_centers))
    
    # Build once
    ds_phase2_base = KpopVocalDataset(group_dir, sr_out=args.sr_out, context_seconds=0.4, group_name=args.group, audio_dir=cache_dir, is_phase2=True)
    
    # Build first focused loader from inital mind centers from Phase 1
    focus_ds = FocusCentersDataset(ds_phase2_base, hard_centers)
    focus_loader = DataLoader(
        focus_ds,
        batch_size=args.batch_size,
        shuffle=True,                 # IMPORTANT: shuffle hard examples
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # Keep same to compare aacross epochs
    val2_ds = KpopVocalDataset(group_dir, sr_out=args.sr_out, context_seconds=args.short_chunk_sec,
                           group_name=args.group, audio_dir=cache_dir, is_phase2=True)

    n_val2 = max(1, int(len(val2_ds) * args.val_split))
    n_train2 = len(val2_ds) - n_val2
    _, val2_subset = random_split(val2_ds, [n_train2, n_val2])

    val2_loader = DataLoader(
        val2_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False
    )
    
    head2 = PresenceHead(emb_dim_fused=fused_dim, num_members=num_members).to(device)
    head2.load_state_dict(head.state_dict())  # start from phase1 solution
    opt2 = torch.optim.AdamW(head2.parameters(), lr=args.lr * 0.3, weight_decay=1e-4)

    phase2_epochs = max(1, args.epochs // 2)  # e.g., 5 if phase1 was 10
    best2 = -1.0
    
    prev_p95_abs = None
    rising_p95_streak = 0
    
    REMINE_EVERY = 1000
    TOP_HARD = 15000 # Cap size to keep training tight
    PAIN_KEEP = 2.0 # Min pain threshold
    THR2 = 0.5 # Phase-2 threshold for mining and eval
    
    for epoch in range(1, phase2_epochs + 1):
        print(f"\n[Phase2] Epoch {epoch}/{phase2_epochs}")
        tr2_loss, tr2_metrics = train_epoch(fused_encoder, head2, focus_loader, device, opt2, thr=THR2, is_phase2=True)
        va2_loss, va2_metrics = eval_epoch(fused_encoder, head2, val2_loader, device, thr=THR2)

        print(f"[Phase2] Train loss {tr2_loss:.4f} microF1 {tr2_metrics['micro_f1']:.4f}")
        print(f"[Phase2] Val   loss {va2_loss:.4f} microF1 {va2_metrics['micro_f1']:.4f}")

        # === SCREAMING GUARDRAILS ===
        # Use the sanity stats you already compute in train_epoch/eval_epoch (mean_abs, p95_abs, max_abs, avg_pred_pos, avg_true_pos)
        # Assuming train_epoch returns them in tr2_metrics["sanity"] (if not, I’ll show where to add it)
        if "sanity" in tr2_metrics:
            s = tr2_metrics["sanity"]
            mean_abs = float(s.get("mean_abs", 0.0))
            p95_abs  = float(s.get("p95_abs", 0.0))
            max_abs  = float(s.get("max_abs", 0.0))
            avg_pred = float(s.get("avg_pred_pos", 0.0))
            avg_true = float(s.get("avg_true_pos", 1e-6))

            # p95 trend check (relapse)
            if prev_p95_abs is not None and p95_abs > prev_p95_abs + 0.5:
                rising_p95_streak += 1
            else:
                rising_p95_streak = 0
            prev_p95_abs = p95_abs

            # "Insane" definitions: these catch your old 75+ p95 meltdown early
            if (p95_abs >= 20.0) or (max_abs >= 60.0) or (rising_p95_streak >= 2):
                print(
                    f"\n\n🚨🚨🚨 LOGIT MELTDOWN DETECTED 🚨🚨🚨\n"
                    f"p95_abs={p95_abs:.2f} mean_abs={mean_abs:.2f} max_abs={max_abs:.2f} "
                    f"(rising_streak={rising_p95_streak})\n"
                    f"This usually means the model is saturating and will start spamming positives.\n"
                )

            # Prediction-rate sanity: stop if it starts screaming "everyone is singing"
            ratio = avg_pred / max(avg_true, 1e-6)
            if ratio >= 2.4:
                print(
                    f"\n\n🚨🚨🚨 FP-SPAM RELAPSE 🚨🚨🚨\n"
                    f"avg_pred_pos={avg_pred:.2f} avg_true_pos={avg_true:.2f} ratio={ratio:.2f}\n"
                    f"The model is predicting far too many singers per window.\n"
                )

            # Also catch the opposite: model becomes too timid
            if ratio <= 0.55:
                print(
                    f"\n⚠️  Phase2 looks overly timid: avg_pred_pos={avg_pred:.2f} avg_true_pos={avg_true:.2f} ratio={ratio:.2f}\n"
                    f"Not necessarily fatal, but it can mean threshold is too high or negatives too strong.\n"
                )

        # Save best
        if va2_metrics["micro_f1"] > best2:
            best2 = va2_metrics["micro_f1"]
            torch.save({
                "state_dict": head2.state_dict(),
                "classes": ds_phase1.class_map.idx_to_name,
                "emb_dim": fused_dim,
                "sr": args.sr_out,
                "chunk_sec": 0.4,
                "group": args.group,
                "eval_thr": THR2,
                "note": "phase2_refinement"
            }, os.path.join(args.save_dir, f"{args.group}_muq_head_phase2.pt"))
            print(f"✅ Saved best Phase2 head (microF1={best2:.4f})")
        
        # === RE-MINE EVERY 3 EPOCHS ===
        if (epoch % REMINE_EVERY) == 0:
            print("\n[Phase2] Re-mining hard centers...")
            # Mine on the *0.4s base dataset* using the current head2
            mine_ds = MiningViewDataset(ds_phase2_base)  # wrapper provides (train_tuple, song, kind, center_frame)
            mine_loader = DataLoader(
                mine_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True, drop_last=False
            )

            new_hard = hard_miner(
                encoder=fused_encoder,
                head=head2,
                loader=mine_loader,
                device=device,
                thr=THR2,
                pain_threshold=PAIN_KEEP,
                max_hard=TOP_HARD * 3,  # allow extra before filtering/dedup
            )

            # Dedup and keep top N
            seen = set()
            dedup = []
            for c in new_hard:
                key = (c["song"], c["kind"], int(c["center_frame"]))
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(c)
                if len(dedup) >= TOP_HARD:
                    break

            hard_centers = dedup
            print(f"[Phase2] Hard centers refreshed: {len(hard_centers)}")

            # Rebuild focus loader with the refreshed set
            focus_ds = FocusCentersDataset(ds_phase2_base, hard_centers)
            focus_loader = DataLoader(
                focus_ds, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
    
if __name__ == "__main__":
    main()