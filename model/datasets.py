import os, glob, json
import numpy as np
from collections import OrderedDict
from typing import List, Tuple, Dict
from torch.utils.data import Dataset, Subset
from torchaudio.transforms import Resample
import torch
import torchaudio
from dataclasses import dataclass

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
                
        # Get song-level complexity stats for curriculum learning
        self.get_all_songs_sorted_by_complexity()
        
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
                              window_presence: np.ndarray, windowVocalFrac: float):
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
    
    def _compute_song_complexities(self):
        """
        Build per-song complexity stats from raw frame arrays.
        Stores:
        self.song_stats[song] = dict(...)
        self.song_complexity[song] = float in [0, 1] (roughly)
        """
        self.song_stats = {}
        self.song_complexity = {}

        for song, data in self.song_cache.items():
            presence = data["presence"][:, :self.num_members].astype(np.int32)  # (T, C)
            lead     = data["lead"][:, :self.num_members].astype(np.int32)
            adlib    = data["adlib"][:, :self.num_members].astype(np.int32)
            backing  = data["backing"][:, :self.num_members].astype(np.int32)

            T = presence.shape[0]
            if T <= 1:
                continue

            activeCount = presence.sum(axis=1)  # (T,)
            vocalMask   = activeCount > 0
            overlapMask = activeCount > 1

            overlapRate = float(overlapMask.mean())
            vocalRate   = float(vocalMask.mean())

            adlibMask   = (adlib.sum(axis=1) > 0)
            backingMask = (backing.sum(axis=1) > 0)
            leadMask    = (lead.sum(axis=1) > 0)

            adlibRate   = float(adlibMask.mean())
            backingRate = float(backingMask.mean())
            leadRate    = float(leadMask.mean())

            # switching: how often the *set* of active singers changes
            # (use XOR-like compare between consecutive frames)
            changes = (presence[1:] != presence[:-1]).any(axis=1)
            switchRate = float(changes.mean())

            # optional: within vocal frames, how often there is a dominant single singer
            # (queue songs have high dominance)
            domFrac = 0.0
            vocalFrames = presence[vocalMask]
            if vocalFrames.shape[0] > 0:
                perSinger = vocalFrames.mean(axis=0)  # (C,)
                denom = float(perSinger.sum()) + 1e-8
                domFrac = float(perSinger.max() / denom)

            # Complexity score: tune weights later
            # Higher overlap/adlib/backing/switch => harder
            # Higher dominance => easier (so subtract it)
            score = (
                0.40 * overlapRate +
                0.20 * backingRate +
                0.15 * adlibRate +
                0.20 * switchRate +
                0.05 * (1.0 - domFrac)
            )

            # clamp to [0,1]
            score = float(max(0.0, min(1.0, score)))

            self.song_stats[song] = {
                "T": T,
                "vocalRate": vocalRate,
                "overlapRate": overlapRate,
                "backingRate": backingRate,
                "adlibRate": adlibRate,
                "leadRate": leadRate,
                "switchRate": switchRate,
                "domFrac": domFrac,
                "complexity": score,
            }
            print(f"Song stats print for {song}: {self.song_stats[song]}")
            self.song_complexity[song] = score

    def get_song_complexity(self, song_name: str) -> float:
        if not hasattr(self, "song_complexity"):
            self._compute_song_complexities()
        return float(self.song_complexity.get(song_name, 0.0))   
    
    def get_all_songs_sorted_by_complexity(self):
        if not hasattr(self, "song_complexity"):
            self._compute_song_complexities()
        return sorted(self.song_complexity.items(), key=lambda kv: kv[1], reverse=True)
    
    def getNonSoloSongs(self, domFracSoloThreshold: float = 0.95):
        # assumes self.song_stats exists; otherwise compute it where you do complexity
        songs = []
        for song, st in self.song_stats.items():
            if st.get("domFrac", 0.0) < domFracSoloThreshold:
                songs.append(song)
        return songs

    def getSongsByDifficultyTiers(self, domFracSoloThreshold: float = 0.95):
        """
        Returns three lists: (easySongs, medSongs, hardSongs), each sorted by complexity ascending.
        Tiers are terciles of the non-solo songs by complexity.
        """
        songs = self.getNonSoloSongs(domFracSoloThreshold=domFracSoloThreshold)
        songs.sort(key=lambda s: self.song_stats[s]["complexity"])  # low -> high

        n = len(songs)
        if n < 3:
            # fallback: everything is 'easy'
            return songs, [], []

        # split into 3 as evenly as possible
        cut1 = n // 3
        cut2 = (2 * n) // 3
        easy = songs[:cut1]
        med  = songs[cut1:cut2]
        hard = songs[cut2:]
        return easy, med, hard

    def pickValidationSongsStratified(
        self,
        valEasy: int = 2,
        valMed: int = 2,
        valHard: int = 1,
        domFracSoloThreshold: float = 0.95,
    ):
        """
        Deterministic split:
        - Exclude solos
        - Create tercile tiers by complexity
        - Choose bottom-of-tier songs for validation (keeps hardest in training)
        Returns: valSongs (set), tierInfo (dict with lists)
        """
        easy, med, hard = self.getSongsByDifficultyTiers(domFracSoloThreshold=domFracSoloThreshold)

        def takeBottom(tierList, k):
            # tierList already sorted low->high complexity
            return tierList[:min(k, len(tierList))]

        valSongs = set()
        valSongs.update(takeBottom(easy, valEasy))
        valSongs.update(takeBottom(med, valMed))
        valSongs.update(takeBottom(hard, valHard))

        tierInfo = {
            "easy": easy,
            "med": med,
            "hard": hard,
            "valSongs": sorted(valSongs, key=lambda s: self.song_stats[s]["complexity"]),
        }
        return valSongs, tierInfo
    
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
