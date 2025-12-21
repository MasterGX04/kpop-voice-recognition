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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
import numpy as np
from torchaudio.transforms import Resample

from tqdm import tqdm
from muq import MuQ
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------
# CLI args
# ---------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Path that contains <group>/<member>/train/Isolated_Vocals")
    ap.add_argument("--group", type=str, required=True, help="Group folder name under root")
    ap.add_argument("--sr_in", type=int, default=22050, help="Expected input sample rate of your files")
    ap.add_argument("--sr_out", type=int, default=16000, help="ECAPA target sample rate")
    ap.add_argument("--chunk-sec", type=float, default=2.0, help="Chunk length in seconds for training")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-dir", type=str, default="./checkpoints")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval_thr", type=float, default=0.7)
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
                chunk_sec: float, group_name: str, min_song_sec: float = 4.0,
                window_hop_ratio: float = 0.5, presence_thresh=0.4, alpha_lead=1.0, 
                alpha_adlib = 0.5, min_weight = 0.2, max_weight = 2.0, k_train: int = 2, useMuq=False):
        super().__init__()
        self.group_dir = group_dir
        self.sr_out = sr_out # This should be 24000 HZ
        self.chunk_sec = chunk_sec
        self.chunk_len = int(round(chunk_sec * sr_out))
        self.min_song_sec = min_song_sec
        
        self.window_hop_ratio = window_hop_ratio
        self.k_train = k_train
        
        self.presence_thresh = presence_thresh
        self.alpha_lead = alpha_lead
        self.alpha_adlib = alpha_adlib
        self.min_weight = min_weight
        self.max_weight = max_weight

        # ----------------------------
        # 1) Discover all JSON label files
        # ----------------------------
        json_pattern = os.path.join(group_dir, "*_frame_labels.json")
        json_files = sorted(glob.glob(json_pattern))
        if not json_files:
            raise RuntimeError(f"No *_frame_labels.json files found under {group_dir}")
        
        # ----------------------------
        # 2) Load first JSON to define members/classes
        # ----------------------------
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
        
        # ----------------------------
        # 1b) Optional: manual harmony annotations
        #     ./saved_labels/<group>/<Song>_test_harmonies.json
        #     ./debug_harmonies/<group>/<Song>/<pair_name>.mp3
        # ---
        self.group_name = group_name
        
        # JSON with manual pairs
        harmony_label_dir = os.path.join(".", "saved_labels", group_name)
        harmony_pattern = os.path.join(harmony_label_dir, "*_test_harmonies.json")
        harmony_files = sorted(glob.glob(harmony_pattern))
        
        # Root where debug harmony mp3 clips live
        self.debug_harmony_root = os.path.join(".", "debug_harmonies", group_name)
        self.manual_harmonies = self._load_manual_harmony_jsons(harmony_files)
        
        self.synthetic_harmony_clips = self._collect_synthetic_harmony_clips()
        
        self.gang_idx = None
        for i, name in enumerate(self.classes):
            if name.lower() == "gang vocal":
                self.gang_idx = i
                break
        
        # Debug stats: how many chunks per category, and a few examples
        self.debug_category_counts = {
            "true_silence": 0,
            "clear_vocal": 0,
            "semi_clear_vocal": 0,
            "ambiguous": 0,
        }

        # store up to N examples per category
        max_debug_examples = 10
        self.debug_category_examples = {
            "true_silence": [],
            "clear_vocal": [],
            "semi_clear_vocal": [],
            "ambiguous": [],
        }
        self._max_debug_examples = max_debug_examples
        
        # ----------------------------
        # 3) Build index of (audio_path, start_sample_out, label_vec)
        # ----------------------------
        self.samples: List[Tuple[str, int,
                         np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray]] = []
        frame_ms = float(self.chunk_duration_ms)
        frames_per_window = int(round(self.chunk_sec * 1000.0 / frame_ms))  # e.g. 2.0s / 40ms = 50
        
        if frames_per_window <= 0:
            raise ValueError("frames_per_window computed as <=0, check chunk_sec and chunkDurationMs")
        
        hop_frames = max(1, int(round(frames_per_window * self.window_hop_ratio)))
        
        if self.k_train is not None:
            hop_frames = max(1, int(self.k_train))
        else:
            hop_frames = max(1, int(round(frames_per_window * self.window_hop_ratio)))

        print(f"[KpopFrameDataset] Frames/window={frames_per_window}, hop_frames={hop_frames}")
        print(f"[KpopFrameDataset] Members={members}")
        
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
        
            song_dur_sec = num_chunks * frame_ms / 1000.0
            if song_dur_sec < self.min_song_sec or num_chunks < frames_per_window:
                print(f"[KpopFrameDataset] Skipping {song_name}: too short ({song_dur_sec:.2f}s)")
                continue
            
            # Find the corresponding vocals audio file
            audio_path = self._find_vocals_audio(song_name)
            if audio_path is None:
                print(f"[KpopFrameDataset] No audio found for song {song_name}, skipping.")
                continue
            
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
                end_frame   = start_frame + frames_per_window
                
                # Sanity check
                if start_frame < 0 or end_frame > num_chunks:
                    continue # Skip weird edge cases
                
                # 2s context labels (for importance)
                window_presence = presence[start_frame:end_frame]   # [F, C]
                window_lead = lead_arr[start_frame:end_frame]
                window_adlib = adlib_arr[start_frame:end_frame]
                window_backingStyle = backing_arr[start_frame:end_frame]
                
                # Fractions over this window (0..1) - still useful for importance weights
                presence_frac = window_presence.mean(axis=0) # how often each siniger is present
                lead_frac = window_lead.mean(axis=0) # How often they are lead (role)
                
                adlib_frac = window_adlib.mean(axis=0) # How often they are adlib (style)
                backingStyle_frac = window_backingStyle.mean(axis=0) # How often they are backing (style)
                
                # Label for THIS example
                frame_label = presence[center_frame] # shape(num_members,), 0/1
                frames_active = window_presence.mean(axis=0) # fraction per singer
                
                any_active_per_frame = (window_presence.sum(axis=1) > 0).astype(np.float32)
                vocal_frac_window = any_active_per_frame.mean()

                overlap_frac_window = (window_presence.sum(axis=1) > 1).astype(np.float32).mean()

                frames_active = window_presence.mean(axis=0)
                dominant_idx = frames_active.argmax()
                dominant_frac = frames_active[dominant_idx] / max(vocal_frac_window, 1e-6)

                any_vocal_center = frame_label.sum() > 0
                
                # For convenience
                has_real_vocals = presence_frac.copy()
                if self.gang_idx is not None:
                    # exclude Gang Vocal from "real member" sum
                    has_real_vocals[self.gang_idx] = 0.0
                any_real = has_real_vocals.sum() > 1e-3
                
                # ------ LOCAL (0.5s) context ------
                local_radius = 6  # approx 500 ms
                lf = max(0, center_frame - local_radius)
                rf = min(num_chunks, center_frame + local_radius + 1)
                local_presence = presence[lf:rf]

                local_any_active = (local_presence.sum(axis=1) > 0).astype(np.float32)
                local_vocal_frac = local_any_active.mean()

                local_frames_active = local_presence.mean(axis=0)
                local_dom_idx = local_frames_active.argmax()
                local_dom_frac = local_frames_active[local_dom_idx] / max(local_vocal_frac, 1e-6)

                label_vec = np.zeros(len(self.classes), dtype=np.float32)
                importance_vec = np.ones(len(self.classes), dtype=np.float32)
                
                TAU_SILENCE_WINDOW = 0.20    # ≤20% of window is vocal → real silence
                TAU_DOMINANT = 0.55          # ≥55% of vocal frames are same singer → clean vocal
                TAU_OVERLAP = 0.30           # ≥30% frames with 2+ singers → ambiguous/gang
                
                # Define thresholds
                W_LEAD = 0.9 # How much lead boosts weight
                W_BACKING = 0.6 # backing/harmony boost
                W_ADLIB = 0.3 # How strongly ad-libs DOWN-weight
                W_GANG_SOLO = 0.6 # Gang vocal alone
                W_GANG_OVERLAP = 0.3 # Gang vocal when members also sing
                
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

                        # Confidence scaling toward 1.0 (keeps 1.0 fixed)
                        scaled = 1.0 + scale * (base - 1.0)

                        w[presentMask] = np.clip(scaled[presentMask], minW, maxW)
                        return w
                
                # CASE 1 → CLEAR VOCAL
                if any_vocal_center:
                    # Label is ALWAYS singing — never flip or skip
                    label_vec[:self.num_members] = frame_label.astype(np.float32)
                    label_vec[self.silence_idx] = 0.0
                    
                    # Gang Vocal should be window-based, not center frame
                    if self.gang_idx is not None:
                        GANG_TAU = 0.1
                        label_vec[self.gang_idx] = 1.0 if presence_frac[self.gang_idx] >= GANG_TAU else 0.0
                    
                    # Check if num_members implies +gang vocals or no
                    importance_singers = np.ones(self.num_members, dtype=np.float32)
                    
                    # GLOBAL clean vocal
                    if dominant_frac >= TAU_DOMINANT and overlap_frac_window <= TAU_OVERLAP:
                        importance_singers = computeSingerWeights(
                            presenceFrac=presence_frac[:self.num_members],
                            leadFrac=lead_frac[:self.num_members],
                            backingFrac=backingStyle_frac[:self.num_members],
                            adlibFrac=adlib_frac[:self.num_members],
                            minW=self.min_weight,
                            maxW=self.max_weight,
                            W_LEAD=W_LEAD, W_BACKING=W_BACKING, W_ADLIB=W_ADLIB,
                            scale=1.0,
                        )
                        category = "clear_vocal"
                    elif local_dom_frac >= 0.65:
                        local_presence_frac = local_presence.mean(axis=0)[:self.num_members]
                        local_lead_frac = lead_arr[lf:rf].mean(axis=0)[:self.num_members]
                        local_backing_frac = backing_arr[lf:rf].mean(axis=0)[:self.num_members]
                        local_adlib_frac = adlib_arr[lf:rf].mean(axis=0)[:self.num_members]

                        importance_singers = computeSingerWeights(
                            presenceFrac=local_presence_frac,
                            leadFrac=local_lead_frac,
                            backingFrac=local_backing_frac,
                            adlibFrac=local_adlib_frac,
                            minW=self.min_weight,
                            maxW=self.max_weight,
                            W_LEAD=W_LEAD, W_BACKING=W_BACKING, W_ADLIB=W_ADLIB,
                            scale=0.65,  # softer than clean
                        )
                        category = "semi_clear_vocal"
                    else:
                        # ambiguous vocal
                        importance_singers = np.full(self.num_members, 0.4, dtype=np.float32)
                        # if you still want to weaken everyone:
                        importance_singers[:] = 0.3
                        category = "ambiguous"
                        
                        category = "ambiguous"
                    
                    importance_vec[:self.num_members] = importance_singers
                    importance_vec[self.silence_idx] = 1.0

                    neg_floor = 0.6
                    center_present = frame_label[:self.num_members].astype(bool)
                    importance_vec[:self.num_members][~center_present] = np.maximum(
                        importance_vec[:self.num_members][~center_present], neg_floor
                    )
                    
                # center frame silent
                else:       
                    # IMPORTANT: don't heavily punish members/gang during silence-center windows,
                    # because the 2s context can contain vocals nearby.
                    importance_vec[:self.num_members] = 0.1  # weak negatives for singers (including gang)
                    if vocal_frac_window <= TAU_SILENCE_WINDOW and local_vocal_frac < 0.15:
                        # real silence
                        importance_vec[:self.num_members] = 0.1
                        label_vec[self.silence_idx] = 1.0
                        importance_vec[self.silence_idx] = 1.0
                        category = "true_silence"
                    else:
                        # ambiguous silence (breath, short pause)
                        importance_vec[:self.num_members] = 0.2 # Super weak negative near vocals
                        label_vec[self.silence_idx] = 0.0
                        importance_vec[self.silence_idx] = 0.0
                        category = "ambiguous"
                
                # ---------------------------
                # Debug accounting
                # ---------------------------
                if category not in self.debug_category_counts:
                    # should never happen, but safe-guard
                    self.debug_category_counts[category] = 0
                    self.debug_category_examples[category] = []

                self.debug_category_counts[category] += 1
                count = self.debug_category_counts[category]

                debug_info = {
                    "song": song_name,
                    "center_frame": int(center_frame),
                    "vocal_frac_window": float(vocal_frac_window),
                    "overlap_frac_window": float(overlap_frac_window),
                    "dominant_idx": int(dominant_idx),
                    "dominant_frac": float(dominant_frac),
                    "frame_label": frame_label.tolist(),
                }

                reservoir = self.debug_category_examples[category]
                max_n = self._max_debug_examples

                if len(reservoir) < max_n:
                    # fill up until we reach max_n
                    reservoir.append(debug_info)
                else:
                    # reservoir sampling: replace an existing one with decreasing probability
                    j = random.randint(0, count - 1)
                    if j < max_n:
                        reservoir[j] = debug_info
                
                # ---------------------------
                # Multi-task: harmony + ad-lib targets
                # ---------------------------
                # Per-singer arrays
                harmony_vec = np.zeros(self.num_members, dtype=np.float32)
                harmony_wts = np.zeros(self.num_members, dtype=np.float32)
                adlib_vec = np.zeros(self.num_members, dtype=np.float32)
                adlib_wts = np.zeros(self.num_members, dtype=np.float32)
                
                # --- Center-band adlib supervision (replaces pos_ad = adlib_frac >= TAU_ADLIB) ---
                band = 1
                cf0 = max(start_frame, center_frame - band)
                cf1 = min(end_frame, center_frame + band + 1)
                lead_votes = lead_arr[cf0:cf1].sum(axis=0)
                center_adlib = adlib_arr[cf0:cf1].max(axis=0) # (C,) 0/1 whether adlib appears
                center_lead = np.zeros(self.num_members, dtype=bool)
                if lead_votes.sum() > 0:
                    center_lead[np.argmax(lead_votes)] = True
                any_lead_center = center_lead.any()
                
                if not any_lead_center:
                    harmony_wts[:] = 0.0
                
                pos_ad = center_adlib.astype(bool)
                adlib_vec[:] = 0.0
                adlib_vec[pos_ad] = 1.0
                
                center_presence = presence[cf0:cf1].max(axis=0).astype(bool)
                center_backing = backing_arr[cf0:cf1].max(axis=0).astype(bool)
                center_overlap = (presence[cf0:cf1].sum(axis=1) > 1).any()
                
                center_primary_adlib = center_backing & center_adlib
                center_secondary_adlib = pos_ad & ~center_backing
                
                # only treat harmony if there's overlap right NOW (not somewhere in the 2s")
                if center_overlap and any_lead_center:
                    pos_harm = center_backing & ~center_lead & ~center_adlib
                    harmony_vec[pos_harm] = 1.0
                    
                    # Base negative: present but not harmony (weak-ish)
                    W_NEG_BASE = 0.6

                    # Strong negative: lead should almost never be harmony
                    W_NEG_LEAD = 1.2
                    
                    # Positive weight: reward true harmony
                    W_POS_HARM = 1.6
    
                    # Weights: only train on singers who are present rn
                    harmony_wts[:] = 0.0
                    harmony_wts[center_presence] = W_NEG_BASE # weak negatives
                    
                    # Make false "lead is harmony" expensive
                    harmony_wts[center_lead & center_presence & ~pos_harm] = W_NEG_LEAD
                    harmony_wts[pos_harm] = W_POS_HARM
                else:
                    harmony_wts[:] = 0.0 # ignore harmony when there is no overlap in the center
                
                valid_ad_mask = center_presence # Only train adlib for singers who are present
                
                # weights: ignore non-present singers for the adlib head
                adlib_wts[:] = 0.0
                
                # This is the cost of a false positive adlib on a present singer.
                W_NEG_SOLO = 0.5 # present, not adlib, not clearly lead
                W_NEG_LEAD = 1.1 # present+lead but not adlib (very important: lead ≠ adlib)
                
                # Positive weight: reward catching real adlibs.
                W_POS_ADLIB = 1.4
                W_POS_ADLIB_PRIMARY = 2.2
                
                # Apply negatives only to present singers by default
                adlib_wts[valid_ad_mask] = W_NEG_SOLO
                
                # Stronger negatives depending on ROLE:
                any_primary_adlib = center_primary_adlib.any()
                if not any_primary_adlib:
                    adlib_wts[center_lead & valid_ad_mask & ~pos_ad] = W_NEG_LEAD
                else:
                    adlib_wts[center_lead & valid_ad_mask & ~pos_ad] = 0.8  # mild, optional
                
                # Positives override negatives
                adlib_wts[center_secondary_adlib] = W_POS_ADLIB
                adlib_wts[center_primary_adlib] = W_POS_ADLIB_PRIMARY
                                
                # Map window start frame -> start sample at sr_out
                start_time_sec = (start_frame * frame_ms) / 1000.0
                start_sample_out = int(round(start_time_sec * self.sr_out))
                                        
                self.samples.append(
                    (
                        audio_path,
                        start_sample_out,
                        label_vec,
                        importance_vec,
                        harmony_vec,
                        harmony_wts,
                        adlib_vec,
                        adlib_wts,
                    )
                )
    
        if not self.samples:
            raise RuntimeError("No training windows built. Check your JSON/audio alignment.")
        
        total_labels = np.zeros(len(self.classes), dtype=np.float64)
        total_weights = np.zeros(len(self.classes), dtype=np.float64)

        for (_, _, label_vec, importance_vec,
             _, _, _, _) in self.samples:
            total_labels += label_vec
            total_weights += importance_vec
            
        print(f"Total labels: {total_labels / len(self.samples)}")
        print(f"Total weights: {total_weights / len(self.samples)}")
        
        n_silence = sum(
            label_vec[self.silence_idx] == 1.0
            for (_, _, label_vec, _, _, _, _, _) in self.samples
        )
        print("silence windows:", n_silence, "/", len(self.samples))
        
        self.base_samples = list(self.samples)
        
        # Test for Dataset to see proportion of Data
        print("\n[KpopFrameDataset] Category counts:")
        total_windows = sum(self.debug_category_counts.values())
        for cat, cnt in self.debug_category_counts.items():
            frac = cnt / max(total_windows, 1)
            print(f"  {cat:13s}: {cnt:7d} ({frac:5.1%})")

        # print("\n[KpopFrameDataset] Example windows per category:")
        # for cat, examples in self.debug_category_examples.items():
        #     print(f"\n  Category: {cat}  (showing {len(examples)} examples)")
        #     for ex in examples:
        #         print(f"    song={ex['song']}, center_frame={ex['center_frame']}, "
        #             f"vocal_frac={ex['vocal_frac_window']:.2f}, "
        #             f"overlap_frac={ex['overlap_frac_window']:.2f}, "
        #             f"dominant_idx={ex['dominant_idx']}, "
        #             f"dominant_frac={ex['dominant_frac']:.2f}, "
        #             f"frame_label={ex['frame_label']}")
        
        # ----------------------------
        # 4) Simple audio cache to avoid re-loading the same song
        # ----------------------------
        self._max_cached_songs = 24       # tune this (8–32 is typical)
        self._wave_cache = OrderedDict()  # path -> wav (1, T_out)
        self._resamplers: Dict[int, Resample] = {}

        print(f"[KpopFrameDataset] total windows: {len(self.samples)}")
    
    def _find_vocals_audio(self, song_name: str) -> str:
        """
        Try to find <song>_vocals.(wav|flac|mp3|m4a) under group_dir.
        """
        exts = [".wav"]
        for ext in exts:
            cand = os.path.join(self.group_dir, f"{song_name}_vocals{ext}")
            if os.path.exists(cand):
                return cand
        return None
    
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
                    "pair_name": str,
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
    
    def __getitem__(self, idx: int):
        # ---- Case 1: base sample (no pitch shift) ----
        (audio_path,
            start_sample_out,
            label_vec,
            importance_vec,
            harmony_vec,
            harmony_wts,
            adlib_vec,
            adlib_wts) = self.base_samples[idx]

        wav = self._load_song_wave(audio_path)  # (1, T_out)
        end = start_sample_out + self.chunk_len
        if end > wav.shape[-1]:
            pad = end - wav.shape[-1]
            seg = torch.nn.functional.pad(wav[..., start_sample_out:], (0, pad))
        else:
            seg = wav[..., start_sample_out:end]  # (1, chunk_len)

        labels_main = torch.from_numpy(label_vec).to(torch.float32)
        weights_main = torch.from_numpy(importance_vec).to(torch.float32)
        labels_harm = torch.from_numpy(harmony_vec).to(torch.float32)
        weights_harm = torch.from_numpy(harmony_wts).to(torch.float32)
        labels_ad   = torch.from_numpy(adlib_vec).to(torch.float32)
        weights_ad   = torch.from_numpy(adlib_wts).to(torch.float32)
        
        return (seg,
                labels_main, weights_main,
                labels_harm, weights_harm,
                labels_ad,   weights_ad)

class MultiTaskHead(nn.Module):
    def __init__(self, emb_dim_fused: int, emb_dim_ctx: int, num_members: int):
        super().__init__()
        self.num_members = num_members
        
        # hidden = 256
        hidden = 256
        # Shared trunk for main/harmony (fused = 2D)
        self.shared_fused = nn.Sequential(
            nn.Linear(emb_dim_fused, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Shared trunk for adlib (ctx = D)
        self.shared_ctx = nn.Sequential(
            nn.Linear(emb_dim_ctx, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        
        # Heads 
        self.presence_head = nn.Linear(hidden, num_members + 1) # members + silence
        self.harmony_head = nn.Linear(hidden, num_members) # harmony per member
        self.adlib_head = nn.Linear(hidden, num_members) # adlib per member
    
    def forward(self, emb_fused, emb_ctx):
        h_fused = self.shared_fused(emb_fused)
        h_ctx   = self.shared_ctx(emb_ctx)

        return {
            "main":    self.presence_head(h_fused),
            "harmony": self.harmony_head(h_fused),
            "adlib":   self.adlib_head(h_ctx),
        }

# ---------------------------
# Model: ECAPA encoder (frozen or trainable) + linear head
# ---------------------------
class MuQEncoderWrapper(nn.Module):
    """
    Wrap MuQ so it behaves like SpeechBrain's encoder.encode_batch(...)
    returning (B, 1, D) so your .squeeze(1) keeps working.
    """
    def __init__(
        self,
        muq_model,
        muq_sr: int = 24000,
        pooling: str = "mean",   # mean is simplest & stable
        debug: bool = True,
    ):
        super().__init__()
        self.muq = muq_model
        self.muq_sr = muq_sr
        self.pooling = pooling
        self.debug = debug
        self.topk_frac=0.3
        self._debug_printed = False
            
    @torch.no_grad()
    def encode_batch(self, wavs: torch.Tensor, ctx_frac: float = 0.2) -> torch.Tensor:
        """
        One MuQ forward. Returns:
          emb_main: (B, 1, D)
          emb_ctx:  (B, 1, D) pooled from the center frames
        """
        if wavs.ndim != 2:
            raise ValueError(f"MuQ expects (B, T), got {wavs.shape}")

        x = wavs

        # MuQ forward: expected to output framewise reps (B, T', D)
        out = self.muq(x, output_hidden_states=False)

        if self.debug and not self._debug_printed:
            print("[MuQ DEBUG] input x:", tuple(x.shape), x.dtype, x.device)
            print("[MuQ DEBUG] out type:", type(out))
            if isinstance(out, dict):
                print("[MuQ DEBUG] out keys:", list(out.keys()))
            else:
                # show common attrs if present
                for k in ["last_hidden_state", "hidden_states"]:
                    print(f"[MuQ DEBUG] has {k}:", hasattr(out, k))
            self._debug_printed = True
            
        # Common conventions: out could be a dict-like or object with last_hidden_state
        feats = getattr(out, "last_hidden_state", None)
        if feats is None and isinstance(out, dict):
            feats = out.get("last_hidden_state", None)
        if feats is None:
            # fallback: if the model returns tensor directly
            if torch.is_tensor(out):
                feats = out
            else:
                raise RuntimeError("Could not find MuQ frame features (last_hidden_state).")

        # feats: (B, T', D)
        B, Tprime, D = feats.shape

        # ---- global pooled ----
        emb_main = self._pool_feats(feats)  # (B, D)

         # ---- center pooled (slice frames, not waveform) ----
        ctx_len = max(1, int(round(Tprime * ctx_frac)))
        mid = Tprime // 2
        half = ctx_len // 2
        start = max(0, mid - half)
        end = min(Tprime, start + ctx_len)
        start = max(0, end - ctx_len)  # keep exact length when possible

        feats_center = feats[:, start:end, :]  # (B, ctx_len, D)
        emb_ctx = self._pool_feats(feats_center)  # (B, D)

        if self.debug and not hasattr(self, "_debug_two_printed"):
            print(f"[MuQ DEBUG] feats={tuple(feats.shape)} center=[{start}:{end}] ctx_len={ctx_len}")
            self._debug_two_printed = True

        return emb_main.unsqueeze(1), emb_ctx.unsqueeze(1)
    
    def _pool_feats(self, feats: torch.Tensor) -> torch.Tensor:
        # feats: (B, T', D) -> (B, D)
        if self.pooling == "mean":
            return feats.mean(dim=1)
        elif self.pooling == "cls":
            return feats[:, 0, :]
        elif self.pooling == "topk":
            # NOTE: make sure self.topk_frac exists in __init__
            scores = feats.norm(p=2, dim=-1)  # (B, T')
            T = feats.size(1)
            k = max(1, int(T * self.topk_frac))
            idx = scores.topk(k, dim=1).indices  # (B, k)
            idx = idx.unsqueeze(-1).expand(-1, -1, feats.size(-1))  # (B,k,D)
            top_feats = feats.gather(dim=1, index=idx)  # (B,k,D)
            
            if self.debug and not hasattr(self, "_debug_topk_printed"):
                print(f"[MuQ DEBUG] topk k={k}/{T} example idx:", idx[0, :, 0].detach().cpu().numpy()[:10])
                self._debug_topk_printed = True
            return top_feats.mean(dim=1)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

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
       
# ---------------------------
# Training / evaluation
# ---------------------------
def train_epoch(encoder, head, loader, device, optimizer, 
                thr=0.5, use_amp=True,
                ctx_frac: float = 0.25,
                lambda_harmony: float = 0.7,
                lambda_adlib: float = 0.7):
    """
    encoder: ECAPA model (SpeechBrain)
    head:   multi-task head taking fused embedding -> dict of logits
    loader: yields (wavs, y_main, w_main, y_harm, w_harm, y_ad, w_ad)
    """
    encoder.eval() # Extract embeddings under no_grad by default
    head.train()
    
    scaler = torch.amp.GradScaler(device=device, enabled=(use_amp and device.type == "cuda"))
    
    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0

    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    
    for batch in tqdm(loader, desc="Train", leave=False):
        (wavs,
         y_main, w_main,
         y_harm, w_harm,
         y_ad,   w_ad) = batch
        
        wavs = wavs.to(device, non_blocking=True)   # (B, 1, T)
        y_main = y_main.to(device, non_blocking=True)
        w_main = w_main.to(device, non_blocking=True)
        y_harm = y_harm.to(device, non_blocking=True)
        w_harm = w_harm.to(device, non_blocking=True)
        y_ad = y_ad.to(device, non_blocking=True)
        w_ad = w_ad.to(device, non_blocking=True)
        
        # Ensure shapes for ECAPA
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs_encoder = wavs.squeeze(1)        # (B, T)
        elif wavs.ndim == 2:
            wavs_encoder = wavs                   # (B, T)
        else:
            raise ValueError(f"Unexpected wavs shape: {wavs.shape}")
        
        amp_ctx = torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda"))
        
        with torch.no_grad():
            # SpeechBrain ECAPA expects (B, T) or (B, 1, T) tensors; encode_batch handles both.
            # print(f"[DEBUG] wavs.shape = {wavs.shape}, dtype={wavs.dtype}, device={wavs.device}")
            emb_main_b1, emb_ctx_b1 = encoder.encode_batch(wavs_encoder, ctx_frac=ctx_frac)
            emb_main = emb_main_b1.squeeze(1)  # (B, D)
            emb_ctx  = emb_ctx_b1.squeeze(1)   # (B, D)
             
        # Fuse multi-window embeddings
        emb_fused = torch.cat([emb_main, emb_ctx], dim=1)
        
        optimizer.zero_grad(set_to_none=True)
        
        with amp_ctx:
            out = head(emb_fused, emb_ctx)
            logits_main = out["main"]      # (B, C_main)
            logits_harm = out["harmony"]   # (B, C_members)
            logits_ad   = out["adlib"]     # (B, C_members)
            
        # --- main head ---
        loss_main_raw = bce(logits_main, y_main)
        loss_main = (loss_main_raw * w_main).mean()
        
        # --- harmony head ---
        loss_harm_raw = bce(logits_harm, y_harm)
        loss_harm = (loss_harm_raw * w_harm).mean()
        
        # --- ad-lib head ---
        loss_ad_raw = bce(logits_ad, y_ad) 
        loss_ad = (loss_ad_raw * w_ad).mean()
        
        # Total loss with task weights
        loss = loss_main + lambda_harmony * loss_harm + lambda_adlib * loss_ad
        
        if scaler.is_enabled():
            scaler.scale(loss).backward()
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
    return avg_loss, {"micro_f1": avg_f1, "precision": avg_p, "recall": avg_r, "subset_acc": avg_subset}

@torch.no_grad()
def eval_epoch(encoder, head, loader, device, thr=0.5, use_amp=True,
               ctx_frac: float = 0.25, lambda_harm: float = 0.7, lambda_ad: float = 0.7):
    encoder.eval()
    head.eval()
    
    total_loss, total_count = 0.0, 0
    total_f1, total_prec, total_rec = 0.0, 0.0, 0.0
    total_subset_acc = 0.0
    
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")
    
    for batch in tqdm(loader, desc="Eval", leave=False):
        (wavs,
         y_main, w_main,
         y_harm, w_harm,
         y_ad,   w_ad) = batch
        
        wavs = wavs.to(device)
        y_main = y_main.to(device)
        w_main = w_main.to(device)
        y_harm = y_harm.to(device)
        w_harm = w_harm.to(device)
        y_ad   = y_ad.to(device)
        w_ad   = w_ad.to(device)
        
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs_encoder = wavs.squeeze(1)
        elif wavs.ndim == 2:
            wavs_encoder = wavs
        else:
            raise ValueError(f"Unexpected wavs shape: {wavs.shape}")

        # Full 2s
        emb_main_b1, emb_ctx_b1 = encoder.encode_batch(wavs_encoder, ctx_frac=ctx_frac)
        emb_main = emb_main_b1.squeeze(1)  # (B, D)
        emb_ctx  = emb_ctx_b1.squeeze(1)  # (B, D)
        
        emb_fused = torch.cat([emb_main, emb_ctx], dim=1)    
        
        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda")):
            out = head(emb_fused, emb_ctx)
            logits_main = out["main"]
            logits_harm = out["harmony"]
            logits_ad   = out["adlib"]
        
        loss_main_raw = bce(logits_main, y_main)
        loss_main = (loss_main_raw * w_main).mean()
        
        loss_harm_raw = bce(logits_harm, y_harm)
        loss_harm = (loss_harm_raw * w_harm).mean()
        
        loss_ad_raw = bce(logits_ad, y_ad)
        loss_ad = (loss_ad_raw * w_ad).mean()
        
        loss = loss_main + lambda_harm * loss_harm + lambda_ad * loss_ad
        
        bsz = wavs.size(0)
        total_loss += loss.item() * bsz
        total_count += bsz

        f1, prec, rec = multilabel_micro_f1(logits_main, y_main, thr=thr)
        total_f1 += f1 * bsz
        total_prec += prec * bsz
        total_rec += rec * bsz
        total_subset_acc += subset_accuracy(logits_main, y_main, thr=thr) * bsz
    
    pred_pos = (torch.sigmoid(logits_main.detach()) > thr).float().sum(dim=1).mean().item()
    true_pos = y_main.sum(dim=1).mean().item()
    print(f"[DEBUG] avg_pred_pos={pred_pos:.2f} avg_true_pos={true_pos:.2f}")
    
    avg_loss = total_loss / max(1, total_count)
    avg_f1 = total_f1 / max(1, total_count)
    avg_p = total_prec / max(1, total_count)
    avg_r = total_rec / max(1, total_count)
    avg_subset = total_subset_acc / max(1, total_count)
    return avg_loss, {"micro_f1": avg_f1, "precision": avg_p, "recall": avg_r, "subset_acc": avg_subset}

# Main
def main():
    args = parse_args()
    set_seed(args.seed)
    torch.set_float32_matmul_precision('high') # Small speed bump on Ampere
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    group_dir = os.path.join(args.root, "training_data", args.group)
    
    # Dataset & split
    full_ds = KpopVocalDataset(group_dir, sr_out=args.sr_out, chunk_sec=args.chunk_sec, group_name=args.group, useMuq=True)
    
    print("Classes:", full_ds.class_map.idx_to_name)
    print("len(full_ds) =", len(full_ds))
    
    n_val = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    print("train/val =", n_train, n_val)
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # Load MuQ ECAPA encoder (pretraiened)
    muq = MuQ.from_pretrained("OpenMuQ/MuQ-large-msd-iter")
    muq.to(device).eval()
    
    encoder = MuQEncoderWrapper(
        muq_model=muq,
        muq_sr=args.sr_out,
        pooling="topk",
        debug=False
    ).to(device)
    
    dummy = torch.zeros(1, int(args.chunk_sec * args.sr_out), device=device)

    # Ensure (B, T)
    if dummy.ndim == 3 and dummy.size(1) == 1:
        dummy = dummy.squeeze(1)

    with torch.no_grad():
        emb_main_b1, _ = encoder.encode_batch(dummy)  # tuple
        emb_dim = emb_main_b1.squeeze(1).shape[-1]
        
    fused_dim = emb_dim * 2
    num_members = full_ds.num_members
        
    head = MultiTaskHead(
        emb_dim_fused=fused_dim, 
        emb_dim_ctx=emb_dim, 
        num_members=num_members
    ).to(device)
    
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)
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
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_metrics = train_epoch(encoder, head, train_loader, device, optimizer, thr=eval_thr)
        va_loss, va_metrics = eval_epoch(encoder, head, val_loader, device, thr=eval_thr)
        
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
                "classes": full_ds.class_map.idx_to_name,
                "emb_dim": emb_dim,
                "sr": args.sr_out,
                "chunk_sec": args.chunk_sec,
                "group": args.group,
                "eval_thr": eval_thr,
            }, ckpt_path)
            print(f"✅ Saved best head to: {ckpt_path} (acc={best_acc:.4f})")
        
    print("\nDone. Best val acc:", best_acc)
    
if __name__ == "__main__":
    main()