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
from speechbrain.inference.speaker import EncoderClassifier
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
    ap.add_argument("--sr_ecapa", type=int, default=16000, help="ECAPA target sample rate")
    ap.add_argument("--chunk-sec", type=float, default=2.0, help="Chunk length in seconds for training")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-split", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--save-dir", type=str, default="./checkpoints")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--eval_thr", type=float, default=0.5)
    ap.add_argument("--freeze-ecapa", action="store_true", help="Freeze ECAPA encoder parameters")
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
                chunk_sec: float, num_workers: int, group_name: str, min_song_sec: float = 4.0,
                pitch_prob: float = 0.3, pitch_semitones_range: Tuple[float, float] = (-2.0, 2.0),
                window_hop_ratio: float = 0.5, presence_thresh=0.4,
                alpha_lead=1.0, alpha_adlib = 0.5, min_weight = 0.2, max_weight = 2.0,
                k_train: int = 2):
        super().__init__()
        self.group_dir = group_dir
        self.sr_out = sr_out
        self.chunk_sec = chunk_sec
        self.chunk_len = int(round(chunk_sec * sr_out))
        self.min_song_sec = min_song_sec

        self.pitch_semitones_range = pitch_semitones_range
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
            secondary_arr = np.asarray(meta["secondaryRole"], dtype=np.int32)
            
        
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
                window_secondary = secondary_arr[start_frame:end_frame]
                window_adlib = adlib_arr[start_frame:end_frame]
                window_backingStyle = backing_arr[start_frame:end_frame]
                
                # Fractions over this window (0..1) - still useful for importance weights
                presence_frac = window_presence.mean(axis=0) # how often each siniger is present
                lead_frac = window_lead.mean(axis=0) # How often they are lead (role)
                secondary_frac = window_secondary.mean(axis=0) # how often they are secondary
                
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
                
                # CASE 1 → CLEAR VOCAL
                if any_vocal_center:
                    # Label is ALWAYS singing — never flip or skip
                    label_vec[:self.num_members] = frame_label.astype(np.float32)
                    label_vec[self.silence_idx] = 0.0
                    
                    # Gang Vocal should be window-based, not center frame
                    if self.gang_idx is not None:
                        GANG_TAU = 0.1
                        label_vec[self.gang_idx] = 1.0 if presence_frac[self.gang_idx] >= GANG_TAU else 0.0
                    
                    # Define thresholds
                    W_LEAD = 0.9 # How much lead boosts weight
                    W_SECONDARY = 0.6
                    W_BACKING = 0.6 # backing/harmony boost
                    W_ADLIB = 0.3 # How strongly ad-libs DOWN-weight
                    W_GANG_SOLO = 0.6 # Gang vocal alone
                    W_GANG_OVERLAP = 0.3 # Gang vocal when members also sing
                    
                    # Check if num_members implies +gang vocals or no
                    importance_singers = np.ones(self.num_members, dtype=np.float32)
                    
                    # GLOBAL clean vocal
                    if dominant_frac >= TAU_DOMINANT and overlap_frac_window <= TAU_OVERLAP:
                        for c in range(self.num_members):
                            if presence_frac[c] < 1e-3: 
                                # not present in this window
                                importance_singers[c] = self.min_weight
                                continue
                            
                            # Handle Gang Vocal specially
                            if self.gang_idx is not None and c == self.gang_idx:
                                if any_real:
                                    # Gang layered under members -> small weight
                                    w = W_GANG_OVERLAP
                                else:
                                    # pure gang section -> moderate weight
                                    w = W_GANG_SOLO
                                importance_singers[c] = np.clip(w, self.min_weight, self.max_weight)
                                continue
                        
                            # Normal member
                            lf = float(lead_frac[c])
                            bf = float(backingStyle_frac[c])
                            sf = float(secondary_frac[c]) # Secondary role (non-lead overlap)
                            af = float(adlib_frac[c])
                            
                            adlibPenalty = W_ADLIB * af

                            # cancel penalty for solo/lead adlibs
                            if af > 0.3 and lf > sf:
                                adlibPenalty *= 0.2   # or even 0.0

                            w = 1.0 + W_LEAD*lf + W_BACKING*bf + W_SECONDARY*sf - adlibPenalty
                            
                            # IF mostly ad-lib and not really lead/backing, don't crash it too low
                            if af > 0.5 and lf < 0.2 and bf < 0.2:
                                w = max(w, 0.7)
                                
                            importance_singers[c] = np.clip(w, self.min_weight, self.max_weight)
                        
                        importance_vec[:self.num_members] = importance_singers
                        category = "clear_vocal"
                    elif local_dom_frac >= 0.65:
                        # not globally clean but locally dominant → short line rescue
                        importance_vec[:self.num_members] = 0.5   # medium weight
                        category = "semi_clear_vocal"
                    else:
                        # ambiguous vocal
                        importance_vec[:self.num_members] = 0.1
                        
                        present = presence_frac[:self.num_members] > 1e-3
                        importance_vec[:self.num_members][present] = 0.4
                        
                        # if gang exists and is present, give it extra weight (this is the whole point)
                        if self.gang_idx is not None and presence_frac[self.gang_idx] > 1e-3:
                            importance_vec[self.gang_idx] = 0.8
                        
                        category = "ambiguous"
                    
                # center frame silent
                else:       
                    # IMPORTANT: don't heavily punish members/gang during silence-center windows,
                    # because the 2s context can contain vocals nearby.
                    importance_vec[:self.num_members] = 0.1  # weak negatives for singers (including gang)
                    if vocal_frac_window <= TAU_SILENCE_WINDOW and local_vocal_frac < 0.15:
                        # real silence
                        importance_vec[:self.num_members] = 0.2
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
                center_secondary = secondary_arr[cf0:cf1].max(axis=0).astype(bool)
                center_backing = backing_arr[cf0:cf1].max(axis=0).astype(bool)
                center_overlap = (presence[cf0:cf1].sum(axis=1) > 1).any()
                
                # only treat harmony if there's overlap right NOW (not somewhere in the 2s")
                if center_overlap and any_lead_center:
                    pos_harm = center_backing & ~center_lead
                    harmony_vec[pos_harm] = 1.0
                    
                    # Base negative: present but not harmony (weak-ish)
                    W_NEG_BASE = 0.6

                    # Strong negative: lead should almost never be harmony
                    W_NEG_LEAD = 1.2

                    # Medium negative: secondary-but-not-backing should not be called harmony
                    W_NEG_SECONDARY = 0.9
                    
                    # Positive weight: reward true harmony
                    W_POS_HARM = 1.6
    
                    # Weights: only train on singers who are present rn
                    harmony_wts[:] = 0.0
                    harmony_wts[center_presence] = W_NEG_BASE # weak negatives
                    harmony_wts[center_secondary & center_presence & ~pos_harm] = W_NEG_SECONDARY
                    
                    # Make false "lead is harmony" expensive
                    harmony_wts[center_lead & center_presence & ~pos_harm] = W_NEG_LEAD
                    harmony_wts[pos_harm] = W_POS_HARM
                else:
                    harmony_wts[:] = 0.0 # ignore harmony when there is no overlap in the center
                
                valid_ad_mask = center_presence # Only train adlib for singers who are present
                
                # weights: ignore non-present singers for the adlib head
                adlib_wts[:] = 0.0
                
                # This is the cost of a false positive adlib on a present singer.
                W_NEG_SOLO = 0.5 # present, not adlib, not clearly lead/secondary (rare)
                W_NEG_SECONDARY = 0.8 # present+secondary but not adlib (important: overlap ≠ adlib)
                W_NEG_LEAD = 1.1 # present+lead but not adlib (very important: lead ≠ adlib)
                
                # Positive weight: reward catching real adlibs.
                W_POS_ADLIB = 1.5
                
                # Apply negatives only to present singers by default
                adlib_wts[valid_ad_mask] = W_NEG_SOLO
                
                # Stronger negatives depending on ROLE:
                # If singer is lead and not adlib, it should be expensive to predict adlib=1.
                adlib_wts[center_lead & valid_ad_mask & ~pos_ad] = W_NEG_LEAD
                
                # If singer is secondary (overlapping/non-lead) and not adlib,
                # make it moderately expensive to predict adlib=1 (prevents "everyone adlibs" shortcut).
                adlib_wts[center_secondary & valid_ad_mask & ~pos_ad] = W_NEG_SECONDARY
                
                # Positives override negatives
                adlib_wts[pos_ad] = W_POS_ADLIB
                
                # if center_overlap and random.random() < 0.01:
                #     print("HARMONY WTS:",
                #         "lead", harmony_wts[center_lead].tolist(),
                #         "secondary", harmony_wts[center_secondary].tolist(),
                #         "pos", harmony_wts[pos_harm].tolist())
                #     print("counts:",
                #         "present", int(center_presence.sum()),
                #         "secondary", int(center_secondary.sum()),
                #         "backing", int(center_backing.sum()),
                #         "lead", int(center_lead.sum()),
                #         "pos_harm", int(pos_harm.sum()))
                                
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
        
        self._append_synthetic_harmony_samples(frames_per_window, frame_ms)
        
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
        self.augmented_samples: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                  torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = []
        
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
        self._cache_path = None
        self._cache_wave = None  # (1, T_out)
        self._resamplers: Dict[int, Resample] = {}
        
        if pitch_prob > 0.0:
            self._build_pitch_augmented_subset(pitch_prob, num_workers)

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
        if sr_src not in self._resamplers:
            self._resamplers[sr_src] = Resample(sr_src, self.sr_out)
        return self._resamplers[sr_src]
    
    def _append_synthetic_harmony_samples(self, frames_per_window: int, frame_ms: int):
        """
        Adds extra training windows from synthetic harmony MP3s.
        These windows are *already overlapped by construction*, so harmony labels are valid.
        """
        if not getattr(self, "synthetic_harmony_clips", None):
            return

        win_sec = (frames_per_window * frame_ms) / 1000.0
        win_samples = int(round(win_sec * self.sr_out))
        hop_samples = max(1, win_samples // 2)  # 50% overlap windows

        added = 0

        for clip in self.synthetic_harmony_clips:
            path = clip["path"]
            lead_idx = clip["lead_idx"]
            harm_idx = clip["harm_idx"]

            try:
                info = torchaudio.info(path)
                dur_sec = info.num_frames / float(info.sample_rate)
                total_out = int(round(dur_sec * self.sr_out))
            except Exception as e:
                print(f"[KpopFrameDataset] Could not read info for {path}: {e}")
                continue

            if total_out <= 0:
                continue

            # windows starting positions in output sample space
            starts = list(range(0, max(1, total_out - win_samples + 1), hop_samples))
            if not starts:
                starts = [0]

            for start_sample_out in starts:
                # --- MAIN head: keep this light (or ignore), since this is for harmony robustness ---
                label_vec = np.zeros(len(self.classes), dtype=np.float32)
                importance_vec = np.zeros(len(self.classes), dtype=np.float32)

                # Mark both singers as present (multi-label). Do NOT mark silence.
                label_vec[lead_idx] = 1.0
                label_vec[harm_idx] = 1.0

                # VERY IMPORTANT: keep weights light so you don't distort main learning.
                importance_vec[lead_idx] = 0.4
                importance_vec[harm_idx] = 0.4
                if self.silence_idx is not None:
                    importance_vec[self.silence_idx] = 0.0

                # --- HARMONY head: the whole point ---
                harmony_vec = np.zeros(self.num_members, dtype=np.float32)
                harmony_wts = np.zeros(self.num_members, dtype=np.float32)

                # Harmony definition: second member is harmony
                harmony_vec[harm_idx] = 1.0

                # Train only on the two active singers in this synthetic clip
                harmony_wts[lead_idx] = 0.3  # negative: lead is present but not harmony
                harmony_wts[harm_idx] = 1.2  # positive

                # --- ADLIB head: optional, but safest is to IGNORE it here ---
                adlib_vec = np.zeros(self.num_members, dtype=np.float32)
                adlib_wts = np.zeros(self.num_members, dtype=np.float32)
                # (If you want to teach "this is NOT adlib", you can set weak negatives instead:)
                # adlib_wts[lead_idx] = 0.1
                # adlib_wts[harm_idx] = 0.1

                self.samples.append(
                    (
                        path,
                        int(start_sample_out),
                        label_vec,
                        importance_vec,
                        harmony_vec,
                        harmony_wts,
                        adlib_vec,
                        adlib_wts,
                    )
                )
                added += 1

        print(f"[KpopFrameDataset] Added synthetic harmony windows: {added}")
    
    def _load_song_wave(self, path: str) -> torch.Tensor:
        """
        Load + mono + resample a *whole song* once, cache it.
        Returns Tensor (1, T_out).
        """
        if self._cache_path == path and self._cache_wave is not None:
            return self._cache_wave

        wav, sr_src = torchaudio.load(path)  # (C, T_src)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        if sr_src != self.sr_out:
            resampler = self._get_resampler(sr_src)
            wav = resampler(wav)

        self._cache_path = path
        self._cache_wave = wav  # (1, T_out)
        return wav
    
    def _build_pitch_augmented_subset(self, pitch_aug_ratio: float, max_workers: int = 4):
        """
        Precompute pitch-shifted copies of a subset of base samples.

        Stores them in self.augmented_samples as (audio_chunk, label_tensor, weight_tensor).
        """
        num_base = len(self.base_samples)
        if num_base == 0:
            return

        num_aug = max(1, int(num_base * pitch_aug_ratio))
        # randomly choose which base samples to augment
        indices = random.sample(range(num_base), num_aug)

        print(f"[KpopVocalDataset] Precomputing {num_aug} pitch-augmented chunks "
              f"out of {num_base} base windows (ratio={pitch_aug_ratio:.2f})")

        def worker(base_idx: int):
            (audio_path,
             start_sample_out,
             label_vec,
             importance_vec,
             harmony_vec,
             harmony_wts,
             adlib_vec,
             adlib_wts) = self.base_samples[base_idx]

            # Load and resample this song independently (no cached state)
            wav, sr_src = torchaudio.load(audio_path)   # (C, T_src)
            if wav.size(0) > 1:
                wav = wav.mean(dim=0, keepdim=True)

            if sr_src != self.sr_out:
                resampler = Resample(sr_src, self.sr_out)
                wav = resampler(wav)                    # (1, T_out)

            end = start_sample_out + self.chunk_len
            if end > wav.shape[-1]:
                pad = end - wav.shape[-1]
                seg = torch.nn.functional.pad(wav[..., start_sample_out:], (0, pad))
            else:
                seg = wav[..., start_sample_out:end]    # (1, chunk_len)

            # Apply pitch shift ONCE here
            seg_aug = self._apply_pitch_shift(seg)
            if seg_aug is None:
                return None
            lab_main = torch.from_numpy(label_vec).to(torch.float32)
            wts_main = torch.from_numpy(importance_vec).to(torch.float32)
            lab_harm = torch.from_numpy(harmony_vec).to(torch.float32)
            wts_harm = torch.from_numpy(harmony_wts).to(torch.float32)
            lab_ad   = torch.from_numpy(adlib_vec).to(torch.float32)
            wts_ad   = torch.from_numpy(adlib_wts).to(torch.float32)
            
            return (seg_aug, lab_main, wts_main,
                    lab_harm, wts_harm,
                    lab_ad, wts_ad)

        results = []
        # If max_workers <= 1, just run sequentially (handy for debugging)
        if max_workers is None or max_workers <= 1:
            for idx in tqdm(indices, desc="Precomputing pitch augs (1 worker)"):
                out = worker(idx)
                if out is not None:
                    results.append(out)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(worker, idx) for idx in indices]
                for fut in tqdm(as_completed(futures),
                                total=len(futures),
                                desc="Precomputing pitch augs"):
                    out = fut.result()
                    if out is not None:
                        results.append(out)

        self.augmented_samples.extend(results)
        print(f"[KpopVocalDataset] Done. Augmented samples: {len(self.augmented_samples)}")
    
    def _apply_pitch_shift(self, seg: torch.Tensor) -> torch.Tensor:
        """
        Apply a random pitch shift to seg (1, T). Always attempts to shift;
        if anything fails, returns seg unchanged.
        """
        n_steps = random.uniform(-2.0, 2.0)  # e.g. ±2 semitones
        try:
            if hasattr(torchaudio.functional, "pitch_shift"):
                seg2 = torchaudio.functional.pitch_shift(seg, self.sr_out, n_steps=n_steps)
            else:
                effects = [
                    ["pitch", f"{n_steps * 100.0}"],
                    ["rate", f"{self.sr_out}"],
                ]
                seg2, _ = torchaudio.sox_effects.apply_effects_tensor(seg, self.sr_out, effects)

            # Ensure exact length chunk_len
            if seg2.shape[-1] >= self.chunk_len:
                return seg2[..., : self.chunk_len]
            else:
                pad = self.chunk_len - seg2.shape[-1]
                return torch.nn.functional.pad(seg2, (0, pad))
        except Exception:
            return seg
    
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
        return len(self.base_samples) + len(self.augmented_samples)

    def __getitem__(self, idx: int):
        base_count = len(self.base_samples)

        # ---- Case 1: base sample (no pitch shift) ----
        if idx < base_count:
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

        # ---- Case 2: augmented sample (precomputed pitch-shifted) ----
        aug_idx = idx - base_count
        (seg_aug,
         lab_main, wts_main,
         lab_harm, wts_harm,
         lab_ad,   wts_ad) = self.augmented_samples[aug_idx]

        # Return copies so we don't accidentally mutate the cached tensors
        return (seg_aug.clone(),
                lab_main.clone(), wts_main.clone(),
                lab_harm.clone(), wts_harm.clone(),
                lab_ad.clone(),   wts_ad.clone())

# ---------------------------
# Model: ECAPA encoder (frozen or trainable) + linear head
# ---------------------------
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
        
        wavs   = wavs.to(device, non_blocking=True)   # (B, 1, T)
        y_main = y_main.to(device, non_blocking=True)
        w_main = w_main.to(device, non_blocking=True)
        y_harm = y_harm.to(device, non_blocking=True)
        w_harm = w_harm.to(device, non_blocking=True)
        y_ad   = y_ad.to(device, non_blocking=True)
        w_ad   = w_ad.to(device, non_blocking=True)
        
        # Ensure shapes for ECAPA
        if wavs.ndim == 3 and wavs.size(1) == 1:
            wavs_ecapa = wavs.squeeze(1)        # (B, T)
        elif wavs.ndim == 2:
            wavs_ecapa = wavs                   # (B, T)
        else:
            raise ValueError(f"Unexpected wavs shape: {wavs.shape}")
        
        with torch.no_grad():
             # SpeechBrain ECAPA expects (B, T) or (B, 1, T) tensors; encode_batch handles both.
             # print(f"[DEBUG] wavs.shape = {wavs.shape}, dtype={wavs.dtype}, device={wavs.device}")
             emb_main = encoder.encode_batch(wavs_ecapa).squeeze(1) # (B, D)
             
             # 400-600ms center context
             ctx_wavs = extract_center_context(wavs, ctx_frac)
             ctx_ecapa = ctx_wavs.squeeze(1)
             emb_ctx = encoder.encode_batch(ctx_ecapa).squeeze(1)
             
        # Fuse multi-window embeddings
        emb_fused = torch.cat([emb_main, emb_ctx], dim=1)
        
        optimizer.zero_grad(set_to_none=True)
        amp_ctx = torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda"))
        
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
            wavs_ecapa = wavs.squeeze(1)
        elif wavs.ndim == 2:
            wavs_ecapa = wavs
        else:
            raise ValueError(f"Unexpected wavs shape: {wavs.shape}")

        with torch.autocast(device_type=device.type, enabled=(use_amp and device.type=="cuda")):
            # Full 2s
            emb_main = encoder.encode_batch(wavs_ecapa).squeeze(1)  # (B, D)
            
            # 0.5s center context
            ctx_wavs = extract_center_context(wavs, ctx_frac=ctx_frac) # (B, 1, T_ctx)
            ctx_ecapa = ctx_wavs.squeeze(1) # (B, T_ctx)
            emb_ctx = encoder.encode_batch(ctx_ecapa).squeeze(1) # (B, D)
            
            emb_fused = torch.cat([emb_main, emb_ctx], dim=1)    
            
            out = head(emb_fused, emb_ctx)
            logits_main = out["main"]
            logits_harm = out["harmony"]
            logits_ad = out["adlib"]
            
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
    
    avg_loss = total_loss / max(1, total_count)
    avg_f1   = total_f1 / max(1, total_count)
    avg_p    = total_prec / max(1, total_count)
    avg_r    = total_rec / max(1, total_count)
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
    full_ds = KpopVocalDataset(group_dir, args.sr_ecapa, args.chunk_sec, num_workers=args.num_workers, group_name=args.group ,pitch_prob=0.0)
    
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
    
    # Load SpeachBrain ECAPA encoder (pretraiened)
    
    encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cuda" if device.type == "cuda" else "cpu"}
    )
    
    if args.freeze_ecapa:
        for p in encoder.parameters():
            p.requires_grad = False
            
    # Probe embedding dim with dummy forward
    dummy = torch.zeros(
        1, int(args.chunk_sec * args.sr_ecapa),
        dtype=torch.float32, device=device   
    )
    with torch.no_grad():
        emb_dim = encoder.encode_batch(dummy).squeeze(1).shape[-1]
    
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
    ckpt_path = os.path.join(args.save_dir, f"{args.group}_ecapa_head.pt")
    
    eval_thr = getattr(args, "eval_thr", 0.5)
    
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
                "sr": args.sr_ecapa,
                "chunk_sec": args.chunk_sec,
                "group": args.group,
                "eval_thr": eval_thr,
            }, ckpt_path)
            print(f"✅ Saved best head to: {ckpt_path} (acc={best_acc:.4f})")
        
    print("\nDone. Best val acc:", best_acc)
    
if __name__ == "__main__":
    main()