import os, json, random, glob
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
from collections import OrderedDict

class BinaryVocalDataset(Dataset):
    """
    Member-agnostic anchor dataset (index -> (song_id, center_ms)).
    Use getItemForMember(memberName, idx) to turn an anchor into (audio, y, weight)
    for binary classification of that member.

    Label JSON format per entry: [memberName, start_chunk, end_chunk, is_backing, is_adlib]
    chunk = 40ms
    """

    def __init__(
        self,
        json_dir,
        cache_dir,          # dict: song_id -> audio filepath
        sr_out=24000,
        context_sec=1.0,
        chunk_ms=40,
        transition_pad_chunks=2, # ±2 chunks (80ms) near boundaries
        overlap_weight=0.4,
        transition_weight=0.3,
        backing_weight=0.5,
        adlib_weight=0.8,
        p_other_member_neg=0.8,  # of negatives, percent that are other-member vocal
        p_pos=0.5, # balanced sampling per call
    ):
        self.training_songs = self._get_wav_files_from_cache(cache_dir)
        self.sr_out = int(sr_out)
        self.context_sec = float(context_sec)
        self.chunk_ms = int(chunk_ms)
        self.transition_pad_chunks = int(transition_pad_chunks)

        self.overlap_weight = float(overlap_weight)
        self.transition_weight = float(transition_weight)
        self.backing_weight = float(backing_weight)
        self.adlib_weight = float(adlib_weight)

        self.p_other_member_neg = float(p_other_member_neg)
        self.p_pos = float(p_pos)

        self.samples_per_window = int(round(self.context_sec * self.sr_out))

        # ====== per-song structures ======
        # song -> dict(member -> bool[T])
        self.memberMask = {}
        # song -> bool[T] (any vocal)
        self.vocalMask = {}
        # song -> bool[T] (>=2 members)
        self.overlapMask = {}
        # song -> bool[T] (uncertain band near boundaries)
        self.transitionMask = {}
        # song -> float[T] (base weight from is_backing/is_adlib, max over members active)
        self.baseWeight = {}

        # Anchors: list of (song_id, center_chunk)
        self.anchors = []

        # Cache audio metadata (length) so we know T
        self._song_num_chunks = {}
        self.json_files = self._get_json_files_for_group(json_dir)
        self.group_members = self._infer_group_members_from_first_json()
        self._build_from_labels()
        sortedSongs = sorted(self.song_complexity.items(), key=lambda x: x[1])
        print("Easiest:", sortedSongs[:5])
        print("Hardest:", sortedSongs[-5:])
                
        # ----------------------------
        # Audio LRU cache
        # ----------------------------
        self._max_cached_songs = 16   # 8–32 is typical. Tune based on RAM.
        self._wave_cache = OrderedDict()  # path -> waveform (1, T_out)
        self._resamplers = {}  # (sr_src, sr_out) -> Resample object

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, idx):
        # member-agnostic anchor
        return self.anchors[idx]

    def _get_wav_files_from_cache(self, training_dir):
        """
        Scans training_dir recursively for *_vocals.wav files,
        excludes files containing 'leading' or 'backing',
        and returns a dict:

            { song_name : full_path_to_song_vocals.wav }
        """

        if not os.path.isdir(training_dir):
            raise ValueError(f"{training_dir} is not a valid directory")

        pattern = os.path.join(training_dir, "**", "*_vocals.wav")
        wav_paths = glob.glob(pattern, recursive=True)

        song_dict = {}

        for path in wav_paths:
            filename = os.path.basename(path)

            # Exclude unwanted stems
            lower_name = filename.lower()
            if "leading" in lower_name or "backing" in lower_name:
                continue

            # Expect format: song_name_vocals.wav
            if not filename.endswith("_vocals.wav"):
                continue

            song_name = filename.replace("_vocals.wav", "")

            # Optional: guard against duplicates
            if song_name in song_dict:
                print(f"[Warning] Duplicate song key detected: {song_name}")
                continue

            song_dict[song_name] = path

        return song_dict
    
    def _get_json_files_for_group(self, group_dir):
        json_files = []
        for fname in os.listdir(group_dir):
            if fname.endswith("_labels.json"):
                json_files.append(os.path.join(group_dir, fname))
        return json_files
    
    def _infer_group_members_from_first_json(self):
        """
        Inspect the first JSON label file in self.json_files
        and extract unique member names.

        Returns:
            list[str]: Sorted list of unique member names.
        """

        if not self.json_files:
            raise ValueError("self.json_files is empty — cannot infer members.")

        first_json_path = self.json_files[0]

        with open(first_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        seen_members = set()

        for entry in data:
            if not entry:
                continue
            member_name = entry[0]
            if member_name != "Cut":
                seen_members.add(member_name)

        members = sorted(list(seen_members))

        if len(members) == 0:
            raise RuntimeError(
                f"No members found in label file: {first_json_path}"
            )

        return members
        
    # ---------------------------
    # Public: member-conditioned
    # ---------------------------
    def getItemForMember(self, memberName: str, idx: int):
        song_id, center_c = self.anchors[idx]

        # pick pos/neg around this song; if anchor is unusable, resample a different anchor
        # (keeps things robust if a song has few positives for a member)
        for _ in range(20):
            is_pos = (random.random() < self.p_pos)
            center_c2, y, w = self._sample_center_for_member(song_id, memberName, want_pos=is_pos)
            if center_c2 is not None:
                audio = self._load_window(song_id, center_c2)
                return audio, torch.tensor([y], dtype=torch.float32), torch.tensor([w], dtype=torch.float32)

            # fallback: try another anchor entirely
            song_id, center_c = random.choice(self.anchors)

        # hard fallback: just return something deterministic
        audio = self._load_window(song_id, center_c)
        return audio, torch.tensor([0.0], dtype=torch.float32), torch.tensor([1.0], dtype=torch.float32)

    # ---------------------------
    # Build masks + anchors
    # ---------------------------
    def _build_from_labels(self):
        self.song_stats = {}
        adlibMaskBySong = {} # song_id -> bool[T]
        backingOnlyMaskBySong = {} # song_id -> bool[T]  (is_backing True AND is_adlib False)
        
        # print(f"training songs: {self.training_songs}")
        # 1) Discover song lengths (in chunks) from audio files
        for song_id, path in self.training_songs.items():
            info = torchaudio.info(path)
            # resampling happens at load time; chunk timeline is in ms so independent of sr
            duration_ms = (info.num_frames / info.sample_rate) * 1000.0
            
            T = int(np.ceil(duration_ms / self.chunk_ms))
            self._song_num_chunks[song_id] = max(T, 1)

        # 2) Initialize empty masks
        for song_id, T in self._song_num_chunks.items():
            self.memberMask[song_id] = {m: np.zeros(T, dtype=bool) for m in self.group_members}
            self.baseWeight[song_id] = np.ones(T, dtype=np.float32)
            
            adlibMaskBySong[song_id] = np.zeros(T, dtype=bool)
            backingOnlyMaskBySong[song_id] = np.zeros(T, dtype=bool)

        # 3) Fill masks from JSON labels
        # We also track boundaries to create transitionMask
        boundaries = defaultdict(list)  # song -> list of (start_c, end_c)
        for json_path in self.json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            song_id = os.path.basename(json_path).replace("_labels.json", "")
            if song_id not in self._song_num_chunks:
                continue
            T = self._song_num_chunks[song_id]

            for entry in data:
                member, start_c, end_c, is_back, is_adlib = entry
                if member not in self.memberMask[song_id]:
                    continue

                start_c = int(max(0, min(T - 1, start_c)))
                end_c = int(max(0, min(T - 1, end_c)))
                if end_c < start_c:
                    start_c, end_c = end_c, start_c

                self.memberMask[song_id][member][start_c:end_c + 1] = True
                boundaries[song_id].append((start_c, end_c))
                
                # Per-chunk adlib / backing-only masks (global, not per-member)
                if is_adlib:
                    adlibMaskBySong[song_id][start_c:end_c + 1] = True
                if is_back and (not is_adlib):
                    # very important: backing-only (exclude adlib)
                    backingOnlyMaskBySong[song_id][start_c:end_c + 1] = True

                w = 1.0
                if is_back:
                    w = self.backing_weight
                elif is_adlib:
                    w = self.adlib_weight

                # if multiple labels overlap in time, keep the max confidence weight
                self.baseWeight[song_id][start_c:end_c + 1] = np.maximum(self.baseWeight[song_id][start_c:end_c + 1], w)

        print(f"Group members inferred from labels: {self.group_members}")
        # 4) Derive vocal/overlap/transition masks
        for song_id, T in self._song_num_chunks.items():
            stacked = np.stack([self.memberMask[song_id][m] for m in self.group_members], axis=0)  # (M, T)
            counts = stacked.sum(axis=0)
            anyVocal = (counts > 0)
            self.vocalMask[song_id] = anyVocal
            overlap = (counts > 1)
            self.overlapMask[song_id] = overlap

            # transition: dilate boundaries by ±pad
            transition = np.zeros(T, dtype=bool)
            pad = self.transition_pad_chunks
            for (s, e) in boundaries.get(song_id, []):
                s2 = max(0, s - pad)
                e2 = min(T - 1, e + pad)
                # mark only the band near edges (not the whole interval)
                transition[s2:min(T, s + pad + 1)] = True
                transition[max(0, e - pad):e2 + 1] = True
            self.transitionMask[song_id] = transition
            
            # ----------------------------
            # Song complexity stats
            # ----------------------------
            totalChunks = int(T)

            # presence matrix: (T, C) float
            # stacked is (C, T) bool, so transpose
            presence = stacked.T.astype(np.float32)  # (T, C)

            # vocal masks already computed
            vocalMask = anyVocal  # (T,) bool

            # any-vocal rate (what you meant by vocalRate)
            vocalChunks = int(np.count_nonzero(vocalMask))
            vocalRate = float(vocalChunks / max(1, totalChunks))

            # solo rate (extra signal you wanted)
            soloChunks = int(np.count_nonzero(counts == 1))
            soloRate = float(soloChunks / max(1, totalChunks))

            # overlap rate
            overlapChunks = int(np.count_nonzero(overlap))
            overlapRate = float(overlapChunks / max(1, totalChunks))

            # adlib/backing-only rates from masks we built while reading labels
            adlibChunks = int(np.count_nonzero(adlibMaskBySong[song_id]))
            adlibRate = float(adlibChunks / max(1, totalChunks))

            backingOnlyChunks = int(np.count_nonzero(backingOnlyMaskBySong[song_id]))
            backingOnlyRate = float(backingOnlyChunks / max(1, totalChunks))

            # switch rate (XOR-like: did the active-singer set change from t-1 to t?)
            if T <= 1:
                changeCount = 0
                switchRate = 0.0
            else:
                frameChanges = np.any(stacked[:, 1:] != stacked[:, :-1], axis=0)  # (T-1,)
                changeCount = int(np.count_nonzero(frameChanges))
                switchRate = float(changeCount / max(1, totalChunks - 1))

            # domFrac computed on vocal frames only
            domFrac = 0.0
            vocalFrames = presence[vocalMask]  # (Tv, C)
            if vocalFrames.shape[0] > 0:
                perSinger = vocalFrames.mean(axis=0)  # (C,)
                denom = float(perSinger.sum()) + 1e-8
                domFrac = float(perSinger.max() / denom)

            self.song_stats[song_id] = {
                "T": totalChunks,

                # rates
                "vocalRate": vocalRate,           # any-vocal fraction
                "soloRate": soloRate,             # solo-only fraction
                "overlapRate": overlapRate,
                "adlibRate": adlibRate,
                "backingOnlyRate": backingOnlyRate,
                "switchRate": switchRate,
                "domFrac": domFrac,

                # optional counts (still useful for debugging)
                "vocalChunks": vocalChunks,
                "soloChunks": soloChunks,
                "overlapChunks": overlapChunks,
                "adlibChunks": adlibChunks,
                "backingOnlyChunks": backingOnlyChunks,
                "changeCount": changeCount,
            }

        # 5) Build anchor list (song_id, center_chunk)
        anchors = []
        for song_id, T in self._song_num_chunks.items():
            # You can anchor on all chunks, or only on vocal/silence subsets.
            # Anchoring on all chunks is simplest; sampling logic chooses pos/neg.
            song_anchors = [(song_id, c) for c in range(T)]
            anchors.extend(song_anchors)

        random.shuffle(anchors)
        self.anchors = anchors
        self._compute_song_complexities()
        
    # ---------------------------
    # Sampling policy
    # ---------------------------
    def _sample_center_for_member(self, song_id: str, memberName: str, want_pos: bool):
        """
        Pick a center *chunk index* to sample a training window around for ONE member's
        binary classifier, and return:

            (centerChunkIndex, label, sampleWeight)

        Where:
        - label = 1.0 means "memberName is present at this center chunk"
        - label = 0.0 means "memberName is NOT present at this center chunk"
        - sampleWeight down-weights ambiguous or lower-confidence regions:
            * overlap chunks (multiple members singing) get overlap_weight
            * transition chunks (near boundaries) get transition_weight
            * backing/adlib weighting is baked into baseWeight per chunk
        """

        if memberName not in self.memberMask[song_id]:
            # Member isn't tracked for this song => can't reliably sample.
            # Return sentinel center=None plus a safe default.
            return None, 0.0, 1.0

        # --- Per-chunk boolean masks for THIS song ---
        memberIsActiveMask = self.memberMask[song_id][memberName]   # True where memberName sings
        anyVocalMask = self.vocalMask[song_id]                     # True where anyone sings (any member)
        isOverlapMask = self.overlapMask[song_id]                  # True where >=2 members sing
        isTransitionMask = self.transitionMask[song_id]            # True near segment boundaries (uncertain)
        perChunkBaseWeight = self.baseWeight[song_id]              # e.g., backing/adlib confidence weight

        # --- Build candidate sets (prefer "clean" chunks: not near transitions) ---
        # Positive candidates: member is active, and NOT in transition region.
        positiveCandidateMask = memberIsActiveMask & ~isTransitionMask

        # Negative candidates type A: member is NOT active, but *someone* is singing (other-member vocal).
        otherMemberVocalNegativeMask = (~memberIsActiveMask) & anyVocalMask & ~isTransitionMask

        # Negative candidates type B: "true silence" (nobody is singing).
        silenceNegativeMask = (~anyVocalMask) & ~isTransitionMask

        # ==========================
        # Sample a POSITIVE example
        # ==========================
        if want_pos:
            positiveCandidateIndices = np.flatnonzero(positiveCandidateMask)
            if positiveCandidateIndices.size == 0:
                # No positives available for this member in this song (or all were excluded by transitions).
                # Returning label=1.0 signals caller that we *wanted* positive but couldn't find one.
                return None, 1.0, 1.0

            centerChunkIndex = int(np.random.choice(positiveCandidateIndices))
            label = 1.0

            # Start from base weight (e.g., backing/adlib weighting) at this chunk.
            sampleWeight = float(perChunkBaseWeight[centerChunkIndex])

            # If multiple members are singing here, down-weight (less clean identity signal).
            if isOverlapMask[centerChunkIndex]:
                sampleWeight *= self.overlap_weight

            # If it's near a boundary, down-weight (can be consonants/hand-offs).
            # NOTE: we already excluded transitions from positiveCandidateMask, but keep this in case
            # you later change masks or add fallback sampling that allows transitions.
            if isTransitionMask[centerChunkIndex]:
                sampleWeight *= self.transition_weight

            return centerChunkIndex, label, sampleWeight

        # ==========================
        # Sample a NEGATIVE example
        # ==========================
        # Choose whether negatives come mostly from "other member is singing" (hard negatives)
        # or from "silence" (easy negatives).
        pickOtherMemberVocalNeg = (random.random() < self.p_other_member_neg)

        if pickOtherMemberVocalNeg:
            negativeCandidateIndices = np.flatnonzero(otherMemberVocalNegativeMask)
        else:
            negativeCandidateIndices = np.flatnonzero(silenceNegativeMask)

        if negativeCandidateIndices.size == 0:
            # Fallback: if the preferred negative pool is empty, allow ANY chunk where member isn't active.
            negativeCandidateIndices = np.flatnonzero(~memberIsActiveMask)
            if negativeCandidateIndices.size == 0:
                # Extremely degenerate case: member is active everywhere (or masks are broken).
                return None, 0.0, 1.0

        centerChunkIndex = int(np.random.choice(negativeCandidateIndices))
        label = 0.0

        # Negatives still get baseWeight (rarely matters, but consistent if you down-weight transitions globally).
        sampleWeight = float(perChunkBaseWeight[centerChunkIndex])

        # If it’s near a boundary, down-weight because “who is singing” can be ambiguous right at hand-offs.
        if isTransitionMask[centerChunkIndex]:
            sampleWeight *= self.transition_weight

        # Note: we do NOT down-weight overlap for negatives, because overlap just means
        # "someone else (not this member) is singing" — that's still a valid negative.
        return centerChunkIndex, label, sampleWeight

    def _get_resampler(self, sr_src: int):
        key = (sr_src, self.sr_out)
        if key not in self._resamplers:
            self._resamplers[key] = torchaudio.transforms.Resample(sr_src, self.sr_out)
        return self._resamplers[key]
    
    def _load_song_wave(self, path: str) -> torch.Tensor:
        """
        Load + mono + resample full song ONCE and cache it (LRU).

        Returns:
            Tensor shape (1, T_out) at self.sr_out
        """

        # ---- Already cached? Refresh LRU and return ----
        if path in self._wave_cache:
            wav = self._wave_cache.pop(path)
            self._wave_cache[path] = wav  # move to end (most recently used)
            return wav

        # ---- Load from disk (slow path) ----
        wav, sr_src = torchaudio.load(path)

        # Convert to mono
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Resample once
        if sr_src != self.sr_out:
            resampler = self._get_resampler(sr_src)
            wav = resampler(wav)

        wav = wav.contiguous()

        # ---- Store in LRU ----
        self._wave_cache[path] = wav

        # Evict oldest if over limit
        if len(self._wave_cache) > self._max_cached_songs:
            self._wave_cache.popitem(last=False)

        return wav
    
    # ---------------------------
    # Window loading
    # ---------------------------
    def _load_window(self, song_id: str, center_c: int):
        """
        Slice a fixed-length window centered at chunk index center_c.
        Uses cached full-song waveform.
        """

        path = self.training_songs[song_id]

        # Convert center chunk -> time
        center_ms = center_c * self.chunk_ms
        half_ms = (self.context_sec * 1000.0) / 2.0

        start_ms = max(0.0, center_ms - half_ms)
        start_sample = int(round((start_ms / 1000.0) * self.sr_out))
        window_length = self.samples_per_window  # precomputed in __init__

        # ---- Load full waveform from cache ----
        wav = self._load_song_wave(path)  # (1, T)

        end_sample = start_sample + window_length

        # ---- Slice ----
        if end_sample > wav.size(1):
            pad = end_sample - wav.size(1)
            chunk = torch.nn.functional.pad(
                wav[:, start_sample:], (0, pad)
            )
        else:
            chunk = wav[:, start_sample:end_sample]

        return chunk
    
    def _compute_song_complexities(self):
        """
        Compute a single scalar complexity score per song using self.song_stats.

        Saves:
            self.song_complexity: dict[song_id] -> float
        """
        self.song_complexity = {}
        
        for song_id, stats in self.song_stats.items():
            overlapRate = float(stats.get("overlapRate", 0.0))
            backingRate = float(stats.get("backingOnlyRate", 0.0))  # backing-only is what we want
            adlibRate = float(stats.get("adlibRate", 0.0))
            switchRate = float(stats.get("switchRate", stats.get("changeRate", 0.0)))
            domFrac = float(stats.get("domFrac", 0.0))

            score = (
                0.40 * overlapRate +
                0.20 * backingRate +
                0.15 * adlibRate +
                0.20 * switchRate +
                0.05 * (1.0 - domFrac)
            )
                
            self.song_complexity[song_id] = float(score)

    def get_song_complexity(self, song_name: str) -> float:
        if not hasattr(self, "song_complexity"):
            self._compute_song_complexities()
        return float(self.song_complexity.get(song_name, 0.0))   
    
    def get_all_songs_sorted_by_complexity(self):
        if not hasattr(self, "song_complexity"):
            self._compute_song_complexities()
        return sorted(self.song_complexity.items(), key=lambda kv: kv[1], reverse=True)
    
    def getNonSoloSongs(self, domFracSoloThreshold: float = 0.95):
        pairs = []
        for song, st in self.song_stats.items():
            if st.get("domFrac", 0.0) < domFracSoloThreshold:
                pairs.append((song, self.get_song_complexity(song)))
        return pairs

    def getSongsByDifficultyTiers(self, domFracSoloThreshold: float = 0.95):
        pairs = self.getNonSoloSongs(domFracSoloThreshold)
        pairs.sort(key=lambda p: p[1])  # sort by complexity

        n = len(pairs)
        if n < 3:
            return pairs, [], []

        cut1 = n // 3
        cut2 = (2 * n) // 3
        return pairs[:cut1], pairs[cut1:cut2], pairs[cut2:]

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
            "valSongs": sorted(valSongs, key=lambda t: (t[1], t[0])),
        }
        return valSongs, tierInfo
    