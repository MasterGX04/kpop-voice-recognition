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
        self._max_cached_songs = 64   # 8–32 is typical. Tune based on RAM.
        self._wave_cache = OrderedDict()  # path -> waveform (1, T_out)
        self._resamplers = {}  # (sr_src, sr_out) -> Resample object
        
        # ----------------------------
        # Embedding cache
        # ----------------------------
        self.embeddingCacheRoot = os.path.join(cache_dir, "embedding_cache")
        os.makedirs(self.embeddingCacheRoot, exist_ok=True)

        # song_id -> np.memmap or np.ndarray
        self._song_emb_cache = OrderedDict()
        self._max_cached_embedding_songs = 16

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
        
    def _makeEmbeddingConfigTag(
        self,
        *,
        contextSec: float,
        srOut: int,
        encoderTag: str = "muq-large-msd-iter",
        pooling: str = "mean",
        pcaTag: str = "none",
    ):
        return f"sr{srOut}_ctx{contextSec:.2f}_{encoderTag}_{pooling}_{pcaTag}"


    def _getSongEmbeddingPaths(
        self,
        songId: str,
        *,
        contextSec: float,
        srOut: int,
        encoderTag: str = "muq-large-msd-iter",
        pooling: str = "mean",
        pcaTag: str = "none",
    ):
        cfg = self._makeEmbeddingConfigTag(
            contextSec=contextSec,
            srOut=srOut,
            encoderTag=encoderTag,
            pooling=pooling,
            pcaTag=pcaTag,
        )

        safeSong = songId.replace(os.sep, "_")
        base = os.path.join(self.embeddingCacheRoot, f"{safeSong}__{cfg}")

        return {
            "npy": base + ".npy",
            "meta": base + ".json",
        }
        
    @torch.no_grad()
    def _buildSongEmbeddingMatrix(
        self,
        songId: str,
        *,
        encoder,
        device,
        batchSize: int = 64,
        contextSec: float = None,
        encoderTag: str = "muq-large-msd-iter",
        pooling: str = "mean",
        pcaTag="pca256"
    ):
        print(F"Building song embedding matrix for {songId} with context {contextSec}s, encoder {encoderTag}, pooling {pooling}")
        if contextSec is None:
            contextSec = self.context_sec

        paths = self._getSongEmbeddingPaths(
            songId,
            contextSec=contextSec,
            srOut=self.sr_out,
            encoderTag=encoderTag,
            pooling=pooling,
            pcaTag=pcaTag,
        )

        if os.path.exists(paths["npy"]):
            return np.load(paths["npy"], mmap_mode="r")

        T = self._song_num_chunks[songId]
        embList = []

        oldContext = self.context_sec
        oldSamplesPerWindow = self.samples_per_window

        # temporarily switch context if needed
        self.context_sec = float(contextSec)
        self.samples_per_window = int(round(self.context_sec * self.sr_out))

        try:
            for start in range(0, T, batchSize):
                end = min(T, start + batchSize)

                wavBatch = []
                for center_c in range(start, end):
                    wav = self._load_window(songId, center_c)   # (1, samples)
                    wavBatch.append(wav)

                wavBatch = torch.stack(wavBatch, dim=0).to(device, non_blocking=True)

                if wavBatch.ndim == 3 and wavBatch.size(1) == 1:
                    wavBatch = wavBatch.squeeze(1)

                embMain, embCtx = encoder.encode_batch(wavBatch, ctx_frac=0.25)

                emb = self._l2Normalize(embMain) + self._l2Normalize(embCtx)
                emb = self._l2Normalize(emb)

                emb = emb.detach().cpu().numpy().astype(np.float32)

                # fix this depending on your wrapper output
                # example: pooled mean over token dimension
                if emb.ndim == 3:
                    emb = emb.mean(dim=1)

                emb = emb.detach().cpu().numpy()

                embList.append(emb.astype(np.float32))

        finally:
            self.context_sec = oldContext
            self.samples_per_window = oldSamplesPerWindow

        fullEmb = np.concatenate(embList, axis=0)
        np.save(paths["npy"], fullEmb)

        meta = {
            "songId": songId,
            "numChunks": int(fullEmb.shape[0]),
            "embDim": int(fullEmb.shape[1]),
            "contextSec": float(contextSec),
            "srOut": int(self.sr_out),
            "chunkMs": int(self.chunk_ms),
            "encoderTag": encoderTag,
            "pooling": pooling,
            "pcaTag": pcaTag,
        }

        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return np.load(paths["npy"], mmap_mode="r")

    def _loadSongEmbeddingMatrix(
        self,
        songId: str,
        *,
        encoder,
        device,
        batchSize: int = 64,
        contextSec: float = None,
        encoderTag: str = "muq-large-msd-iter",
        pooling: str = "mean",
        pcaTag: str = "none",
    ):
        cacheKey = (songId, contextSec or self.context_sec, self.sr_out, encoderTag, pooling, pcaTag)

        if cacheKey in self._song_emb_cache:
            arr = self._song_emb_cache.pop(cacheKey)
            self._song_emb_cache[cacheKey] = arr
            return arr

        arr = self._buildSongEmbeddingMatrix(
            songId,
            encoder=encoder,
            device=device,
            batchSize=batchSize,
            contextSec=contextSec,
            encoderTag=encoderTag,
            pooling=pooling,
            pcaTag=pcaTag,
        )

        self._song_emb_cache[cacheKey] = arr

        if len(self._song_emb_cache) > self._max_cached_embedding_songs:
            self._song_emb_cache.popitem(last=False)

        return arr
    
    def getEmbeddingForCenter(
        self,
        songId: str,
        centerChunk: int,
        *,
        encoder,
        device,
        batchSize: int = 64,
        contextSec: float = None,
        encoderTag: str = "muq-large-msd-iter",
        pooling: str = "mean",
        pca = None,
        pcaTag: str = "none",
    ):
        arr = self._loadSongEmbeddingMatrix(
            songId,
            encoder=encoder,
            device=device,
            batchSize=batchSize,
            contextSec=contextSec,
            encoderTag=encoderTag,
            pooling=pooling,
            pca=pca,
            pcaTag=pcaTag,
        )

        return torch.from_numpy(np.asarray(arr[centerChunk])).float()
    
    # ---------------------------
    # Build masks + anchors
    # ---------------------------
    def _build_from_labels(self):
        """
        Build all dataset structures from labeled training songs.

        Main steps:
        1. Discover how long each song is in chunk units.
        2. Initialize empty masks for members / weights / adlibs / backing vocals.
        3. Read JSON labels and fill the masks.
        4. Derive global masks like vocal / overlap / transition.
        5. Compute per-song stats for debugging and curriculum ideas.
        6. Build anchor list used for sampling.
        """
        self.song_stats = {}

        # Step 1: figure out how many chunks each song has
        self._discover_song_chunk_lengths()

        # Step 2: create empty arrays for all masks
        adlibMaskBySong, backingOnlyMaskBySong = self._initialize_empty_song_masks()

        # Step 3: read label JSON files and fill per-song masks
        boundaries = self._fill_masks_from_json_labels(
            adlibMaskBySong=adlibMaskBySong,
            backingOnlyMaskBySong=backingOnlyMaskBySong,
        )

        print(f"Group members inferred from labels: {self.group_members}")

        # Step 4 + 5: derive global masks and compute stats
        self._finalize_song_masks_and_stats(
            boundaries=boundaries,
            adlibMaskBySong=adlibMaskBySong,
            backingOnlyMaskBySong=backingOnlyMaskBySong,
        )
        
        # Step 6: Precompute which centers can be sampled depending on the current stage
        self._precompute_candidate_indices()

        # Step 6: build the list of anchor chunks used by sampling
        self._build_anchor_list()

        # Optional curriculum / difficulty scores
        self._compute_song_complexities()
    
    def _discover_song_chunk_lengths(self):
        """
        Determine how many chunk positions each song has.

        We use the audio file duration and convert it into chunk units based on
        self.chunk_ms. This gives us the timeline length T for each song.
        """
        for song_id, path in self.training_songs.items():
            info = torchaudio.info(path)

            # Convert audio length into milliseconds.
            # Resampling happens later when loading audio, so timeline length is
            # based on original duration, not sample rate after resampling.
            duration_ms = (info.num_frames / info.sample_rate) * 1000.0

            # Number of chunk positions in this song.
            T = int(np.ceil(duration_ms / self.chunk_ms))
            self._song_num_chunks[song_id] = max(T, 1)
    
    def _initialize_empty_song_masks(self):
        """
        Create empty per-song arrays before reading labels.

        Returns:
            adlibMaskBySong: song_id -> bool[T]
            backingOnlyMaskBySong: song_id -> bool[T]

        We keep these as temporary global masks because they are useful later for
        song-level stats and sampling logic.
        """
        adlibMaskBySong = {}
        backingOnlyMaskBySong = {}

        for song_id, T in self._song_num_chunks.items():
            # One boolean mask per member: True means that member is active on that chunk.
            self.memberMask[song_id] = {
                m: np.zeros(T, dtype=bool) for m in self.group_members
            }

            # Base training weight per chunk.
            self.baseWeight[song_id] = np.ones(T, dtype=np.float32)

            # Global song masks for special vocal types.
            adlibMaskBySong[song_id] = np.zeros(T, dtype=bool)
            backingOnlyMaskBySong[song_id] = np.zeros(T, dtype=bool)

        return adlibMaskBySong, backingOnlyMaskBySong
    
    def _fill_masks_from_json_labels(self, *, adlibMaskBySong, backingOnlyMaskBySong):
        """
        Read all JSON label files and use them to fill member activity masks,
        special vocal masks, and chunk weights.

        Returns:
            boundaries: dict[song_id] -> list[(start_c, end_c)]

        boundaries are saved so we can later build transition masks around
        segment edges.
        """
        boundaries = defaultdict(list)

        for json_path in self.json_files:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            song_id = os.path.basename(json_path).replace("_labels.json", "")
            if song_id not in self._song_num_chunks:
                continue

            T = self._song_num_chunks[song_id]

            for entry in data:
                member, start_c, end_c, is_back, is_adlib = entry

                # Skip labels for members we are not modeling.
                if member not in self.memberMask[song_id]:
                    continue

                start_c, end_c = self._sanitize_label_range(start_c, end_c, T)

                # Mark this member as active over the labeled interval.
                self.memberMask[song_id][member][start_c:end_c + 1] = True

                # Save boundaries so we can later mark transitions near edges.
                boundaries[song_id].append((start_c, end_c))

                # Update global adlib / backing-only masks.
                if is_adlib:
                    adlibMaskBySong[song_id][start_c:end_c + 1] = True

                if is_back and (not is_adlib):
                    # backing-only means backing vocal that is NOT also marked adlib
                    backingOnlyMaskBySong[song_id][start_c:end_c + 1] = True

                # Update training weights for this labeled region.
                weight = self._get_label_weight(is_back=is_back, is_adlib=is_adlib)

                # If multiple labels overlap, keep the maximum weight.
                self.baseWeight[song_id][start_c:end_c + 1] = np.maximum(
                    self.baseWeight[song_id][start_c:end_c + 1],
                    weight
                )

        return boundaries
    
    def _sanitize_label_range(self, start_c, end_c, T):
        """
        Clamp label boundaries into valid chunk range [0, T-1] and ensure
        start <= end.
        """
        start_c = int(max(0, min(T - 1, start_c)))
        end_c = int(max(0, min(T - 1, end_c)))

        if end_c < start_c:
            start_c, end_c = end_c, start_c

        return start_c, end_c
    
    def _sanitize_label_range(self, start_c, end_c, T):
        """
        Clamp label boundaries into valid chunk range [0, T-1] and ensure
        start <= end.
        """
        start_c = int(max(0, min(T - 1, start_c)))
        end_c = int(max(0, min(T - 1, end_c)))

        if end_c < start_c:
            start_c, end_c = end_c, start_c

        return start_c, end_c

    def _get_label_weight(self, *, is_back, is_adlib):
        """
        Choose the base training weight for a labeled interval.

        Priority:
        - backing uses backing_weight
        - adlib uses adlib_weight
        - otherwise default is 1.0
        """
        if is_back:
            return self.backing_weight
        elif is_adlib:
            return self.adlib_weight
        return 1.0

    def _finalize_song_masks_and_stats(
        self,
        *,
        boundaries,
        adlibMaskBySong,
        backingOnlyMaskBySong,
    ):
        """
        For each song:
        - derive vocal / overlap / transition masks
        - compute useful summary stats

        This keeps all 'post-processing' logic in one place after raw labels have
        already been loaded.
        """
        for song_id, T in self._song_num_chunks.items():
            stacked, counts, anyVocal, overlap = self._compute_basic_song_masks(song_id)

            self.vocalMask[song_id] = anyVocal
            self.overlapMask[song_id] = overlap
            self.transitionMask[song_id] = self._build_transition_mask(
                T=T,
                boundaries=boundaries.get(song_id, []),
            )

            self.song_stats[song_id] = self._compute_song_stats(
                song_id=song_id,
                T=T,
                stacked=stacked,
                counts=counts,
                anyVocal=anyVocal,
                overlap=overlap,
                adlibMask=adlibMaskBySong[song_id],
                backingOnlyMask=backingOnlyMaskBySong[song_id],
            )
            
    def _compute_basic_song_masks(self, song_id):
        """
        Build the stacked presence matrix for one song and derive basic vocal masks.

        Returns:
            stacked: (M, T) bool
            counts: (T,) int
            anyVocal: (T,) bool
            overlap: (T,) bool
        """
        stacked = np.stack(
            [self.memberMask[song_id][m] for m in self.group_members],
            axis=0
        )  # (M, T)

        counts = stacked.sum(axis=0)
        anyVocal = (counts > 0)
        overlap = (counts > 1)

        return stacked, counts, anyVocal, overlap
    
    def _build_transition_mask(self, *, T, boundaries):
        """
        Build a boolean mask that marks chunks near labeled segment edges.

        We do NOT mark the entire vocal segment as transition.
        We only mark a padded band around the start and end boundaries.
        """
        transition = np.zeros(T, dtype=bool)
        pad = self.transition_pad_chunks

        for (s, e) in boundaries:
            s2 = max(0, s - pad)
            e2 = min(T - 1, e + pad)

            # Left edge band near the start
            transition[s2:min(T, s + pad + 1)] = True

            # Right edge band near the end
            transition[max(0, e - pad):e2 + 1] = True

        return transition
    
    def _precompute_candidate_indices(self):
        """
        Precompute per-song, per-member candidate chunk indices for each training stage.

        Why this exists:
        - Sampling is called many times during training.
        - Rebuilding boolean masks and calling np.flatnonzero every time is wasteful.
        - These masks depend only on labels, so we can compute them once after
        _build_from_labels() has finished.

        We store pools for:
        - stage 1 clean positives / negatives
        - stage 2 included positives / negatives

        Naming idea:
        self.candidateIdx[song_id][memberName]["stage1"]["pos_clean"]
        self.candidateIdx[song_id][memberName]["stage1"]["neg_other_vocal"]
        self.candidateIdx[song_id][memberName]["stage1"]["neg_silence"]
        etc.
        """
        self.candidateIdx = {}

        for song_id in self._song_num_chunks:
            self.candidateIdx[song_id] = {}

            anyVocalMask = self.vocalMask[song_id]
            isOverlapMask = self.overlapMask[song_id]
            isTransitionMask = self.transitionMask[song_id]

            # Song-level silence pools do not depend on member
            cleanSilenceMask = (~anyVocalMask) & (~isTransitionMask)
            anySilenceMask = ~anyVocalMask

            for memberName in self.group_members:
                if memberName not in self.memberMask[song_id]:
                    continue

                memberIsActiveMask = self.memberMask[song_id][memberName]

                # -------------------------
                # Stage 1 = clean identity
                # -------------------------
                stage1PosCleanMask = memberIsActiveMask & (~isTransitionMask) & (~isOverlapMask)

                stage1NegOtherVocalMask = (
                    (~memberIsActiveMask) &
                    anyVocalMask &
                    (~isTransitionMask) &
                    (~isOverlapMask)
                )

                stage1NegSilenceMask = cleanSilenceMask

                # -------------------------
                # Stage 2 = include messy positives
                # -------------------------
                # Allow overlap positives, but still avoid transitions.
                stage2PosIncludedMask = memberIsActiveMask & (~isTransitionMask)

                # Hard negative: target absent, somebody else singing, transitions removed.
                stage2NegOtherVocalMask = (
                    (~memberIsActiveMask) &
                    anyVocalMask &
                    (~isTransitionMask)
                )

                # Silence remains useful in stage 2 too.
                stage2NegSilenceMask = cleanSilenceMask

                # Very loose fallbacks if clean pools are empty
                fallbackAnyPositiveMask = memberIsActiveMask
                fallbackAnyNegativeMask = ~memberIsActiveMask
                fallbackAnySilenceMask = anySilenceMask

                self.candidateIdx[song_id][memberName] = {
                    "stage1": {
                        "pos_clean": np.flatnonzero(stage1PosCleanMask),
                        "neg_other_vocal": np.flatnonzero(stage1NegOtherVocalMask),
                        "neg_silence": np.flatnonzero(stage1NegSilenceMask),
                    },
                    "stage2": {
                        "pos_included": np.flatnonzero(stage2PosIncludedMask),
                        "neg_other_vocal": np.flatnonzero(stage2NegOtherVocalMask),
                        "neg_silence": np.flatnonzero(stage2NegSilenceMask),
                    },
                    "fallback": {
                        "pos_any": np.flatnonzero(fallbackAnyPositiveMask),
                        "neg_any": np.flatnonzero(fallbackAnyNegativeMask),
                        "silence_any": np.flatnonzero(fallbackAnySilenceMask),
                    }
                }
    
    def _compute_song_stats(
        self,
        *,
        song_id,
        T,
        stacked,
        counts,
        anyVocal,
        overlap,
        adlibMask,
        backingOnlyMask,
    ):
        """
        Compute descriptive stats for one song.

        These stats are mainly useful for:
        - debugging label quality
        - curriculum / difficulty sampling
        - understanding how messy each song is
        """
        totalChunks = int(T)

        # Convert stacked from (C, T) bool -> (T, C) float for easier averaging.
        presence = stacked.T.astype(np.float32)

        vocalChunks = int(np.count_nonzero(anyVocal))
        vocalRate = float(vocalChunks / max(1, totalChunks))

        soloChunks = int(np.count_nonzero(counts == 1))
        soloRate = float(soloChunks / max(1, totalChunks))

        overlapChunks = int(np.count_nonzero(overlap))
        overlapRate = float(overlapChunks / max(1, totalChunks))

        adlibChunks = int(np.count_nonzero(adlibMask))
        adlibRate = float(adlibChunks / max(1, totalChunks))

        backingOnlyChunks = int(np.count_nonzero(backingOnlyMask))
        backingOnlyRate = float(backingOnlyChunks / max(1, totalChunks))

        if T <= 1:
            changeCount = 0
            switchRate = 0.0
        else:
            # True whenever the active singer set changes from one chunk to the next.
            frameChanges = np.any(stacked[:, 1:] != stacked[:, :-1], axis=0)
            changeCount = int(np.count_nonzero(frameChanges))
            switchRate = float(changeCount / max(1, totalChunks - 1))

        # Dominance fraction:
        # among vocal frames only, how concentrated is the activity toward the
        # most active singer?
        domFrac = 0.0
        vocalFrames = presence[anyVocal]
        if vocalFrames.shape[0] > 0:
            perSinger = vocalFrames.mean(axis=0)
            denom = float(perSinger.sum()) + 1e-8
            domFrac = float(perSinger.max() / denom)

        return {
            "T": totalChunks,

            "vocalRate": vocalRate,
            "soloRate": soloRate,
            "overlapRate": overlapRate,
            "adlibRate": adlibRate,
            "backingOnlyRate": backingOnlyRate,
            "switchRate": switchRate,
            "domFrac": domFrac,

            "vocalChunks": vocalChunks,
            "soloChunks": soloChunks,
            "overlapChunks": overlapChunks,
            "adlibChunks": adlibChunks,
            "backingOnlyChunks": backingOnlyChunks,
            "changeCount": changeCount,
        }
        
    def _build_anchor_list(self):
        """
        Build the master list of anchor positions used by sampling.

        Each anchor is just (song_id, center_chunk). Later sampling logic decides
        whether that anchor becomes a positive or negative example for a member.
        """
        anchors = []

        for song_id, T in self._song_num_chunks.items():
            song_anchors = [(song_id, c) for c in range(T)]
            anchors.extend(song_anchors)

        random.shuffle(anchors)
        self.anchors = anchors
    
    def buildStage1ExamplesByMember(
        self,
        *,
        allowedSongs=None,
        totalExamplesPerMember=None,
        negOtherFrac: float = 0.70,
        seed: int = 1337,
        maxWorkers: int = 8,
    ):
        """
        Build fixed stage-1 examples grouped by member.

        Stage-1 policy:
        - positives come only from stage1/pos_clean
        - negatives are split between:
            - stage1/neg_other_vocal
            - stage1/neg_silence
        - examples are sampled WITHOUT replacement
        - overall ratio is 50% positive / 50% negative
        - within negatives, target split is negOtherFrac / (1-negOtherFrac)

        Returns:
            examplesByMember: dict[str, list[dict]]

            Each example dict contains:
                songId, centerChunk, memberName, label, weight, source
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm
        import math

        if allowedSongs is None:
            allowedSongs = set(self._song_num_chunks.keys())
        else:
            allowedSongs = set(allowedSongs)

        if not (0.0 < negOtherFrac < 1.0):
            raise ValueError("negOtherFrac must be between 0 and 1.")

        negSilFrac = 1.0 - negOtherFrac

        masterRng = np.random.default_rng(seed)
        memberSeeds = {
            memberName: int(masterRng.integers(0, 2**31 - 1))
            for memberName in self.group_members
        }

        def collectPoolEntries(memberName: str):
            """
            Collect all clean stage-1 candidate examples for one member across the
            allowed song set.
            """
            posPool = []
            negOtherPool = []
            negSilPool = []

            for song_id in allowedSongs:
                if song_id not in self.candidateIdx:
                    continue
                if memberName not in self.candidateIdx[song_id]:
                    continue

                stage1Pools = self.candidateIdx[song_id][memberName]["stage1"]

                for c in stage1Pools["pos_clean"]:
                    c = int(c)
                    posPool.append({
                        "songId": song_id,
                        "centerChunk": c,
                        "memberName": memberName,
                        "label": 1.0,
                        "weight": float(self.baseWeight[song_id][c]),
                        "source": "pos_clean",
                    })

                for c in stage1Pools["neg_other_vocal"]:
                    c = int(c)
                    negOtherPool.append({
                        "songId": song_id,
                        "centerChunk": c,
                        "memberName": memberName,
                        "label": 0.0,
                        "weight": float(self.baseWeight[song_id][c]),
                        "source": "neg_other_vocal",
                    })

                for c in stage1Pools["neg_silence"]:
                    c = int(c)
                    negSilPool.append({
                        "songId": song_id,
                        "centerChunk": c,
                        "memberName": memberName,
                        "label": 0.0,
                        "weight": float(self.baseWeight[song_id][c]),
                        "source": "neg_silence",
                    })

            return posPool, negOtherPool, negSilPool

        def chooseBalancedCounts(posAvail, negOtherAvail, negSilAvail, totalExamplesCap):
            """
            Decide how many examples to use so that:
            - positives == negatives
            - negatives are split ~ negOtherFrac / negSilFrac
            - sampling is without replacement
            """
            maxPos = posAvail

            # If total negatives = nPos, then:
            # nOther ≈ nPos * negOtherFrac
            # nSil   ≈ nPos * negSilFrac
            #
            # Need both buckets to support that split.
            maxByOther = int(math.floor(negOtherAvail / negOtherFrac)) if negOtherFrac > 0 else 10**18
            maxBySil = int(math.floor(negSilAvail / negSilFrac)) if negSilFrac > 0 else 10**18
            maxNegBalanced = min(maxByOther, maxBySil)

            nPos = min(maxPos, maxNegBalanced)

            if totalExamplesCap is not None:
                # total = pos + neg = 2*nPos
                nPos = min(nPos, totalExamplesCap // 2)

            while nPos > 0:
                nNeg = nPos
                nOther = int(round(nNeg * negOtherFrac))
                nSil = nNeg - nOther

                if nOther <= negOtherAvail and nSil <= negSilAvail:
                    return nPos, nOther, nSil

                nPos -= 1

            return 0, 0, 0

        def buildForMember(memberName: str):
            """
            Build the final fixed example list for one member.
            """
            rng = np.random.default_rng(memberSeeds[memberName])

            posPool, negOtherPool, negSilPool = collectPoolEntries(memberName)

            posAvail = len(posPool)
            negOtherAvail = len(negOtherPool)
            negSilAvail = len(negSilPool)

            nPos, nOther, nSil = chooseBalancedCounts(
                posAvail=posAvail,
                negOtherAvail=negOtherAvail,
                negSilAvail=negSilAvail,
                totalExamplesCap=totalExamplesPerMember,
            )

            if nPos == 0:
                return {
                    "memberName": memberName,
                    "examples": [],
                    "stats": {
                        "posAvail": posAvail,
                        "negOtherAvail": negOtherAvail,
                        "negSilAvail": negSilAvail,
                        "usedPos": 0,
                        "usedNegOther": 0,
                        "usedNegSil": 0,
                    },
                }

            posIdx = rng.choice(posAvail, size=nPos, replace=False)
            negOtherIdx = rng.choice(negOtherAvail, size=nOther, replace=False)
            negSilIdx = rng.choice(negSilAvail, size=nSil, replace=False)

            examples = [posPool[i] for i in posIdx]
            examples.extend(negOtherPool[i] for i in negOtherIdx)
            examples.extend(negSilPool[i] for i in negSilIdx)

            rng.shuffle(examples)

            return {
                "memberName": memberName,
                "examples": examples,
                "stats": {
                    "posAvail": posAvail,
                    "negOtherAvail": negOtherAvail,
                    "negSilAvail": negSilAvail,
                    "usedPos": nPos,
                    "usedNegOther": nOther,
                    "usedNegSil": nSil,
                },
            }

        examplesByMember = {}
        memberStats = {}

        workers = max(1, min(maxWorkers, len(self.group_members)))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {
                ex.submit(buildForMember, memberName): memberName
                for memberName in self.group_members
            }

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Building stage-1 examples by member"):
                result = fut.result()
                memberName = result["memberName"]
                examplesByMember[memberName] = result["examples"]
                memberStats[memberName] = result["stats"]

        print("\nStage-1 example summary by member:")
        for memberName in self.group_members:
            st = memberStats.get(memberName, {})
            print(
                f"{memberName}: "
                f"posAvail={st.get('posAvail', 0)}, "
                f"negOtherAvail={st.get('negOtherAvail', 0)}, "
                f"negSilAvail={st.get('negSilAvail', 0)}, "
                f"usedPos={st.get('usedPos', 0)}, "
                f"usedNegOther={st.get('usedNegOther', 0)}, "
                f"usedNegSil={st.get('usedNegSil', 0)}, "
                f"total={len(examplesByMember.get(memberName, []))}"
            )

        return examplesByMember
      
    # ---------------------------
    # Sampling policy
    # ---------------------------
    def _sample_center_for_member(self, song_id: str, memberName: str, want_pos: bool, stage: int = 1):
        """
        Pick a center chunk index for one member's binary classifier.

        Returns:
            (centerChunkIndex, label, sampleWeight)

        label meanings:
        - 1.0 => memberName is present at this center chunk
        - 0.0 => memberName is NOT present at this center chunk

        stage meanings:
        - stage 1:
            Learn clean vocal identity only.
            Positives are solo, non-transition chunks.
            Negatives are clean other-member vocal chunks or silence.
        - stage 2:
            Learn robustness to messier audio.
            Positives may include overlap / included-presence chunks,
            but still avoid transition regions by default.

        Notes:
        - This function assumes candidate pools were already precomputed by
        _precompute_candidate_indices().
        - baseWeight already includes backing/adlib confidence scaling.
        """
        if memberName not in self.memberMask[song_id]:
            return None, 0.0, 1.0

        if stage not in (1, 2):
            raise ValueError(f"Unsupported stage={stage}. Expected 1 or 2.")

        perChunkBaseWeight = self.baseWeight[song_id]
        isOverlapMask = self.overlapMask[song_id]
        isTransitionMask = self.transitionMask[song_id]

        pools = self.candidateIdx[song_id][memberName]
        stageKey = f"stage{stage}"

        # ==========================
        # POSITIVE sampling
        # ==========================
        if want_pos:
            if stage == 1:
                positiveCandidateIndices = pools[stageKey]["pos_clean"]
            else:
                positiveCandidateIndices = pools[stageKey]["pos_included"]

            if positiveCandidateIndices.size == 0:
                # Fallback to any positive chunk for this member
                positiveCandidateIndices = pools["fallback"]["pos_any"]
                if positiveCandidateIndices.size == 0:
                    return None, 1.0, 1.0

            centerChunkIndex = int(np.random.choice(positiveCandidateIndices))
            label = 1.0

            sampleWeight = float(perChunkBaseWeight[centerChunkIndex])

            # In stage 2, overlap positives are valid but lower confidence.
            if isOverlapMask[centerChunkIndex]:
                sampleWeight *= self.overlap_weight

            # We usually exclude transitions from candidate pools, but keep this
            # as a safety guard for fallback cases.
            if isTransitionMask[centerChunkIndex]:
                sampleWeight *= self.transition_weight

            return centerChunkIndex, label, sampleWeight

        # ==========================
        # NEGATIVE sampling
        # ==========================
        pickOtherMemberVocalNeg = (random.random() < self.p_other_member_neg)

        if pickOtherMemberVocalNeg:
            negativeCandidateIndices = pools[stageKey]["neg_other_vocal"]
        else:
            negativeCandidateIndices = pools[stageKey]["neg_silence"]

        if negativeCandidateIndices.size == 0:
            # Prefer silence fallback if we specifically wanted silence
            if not pickOtherMemberVocalNeg:
                negativeCandidateIndices = pools["fallback"]["silence_any"]

            # Final fallback: any chunk where this member is absent
            if negativeCandidateIndices.size == 0:
                negativeCandidateIndices = pools["fallback"]["neg_any"]

            if negativeCandidateIndices.size == 0:
                return None, 0.0, 1.0

        centerChunkIndex = int(np.random.choice(negativeCandidateIndices))
        label = 0.0

        sampleWeight = float(perChunkBaseWeight[centerChunkIndex])

        # Transition negatives are lower confidence if a fallback ever lands there.
        if isTransitionMask[centerChunkIndex]:
            sampleWeight *= self.transition_weight

        # For negatives, overlap is still a valid negative if target is absent.
        # We do not need to down-weight it by default.
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
    