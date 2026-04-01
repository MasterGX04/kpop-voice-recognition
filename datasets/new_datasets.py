import os, json, random, glob

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset
import torch.nn.functional as F
from .audio_handler import AudioHandler
from .vocal_metadata import VocalMetadataManager
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
        transition_pad_chunks=10, # ±10 chunks (400ms) near boundaries
        overlap_weight=0.4,
        transition_weight=0.3,
        backing_weight=0.5,
        adlib_weight=0.8,
        p_other_member_neg=0.8,  # of negatives, percent that are other-member vocal
        p_pos=0.5, # balanced sampling per call
    ):
        raw_training_songs = self._get_wav_files_from_cache(cache_dir)
        self.sr_out = int(sr_out)
        self.context_sec = float(context_sec)
        self.chunk_ms = int(chunk_ms)

        self.samples_per_window = int(round(self.context_sec * self.sr_out))        

        # ----------------------------
        # Audio LRU cache
        # ----------------------------
        self.audio_handler = AudioHandler(
            sr_out=self.sr_out, 
            max_cached_songs=64 # You can pass this via kwargs if you want
        )
        
        # Get raw WAV files
        raw_training_songs = self._get_wav_files_from_cache(cache_dir)

        # Metadata Manager Initialization (Handles JSONs, masks, filtering, etc.)
        self.metadata = VocalMetadataManager(
            json_dir=json_dir,
            training_songs=raw_training_songs,
            chunk_ms=chunk_ms,
            transition_pad_chunks=transition_pad_chunks,
            overlap_weight=overlap_weight,
            transition_weight=transition_weight,
            backing_weight=backing_weight,
            adlib_weight=adlib_weight,
            p_other_member_neg=p_other_member_neg,
            p_pos=p_pos
        )
        
        # Pull processed properties back up for easy access
        self.training_songs = self.metadata.training_songs
        self.group_members = self.metadata.group_members
        
        sortedSongs = sorted(self.metadata.song_complexity.items(), key=lambda x: x[1])
        print("Easiest:", sortedSongs[:5])
        print("Hardest:", sortedSongs[-5:])

        # Build Anchor List for Dataset __len__
        self.anchors = []
        self._build_anchor_list()
        
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

    def _l2Normalize(self, x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return F.normalize(x, p=2, dim=-1, eps=eps)

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
        from tqdm import tqdm

        print(f"Building song embedding matrix for {songId} with context {contextSec}s, encoder {encoderTag}, pooling {pooling}")
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

        T = self.metadata._song_num_chunks[songId]
        embList = []

        oldContext = self.context_sec
        oldSamplesPerWindow = self.samples_per_window

        # temporarily switch context if needed
        self.context_sec = float(contextSec)
        self.samples_per_window = int(round(self.context_sec * self.sr_out))

        try:
            for start in tqdm(range(0, T, batchSize), desc=f"Encoding {songId}", unit="batch", leave=False):
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

                emb_np = emb.detach().cpu().numpy().astype(np.float32)

                embList.append(emb_np)

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
     
    def _load_window(self, song_id: str, center_c: int):
        """
        Wrapper that passes dataset parameters to the AudioHandler.
        """
        path = self.training_songs[song_id]
        
        return self.audio_handler.load_window(
            path=path,
            center_c=center_c,
            chunk_ms=self.chunk_ms,
            context_sec=self.context_sec,
            samples_per_window=self.samples_per_window
        )
  
    def _build_anchor_list(self):
        anchors = []
        for song_id, T in self.metadata._song_num_chunks.items():
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
            allowedSongs = set(self.metadata._song_num_chunks.keys())
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
                if song_id not in self.metadata.candidateIdx:
                    continue
                if memberName not in self.metadata.candidateIdx[song_id]:
                    continue

                stage1Pools = self.metadata.candidateIdx[song_id][memberName]["stage1"]

                for c in stage1Pools["pos_clean"]:
                    c = int(c)
                    posPool.append({
                        "songId": song_id,
                        "centerChunk": c,
                        "memberName": memberName,
                        "label": 1.0,
                        "weight": float(self.metadata.baseWeight[song_id][c]),
                        "source": "pos_clean",
                    })

                for c in stage1Pools["neg_other_vocal"]:
                    c = int(c)
                    negOtherPool.append({
                        "songId": song_id,
                        "centerChunk": c,
                        "memberName": memberName,
                        "label": 0.0,
                        "weight": float(self.metadata.baseWeight[song_id][c]),
                        "source": "neg_other_vocal",
                    })

                for c in stage1Pools["neg_silence"]:
                    c = int(c)
                    negSilPool.append({
                        "songId": song_id,
                        "centerChunk": c,
                        "memberName": memberName,
                        "label": 0.0,
                        "weight": float(self.metadata.baseWeight[song_id][c]),
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
        if memberName not in self.metadata.memberMask[song_id]:
            return None, 0.0, 1.0

        if stage not in (1, 2):
            raise ValueError(f"Unsupported stage={stage}. Expected 1 or 2.")

        perChunkBaseWeight = self.metadata.baseWeight[song_id]
        isOverlapMask = self.metadata.overlapMask[song_id]
        isTransitionMask = self.metadata.transitionMask[song_id]

        pools = self.metadata.candidateIdx[song_id][memberName]
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
                sampleWeight *= self.metadata.overlap_weight

            # We usually exclude transitions from candidate pools, but keep this
            # as a safety guard for fallback cases.
            if isTransitionMask[centerChunkIndex]:
                sampleWeight *= self.metadata.transition_weight

            return centerChunkIndex, label, sampleWeight

        # ==========================
        # NEGATIVE sampling
        # ==========================
        pickOtherMemberVocalNeg = (random.random() < self.metadata.p_other_member_neg)

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
            sampleWeight *= self.metadata.transition_weight

        # For negatives, overlap is still a valid negative if target is absent.
        # We do not need to down-weight it by default.
        return centerChunkIndex, label, sampleWeight

    def export_debug_samples(self, memberName: str, output_dir: str, num_samples: int = 10, stage: int = 1):
        """
        Randomly samples audio windows using the actual dataset logic and saves them
        to a directory for manual listening/verification.
        """
        os.makedirs(output_dir, exist_ok=True)
        print(f"Exporting debug samples for {memberName} to {output_dir}...")

        # We'll try to get 50% Positives and 50% Negatives
        counts = {"pos": 0, "neg": 0}
        target_per_type = num_samples // 2

        # To sample effectively, we'll iterate through songs
        all_songs = list(self.training_songs.keys())
        random.shuffle(all_songs)

        for song_id in all_songs:
            for want_pos in [True, False]:
                type_key = "pos" if want_pos else "neg"
                if counts[type_key] >= target_per_type:
                    continue

                # Use your REAL sampling logic
                center_c, label, weight = self._sample_center_for_member(
                    song_id, memberName, want_pos=want_pos, stage=stage
                )

                if center_c is None:
                    continue

                # Load the EXACT window the encoder sees
                audio_window = self._load_window(song_id, center_c) # (1, num_samples)
                
                # Filename: {member}_{label}_{weight}_{song}_{chunk}.wav
                clean_song_id = song_id.replace(" ", "_").replace("/", "_")
                filename = f"{type_key}_{memberName}_w{weight:.2f}_{clean_song_id}_c{center_c}.wav"
                save_path = os.path.join(output_dir, filename)

                torchaudio.save(save_path, audio_window, self.sr_out)
                counts[type_key] += 1

                if sum(counts.values()) >= num_samples:
                    break
            if sum(counts.values()) >= num_samples:
                break

        print(f"Done! Saved {counts['pos']} positives and {counts['neg']} negatives.")
        
    def pickValidationSongsStratified(self, **kwargs):
        return self.metadata.pickValidationSongsStratified(**kwargs)