import os
import json
import numpy as np
import torchaudio
from collections import defaultdict

class VocalMetadataManager:
    """
    Handles JSON label parsing, mask generation, candidate precomputation,
    and song complexity analytics.
    """
    def __init__(
        self,
        json_dir,
        training_songs,
        chunk_ms=40,
        transition_pad_chunks=10,
        overlap_weight=0.4,
        transition_weight=0.3,
        backing_weight=0.5,
        adlib_weight=0.8,
        p_other_member_neg=0.8,
        p_pos=0.5
    ):
        self.chunk_ms = chunk_ms
        self.transition_pad_chunks = transition_pad_chunks
        self.overlap_weight = overlap_weight
        self.transition_weight = transition_weight
        self.backing_weight = backing_weight
        self.adlib_weight = adlib_weight
        self.p_other_member_neg = p_other_member_neg
        self.p_pos = p_pos

        self.json_files = self._get_json_files_for_group(json_dir)
        self.group_members = self._infer_group_members_from_first_json()
        
        # Filter training songs to only those with labels
        valid_labeled_songs = {os.path.basename(j).replace("_labels.json", "") for j in self.json_files}
        self.training_songs = {}
        for song_id, path in training_songs.items():
            if song_id in valid_labeled_songs:
                self.training_songs[song_id] = path
            else:
                print(f"[Warning] Excluding '{song_id}' from training: No label JSON found.")

        # ====== per-song structures ======
        self._song_num_chunks = {}
        self.memberMask = {}
        self.vocalMask = {}
        self.overlapMask = {}
        self.transitionMask = {}
        self.baseWeight = {}
        self.song_stats = {}
        self.candidateIdx = {}
        self.song_complexity = {}

        # Build everything immediately upon initialization
        self._build_from_labels()

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
        """
        if not self.json_files:
            raise ValueError("json_files is empty — cannot infer members.")

        first_json_path = self.json_files[0]
        print(f"First json path: {first_json_path}")
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
            raise RuntimeError(f"No members found in label file: {first_json_path}")

        return members

    def _build_from_labels(self):
        """
        Build all dataset structures from labeled training songs.
        """
        self.song_stats = {}

        # 1. figure out how many chunks each song has
        self._discover_song_chunk_lengths()

        # 2. create empty arrays for all masks
        adlibMaskBySong, backingOnlyMaskBySong = self._initialize_empty_song_masks()

        # 3. read label JSON files and fill per-song masks
        boundaries = self._fill_masks_from_json_labels(
            adlibMaskBySong=adlibMaskBySong,
            backingOnlyMaskBySong=backingOnlyMaskBySong,
        )

        print(f"Group members inferred from labels: {self.group_members}")

        # 4 + 5. derive global masks and compute stats
        self._finalize_song_masks_and_stats(
            boundaries=boundaries,
            adlibMaskBySong=adlibMaskBySong,
            backingOnlyMaskBySong=backingOnlyMaskBySong,
        )

        # 6. Precompute candidate indices
        self._precompute_candidate_indices()

        # 7. Compute complexities
        self._compute_song_complexities()

    def _discover_song_chunk_lengths(self):
        for song_id, path in self.training_songs.items():
            info = torchaudio.info(path)
            duration_ms = (info.num_frames / info.sample_rate) * 1000.0
            T = int(np.ceil(duration_ms / self.chunk_ms))
            self._song_num_chunks[song_id] = max(T, 1)

    def _initialize_empty_song_masks(self):
        adlibMaskBySong = {}
        backingOnlyMaskBySong = {}

        for song_id, T in self._song_num_chunks.items():
            self.memberMask[song_id] = {m: np.zeros(T, dtype=bool) for m in self.group_members}
            self.baseWeight[song_id] = np.ones(T, dtype=np.float32)
            adlibMaskBySong[song_id] = np.zeros(T, dtype=bool)
            backingOnlyMaskBySong[song_id] = np.zeros(T, dtype=bool)

        return adlibMaskBySong, backingOnlyMaskBySong

    def _fill_masks_from_json_labels(self, *, adlibMaskBySong, backingOnlyMaskBySong):
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

                if member not in self.memberMask[song_id]:
                    continue

                start_c, end_c = self._sanitize_label_range(start_c, end_c, T)
                self.memberMask[song_id][member][start_c:end_c + 1] = True
                boundaries[song_id].append((start_c, end_c))

                if is_adlib:
                    adlibMaskBySong[song_id][start_c:end_c + 1] = True

                if is_back and (not is_adlib):
                    backingOnlyMaskBySong[song_id][start_c:end_c + 1] = True

                weight = self._get_label_weight(is_back=is_back, is_adlib=is_adlib)
                self.baseWeight[song_id][start_c:end_c + 1] = np.maximum(
                    self.baseWeight[song_id][start_c:end_c + 1], weight
                )

        return boundaries

    def _sanitize_label_range(self, start_c, end_c, T):
        start_c = int(max(0, min(T - 1, start_c)))
        end_c = int(max(0, min(T - 1, end_c)))
        if end_c < start_c:
            start_c, end_c = end_c, start_c
        return start_c, end_c

    def _get_label_weight(self, *, is_back, is_adlib):
        if is_back:
            return self.backing_weight
        elif is_adlib:
            return self.adlib_weight
        return 1.0

    def _finalize_song_masks_and_stats(self, *, boundaries, adlibMaskBySong, backingOnlyMaskBySong):
        for song_id, T in self._song_num_chunks.items():
            stacked, counts, anyVocal, overlap = self._compute_basic_song_masks(song_id)

            self.vocalMask[song_id] = anyVocal
            self.overlapMask[song_id] = overlap
            self.transitionMask[song_id] = self._build_transition_mask(
                T=T, boundaries=boundaries.get(song_id, [])
            )

            self.song_stats[song_id] = self._compute_song_stats(
                song_id=song_id, T=T, stacked=stacked, counts=counts,
                anyVocal=anyVocal, overlap=overlap,
                adlibMask=adlibMaskBySong[song_id],
                backingOnlyMask=backingOnlyMaskBySong[song_id],
            )

    def _compute_basic_song_masks(self, song_id):
        stacked = np.stack([self.memberMask[song_id][m] for m in self.group_members], axis=0)
        counts = stacked.sum(axis=0)
        anyVocal = (counts > 0)
        overlap = (counts > 1)
        return stacked, counts, anyVocal, overlap

    def _build_transition_mask(self, *, T, boundaries):
        transition = np.zeros(T, dtype=bool)
        pad = self.transition_pad_chunks
        for (s, e) in boundaries:
            transition[max(0, s - pad):min(T, s + pad + 1)] = True
            transition[max(0, e - pad):min(T - 1, e + pad) + 1] = True
        return transition

    def _precompute_candidate_indices(self):
        self.candidateIdx = {}
        for song_id in self._song_num_chunks:
            self.candidateIdx[song_id] = {}
            anyVocalMask = self.vocalMask[song_id]
            isOverlapMask = self.overlapMask[song_id]
            isTransitionMask = self.transitionMask[song_id]

            cleanSilenceMask = (~anyVocalMask) & (~isTransitionMask)
            anySilenceMask = ~anyVocalMask

            for memberName in self.group_members:
                if memberName not in self.memberMask[song_id]:
                    continue

                memberIsActiveMask = self.memberMask[song_id][memberName]

                stage1PosCleanMask = memberIsActiveMask & (~isTransitionMask) & (~isOverlapMask)
                stage1NegOtherVocalMask = (~memberIsActiveMask) & anyVocalMask & (~isTransitionMask) & (~isOverlapMask)
                
                stage2PosIncludedMask = memberIsActiveMask & (~isTransitionMask)
                stage2NegOtherVocalMask = (~memberIsActiveMask) & anyVocalMask & (~isTransitionMask)

                self.candidateIdx[song_id][memberName] = {
                    "stage1": {
                        "pos_clean": np.flatnonzero(stage1PosCleanMask),
                        "neg_other_vocal": np.flatnonzero(stage1NegOtherVocalMask),
                        "neg_silence": np.flatnonzero(cleanSilenceMask),
                    },
                    "stage2": {
                        "pos_included": np.flatnonzero(stage2PosIncludedMask),
                        "neg_other_vocal": np.flatnonzero(stage2NegOtherVocalMask),
                        "neg_silence": np.flatnonzero(cleanSilenceMask),
                    },
                    "fallback": {
                        "pos_any": np.flatnonzero(memberIsActiveMask),
                        "neg_any": np.flatnonzero(~memberIsActiveMask),
                        "silence_any": np.flatnonzero(anySilenceMask),
                    }
                }

    def _compute_song_stats(self, *, song_id, T, stacked, counts, anyVocal, overlap, adlibMask, backingOnlyMask):
        totalChunks = int(T)
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
            changeCount, switchRate = 0, 0.0
        else:
            frameChanges = np.any(stacked[:, 1:] != stacked[:, :-1], axis=0)
            changeCount = int(np.count_nonzero(frameChanges))
            switchRate = float(changeCount / max(1, totalChunks - 1))

        domFrac = 0.0
        vocalFrames = presence[anyVocal]
        if vocalFrames.shape[0] > 0:
            perSinger = vocalFrames.mean(axis=0)
            domFrac = float(perSinger.max() / (float(perSinger.sum()) + 1e-8))

        return {
            "T": totalChunks, "vocalRate": vocalRate, "soloRate": soloRate,
            "overlapRate": overlapRate, "adlibRate": adlibRate, "backingOnlyRate": backingOnlyRate,
            "switchRate": switchRate, "domFrac": domFrac, "vocalChunks": vocalChunks,
            "soloChunks": soloChunks, "overlapChunks": overlapChunks, "adlibChunks": adlibChunks,
            "backingOnlyChunks": backingOnlyChunks, "changeCount": changeCount,
        }

    def _compute_song_complexities(self):
        for song_id, stats in self.song_stats.items():
            overlapRate = float(stats.get("overlapRate", 0.0))
            backingRate = float(stats.get("backingOnlyRate", 0.0))
            adlibRate = float(stats.get("adlibRate", 0.0))
            switchRate = float(stats.get("switchRate", stats.get("changeRate", 0.0)))
            domFrac = float(stats.get("domFrac", 0.0))

            score = 0.40 * overlapRate + 0.20 * backingRate + 0.15 * adlibRate + 0.20 * switchRate + 0.05 * (1.0 - domFrac)
            self.song_complexity[song_id] = float(score)

    def get_song_complexity(self, song_name: str) -> float:
        return float(self.song_complexity.get(song_name, 0.0))   

    def get_all_songs_sorted_by_complexity(self):
        return sorted(self.song_complexity.items(), key=lambda kv: kv[1], reverse=True)

    def getNonSoloSongs(self, domFracSoloThreshold: float = 0.95):
        pairs = []
        for song, st in self.song_stats.items():
            if st.get("domFrac", 0.0) < domFracSoloThreshold:
                pairs.append((song, self.get_song_complexity(song)))
        return pairs

    def getSongsByDifficultyTiers(self, domFracSoloThreshold: float = 0.95):
        pairs = self.getNonSoloSongs(domFracSoloThreshold)
        pairs.sort(key=lambda p: p[1])

        n = len(pairs)
        if n < 3: return pairs, [], []
        cut1, cut2 = n // 3, (2 * n) // 3
        return pairs[:cut1], pairs[cut1:cut2], pairs[cut2:]

    def pickValidationSongsStratified(self, valEasy: int = 2, valMed: int = 2, valHard: int = 1, domFracSoloThreshold: float = 0.95):
        easy, med, hard = self.getSongsByDifficultyTiers(domFracSoloThreshold=domFracSoloThreshold)
        
        def takeBottom(tierList, k): return tierList[:min(k, len(tierList))]

        valSongs = set()
        valSongs.update(takeBottom(easy, valEasy))
        valSongs.update(takeBottom(med, valMed))
        valSongs.update(takeBottom(hard, valHard))

        tierInfo = {
            "easy": easy, "med": med, "hard": hard,
            "valSongs": sorted(valSongs, key=lambda t: (t[1], t[0])),
        }
        return valSongs, tierInfo