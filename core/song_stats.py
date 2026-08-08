import json
import os
import statistics

CHUNK_MS = 40  # matches CHUNK_DURATION in core/audio_processing.py

# Weighting for fairness only -- song length always sums every label unweighted.
# Kept separate from datasets/vocal_metadata.py's backing_weight/adlib_weight (0.5/0.8),
# which are training-loss weights for a different purpose.
BACKING_WEIGHT = 0.7
ADLIB_WEIGHT = 1.0

# Reserved pseudo-member names used by the labeling UI (gui/audio_tester.py's
# self.bannedNames) -- not real group members, so they must never enter
# per-member fairness stats. "Cut" marks edited-out footage (not sung time
# at all), so it's also excluded from song length.
GANG_VOCAL_NAME = "Gang Vocal"
CUT_NAME = "Cut"
NON_MEMBER_NAMES = {GANG_VOCAL_NAME, CUT_NAME}

SAVED_LABELS_DIR = "saved_labels"

# Bump whenever the stat-computation logic changes, so stale cached values
# (computed under old logic) get recomputed instead of trusted via mtime match.
_CACHE_VERSION = 2


def labelsPathFor(group, songName):
    return os.path.join(SAVED_LABELS_DIR, group, f"{songName}_labels.json")


def _cachePathFor(group):
    return os.path.join(SAVED_LABELS_DIR, group, "_song_stats_cache.json")


def loadRawLabels(group, songName):
    path = labelsPathFor(group, songName)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return []
    return data if isinstance(data, list) else []


def _chunkSpanSeconds(startChunk, endChunk):
    return max(0, int(endChunk) - int(startChunk) + 1) * CHUNK_MS / 1000.0


def computeSongLengthSeconds(labels):
    if not labels:
        return None
    total = 0.0
    for entry in labels:
        try:
            member, startChunk, endChunk, _, _ = entry
        except (ValueError, TypeError):
            continue
        if member == CUT_NAME:
            continue  # edited-out footage, not sung time
        total += _chunkSpanSeconds(startChunk, endChunk)
    return total if total > 0 else None


def _labelWeight(isBacking, isAdlib):
    if isBacking:
        return BACKING_WEIGHT
    if isAdlib:
        return ADLIB_WEIGHT
    return 1.0


def computeSongFairness(labels):
    """1 - pstdev/mean over each member's weighted seconds sung. Mirrors
    LineDistributionPanel.computeStats() in gui/line_distribution_panel.py,
    but backing lines are weighted at BACKING_WEIGHT instead of full value."""
    if not labels:
        return None

    memberSeconds = {}
    for entry in labels:
        try:
            member, startChunk, endChunk, isBacking, isAdlib = entry
        except (ValueError, TypeError):
            continue
        if member in NON_MEMBER_NAMES:
            continue  # "Gang Vocal"/"Cut" aren't real members -- would skew mean/stdev
        duration = _chunkSpanSeconds(startChunk, endChunk)
        weight = _labelWeight(bool(isBacking), bool(isAdlib))
        memberSeconds[member] = memberSeconds.get(member, 0.0) + duration * weight

    values = [v for v in memberSeconds.values() if v > 1e-6]
    if not values:
        return None
    if len(values) == 1:
        return 1.0

    meanS = statistics.mean(values)
    stdS = statistics.pstdev(values)
    if meanS <= 1e-9:
        return None

    return 1.0 - (stdS / meanS)


def _readCache(group):
    path = _cachePathFor(group)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _writeCache(group, cache):
    path = _cachePathFor(group)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, separators=(",", ":"))


def _computeEntry(group, songName):
    labels = loadRawLabels(group, songName)
    return {
        "length_seconds": computeSongLengthSeconds(labels),
        "fairness": computeSongFairness(labels),
    }


def _currentMtime(group, songName):
    path = labelsPathFor(group, songName)
    return os.path.getmtime(path) if os.path.exists(path) else None


def getGroupSongStats(group, songNames):
    """Returns {songName: {"length_seconds": float|None, "fairness": float|None}},
    recomputing (and persisting to disk) only songs whose labels file is new or
    has changed since the last cached mtime."""
    cache = _readCache(group)
    changed = False
    result = {}

    for songName in songNames:
        mtime = _currentMtime(group, songName)
        entry = cache.get(songName)
        stale = entry is None or entry.get("mtime") != mtime or entry.get("cacheVersion") != _CACHE_VERSION
        if stale:
            entry = {"mtime": mtime, "cacheVersion": _CACHE_VERSION, **_computeEntry(group, songName)}
            cache[songName] = entry
            changed = True
        result[songName] = {
            "length_seconds": entry.get("length_seconds"),
            "fairness": entry.get("fairness"),
        }

    if changed:
        _writeCache(group, cache)

    return result


def invalidateSongStats(group, songName):
    """Recompute a single song's stats immediately and persist to the cache,
    called right after its labels are saved so the picker doesn't need to wait
    for a lazy mtime check."""
    cache = _readCache(group)
    cache[songName] = {
        "mtime": _currentMtime(group, songName),
        "cacheVersion": _CACHE_VERSION,
        **_computeEntry(group, songName),
    }
    _writeCache(group, cache)
