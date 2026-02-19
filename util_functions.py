import subprocess
import json
import os, sys
from pathlib import Path
import re
from dataclasses import dataclass

def hexToRgb01(hexColor: str):
    hexColor = hexColor.strip().lstrip("#")
    if len(hexColor) == 3:
        hexColor = "".join([c * 2 for c in hexColor])
    r = int(hexColor[0:2], 16) / 255.0
    g = int(hexColor[2:4], 16) / 255.0
    b = int(hexColor[4:6], 16) / 255.0
    return r, g, b

def rgb01ToHex(rgb):
    r, g, b = rgb
    return "#{:02x}{:02x}{:02x}".format(
        int(round(max(0, min(1, r)) * 255)),
        int(round(max(0, min(1, g)) * 255)),
        int(round(max(0, min(1, b)) * 255)),
    )

def srgbToLinear(c: float) -> float:
    # WCAG / sRGB standard conversion
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def relativeLuminance(rgb01):
    r, g, b = rgb01
    rL = srgbToLinear(r)
    gL = srgbToLinear(g)
    bL = srgbToLinear(b)
    # WCAG coefficients
    return 0.2126 * rL + 0.7152 * gL + 0.0722 * bL

def contrastRatio(fgRgb01, bgRgb01):
    L1 = relativeLuminance(fgRgb01)
    L2 = relativeLuminance(bgRgb01)
    lighter = max(L1, L2)
    darker = min(L1, L2)
    return (lighter + 0.05) / (darker + 0.05)

def mix(rgbA, rgbB, t: float):
    # linear interpolation in sRGB space (good enough for UI)
    return (
        rgbA[0] * (1 - t) + rgbB[0] * t,
        rgbA[1] * (1 - t) + rgbB[1] * t,
        rgbA[2] * (1 - t) + rgbB[2] * t,
    )

def ensureReadableOnBackground(hexColor: str, bgHex: str = "#ffffff", minContrast: float = 4.5):
    """
    Returns a hex color adjusted (darkened toward black) until contrast vs bg meets minContrast.
    4.5 is WCAG-ish for normal text. If your text is big/bold, you can use 3.0.
    """
    fg = hexToRgb01(hexColor)
    bg = hexToRgb01(bgHex)

    if contrastRatio(fg, bg) >= minContrast:
        return hexColor if hexColor.startswith("#") else f"#{hexColor}"

    # Darken by blending toward black until the contrast passes.
    black = (0.0, 0.0, 0.0)

    lo, hi = 0.0, 1.0  # t in [0,1] where t=1 means fully black
    for _ in range(24):  # binary search for minimal darkening
        mid = (lo + hi) / 2.0
        candidate = mix(fg, black, mid)
        if contrastRatio(candidate, bg) >= minContrast:
            hi = mid
        else:
            lo = mid

    adjusted = mix(fg, black, hi)
    return rgb01ToHex(adjusted)

def pickTextColorForBg(bgHex: str) -> str:
    bg = hexToRgb01(bgHex)
    L = relativeLuminance(bg)

    # If background is reasonably dark, prefer white
    if L < 0.45:
        return "#ffffff"

    # Otherwise prefer black
    return "#000000"

def getVideoResolution(videoPath):
    """Return (width, height) of video using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        videoPath
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    return int(stream["width"]), int(stream["height"])

def getHalfCpuThreads(minThreads=2, maxThreads=16):
    cores = os.cpu_count() or 4
    threads = max(minThreads, cores // 2)
    return min(threads, maxThreads)  # optional cap so you don't go crazy

def resourcePath(*parts: str) -> str:
    # When packaged (PyInstaller), sys._MEIPASS points to the temp extracted dir
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)

def getCached720pVideo(videoPath, cacheDir="./cache_audio"):
    """
    Returns path to a 720p-safe video.
    If original is <= 720p, returns original path.
    If > 720p, creates/uses cached 720p version.
    """
    os.makedirs(cacheDir, exist_ok=True)

    baseName = os.path.splitext(os.path.basename(videoPath))[0]
    cachedPath = os.path.join(cacheDir, f"{baseName}_720p.mp4")

    # If cached already exists, just use it
    if os.path.exists(cachedPath):
        return cachedPath

    width, height = getVideoResolution(videoPath)
    if height <= 720:
        return videoPath  # already safe

    threads = getHalfCpuThreads()
    print(f"🎥 Downscaling video {width}x{height} → 720p cache")
    ffmpegPath = resourcePath("ffmpeg.exe") 
    cmd = [
        ffmpegPath,
        "-y",
        "-i", videoPath,
        "-vf", "scale=-2:720",  # preserve aspect ratio, divisible by 2
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-threads", str(threads),
        "-an",  # drop audio (you already handle audio separately)
        cachedPath
    ]

    subprocess.run(cmd, check=True)
    return cachedPath

_AUDIO_EXT_PRIORITY = {
    ".wav": 100,   # uncompressed PCM (usually)
    ".flac": 95,   # lossless compressed
    ".aiff": 90,   # uncompressed (often)
    ".aif": 90,
    ".m4a": 80,    # container, usually AAC/ALAC (could be lossless ALAC)
    ".alac": 80,   # sometimes used as extension (less common)
    ".aac": 75,    # lossy but decent
    ".mp3": 70,    # lossy, very common (your #2)
    ".ogg": 60,    # Vorbis/Opus container
    ".opus": 60,
    ".wma": 40,    # legacy Windows format
}

_STEM_SUFFIXES_TO_STRIP = (
    "_leading_vocals",
    "_backing_vocals",
    "_vocals",
)

def _normalizeSongStem(stem: str) -> str:
    """
    Convert a filename stem into a canonical song name:
    - remove known suffixes like _leading_vocals, _backing_vocals, _vocals
    - collapse extra whitespace/underscores
    - keep the rest intact
    """
    s = stem

    # Strip any of the suffixes (case-insensitive) if present at end.
    lowered = s.lower()
    for suf in _STEM_SUFFIXES_TO_STRIP:
        if lowered.endswith(suf):
            s = s[: len(s) - len(suf)]
            lowered = s.lower()
            break

    # Optional: clean trailing separators like "_" or "-" after stripping
    s = re.sub(r"[\s_-]+$", "", s)

    # Normalize internal whitespace a bit (don’t overdo it; preserve user’s naming)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _audioExtScore(ext: str) -> int:
    return _AUDIO_EXT_PRIORITY.get(ext.lower(), 0)

@dataclass(frozen=True)
class AudioPick:
    songName: str          # normalized base name used in UI
    path: str              # best file path for that song
    ext: str               # extension chosen
    score: int             # extension priority
    sizeBytes: int         # file size tie-breaker

def findBestAudioFiles(songDir: str):
    """
    Scans songDir for audio files, groups by normalized song name, and selects the best candidate
    per song using:
      1) extension priority (wav highest, mp3 second, then other common types)
      2) file size as tie-breaker for same extension
    Returns:
      - sortedSongNames: list[str] of unique song names for the picker
      - bestBySong: dict[songName -> AudioPick] containing the best path to load later
    """
    p = Path(songDir)
    if not p.exists() or not p.is_dir():
        return [], {}

    bestBySong = {}

    for entry in p.iterdir():
        if not entry.is_file():
            continue

        ext = entry.suffix.lower()
        score = _audioExtScore(ext)
        if score <= 0:
            continue  # not a recognized audio type

        stem = entry.stem  # filename without extension
        songName = _normalizeSongStem(stem)
        if not songName:
            continue

        try:
            sizeBytes = entry.stat().st_size
        except Exception:
            sizeBytes = 0

        candidate = AudioPick(
            songName=songName,
            path=str(entry),
            ext=ext,
            score=score,
            sizeBytes=sizeBytes
        )

        current = bestBySong.get(songName)
        if current is None:
            bestBySong[songName] = candidate
            continue

        # Prefer higher extension score; if tied, prefer larger file
        if (candidate.score > current.score) or (
            candidate.score == current.score and candidate.sizeBytes > current.sizeBytes
        ):
            bestBySong[songName] = candidate

    sortedSongNames = sorted(bestBySong.keys(), key=str.lower)
    return sortedSongNames, bestBySong

def pickBestAudioForStem(songDir: str, stem: str):
    """
    Find best audio file matching exactly "<stem>.<ext>" in songDir.
    Priority: extension score, then file size.
    Returns full path or None if not found.
    """
    p = Path(songDir)
    if not p.exists():
        return None

    bestPath = None
    bestScore = -1
    bestSize = -1

    # Only match exact stem; no fuzzy matching (keeps behavior predictable)
    for ext in _AUDIO_EXT_PRIORITY.keys():
        candidate = p / f"{stem}{ext}"
        if not candidate.exists() or not candidate.is_file():
            continue

        score = _audioExtScore(ext)
        try:
            size = candidate.stat().st_size
        except Exception:
            size = 0

        if (score > bestScore) or (score == bestScore and size > bestSize):
            bestPath = str(candidate)
            bestScore = score
            bestSize = size

    return bestPath

class ModalGuard:
    _open_modals = set()

    @classmethod
    def try_open(cls, key: str) -> bool:
        """
        Attempt to open a modal identified by `key`.
        Returns True if allowed, False if already open.
        """
        if key in cls._open_modals:
            return False
        cls._open_modals.add(key)
        return True

    @classmethod
    def close(cls, key: str):
        """Mark a modal as closed."""
        if key in cls._open_modals:
            cls._open_modals.discard(key)
        
    @classmethod
    def is_open(cls, key: str) -> bool:
        return key in cls._open_modals