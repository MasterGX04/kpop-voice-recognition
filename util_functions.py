import subprocess
import json
import os

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

    cmd = [
        "ffmpeg",
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
        else:
            print(f"Warning: Attempted to close modal '{key}' which was not open.")

    @classmethod
    def is_open(cls, key: str) -> bool:
        return key in cls._open_modals