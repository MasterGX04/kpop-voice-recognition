"""
Saved tap-along takes: one file per song, `saved_labels/<group>/taps/<song>_taps.json`, keyed by lyricId. See
.claude/KARAOKE_PLAN.md Part 5.

A take is the chunk each word was tapped at (lag already removed), plus a hash of the word list it was tapped
against. It lives in its own file - not `_lyrics.json` - because the Tk Lyric Editor rebuilds every lyric entry from
its LyricBox and would silently drop unknown keys, and in a `taps/` subfolder to keep the song folders uncluttered.

A take is only applied while the lyric still segments into the same words (`wordsHash`). Edit the wording, add a
pause marker or split a kanji and the hash no longer matches: `getTake` reports it as stale and the caller falls back
to the estimate - stale times are never applied. Absolute chunks, so moving the card's own startChunk is harmless.

Format: {lyricId: {"hash": str, "lag": float|null, "words": {"<wordIndex>": chunk}, "raw": {"<wordIndex>": chunk}|absent,
"rate": float|absent, "takenAt": iso8601}}. `words` is what is applied (lag removed, any fine-tune nudges included);
`raw` is the taps exactly as the page clock read them, with the playback `rate`, so a take can be re-derived when the
lag logic improves. Takes saved before `raw` existed simply have none.
Written via temp file + replace so a crash cannot leave a half-written file; an unreadable file is refused, never
overwritten.
"""

import hashlib
import json
import os
from datetime import datetime, timezone


class TapStoreError(ValueError):
    """A problem the user should read (unreadable file...), not a bug."""


def tapsPath(group: str, song: str, root: str = ".") -> str:
    return os.path.join(root, "saved_labels", group, "taps", f"{song}_taps.json")


def wordsHash(words: list) -> str:
    """Fingerprint of the words one tap is taken for (`timeLine(...)["words"]`), order included."""
    return hashlib.sha1("\x1f".join(words).encode("utf-8")).hexdigest()[:16]


def _load(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TapStoreError(f"Could not read {path}: {exc}")
    if not isinstance(data, dict):
        raise TapStoreError(f"Unexpected format in {path}")
    return data


def getTake(group: str, song: str, lyricId: str, words: list, root: str = ".") -> dict:
    """{"status": "none" | "stale" | "ok", "anchors": {wordIndex: chunk}, "lag": float|None, "takenAt": str|None,
    "raw": {wordIndex: chunk}|None, "rate": float|None}. `anchors` and `raw` are empty/None unless status is "ok"."""
    entry = _load(tapsPath(group, song, root)).get(lyricId) if lyricId else None
    if not entry:
        return {"status": "none", "anchors": {}, "lag": None, "takenAt": None, "raw": None, "rate": None}
    if entry.get("hash") != wordsHash(words):
        return {"status": "stale", "anchors": {}, "lag": entry.get("lag"), "takenAt": entry.get("takenAt"),
                "raw": None, "rate": None}
    anchors = {int(i): chunk for i, chunk in (entry.get("words") or {}).items() if 0 <= int(i) < len(words)}
    raw = entry.get("raw")
    return {"status": "ok", "anchors": anchors, "lag": entry.get("lag"), "takenAt": entry.get("takenAt"),
            "raw": {int(i): chunk for i, chunk in raw.items() if 0 <= int(i) < len(words)} if raw else None,
            "rate": entry.get("rate")}


def saveTake(group: str, song: str, lyricId: str, words: list, anchors: dict, lag=None, root: str = ".",
             raw: dict = None, rate: float = None) -> None:
    """Replace the lyric's take (an empty `anchors` clears it). Other lyrics' takes in the file are untouched.
    `raw` / `rate`: the uncorrected taps and playback speed, kept beside the applied `anchors`."""
    if not lyricId:
        raise TapStoreError("This lyric has no id, so a take cannot be saved for it.")
    path = tapsPath(group, song, root)
    data = _load(path)
    if anchors:
        data[lyricId] = {
            "hash": wordsHash(words),
            "lag": lag,
            "words": {str(i): round(float(chunk), 1) for i, chunk in sorted(anchors.items())},
            "takenAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        if raw:
            data[lyricId]["raw"] = {str(i): round(float(chunk), 2) for i, chunk in sorted(raw.items())}
        if rate is not None:
            data[lyricId]["rate"] = float(rate)
    else:
        data.pop(lyricId, None)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
