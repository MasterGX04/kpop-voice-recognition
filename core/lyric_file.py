"""
Edit one lyric's pause markers directly in a song's `_lyrics.json`, for callers that are not the Tk app -
the flashcard window (its own process) uses this to fix a card's pauses without opening the Lyric Editor.

Why not drive the Tk Lyric Editor: it is welded to the ONE song the app has loaded (its save path comes from
`app.selectedGroup`/`app.songName` and saving rebuilds canvas LyricBoxes), so editing another song through it
would mean switching the app's whole song from another process. This touches the file only.

Deliberately narrow: only the pause markers (core.lyric_text.PAUSE_MARK) may change. Any change to the
wording or the "|" colour split is refused, because the vocab DB holds a clean copy of this text and a silent
edit would leave it stale (that is the Lyric Editor's job, followed by Compile Vocab). The file is written in
the Lyric Editor's own format (UTF-8, indent=4, ensure_ascii=False) keeping the file's existing line endings
(8 of the 36 real lyrics files are CRLF, the rest LF), via a temp file + replace, so a crash can't leave a
half-written lyrics file and no unrelated diff appears.
"""

import json
import os

from core.lyric_text import findRawLyricEntry, stripForDisplay


class LyricEditError(ValueError):
    """A problem the user should read (lyric not found, wording changed...), not a bug."""


def lyricsPath(group: str, song: str, root: str = ".") -> str:
    return os.path.join(root, "saved_labels", group, f"{song}_lyrics.json")


def _load(path: str):
    """(entries, newline): `newline` is the file's own line ending, to write it back unchanged."""
    if not os.path.exists(path):
        raise LyricEditError(f"No lyrics file at {path}")
    with open(path, "rb") as f:
        raw = f.read()
    newline = "\r\n" if b"\r\n" in raw else "\n"
    try:
        entries = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        # Never overwrite a file we could not read - the Lyric Editor would silently treat it as empty.
        raise LyricEditError(f"Could not read {path}: {exc}")
    if not isinstance(entries, list):
        raise LyricEditError(f"Unexpected format in {path}")
    return entries, newline


def getRawLyric(group: str, song: str, lyricId, cleanLine: str, root: str = "."):
    """(lyricId, stored text with markers) of the lyric an occurrence came from."""
    entries, _ = _load(lyricsPath(group, song, root))
    entry = findRawLyricEntry(entries, lyricId, cleanLine)
    if entry is None:
        raise LyricEditError(f"Couldn't find this lyric in {song}'s lyrics file.")
    return entry.get("lyricId"), entry.get("korean") or ""


def setPauseMarkers(group: str, song: str, lyricId, cleanLine: str, newStoredText: str, root: str = "."):
    """Replace the lyric's `korean` text with `newStoredText` (real markers, not the editor stand-in), provided
    only pause markers differ. Returns the stored text that was written."""
    path = lyricsPath(group, song, root)
    entries, newline = _load(path)
    entry = findRawLyricEntry(entries, lyricId, cleanLine)
    if entry is None:
        raise LyricEditError(f"Couldn't find this lyric in {song}'s lyrics file.")

    newText = newStoredText.strip()
    if stripForDisplay(newText) != stripForDisplay(entry.get("korean") or "").strip():
        raise LyricEditError(
            "Only pause markers can be changed here - the words or the | colour split differ. "
            "Edit wording in the Lyric Editor (then Compile Vocab)."
        )
    if newText == (entry.get("korean") or ""):
        return newText

    entry["korean"] = newText
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline=newline) as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)
    os.replace(tmp, path)
    return newText
