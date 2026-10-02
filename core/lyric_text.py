"""
The one place that knows about the two in-text markers a lyric line can carry. See
.claude/KARAOKE_PLAN.md Part 4.

    "|"        splits a line into two member colours (Tk lyric box only; see LyricBox._createColorCodedText).
    PAUSE_MARK the singer pauses here - an invisible U+2063 INVISIBLE SEPARATOR, so the lyric box draws
               it at width 0 (verified in Pretendard Variable and the common fallbacks). It is the
               author's escape hatch for a pause the label rows can't place (core.karaoke_timing).

    READING    `恋<open>こ<PAUSE_MARK>い<close>` - a kana reading attached to the kanji word before it, so a pause can
               sit INSIDE the word (Doughnut: Sana sings 恋 as こ ... い across two label rows). Both brackets are
               invisible (U+2064 / U+2061), the reading is timing-only and every reader strips the whole
               annotation (the kanji stays the lyric); the editor shows it as `恋《こ▾い》`.

It is NOT whitespace to Python or JS (`"\\u2063".strip()` keeps it) and fugashi / Intl.Segmenter both emit
it as its own token, so every reader of lyric text must strip it through this module and never inline.

Because it is invisible, the lyric editor shows a visible stand-in, EDITOR_PAUSE_GLYPH, while editing:
`toEditorText` on load, `fromEditorText` on save. Stored/drawn/analysed text never contains the stand-in.
"""

import re

PAUSE_MARK = chr(0x2063)   # U+2063 INVISIBLE SEPARATOR (kept as chr() so the source never holds an invisible char)
EDITOR_PAUSE_GLYPH = chr(0x25BE)   # BLACK DOWN-POINTING SMALL TRIANGLE, the visible stand-in
READING_OPEN = chr(0x2064)         # U+2064 INVISIBLE PLUS
READING_CLOSE = chr(0x2061)        # U+2061 FUNCTION APPLICATION
EDITOR_READING_OPEN = chr(0x300A)  # 《
EDITOR_READING_CLOSE = chr(0x300B)  # 》

# An annotation is only meaningful right after a kanji run; anywhere else (or unclosed) it is just noise to drop.
READING_ANNOTATION = re.compile(READING_OPEN + "([^" + READING_OPEN + READING_CLOSE + "]*)" + READING_CLOSE)


def hasPauseMarks(text: str) -> bool:
    return PAUSE_MARK in (text or "")


def stripForDisplay(text: str) -> str:
    """Drop the pause marker, keep "|" (the Tk lyric box still needs it for the colour split)."""
    return _dropReadings(text or "").replace(PAUSE_MARK, "")


def _dropReadings(text: str) -> str:
    """Remove every `<open>reading<close>` annotation (and any stray bracket)."""
    return READING_ANNOTATION.sub("", text).replace(READING_OPEN, "").replace(READING_CLOSE, "")


def stripAll(text: str) -> str:
    """Drop both markers: what analysis, the vocab DB and the flashcard should see."""
    return _dropReadings(text or "").replace(PAUSE_MARK, "").replace("|", "")


def stripAllWithSelection(text: str, start: int, end: int):
    """`stripAll(text)` plus the selection [start, end) re-expressed in the stripped text, for callers that
    analyse a highlighted range (offsets into the editor text would otherwise drift past a marker)."""
    text = text or ""
    dropped = set()
    for m in READING_ANNOTATION.finditer(text):
        dropped.update(range(m.start(), m.end()))
    kept = []
    newStart = newEnd = 0
    for i, ch in enumerate(text):
        if i == start:
            newStart = len(kept)
        if i == end:
            newEnd = len(kept)
        if i not in dropped and ch not in (PAUSE_MARK, "|", READING_OPEN, READING_CLOSE):
            kept.append(ch)
    if start >= len(text):
        newStart = len(kept)
    if end >= len(text):
        newEnd = len(kept)
    return "".join(kept), newStart, newEnd


def toEditorText(text: str) -> str:
    """Stored text -> what the editor widget shows (marker made visible)."""
    return ((text or "").replace(PAUSE_MARK, EDITOR_PAUSE_GLYPH)
            .replace(READING_OPEN, EDITOR_READING_OPEN).replace(READING_CLOSE, EDITOR_READING_CLOSE))


def fromEditorText(text: str) -> str:
    """What the editor widget holds -> stored text (stand-in made invisible again)."""
    return ((text or "").replace(EDITOR_PAUSE_GLYPH, PAUSE_MARK)
            .replace(EDITOR_READING_OPEN, READING_OPEN).replace(EDITOR_READING_CLOSE, READING_CLOSE))


def findRawLyricEntry(entries: list, lyricId=None, cleanLine: str = ""):
    """The stored lyric entry a clean occurrence line came from: matched by `lyricId` when there is one, else
    by the entry whose stripped text equals `cleanLine`. None if absent."""
    if lyricId:
        for entry in entries or []:
            if entry.get("lyricId") == lyricId:
                return entry
    if cleanLine:
        target = stripAll(cleanLine).strip()
        for entry in entries or []:
            raw = entry.get("korean") or ""
            if raw and stripAll(raw).strip() == target:
                return entry
    return None


def findRawLyricText(entries: list, lyricId=None, cleanLine: str = ""):
    """The stored (marker-bearing) `korean` text of that lyric, or None."""
    entry = findRawLyricEntry(entries, lyricId, cleanLine)
    return None if entry is None else (entry.get("korean") or "")
