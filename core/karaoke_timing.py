"""
Estimate when each word of a lyric card is sung, for the flashcard's "play from this word" (and, later,
karaoke highlighting). See .claude/KARAOKE_PLAN.md.

The data only has per-card timing plus the hand-made label rows (one row = one stretch of continuous
singing, with a pause either side). No word timestamps exist, so everything here is an ESTIMATE built
from two signals the data does have:

1. The label rows are the pauses. A card's span is split into "stretches" of continuous singing (rows
   of the card's singer(s), overlapping/near-touching rows fused, ad-lib rows ignored). Words are never
   spread across a pause; the highlight just waits through the gap.
2. Within a stretch, a word takes time in proportion to how many beats (morae / syllables) it is sung
   with - read from its kana reading, not its character count (君 is 1 character but キミ = 2 beats).

Which words belong to which stretch is the only real decision: pick the cut points (preferably at line
breaks, then after a particle) so each stretch's share of the text matches its share of the time.
A one-row card, or a card with no rows, simply gets one stretch - a continuous rap needs no markers.
"""

import re

from core.label_runs import _asMemberList, _isAdLibRow, _memberMatches
from core.lyric_text import PAUSE_MARK, READING_ANNOTATION
from core.util_functions import CHUNK_DURATION_MS, WORD_LEAD_MS

# Label rows are the author's exact timing of when a singer starts and stops, so EVERY gap between two rows is
# a real boundary, however small (the corpus is full of 2-5 chunk gaps, ~750-900 of each; an earlier version
# fused gaps under 4 chunks as "slop" and starved the pause marker of stretches to land on - Stay Gold V, 5
# rows 3 chunks apart, came out as 2 stretches). Only rows that genuinely OVERLAP (a member's duplicate or
# nested track) are one stretch: they fuse when a row starts before the previous one ended.
MIN_PAUSE_CHUNKS = 0

# Cut-point priors for the stretch assignment. Added to the (text share - time share)^2 mismatch, which
# is ~0.01-0.05 for a plausible split, so these are tie-breakers that favour natural phrase breaks.
_CUT_AT_LINE_BREAK = 0.0
_CUT_AFTER_PARTICLE = 0.02
_CUT_MID_PHRASE = 0.05
# A row is a stretch where the member is singing, so it should hold words: an empty row only when there are more
# rows than words (or humming not in the lyrics). Heavy on purpose - at 0.03 the guess left a row empty on 82 of
# 647 cards (Doughnut Sana: all four "Na" crammed into one row, the next row empty); at 1.0 only the 33 cards
# with more rows than words keep one.
_EMPTY_STRETCH = 1.0
# An author-placed pause marker (core.lyric_text.PAUSE_MARK) is a strong attractor for a cut, large enough
# to beat any text/time proportion mismatch (sum of squared share errors is <= 2) but still a cost, not a
# constraint: with fewer label gaps than markers the unused markers fall back to a rest inside a stretch.
_CUT_AT_MARKER = -1.0
REST_BEATS = 1.5             # length of an unused marker's pause inside a stretch, in morae

# Beats that are sometimes folded into the syllable before them and sometimes sung on their own (ふっ|て vs ふ-っ-て,
# だん vs だ-ん): a held vowel, a stopped consonant, a moraic nasal. The tap dialog can step over them ("tap ー っ ん").
_HOLD_BEATS = {"ー", "っ", "ッ", "ん", "ン"}
_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_KANA = re.compile(r"[぀-ヿ]")
_JP_SCRIPT = re.compile(r"[぀-ヿ一-鿿]")
_HANGUL = re.compile(r"[가-힣]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
_TRAILING_KANJI = re.compile(r"[一-鿿々〆ヶ]+$")
_ATTACHES_TO_PREVIOUS = {"助詞", "助動詞", "接尾辞"}   # particle / auxiliary / suffix join the word before


def kanaMorae(reading: str) -> int:
    """Beats in a kana reading: every kana counts, small ゃゅょ/ぁ.. fuse with the kana before them.
    ッ (pause), ン and ー (held vowel) are beats of their own (all inside the kana range)."""
    return sum(1 for ch in reading if _KANA.match(ch) and ch not in _SMALL_KANA)


def englishSyllables(word: str) -> int:
    """Rough syllable count for a Latin-script word sung inside a JA/KO lyric (Baby=2, Slowly=2)."""
    w = word.lower()
    count = len(re.findall(r"[aeiouy]+", w))
    if count > 1 and w.endswith("e") and not w.endswith(("le", "ee", "ye")):
        count -= 1
    return max(1, count)


def splitMorae(kana: str) -> list:
    """The beats of a kana string, one entry each: every kana is a beat, small ゃゅょぁ.. fuse with the kana before
    them, and ー (held) and っ/ッ (pause) are beats of their own - counted out loud like ひ・と・つ, one count each."""
    units = []
    for ch in kana:
        if not _KANA.match(ch):
            continue
        if ch in _SMALL_KANA and units:
            units[-1] += ch
        else:
            units.append(ch)
    return units


def _toHiragana(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


_VOWEL_OF = {ch: vowel for vowel, chars in {
    "a": "あかがさざただなはばぱまやゃらわゎぁ", "i": "いきぎしじちぢにひびぴみりぃ", "u": "うくぐすずつづぬふぶぷむゆゅるぅゔ",
    "e": "えけげせぜてでねへべぺめれぇ", "o": "おこごそぞとどのほぼぽもよょろをぉ"}.items() for ch in chars}


def isHeldVowel(prev: str, beat: str) -> bool:
    """Is hiragana `beat` just the previous beat's vowel held on, as in the long vowels of かんじょう (o + う),
    じゅう (u + う), せい (e + い) and おお? Then it is not a new sung syllable of its own for tapping purposes."""
    vowel = _VOWEL_OF.get(prev[-1:]) if prev else None
    return (beat == "う" and vowel in ("u", "o")) or (beat == "い" and vowel == "e") or (beat == "お" and vowel == "o")


def splitUnits(piece: dict, language: str) -> list:
    """What one word is tapped as in syllable mode: [{"label", "weight"}]. Japanese: one unit per beat of its kana
    reading (labelled in the word's own script: katakana for a katakana word, else hiragana, so 恋 -> こ, い).
    Korean: one per Hangul block (an English run inside is one unit). A Latin word, or anything that does not add up
    to the word's own weight, stays ONE unit - never split on a guess."""
    whole = [{"label": piece.get("label", piece["text"]), "weight": piece["weight"]}]
    kana = piece.get("kana")
    if kana:
        beats = splitMorae(kana)
        if len(beats) != piece["weight"] or not beats:
            return whole
        katakana = bool(re.search(r"[゠-ヿ]", piece["text"])) and not re.search(r"[ぁ-ゟ一-鿿]", piece["text"])
        hira = [_toHiragana(b) for b in beats]
        return [{"label": b if katakana else hira[i], "weight": 1, "hold": b in _HOLD_BEATS,
                 "held": i > 0 and isHeldVowel(hira[i - 1], hira[i])} for i, b in enumerate(beats)]
    if language == "Korean":
        units = [{"label": m.group(), "weight": 1 if _HANGUL.match(m.group()) else englishSyllables(m.group())}
                 for m in re.finditer(r"[가-힣]|[A-Za-z]+", piece["text"])]
        if units and sum(u["weight"] for u in units) == piece["weight"]:
            return units
    return whole


def _newPiece(text, weight=0, isWord=False, particleEnd=False, prefix=False):
    return {"text": text, "weight": weight, "isWord": isWord, "particleEnd": particleEnd, "prefix": prefix,
            "pauseAfter": False, "kana": ""}


def _segmentJapanese(text: str) -> list:
    from core.japanese_utils import getTagger, tokenReading, rawTokenReading   # heavy import (fugashi); only JA needs it

    pieces, pos, prevWord = [], 0, None
    for word in getTagger()(text):
        surface = word.surface
        at = text.find(surface, pos)
        if at < 0:
            continue
        if at > pos:                                    # whitespace/newlines MeCab drops
            pieces.append(_newPiece(text[pos:at]))
        pos = at + len(surface)

        reading = ""
        if _JP_SCRIPT.search(surface):
            reading = tokenReading(word, prevWord)
            weight = kanaMorae(reading)
            raw = rawTokenReading(word, prevWord)       # the same beats with a real ー kept (shown/tapped as ー)
            if kanaMorae(raw) == weight:
                reading = raw
            prevWord = word
        elif _LATIN_WORD.fullmatch(surface):
            weight = englishSyllables(surface)
        else:
            weight = 0

        pos1 = getattr(word.feature, "pos1", None)
        if weight == 0:
            pieces.append(_newPiece(surface))
        elif pieces and pieces[-1]["isWord"] and (pieces[-1]["prefix"] or pos1 in _ATTACHES_TO_PREVIOUS):
            last = pieces[-1]                           # 君 + を -> "君を"; 無 + 防備 -> "無防備"
            last.update(text=last["text"] + surface, weight=last["weight"] + weight,
                        particleEnd=(pos1 == "助詞"), prefix=False, kana=last["kana"] + reading)
        else:
            pieces.append(_newPiece(surface, weight, True, particleEnd=(pos1 == "助詞"),
                                    prefix=(pos1 == "接頭辞")))
            pieces[-1]["kana"] = reading
    if pos < len(text):
        pieces.append(_newPiece(text[pos:]))
    return pieces


def _segmentSpaced(text: str, countBeats) -> list:
    """Korean (and fallback): each whitespace-separated chunk is one word."""
    pieces, pos = [], 0
    for m in re.finditer(r"\S+", text):
        if m.start() > pos:
            pieces.append(_newPiece(text[pos:m.start()]))
        weight = countBeats(m.group())
        pieces.append(_newPiece(m.group(), weight, weight > 0))
        pos = m.end()
    if pos < len(text):
        pieces.append(_newPiece(text[pos:]))
    return pieces


def _koreanBeats(chunk: str) -> int:
    return len(_HANGUL.findall(chunk)) + sum(englishSyllables(w) for w in _LATIN_WORD.findall(chunk))


def _segmentChunk(text: str, language: str) -> list:
    if language == "Japanese":
        return _segmentJapanese(text)
    return _segmentSpaced(text, _koreanBeats)


def _splitReadings(text: str) -> list:
    """Cut `text` at its reading annotations (core.lyric_text): [("text", str) | ("reading", base, reading)].
    The base is the kanji run right before the annotation; an annotation with no kanji before it is dropped."""
    segments, pos = [], 0
    for m in READING_ANNOTATION.finditer(text):
        before = text[pos:m.start()]
        pos = m.end()
        base = _TRAILING_KANJI.search(before)
        if base is None:
            segments.append(("text", before))
            continue
        segments.append(("text", before[:base.start()]))
        segments.append(("reading", base.group(), m.group(1)))
    segments.append(("text", text[pos:]))
    return [seg for seg in segments if seg[0] != "text" or seg[1]]


def _readingPieces(base: str, reading: str) -> list:
    """One word piece per kana part of an annotated kanji word. The first carries the kanji (`text`); the rest
    are zero-text continuations that timeLine folds back into it as `parts`. A part break is a pause, so
    `pauseAfter` is set on every part but the last. No kana in the reading -> None (annotation ignored)."""
    parts = [p for p in reading.split(PAUSE_MARK) if kanaMorae(p) > 0]
    if not parts:
        return None
    pieces = []
    for n, part in enumerate(parts):
        piece = _newPiece(base if n == 0 else "", kanaMorae(part), True)
        piece["pauseAfter"] = n < len(parts) - 1
        piece["continuation"] = n > 0
        piece["label"] = f"{base}({part})"
        piece["reading"] = part
        piece["kana"] = part
        pieces.append(piece)
    return pieces


def segmentLine(text: str, language: str) -> list:
    """Pieces that reassemble to `text` minus pause markers and reading annotations: [{text, weight, isWord,
    particleEnd, pauseAfter}]. Words (isWord) carry a beat `weight`; whitespace, newlines and punctuation are
    weight-0 non-words. A PAUSE_MARK is a hard word break (the text is segmented either side of it separately)
    and sets `pauseAfter` on the last word before it; a marker with no word before it is ignored.

    A kanji word followed by a reading annotation (`恋<open>こ<mark>い<close>`) is one word timed from the
    annotation's kana, not the dictionary: its pieces come back as the kanji word plus zero-text
    `continuation` pieces, one per part (see _readingPieces)."""
    pieces = []

    def markPause():
        for piece in reversed(pieces):
            if piece["isWord"]:
                piece["pauseAfter"] = True
                break

    for segment in _splitReadings(text):
        if segment[0] == "reading":
            made = _readingPieces(segment[1], segment[2])
            if made is not None:
                pieces.extend(made)
                continue
            segment = ("text", segment[1])
        chunks = segment[1].split(PAUSE_MARK)
        for n, chunk in enumerate(chunks):
            if chunk:
                pieces.extend(_segmentChunk(chunk, language))
            if n < len(chunks) - 1:
                markPause()
    return pieces


def _rowWindows(rows: list, members, spanStart: int, spanEnd: int) -> list:
    """The card singer(s)' label rows clipped to the span, sorted. Ad-lib rows are skipped."""
    wanted = _asMemberList(members)
    return sorted(
        (max(r[1], spanStart), min(r[2], spanEnd))
        for r in rows
        if len(r) >= 3 and not _isAdLibRow(r) and _memberMatches(wanted, r[0])
        and r[2] > spanStart and r[1] < spanEnd and min(r[2], spanEnd) > max(r[1], spanStart)
    )


def buildStretches(rows: list, members, spanStart: int, spanEnd: int) -> list:
    """[(start, end)] stretches of continuous singing inside the span, from the label rows of the
    card's singer(s). Ad-lib rows are skipped; overlapping or near-touching rows are fused. With no
    usable row the whole span is one stretch."""
    clipped = _rowWindows(rows, members, spanStart, spanEnd)
    stretches = []
    for start, end in clipped:
        if stretches and start - stretches[-1][1] < MIN_PAUSE_CHUNKS:
            stretches[-1] = (stretches[-1][0], max(stretches[-1][1], end))
        else:
            stretches.append((start, end))
    return stretches or [(spanStart, spanEnd)]


def _cutPenalty(wordPieces: list, lineBreakAfter: list, cut: int) -> float:
    """Penalty for a pause falling between word cut-1 and word cut."""
    if wordPieces[cut - 1]["pauseAfter"]:
        return _CUT_AT_MARKER
    if lineBreakAfter[cut - 1]:
        return _CUT_AT_LINE_BREAK
    return _CUT_AFTER_PARTICLE if wordPieces[cut - 1]["particleEnd"] else _CUT_MID_PHRASE


def assignWordsToStretches(wordPieces: list, lineBreakAfter: list, stretches: list) -> list:
    """Returns cuts [c_0=0, c_1, ..., c_m=n]: stretch k sings words [c_k, c_{k+1}). Chosen by DP to
    minimise sum((text share - time share)^2) + cut-point priors + empty-stretch penalties."""
    n, m = len(wordPieces), len(stretches)
    if m == 1 or n == 0:
        return [0] * m + [n]

    # Every pause marked: rows - 1 markers fully describe the card (Doughnut Sana/Tzuyu: a held syllable, a slow
    # phrase, then two lines sung as ONE long row - a line break that is not a pause, which nothing in the
    # text can reveal). The markers ARE the row boundaries; line breaks are ignored.
    markerCuts = [c for c in range(1, n) if wordPieces[c - 1]["pauseAfter"]]
    if markerCuts and len(markerCuts) + 1 == m:
        return [0] + markerCuts + [n]

    # Author-asserted structure wins. A pause marker means the author has described this card's phrases
    # (markers + line breaks). If that gives exactly one phrase per row, each phrase IS one row: do not let
    # the text-length-vs-row-length guess shuffle words across rows - it assumes one even tempo, and real
    # singing is not (Doughnut, Nayeon: a held 4-mora phrase fills a 105-chunk row while 10 morae fit in 68).
    # Deliberately limited to marked cards: forcing this on unmarked cards made the speed between rows absurd
    # on a dozen of them (up to 87x) where line count == row count only by coincidence, so those still go
    # through the tempo-aware search below.
    if any(p["pauseAfter"] for p in wordPieces):
        boundaries = [c for c in range(1, n) if lineBreakAfter[c - 1] or wordPieces[c - 1]["pauseAfter"]]
        if len(boundaries) + 1 == m:
            return [0] + boundaries + [n]

    prefix = [0]
    for p in wordPieces:
        prefix.append(prefix[-1] + p["weight"])
    totalWeight = prefix[-1] or 1
    durations = [e - s for s, e in stretches]
    totalDuration = sum(durations) or 1
    timeShare = [d / totalDuration for d in durations]

    INF = float("inf")
    # best[k][c]: min cost of stretches 0..k-1 consuming the first c words. back[k][c]: previous cut.
    best = [[INF] * (n + 1) for _ in range(m + 1)]
    back = [[0] * (n + 1) for _ in range(m + 1)]
    best[0][0] = 0.0
    for k in range(1, m + 1):
        for c in range(n + 1):
            for prev in range(c + 1):
                if best[k - 1][prev] == INF:
                    continue
                cost = best[k - 1][prev] + ((prefix[c] - prefix[prev]) / totalWeight - timeShare[k - 1]) ** 2
                if c == prev:
                    cost += _EMPTY_STRETCH
                if k < m and 0 < c < n:
                    penalty = _cutPenalty(wordPieces, lineBreakAfter, c)
                    # The marker bonus is for a stretch that really ends at the marker, never for empty
                    # stretches stacked on the same spot (that would farm the bonus).
                    if not (penalty < 0 and c == prev):
                        cost += penalty
                if cost < best[k][c]:
                    best[k][c], back[k][c] = cost, prev
    cuts = [n]
    for k in range(m, 0, -1):
        cuts.append(back[k][cuts[-1]])
    return cuts[::-1]


# A human reaction lag is a fraction of a second; 30 chunks (1.2 s) is generous. A gap outside this window means a tap
# was matched with the wrong row start, not that the person was slow.
_PLAUSIBLE_LAG = (-3, 30)


def _medianSpread(gaps: list):
    gaps = sorted(gaps)
    mid = len(gaps) // 2
    return (gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2), gaps[-1] - gaps[0]


def calibrateLag(taps: dict, rowStarts: dict, stretchStarts: list = None, minLag: float = 3.0):
    """Human tap lag, in chunks, and how steady it was (max - min of the gaps): (lag, spread), or (None, None) when it
    cannot be told.

    1. By unit: the median of (tap - exact row start) over units that are BOTH tapped and the first unit of a label
       row (`timeLine(...)["rowStarts"]`), used only while every such gap is plausible - if the text put a unit in
       the wrong row (markers that disagree with the label rows), those gaps are wild and this is not trusted.
    2. By time: for each row start (`timeLine(...)["stretchStarts"]`), the first tap at least `minLag` chunks after
       it (nobody reacts faster: a tap sooner than that is the PREVIOUS row's last beat landing late - rows are often
       only ~4 chunks apart and a reaction is ~5); the median of those gaps. Needs two rows. `minLag` is song time:
       scale it by the playback speed on a slowed clip."""
    gaps = [taps[i] - rowStarts[i] for i in taps if i in rowStarts]
    if len(gaps) >= 2 and all(_PLAUSIBLE_LAG[0] <= g <= _PLAUSIBLE_LAG[1] for g in gaps):
        return _medianSpread(gaps)
    if stretchStarts:
        times = sorted(taps.values())
        near = []
        for start in stretchStarts:
            first = next((t for t in times if t >= start + minLag), None)
            if first is not None and first - start <= _PLAUSIBLE_LAG[1]:
                near.append(first - start)
        if len(near) >= 2:
            return _medianSpread(near)
    return None, None


def rowCheck(correctedTaps: dict, stretchStarts: list) -> list:
    """How well a take agrees with the author's label rows, independent of how the text was split: for each row start,
    the first (lag-corrected) tap within [start - 2, start + 30] chunks and its gap to the start, in chunks. A take that
    matches the labels has gaps near 0 (a row's first beat IS its start); a big gap means the take and the labels
    disagree - a mis-tap, or the labels/audio are out of step. Rows with no tap near their start are left out."""
    times = sorted(correctedTaps.values())
    gaps = []
    for start in stretchStarts:
        first = next((t for t in times if t >= start - 2), None)
        if first is not None and first - start <= 30:
            gaps.append(round(first - start, 1))
    return gaps


NUDGE_LOCKED, NUDGE_MOVED, NUDGE_EDGE, NUDGE_RESET = "locked", "moved", "edge", "reset"


def nudgeAnchor(timing: dict, anchors: dict, index: int, deltaChunks) -> tuple:
    """Move ONE beat of a take by `deltaChunks` (the tap dialog's fine-tune pass) or, with `deltaChunks` None, put
    it back to the estimate. `timing` is `timeLine(..., anchors=anchors)`; `anchors` are lag-CORRECTED chunks, so
    nothing here subtracts a lag. Returns (new anchors, status):
      "locked": the first beat of a label row sits on the author's exact row start and never moves;
      "edge": already at the limit - it stays inside its own row and never crosses the nearest fixed (tapped or
              row-start) beat on either side; estimated beats between are re-spread, so they do not block;
      "moved" / "reset".
    An estimated beat that is nudged becomes a tapped one (the anchor is its current time plus the nudge)."""
    slots = timing["slots"]
    if not 0 <= index < len(slots):
        raise ValueError(f"No beat {index} in this line.")
    slot = slots[index]
    if slot["source"] == "row":
        return dict(anchors), NUDGE_LOCKED
    out = dict(anchors)
    if deltaChunks is None:
        return (out, NUDGE_RESET) if out.pop(index, None) is not None else (out, NUDGE_EDGE)
    rowStart, rowEnd = timing["rows"][slot["row"]][:2] if timing["rows"] else (slot["startChunk"], slot["startChunk"])
    low, high = rowStart, rowEnd
    for other in reversed(slots[:index]):
        if other["source"] in ("tapped", "row") and other["row"] == slot["row"]:
            low = max(low, other["startChunk"])
            break
    for other in slots[index + 1:]:
        if other["source"] == "tapped" and other["row"] == slot["row"]:
            high = min(high, other["startChunk"])
            break
    target = round(min(high, max(low, slot["startChunk"] + deltaChunks)), 1)
    if abs(target - slot["startChunk"]) < 0.05:
        return out, NUDGE_EDGE
    out[index] = target
    return out, NUDGE_MOVED


# Taps per second a person can keep up with; above this, the tap dialog suggests slowing the clip down.
TAP_MAX_PER_SEC = 4.0
SLOW_RATES = (1.0, 0.75, 0.5, 0.35)


def suggestRate(fastestPerSec: float) -> float:
    """The fastest playback speed (from SLOW_RATES) at which the busiest stretch needs <= TAP_MAX_PER_SEC taps/s."""
    for rate in SLOW_RATES:
        if fastestPerSec * rate <= TAP_MAX_PER_SEC:
            return rate
    return SLOW_RATES[-1]


def timeLine(text: str, language: str, spanStart: int, spanEnd: int, rows: list, members=None,
             anchors: dict = None, unit: str = "word") -> dict:
    """
    Word timing for one lyric card. `rows` are the song's label rows
    ([member, start, end, isBacking, isAdLib]); `members` the card's singers (None/"All" = anyone).

    Returns {"stretches": n, "pieces": [...], "rows": [(start, end, [words])], "markers": n, "words": [...],
    "rowStarts": {...}, "unit": unit, "pace": {...}}. Pieces
    reassemble to `text` minus pause markers; each word piece
    also has startChunk/endChunk, `fraction` (0..1: how far into [spanStart, spanEnd) it starts) and
    `playFraction` (where play-from-here should begin: `fraction` minus a lead-in, clamped to the word's own
    stretch - pass it to playOccurrenceAudio with exact=True).

    `unit` is what one tap is taken for: "word" (a word, or one kana part of a split kanji) or "syllable" (every
    beat of a word's reading, see splitUnits; each word piece then carries `syllables`). `anchors` ({unit index:
    chunk}, core.tap_store) are tapped onsets: a tapped unit starts exactly at its tap (clamped into its own
    stretch); units between two anchors are spread by beat weight over just that gap. The first unit of a label
    row always stays on the row start - the author's labels win over a tap. A word starts where its first unit
    does. Each word piece reports `source`: "tapped" | "row" (exact row start) | "estimated"; `words` lists the
    unit labels in tap order and `rowStarts` the units that sit on an exact row start (index -> chunk).
    """
    pieces = segmentLine(text, language)
    spanLength = max(1, spanEnd - spanStart)
    wordIdx = [i for i, p in enumerate(pieces) if p["isWord"]]
    words = [pieces[i] for i in wordIdx]

    # Does a line break follow word j (i.e. before the next word)?
    lineBreakAfter = []
    for a, i in enumerate(wordIdx):
        nxt = wordIdx[a + 1] if a + 1 < len(wordIdx) else len(pieces)
        lineBreakAfter.append(any("\n" in pieces[j]["text"] for j in range(i + 1, nxt)))

    stretches = buildStretches(rows, members, spanStart, spanEnd)
    haveRows = bool(_rowWindows(rows, members, spanStart, spanEnd))
    cuts = assignWordsToStretches(words, lineBreakAfter, stretches)
    anchors = anchors or {}
    syllables = unit == "syllable"
    rowStarts, fastest = {}, 0.0

    # Every unit that can be tapped, in order, each remembering its word and the row (stretch) it sits in.
    wordStretch = [k for k in range(len(stretches)) for _ in range(cuts[k], cuts[k + 1])]
    slots = []
    for wi, w in enumerate(words):
        parts = splitUnits(w, language) if syllables else [{"label": w.get("label", w["text"]), "weight": w["weight"]}]
        for n, part in enumerate(parts):
            slots.append({"w": wi, "label": part["label"], "weight": part["weight"], "hold": bool(part.get("hold")), "held": bool(part.get("held")),
                          "last": n == len(parts) - 1, "k": wordStretch[wi]})
    for slot in slots:
        slot["k0"] = slot["k"]                       # the row the TEXT (markers, line breaks, tempo) chose

    # Taps are evidence: a unit tapped inside another row than the text chose belongs to the row it was tapped in
    # (Funny Valentine: the markers said Moon night | あなた | ..., the rows say Moon night あなたが | くれた | ...).
    # Untapped units keep their row, but never outside the rows of the taps around them.
    if anchors and len(stretches) > 1:
        def rowOf(chunk):
            return min(range(len(stretches)), key=lambda k: (
                0 if stretches[k][0] <= chunk <= stretches[k][1] else min(abs(chunk - stretches[k][0]), abs(chunk - stretches[k][1])), k))
        tapped = sorted(i for i in anchors if 0 <= i < len(slots))
        floor = 0
        tappedRow = {}
        for i in tapped:                              # rows never go backwards along the line
            floor = tappedRow[i] = max(floor, rowOf(anchors[i]))
        low, running = 0, 0
        for i, slot in enumerate(slots):
            if i in tappedRow:
                low = tappedRow[i]
                slot["k"] = low
            else:
                nextTapped = next((tappedRow[j] for j in tapped if j > i), len(stretches) - 1)
                slot["k"] = min(max(slot["k"], low), nextTapped)
            running = slot["k"] = max(slot["k"], running)

    for i, slot in enumerate(slots):
        nxt = slots[i + 1] if i + 1 < len(slots) else None
        # A marker that did not become a stretch cut (no label gap there) is a short rest inside the stretch.
        slot["rest"] = REST_BEATS if (slot["last"] and words[slot["w"]]["pauseAfter"] and nxt is not None
                                      and nxt["k"] == slot["k"]) else 0

    for k, (sStart, sEnd) in enumerate(stretches):
        first = next((i for i, slot in enumerate(slots) if slot["k"] == k), None)
        if first is None:
            continue
        group = [slot for slot in slots if slot["k"] == k]

        # Fixed points inside the row: its start (the label row, exact), every tap (clamped into the row, never
        # earlier than the point before it), and its end. Units between two points share that gap by beat.
        points = {0: sStart}
        firstTap = anchors.get(first) if not haveRows else None
        if firstTap is not None:                 # no labelled rows: nothing exact to snap the first unit to
            points[0] = min(sEnd, max(sStart, firstTap))
        for i in range(1, len(group)):
            tap = anchors.get(first + i)
            if tap is not None:
                points[i] = min(sEnd, max(points[max(points)], sStart, tap))
        points[len(group)] = sEnd
        fixed = sorted(points)
        for ia, ib in zip(fixed, fixed[1:]):
            segWeight = sum(group[i]["weight"] + group[i]["rest"] for i in range(ia, ib)) or 1
            elapsed = 0
            for i in range(ia, ib):
                slot = group[i]
                slot["startChunk"] = round(points[ia] + (points[ib] - points[ia]) * elapsed / segWeight, 1)
                elapsed += slot["weight"]
                slot["endChunk"] = round(points[ia] + (points[ib] - points[ia]) * elapsed / segWeight, 1)
                elapsed += slot["rest"]
        for i, slot in enumerate(group):
            if i == 0 and haveRows:
                slot["source"] = "row"
                rowStarts[first] = sStart
            else:
                slot["source"] = "tapped" if i in points and (i > 0 or firstTap is not None) else "estimated"
        if sEnd > sStart:
            fastest = max(fastest, len(group) / ((sEnd - sStart) * CHUNK_DURATION_MS / 1000))

    for wi, w in enumerate(words):
        mine = [slot for slot in slots if slot["w"] == wi]
        sStart = stretches[mine[0]["k"]][0]
        w["startChunk"], w["endChunk"], w["source"] = mine[0]["startChunk"], mine[-1]["endChunk"], mine[0]["source"]
        if syllables:
            w["syllables"] = [{k2: slot[k2] for k2 in ("label", "startChunk", "endChunk", "source", "hold", "held")}
                              for slot in mine]
        w["fraction"] = round(min(1.0, max(0.0, (w["startChunk"] - spanStart) / spanLength)), 4)
        # Where "play from this word" should actually start: a lead-in before the word (its onset inside
        # a stretch is an estimate), but NEVER before its own stretch's start. The first word of a stretch
        # sits on a row start the author labelled exactly, and the gap before it is only ~0.3 s, so a
        # plain 300 ms lead reached back into the previous line (Doughnut: clicking line 2 played line 1).
        playChunk = max(sStart, w["startChunk"] - WORD_LEAD_MS / CHUNK_DURATION_MS)
        w["playFraction"] = round(min(1.0, max(0.0, (playChunk - spanStart) / spanLength)), 4)
    unitLabels = [slot["label"] for slot in slots]
    # Fold each annotated word's continuation parts back into the word: it is ONE word on screen, but `parts`
    # keeps when each kana part starts (a part can sit in a different label row - the word spans a pause).
    head = None
    for w in words:
        if w.get("continuation") and head is not None:
            head["parts"].append({"reading": w["reading"], "startChunk": w["startChunk"], "endChunk": w["endChunk"]})
            head["endChunk"] = w["endChunk"]
            if syllables:
                head["syllables"] += w["syllables"]
        elif w.get("reading"):
            head = w
            w["parts"] = [{"reading": w["reading"], "startChunk": w["startChunk"], "endChunk": w["endChunk"]}]
    pieces = [p for p in pieces if not p.get("continuation")]
    return {
        "stretches": len(stretches),
        "pieces": pieces,
        # Which words landed in which row, for the pause editor's live preview: [(start, end, [words])...].
        "rows": [(sStart, sEnd, [w.get("label", w["text"]) for wi, w in enumerate(words)
                                 if next(slot["k"] for slot in slots if slot["w"] == wi) == k])
                 for k, (sStart, sEnd) in enumerate(stretches)],
        "markers": sum(1 for w in words if w["pauseAfter"]),
        # For the tap tool: the units one tap each is taken for, in order, and which of them sit on an exact row
        # start (index -> chunk), the ground truth `calibrateLag` measures taps against.
        "words": unitLabels,
        "rowStarts": rowStarts,
        "stretchStarts": [start for start, _ in stretches],
        # Every tap unit with its own timing, source and label row (index into `rows`): what a nudge needs to know.
        "slots": [{"label": slot["label"], "startChunk": slot["startChunk"], "endChunk": slot["endChunk"],
                   "source": slot["source"], "row": slot["k"], "hold": slot["hold"], "held": slot["held"]} for slot in slots],
        # How many units the taps moved to a different label row than the text (markers / line breaks) chose.
        "reassigned": sum(1 for slot in slots if slot["k"] != slot["k0"]),
        "unit": unit,
        # How busy the busiest stretch is (units per second, averaged over its row - real bursts are faster) and the
        # playback speed to suggest for tapping it.
        "pace": {"fastest": round(fastest, 1), "suggestedRate": suggestRate(fastest)},
    }
