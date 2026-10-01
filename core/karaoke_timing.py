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
from core.lyric_text import PAUSE_MARK

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
_EMPTY_STRETCH = 0.03        # a pause with no words in it (breath / someone else's tail)
# An author-placed pause marker (core.lyric_text.PAUSE_MARK) is a strong attractor for a cut, large enough
# to beat any text/time proportion mismatch (sum of squared share errors is <= 2) but still a cost, not a
# constraint: with fewer label gaps than markers the unused markers fall back to a rest inside a stretch.
_CUT_AT_MARKER = -1.0
REST_BEATS = 1.5             # length of an unused marker's pause inside a stretch, in morae

_SMALL_KANA = set("ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ")
_KANA = re.compile(r"[぀-ヿ]")
_JP_SCRIPT = re.compile(r"[぀-ヿ一-鿿]")
_HANGUL = re.compile(r"[가-힣]")
_LATIN_WORD = re.compile(r"[A-Za-z]+")
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


def _newPiece(text, weight=0, isWord=False, particleEnd=False, prefix=False):
    return {"text": text, "weight": weight, "isWord": isWord, "particleEnd": particleEnd, "prefix": prefix,
            "pauseAfter": False}


def _segmentJapanese(text: str) -> list:
    from core.japanese_utils import getTagger, tokenReading   # heavy import (fugashi); only JA needs it

    pieces, pos, prevWord = [], 0, None
    for word in getTagger()(text):
        surface = word.surface
        at = text.find(surface, pos)
        if at < 0:
            continue
        if at > pos:                                    # whitespace/newlines MeCab drops
            pieces.append(_newPiece(text[pos:at]))
        pos = at + len(surface)

        if _JP_SCRIPT.search(surface):
            weight = kanaMorae(tokenReading(word, prevWord))
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
                        particleEnd=(pos1 == "助詞"), prefix=False)
        else:
            pieces.append(_newPiece(surface, weight, True, particleEnd=(pos1 == "助詞"),
                                    prefix=(pos1 == "接頭辞")))
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


def segmentLine(text: str, language: str) -> list:
    """Pieces that reassemble to `text` minus any pause markers: [{text, weight, isWord, particleEnd,
    pauseAfter}]. Words (isWord) carry a beat `weight`; whitespace, newlines and punctuation are weight-0
    non-words. A PAUSE_MARK is a hard word break (the text is segmented either side of it separately) and
    sets `pauseAfter` on the last word before it; a marker with no word before it is ignored."""
    chunks = text.split(PAUSE_MARK)
    pieces = []
    for n, chunk in enumerate(chunks):
        if chunk:
            pieces.extend(_segmentChunk(chunk, language))
        if n < len(chunks) - 1:
            for piece in reversed(pieces):
                if piece["isWord"]:
                    piece["pauseAfter"] = True
                    break
    return pieces


def buildStretches(rows: list, members, spanStart: int, spanEnd: int) -> list:
    """[(start, end)] stretches of continuous singing inside the span, from the label rows of the
    card's singer(s). Ad-lib rows are skipped; overlapping or near-touching rows are fused. With no
    usable row the whole span is one stretch."""
    wanted = _asMemberList(members)
    clipped = sorted(
        (max(r[1], spanStart), min(r[2], spanEnd))
        for r in rows
        if len(r) >= 3 and not _isAdLibRow(r) and _memberMatches(wanted, r[0])
        and r[2] > spanStart and r[1] < spanEnd and min(r[2], spanEnd) > max(r[1], spanStart)
    )
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


def timeLine(text: str, language: str, spanStart: int, spanEnd: int, rows: list, members=None) -> dict:
    """
    Word timing for one lyric card. `rows` are the song's label rows
    ([member, start, end, isBacking, isAdLib]); `members` the card's singers (None/"All" = anyone).

    Returns {"stretches": n, "pieces": [...]}. Pieces reassemble to `text` minus pause markers; each word piece
    also has startChunk/endChunk and `fraction` (0..1: how far into [spanStart, spanEnd) it starts,
    the unit the flashcard's play-from-here takes).
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
    cuts = assignWordsToStretches(words, lineBreakAfter, stretches)

    for k, (sStart, sEnd) in enumerate(stretches):
        group = words[cuts[k]:cuts[k + 1]]
        # A marker that did not become a stretch cut (no label gap there) is a short rest inside the stretch.
        restAfter = [REST_BEATS if w["pauseAfter"] and j < len(group) - 1 else 0 for j, w in enumerate(group)]
        groupWeight = (sum(w["weight"] for w in group) + sum(restAfter)) or 1
        elapsed = 0
        for j, w in enumerate(group):
            w["startChunk"] = round(sStart + (sEnd - sStart) * elapsed / groupWeight, 1)
            elapsed += w["weight"]
            w["endChunk"] = round(sStart + (sEnd - sStart) * elapsed / groupWeight, 1)
            elapsed += restAfter[j]
            w["fraction"] = round(min(1.0, max(0.0, (w["startChunk"] - spanStart) / spanLength)), 4)
    return {"stretches": len(stretches), "pieces": pieces}
