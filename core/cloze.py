"""
Grammar cloze drills (Milestone 5 of .claude/FLASHCARD_WEB_UPGRADE_PLAN.md): blank a due word's
own surface span out of one of its real lyric-line occurrences. Deliberately thin - reuses
core.grammar_breakdown.breakdownLine()/core.korean_grammar_breakdown.breakdownLine() completely
unmodified rather than re-tokenizing, matching by LEMMA (the stable dictionary key a vocab_ja/
vocab_ko row is stored under), not by the row's own `surface` column - that's just one example
form, while the real occurrence line may show the word in a different inflected form.

Blanking is a plain first-occurrence string replace of the matched entry's real surface text
within the line, not character-offset splicing - neither breakdownLine() implementation exposes
token offsets externally today, and doing so is unneeded complexity: the only failure mode is the
rare case of the identical surface text appearing twice in one line, which still produces a valid
blank (a different real occurrence of the exact same word-form) - an accepted, documented edge
case, not a bug.
"""

import random

from core.grammar_breakdown import breakdownLine as breakdownJapaneseLine
from core.korean_grammar_breakdown import breakdownLine as breakdownKoreanLine

BLANK_MARKER = "＿＿＿＿"


def orderOccurrencesForCloze(occurrences: list) -> list:
    """
    Real user feedback: a uniformly random occurrence line made cloze feel like "which song is
    this from" rather than "do you know this word" - a word whose only tested line comes from a
    long, dense, unfamiliar rap verse needs the whole surrounding passage recalled before the
    blank is even attemptable, while a short, already-memorized line (a sung chorus/hook) tests
    the word itself, which is the actual point of a cloze drill.

    Returns `occurrences` reordered shorter-line-first: the shorter half (by lyric_line length,
    rounded up so a single leftover occurrence counts as "short") is shuffled and placed before the
    longer half (also shuffled) - never discarding a word's only occurrence just because it's
    long, only deprioritizing it when a shorter alternative exists. The caller (see
    gui.vocab_review_api.getClozeCardDetail) still tries every entry in order until one actually
    produces a valid cloze card, so this only changes WHICH one is tried first, never whether a
    word with no short occurrence at all can still be quizzed.
    """
    if not occurrences:
        return []

    ranked = sorted(occurrences, key=lambda o: len(o.get("lyricLine") or ""))
    mid = (len(ranked) + 1) // 2
    shorter, longer = ranked[:mid], ranked[mid:]
    random.shuffle(shorter)
    random.shuffle(longer)
    return shorter + longer


def buildClozeCard(language: str, lemma: str, lyricLine: str):
    """
    Returns {"blankedLine", "answerSurface", "gloss"} for the first content entry in `lyricLine`
    whose lemma matches `lemma`, or None if no entry in this line matches (the caller should try
    another real occurrence of the same word - see gui.vocab_review_api.getClozeCardDetail()).
    """
    breakdown = breakdownJapaneseLine if language == "Japanese" else breakdownKoreanLine
    entries = breakdown(lyricLine)
    match = next((e for e in entries if e["role"] == "content" and e["lemma"] == lemma), None)
    if match is None:
        return None

    surface = match["surface"]
    return {
        "blankedLine": lyricLine.replace(surface, BLANK_MARKER, 1),
        "answerSurface": surface,
        "gloss": match.get("gloss"),
    }
