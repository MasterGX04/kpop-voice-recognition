"""
Hanja converter: given a highlighted Hangul word, list every Hanja (Chinese character) word it
could represent. Kept separate from core/korean_grammar_breakdown.py, same reasoning as keeping
core/kanji_reference.py separate from core/grammar_breakdown.py - a per-word dictionary lookup,
not a whole-sentence structural view, even though both draw on the same
core/korean_dictionary.py-loaded index. Design reference: .claude/KOREAN_HANJA_PLAN.md.
"""

from core.kanji_reference import lookupChineseCognate
from core.korean_dictionary import allEntries, wikiPosForTag


def _hanjaPinyin(hanja: str) -> str:
    """
    Mandarin pinyin for a Hanja candidate, reusing the same CC-CEDICT lookup already built for
    Japanese Kanji (core/kanji_reference.py: lookupChineseCognate) - Korean Hanja are themselves
    Traditional Chinese character forms (Korean never adopted Simplified or Japan's Shinjitai),
    so no script conversion is needed first, unlike the Japanese case.

    A `hanja` value can list more than one orthographic variant separated by "/" (e.g. "畫/畵",
    confirmed via testing real index data) - both variants are pronounced identically in
    Mandarin, so only the first is looked up. Falls back to the character-by-character
    "not_attested" pinyin (never fabricated as "real Chinese") for the rare case where the exact
    multi-character Hanja spelling isn't itself a CC-CEDICT word.
    """
    firstVariant = hanja.split("/", 1)[0]
    cognate = lookupChineseCognate(firstVariant)
    return cognate["pinyin"] if cognate["status"] == "confirmed" else cognate["pinyinFallback"]


def lookupHanja(hangulWord: str, tag: str = None) -> list:
    """
    Return every real Hanja candidate for a Hangul spelling, as a list of
    `{"hanja", "gloss", "pos", "pinyin"}` dicts - always a list, since a genuinely ambiguous word (e.g. 화,
    which has six unrelated real Hanja words - 火/禍/和/化/畫/靴 - sharing the same modern
    pronunciation) must show every real candidate rather than silently guessing one. An
    unambiguous Sino-Korean word (학교 -> 學校) naturally returns a one-item list; a purely native
    word with no Hanja origin at all (너무, 아니, 목소리) naturally returns an empty list - no
    special-casing needed for either case, since both just fall out of filtering by which index
    entries happen to carry a "hanja" field.

    When the caller's token `tag` is given (as core/korean_grammar_breakdown.py always does),
    narrows to just the entries whose Wiktionary "pos" matches it, same disambiguate-by-tag idea
    as core/korean_dictionary.py: lookupKoreanMeaning(). This matters because a short, common
    Hangul syllable can be a real, unrelated Hanja word under a completely different part of
    speech than the one actually being used - found via direct user testing: 이 as the native
    proximal determiner ("this", tag MM/pos "det", hanja=null) was showing six unrelated Hanja
    candidates - 二(two)/李(surname)/理/利/釐/伊 - that only ever surface as a numeral, a surname,
    etc. (different tags entirely), never as this determiner. Filtering to the matching POS
    bucket correctly finds that bucket's own entry is hanja=null, so nothing is shown - exactly
    the same "same syllable, unrelated homograph" problem 화 demonstrates the *right* way to
    handle when the homographs really do share the token's actual part of speech (all six of
    화's candidates are real for an ordinary noun-tagged 화; 이's are not for a determiner-tagged
    이). Called with no `tag` (e.g. a direct/manual lookup with no grammatical context), the
    original unfiltered behavior is preserved - every real Hanja candidate across every homograph,
    letting the caller judge for itself.

    Never fabricates a candidate: a word missing from the index, or present only with native
    (non-Sino) senses (or senses under an unrelated part of speech, once `tag` narrows it), returns
    [] rather than guessing at a character-level reading (that weaker, unconfirmed tier is the
    cihai/Unihan fallback deferred to Phase 2+, not built here).
    """
    entries = allEntries(hangulWord)

    wikiPos = wikiPosForTag(tag) if tag is not None else None
    if wikiPos is not None:
        entries = [e for e in entries if e["pos"] == wikiPos]

    return [
        {"hanja": c["hanja"], "gloss": c["gloss"], "pos": c["pos"], "pinyin": _hanjaPinyin(c["hanja"])}
        for c in entries
        if c["hanja"]
    ]
