"""
Shared lookup against the derived Korean Wiktionary index (data/ko_wiktionary/index.json) - kept
separate from core/korean_grammar_breakdown.py and the not-yet-built core/korean_hanja.py since
both features query the same index for different reasons (English meaning vs. Hanja candidates).
"""

import codecs
import json
import os

_INDEX_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "ko_wiktionary", "index.json")

_index = None

# Maps a Kiwi/Sejong POS tag to the Wiktionary "pos" string, so a homograph entry can be picked by
# matching part of speech instead of blindly trusting entry order - found necessary via a real
# case: 다 has 6 Wiktionary entries, and the first-listed one ("pos": "suffix") is a bare redirect
# note ("For the verb-final suffix, see the entry at -다") rather than a usable gloss, while the
# real adverb sense ("all, completely") a MAG-tagged token wants is listed second. Same
# disambiguate-by-matching-signal idea as core/kanji_reference.py: lookupJapaneseMeaning() picking
# the JMdict entry whose reading list actually matches, just keyed on POS here since Korean has no
# separate reading to match on.
_TAG_TO_WIKI_POS = {
    "NNG": "noun", "NNP": "noun", "NNB": "noun", "NR": "noun",
    "NP": "pron",
    "VV": "verb",
    "VA": "adj",
    "MAG": "adv",
    "MM": "det",
    "IC": "intj",
    "MAJ": "conj",
    "XR": "root",
    "XPN": "prefix",
    "XSN": "suffix", "XSV": "suffix", "XSA": "suffix",
}


def _loadIndex():
    global _index
    if _index is None:
        with codecs.open(_INDEX_PATH, "r", encoding="utf-8") as f:
            _index = json.load(f)
    return _index


def allEntries(word: str) -> list:
    """
    Public accessor for every raw index entry (`{"pos", "gloss", "hanja"}`) recorded for a Hangul
    spelling, in index order - used by core/korean_hanja.py: lookupHanja() to filter down to just
    the Hanja-bearing ones. Returns [] for a word not in the index at all.
    """
    return _loadIndex().get(word, [])


def wikiPosForTag(tag: str):
    """
    Public accessor for the Kiwi/Sejong tag -> Wiktionary "pos" mapping - reused by
    core/korean_hanja.py: lookupHanja() so it can narrow a homograph's real candidates down to
    the one(s) matching how a specific token is actually being used, not just the whole set of
    unrelated words sharing that Hangul spelling. Returns None for an unmapped tag.
    """
    return _TAG_TO_WIKI_POS.get(tag)


def lookupKoreanMeaning(word: str, tag: str = None) -> dict:
    """
    Look up a word's own English meaning via the Korean Wiktionary index - run for every content
    word regardless of whether it's Sino-Korean, same "unconditional" role
    core/kanji_reference.py: lookupJapaneseMeaning() plays for Japanese.

    A Hangul spelling can map to more than one entry (real homographs, e.g. 화 has 6 in the built
    index - see data/ko_wiktionary/build_index.py's docstring). When the caller's token `tag` is
    given, prefers
    the entry whose Wiktionary "pos" matches it (see _TAG_TO_WIKI_POS); otherwise, or if nothing
    matches, falls back to the first-listed entry - same first-listed-sense convention
    CC-CEDICT/JMdict lookups already use elsewhere in this project.

    Returns {"status": "found", "pos": ..., "gloss": [...]} or {"status": "not_found"} - never
    fabricates a gloss for a word missing from the index.
    """
    candidates = allEntries(word)
    if not candidates:
        return {"status": "not_found"}

    wikiPos = wikiPosForTag(tag)
    match = next((c for c in candidates if c["pos"] == wikiPos), candidates[0]) if wikiPos else candidates[0]
    return {"status": "found", "pos": match["pos"], "gloss": match["gloss"]}
