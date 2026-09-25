"""
Korean word-level vocab extraction, parity with core.kanji_reference.analyzeSelection() for
Japanese - see .claude/ plan doc. Deliberately its own shape rather than forced to match Japanese's
field names where the concepts don't correspond (no onyomi/kunyomi category; Hanja candidates are
a list, not a single cognate).

Reuses core.korean_grammar_breakdown.breakdownLine() rather than re-tokenizing from scratch - that
function already does the hard work of Hanja-gating a dependent noun like 수 (see its own
_NATIVE_DEPENDENT_NOUNS), resolving XR+하다 compounds, and merging vowel-contracted syllable
groups. Only content-word entries are kept (particles/endings aren't "vocab" in the flashcard
sense); each one's own meaning is re-looked-up as a structured {"status","pos","gloss"} dict
(lookupKoreanMeaning) rather than reusing breakdownLine's already-flattened display string, so it
matches the shape core.vocab_store_ko.upsertVocab expects.
"""

from core.korean_dictionary import lookupKoreanMeaning
from core.korean_grammar_breakdown import breakdownLine


def analyzeKoreanSelection(fullText: str) -> list:
    entries = breakdownLine(fullText)
    results = []
    for entry in entries:
        if entry["role"] != "content":
            continue
        # A merged group's tag can be a "+"-joined compound (e.g. "XR+XSA") that isn't one of
        # wikiPosForTag's real tags - passing it through is harmless (it just returns None, and
        # lookupKoreanMeaning already falls back to the first-listed entry when no tag matches),
        # so no special-casing is needed here.
        meaning = lookupKoreanMeaning(entry["lemma"], entry["tag"])
        results.append({
            "surface": entry["surface"],
            "lemma": entry["lemma"],
            "meaning": meaning,
            "hanjaCandidates": entry["hanja"] or [],
        })
    return results
