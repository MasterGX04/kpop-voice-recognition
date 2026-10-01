"""
Korean word-level vocab extraction, parity with core.kanji_reference.analyzeSelection() for
Japanese - see .claude/ plan doc. Deliberately its own shape rather than forced to match Japanese's
field names where the concepts don't correspond (no onyomi/kunyomi category; Hanja candidates are
a list, not a single cognate).

Reuses core.korean_grammar_breakdown.breakdownLine() rather than re-tokenizing from scratch - that
function already does the hard work of Hanja-gating a dependent noun like 수 (see its own
_NATIVE_DEPENDENT_NOUNS), resolving XR+하다/ㅂ-irregular compounds, and merging vowel-contracted
syllable groups. Only content-word entries are kept (particles/endings aren't "vocab" in the
flashcard sense); each one's `meaning` is breakdownLine's own already-resolved {"status","pos",
"gloss"} dict, not re-derived here - a from-scratch `lookupKoreanMeaning(entry["lemma"],
entry["tag"])` call used to live in this function instead, but that silently discarded every one of
breakdownLine's fallback lookups (XR+하다 composition, ㅂ-irregular -어하다 reversal, MAG root
stripping) the moment a merged group's `lemma`/`tag` became a "+"-joined compound that can't be
looked up directly - a real bug found via a live-DB audit (see core/korean_grammar_breakdown.py:
_mergeGroup's own comment for the numbers). Reusing the already-resolved dict here means there is
exactly one place this resolution logic lives, not two copies that can drift apart again.
"""

from core.korean_grammar_breakdown import breakdownLine


def analyzeKoreanSelection(fullText: str) -> list:
    entries = breakdownLine(fullText)
    results = []
    for entry in entries:
        if entry["role"] != "content":
            continue
        results.append({
            "surface": entry["surface"],
            "lemma": entry["lemma"],
            "meaning": entry["meaning"],
            "hanjaCandidates": entry["hanja"] or [],
        })
    return results
