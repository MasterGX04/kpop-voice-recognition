"""
One-time build script: parse kaikki.org's Korean Wiktionary extraction (not committed - see
NOTICE.md) into the compact data/ko_wiktionary/index.json used by core/korean_hanja.py (Hanja
lookup) and core/korean_grammar_breakdown.py (English gloss for content words).

Usage: python build_index.py /path/to/kaikki.org-dictionary-Korean.jsonl
Download source: https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl

One parse of the 201MB file serves both features (see .claude/KOREAN_GRAMMAR_BREAKDOWN_PLAN.md's
"What's reused vs. what's new" section) - every lang_code=="ko" word entry keeps a {pos, gloss}
record, and additionally a "hanja" field whenever the entry's own head_templates carry one (that
field is what makes a record usable for .claude/KOREAN_HANJA_PLAN.md's Hangul->Hanja lookup;
absence just means "not a Sino-Korean word", not "not indexed").

Indexed by **Hangul word**, one entry per JSONL line (each line is already one word/etymology/
part-of-speech - confirmed live: 화 alone appears as 7 separate lines, one per etymology, each
with its own distinct "hanja" field or none at all) - mirrors data/cedict/index.json's own
{headword: [entries]} shape, since Korean's homophone ambiguity (6+ unrelated Hanja words sharing
one Hangul spelling) is the same "one spelling, multiple real entries" shape CC-CEDICT/JMdict
already handle elsewhere in this project.

A "syllable" pos entry (e.g. 화's 7th entry, "Korean reading of various Chinese characters") is a
reference note about the syllable block itself, not a real word - skipped, since neither feature
ever wants to surface it as a word.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def _hanjaOf(entry: dict):
    for template in entry.get("head_templates", []):
        hanja = template.get("args", {}).get("hanja")
        if hanja:
            return hanja
    return None


def buildIndex(jsonlPath: str) -> dict:
    index = defaultdict(list)
    with open(jsonlPath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)

            if entry.get("lang_code") != "ko":
                continue
            if entry.get("pos") == "syllable":
                continue  # a note about the syllable block, not a real word - see docstring

            word = entry.get("word")
            if not word:
                continue

            glosses = []
            for sense in entry.get("senses", []):
                glosses.extend(sense.get("glosses", []))
            if not glosses:
                continue  # nothing to show for this entry, not usable by either feature

            record = {"pos": entry.get("pos"), "gloss": glosses, "hanja": _hanjaOf(entry)}
            index[word].append(record)

    return dict(index)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_index.py /path/to/kaikki.org-dictionary-Korean.jsonl")
        sys.exit(1)

    builtIndex = buildIndex(sys.argv[1])
    outPath = Path(__file__).parent / "index.json"
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(builtIndex, f, ensure_ascii=False, separators=(",", ":"))

    withHanja = sum(1 for records in builtIndex.values() if any(r["hanja"] for r in records))
    print(f"Wrote {len(builtIndex)} Korean headwords ({withHanja} with a Hanja entry) to {outPath}")
