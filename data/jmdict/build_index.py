"""
One-time build script: parse JMdict's English-glosses subset (not committed - see NOTICE.md)
into the compact data/jmdict/index.json used by core/kanji_reference.py.

Usage: python build_index.py /path/to/JMdict_e
Download source: http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz

JMdict_e is plain XML with an inline DOCTYPE declaring all the part-of-speech/field/etc.
abbreviation entities (e.g. "&v5r;" -> "Godan verb with 'ru' ending") right in the file - stdlib
xml.etree.ElementTree (via expat) resolves those internal entities on its own, no external DTD
fetch or hand-written entity table needed.

Indexed by **kanji spelling only** (k_ele/keb) - every word core/kanji_reference.py looks up
already contains Kanji (see analyzeSelection()), so kana-only JMdict entries (particles,
onomatopoeia, etc.) are skipped; they're never queried. A kanji spelling can map to *multiple*
JMdict entries with different readings and unrelated meanings (e.g. 湯 read "ゆ" = "hot water"
vs the same kanji read "タン", borrowed specifically for the Chinese sense "soup") - so each
kanji key maps to a list of {"readings": [...], "senses": [{"pos": [...], "gloss": [...]}]}
records, one per JMdict <entry>, letting lookupJapaneseMeaning() pick the record whose readings
list actually contains the word's real (in-context) reading.
"""

import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def buildIndex(jmdictPath: str) -> dict:
    tree = ET.parse(jmdictPath)
    root = tree.getroot()

    index = defaultdict(list)
    for entry in root.findall("entry"):
        kebs = [k.findtext("keb") for k in entry.findall("k_ele")]
        if not kebs:
            continue  # kana-only entry - never looked up (every lemma we query has Kanji)

        rebs = [r.findtext("reb") for r in entry.findall("r_ele")]

        senses = []
        for sense in entry.findall("sense"):
            glosses = [g.text for g in sense.findall("gloss") if g.text]
            if not glosses:
                continue
            pos = [p.text for p in sense.findall("pos")]
            senses.append({"pos": pos, "gloss": glosses})

        if not senses:
            continue

        record = {"readings": rebs, "senses": senses}
        for keb in kebs:
            index[keb].append(record)

    return dict(index)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_index.py /path/to/JMdict_e")
        sys.exit(1)

    builtIndex = buildIndex(sys.argv[1])
    outPath = Path(__file__).parent / "index.json"
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(builtIndex, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Wrote {len(builtIndex)} kanji headwords to {outPath}")
