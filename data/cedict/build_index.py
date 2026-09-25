"""
One-time build script: parse CC-CEDICT's plain-text dictionary (not committed - see NOTICE.md)
into the compact data/cedict/index.json used by core/kanji_reference.py.

Usage: python build_index.py /path/to/cedict_1_0_ts_utf-8_mdbg.txt
Download source: https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz

Line format (comments start with "#"):
    Traditional Simplified [pin1 yin1] /gloss one/gloss two/.../
A headword can appear on multiple lines (different readings/senses), so each Traditional
headword maps to a *list* of {pinyin, gloss} entries.
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_LINE_RE = re.compile(r"^(\S+)\s+\S+\s+\[([^\]]+)\]\s+/(.+)/\s*$")


def buildIndex(cedictPath: str) -> dict:
    index = defaultdict(list)

    with open(cedictPath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue

            match = _LINE_RE.match(line)
            if not match:
                continue

            traditional, pinyin, glossField = match.groups()
            glosses = [g for g in glossField.split("/") if g]
            index[traditional].append({"pinyin": pinyin, "gloss": glosses})

    return dict(index)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_index.py /path/to/cedict_1_0_ts_utf-8_mdbg.txt")
        sys.exit(1)

    index = buildIndex(sys.argv[1])
    outPath = Path(__file__).parent / "index.json"
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Wrote {len(index)} headwords to {outPath}")
