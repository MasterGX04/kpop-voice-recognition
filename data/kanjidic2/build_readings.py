"""
One-time build script: parse kanjidic2.xml (not committed - see NOTICE.md) into the
compact data/kanjidic2/readings.json used by core/kanji_reference.py.

Usage: python build_readings.py /path/to/kanjidic2.xml
Download source: http://www.edrdg.org/kanjidic/kanjidic2.xml.gz
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def buildReadings(kanjidicXmlPath: str) -> dict:
    tree = ET.parse(kanjidicXmlPath)
    root = tree.getroot()

    readings = {}
    for char in root.findall("character"):
        literal = char.findtext("literal")
        if not literal:
            continue

        onList = []
        kunList = []
        pinyinList = []
        for rm in char.findall("reading_meaning/rmgroup"):
            for r in rm.findall("reading"):
                rType = r.get("r_type")
                text = (r.text or "").strip()
                if not text:
                    continue
                if rType == "ja_on":
                    onList.append(text)
                elif rType == "ja_kun":
                    # Strip position-only markers ("-" prefix/suffix means this
                    # reading only occurs as the first/last element of a compound);
                    # keep "." (marks the okurigana boundary, e.g. "あ.う").
                    kunList.append(text.strip("-"))
                elif rType == "pinyin":
                    # Mandarin reading(s), e.g. "shi2", "hui4"/"kuai4" for a polyphonic
                    # character. Used only as a last-resort constructed pinyin for
                    # wasei-kango not found in CC-CEDICT (Milestone 3) - never presented
                    # as an attested Chinese reading.
                    pinyinList.append(text)

        if onList or kunList or pinyinList:
            readings[literal] = {"on": onList, "kun": kunList, "pinyin": pinyinList}

    return readings


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python build_readings.py /path/to/kanjidic2.xml")
        sys.exit(1)

    readings = buildReadings(sys.argv[1])
    outPath = Path(__file__).parent / "readings.json"
    with open(outPath, "w", encoding="utf-8") as f:
        json.dump(readings, f, ensure_ascii=False, separators=(",", ":"))

    print(f"Wrote {len(readings)} characters to {outPath}")
