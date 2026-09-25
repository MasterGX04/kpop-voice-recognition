# Source of `readings.json`

`readings.json` is a compact derivative of **KANJIDIC2**, published by the
[Electronic Dictionary Research and Development Group (EDRDG)](https://www.edrdg.org/), extracting
just the per-character `ja_on` (on'yomi, katakana) and `ja_kun` (kun'yomi, hiragana) reading fields
via `build_readings.py`.

- Source file: `kanjidic2.xml.gz` from http://www.edrdg.org/kanjidic/kanjidic2.xml.gz (not committed
  here — only the small derived `readings.json` is; re-run `build_readings.py` against a fresh copy
  to update).
- License: EDRDG's dictionary files, including KANJIDIC2, are made available under a
  Creative Commons Attribution-ShareAlike 4.0 licence — see https://www.edrdg.org/edrdg/licence.html.
  This project is not affiliated with EDRDG.

Used by `core/kanji_reference.py: classifyReading()` to determine whether a word's reading is
on'yomi, kun'yomi, mixed, or a non-decomposable special reading (jukujigo).
