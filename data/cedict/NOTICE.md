# Source of `index.json`

`index.json` is a compact derivative of **CC-CEDICT**, a community-maintained free Chinese-English
dictionary published by [MDBG](https://www.mdbg.net/), extracting just the Traditional headword,
pinyin, and gloss fields via `build_index.py`.

- Source file: `cedict_1_0_ts_utf-8_mdbg.txt.gz` from
  https://www.mdbg.net/chinese/export/cedict/cedict_1_0_ts_utf-8_mdbg.txt.gz (not committed here —
  only the small derived `index.json` is; re-run `build_index.py` against a fresh copy to update).
- License: CC-CEDICT is licensed under a Creative Commons Attribution-ShareAlike 4.0 International
  License — see https://creativecommons.org/licenses/by-sa/4.0/. Referenced work: CEDICT, copyright
  1997, 1998 Paul Andrew Denisowski. This project is not affiliated with MDBG.

Used by `core/kanji_reference.py: lookupChineseCognate()` to decide whether an on'yomi/mixed
Japanese word has a real, attested Chinese equivalent (vs. being a Japan-coined term not found in
Chinese) — see `.claude/KANJI_REFERENCE_PLAN.md` for the full algorithm.
