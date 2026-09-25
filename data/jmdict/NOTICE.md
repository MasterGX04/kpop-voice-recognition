# Source of `index.json`

`index.json` is a compact derivative of **JMdict**, published by the
[Electronic Dictionary Research and Development Group (EDRDG)](https://www.edrdg.org/), extracting
just the kanji-headword -> {readings, part-of-speech, English gloss} fields via `build_index.py`.

- Source file: `JMdict_e` (the English-glosses-only subset) from
  http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz (not committed here — only the derived
  `index.json` is; re-run `build_index.py` against a fresh copy to update).
- License: EDRDG's dictionary files, including JMdict, are made available under a
  Creative Commons Attribution-ShareAlike 4.0 licence — see https://www.edrdg.org/edrdg/licence.html.
  This project is not affiliated with EDRDG.

Used by `core/kanji_reference.py: lookupJapaneseMeaning()` to show a word's own English meaning -
run for **every** Kanji word regardless of on'yomi/kun'yomi classification, unlike
`lookupChineseCognate()` (CC-CEDICT, onyomi/mixed only). This is the piece that covers kun'yomi
words (native Japanese vocabulary with no Chinese-cognate shortcut, e.g. 離す) and catches
same-reading/different-kanji homophones (離す "to separate" vs 話す "to speak", both はなす) and
same-kanji/different-reading splits (湯 read ゆ = "hot water" vs the same kanji read タン,
borrowed for the Chinese sense "soup") - see `.claude/KANJI_REFERENCE_PLAN.md` for the full
Milestone 6 writeup.
