# Source of `index.json`

`index.json` is a compact derivative of the **Korean edition of English Wiktionary**, extracted
via [`kaikki.org`](https://kaikki.org)'s `wiktextract` tool, keeping just the headword, part of
speech, English gloss(es), and (when present) the Hanja spelling declared on the entry, via
`build_index.py`.

- Source file: `kaikki.org-dictionary-Korean.jsonl` from
  https://kaikki.org/dictionary/Korean/kaikki.org-dictionary-Korean.jsonl (not committed here -
  only the derived `index.json` is; re-run `build_index.py` against a fresh copy to update).
- License: Wiktionary content is dual-licensed under Creative Commons
  Attribution-ShareAlike 4.0 International (CC BY-SA) and the GNU Free Documentation License
  (GFDL) - see https://creativecommons.org/licenses/by-sa/4.0/. `wiktextract`
  (https://github.com/tatuylonen/wiktextract) is the open-source extraction tool `kaikki.org` runs
  to publish Wiktionary as structured JSONL. This project is not affiliated with Wiktionary,
  Wikimedia, or kaikki.org.

Used by `core/korean_hanja.py: lookupHanja()` to list every Hanja (Chinese character) candidate a
Hangul word could represent (see .claude/KOREAN_HANJA_PLAN.md), and by
`core/korean_grammar_breakdown.py` for content-word English glosses (see
.claude/KOREAN_GRAMMAR_BREAKDOWN_PLAN.md) - one parse of the source file serves both, since most
entries carry a usable gloss regardless of whether they're also tagged Sino-Korean.
