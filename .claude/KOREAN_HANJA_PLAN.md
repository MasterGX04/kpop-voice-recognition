# Korean Hanja Converter — Technical Plan

Design doc for extending this tool's "show the Chinese-character connection" feature to Korean:
highlight a Korean word in the Lyrics Editor and see its Hanja (Chinese character) origin, mirroring
the existing Japanese "Add Kanji to Document" button. This is the durable version of the planning
conversation that scoped it, kept the same way `.claude/KANJI_REFERENCE_PLAN.md` preserved that
project's design reasoning - including a real dead end (a Korean government API requiring
phone/carrier verification) and the pivot away from it, so that dead end isn't rediscovered later.

## Why this is a genuinely harder problem than the Japanese case

The Japanese Kanji Reference Collector's core trick was: Japanese text still *writes* Kanji
directly, so "finding the Chinese connection" is a character-substitution problem
(`toTraditional()`, Shinjitai → Traditional) followed by a dictionary lookup. **Modern Korean text
no longer writes Hanja at all** - it's 100% Hangul. So this isn't a substitution problem; it's a
**homophone disambiguation** problem: given a Hangul spelling, which of potentially several
unrelated Hanja characters/words does it represent? Confirmed directly against real data (see
below): 화 alone corresponds to at least six unrelated Hanja words in real usage - 火 (fire), 禍
(misfortune), 和 (harmony), 化 (suffix), 畫 (picture), 靴 (shoe) - each a real, distinct
Sino-Korean word that happens to share the same modern pronunciation. There is no shortcut around
needing a real **word-level** dictionary that records each Hangul word's specific Hanja origin -
character-level reading data alone (e.g. Unicode's Unihan `kHangul` field) can only ever produce
an unconfirmed multi-candidate guess, the same category of honesty problem `_fallbackPinyin()`
already solves for wasei-kango by clearly labeling itself "constructed, not real."

## Data source investigation (what was actually checked, in order)

1. **표준국어대사전 (Standard Korean Dictionary) / 우리말샘 (Open Korean Dictionary)**, both
   published by South Korea's National Institute of Korean Language, confirmed real via direct
   fetch (`stdict.korean.go.kr`, `opendict.korean.go.kr`) and confirmed as the underlying data
   source behind `gukhanmun` (a real Rust tool doing the *reverse* direction - Hanja→Hangul -
   confirming these two dictionaries do contain adequate Hanja-word data). **Dead end for this
   user**: their Open API developer registration requires Korean real-name verification
   (본인인증) via a Korean mobile carrier or resident ID - not available to someone without a
   Korean phone plan. Also checked `krdict.korean.go.kr` (the same institute's dictionary
   specifically for foreign learners) as a possible workaround - it also routes through the same
   government Open API portal, so it's not confirmed to avoid the same wall. **Not pursued
   further** - flagging here so this path isn't re-investigated without a reason to believe the
   verification requirement has changed.
2. **`hanja` PyPI package (`suminb/hanja`)** - the option an earlier session had already flagged as
   weak; re-verified directly via its own README example (`hanja.translate('大韓民國은
   民主共和國이다.', 'substitution')`). Confirmed why it's the wrong shape: it's built for
   **Hanja→Hangul** substitution (the easy, deterministic direction - each Hanja character has a
   small fixed set of Korean readings), not Hangul→Hanja disambiguation (the hard, ambiguous
   direction this feature actually needs). Not usable as the primary source.
3. **`hanja-tagger` (`kaniblu/hanja-tagger`)**, suggested by the user - does perform real
   Hangul→Hanja tagging (`안녕하세요` → `안녕(安寧)하세요`), but it's a **live web scraper** against
   a third-party site (Hanjaro, `hanjaro.juntong.or.kr`), not a local dictionary - fragile
   (depends on an external site staying up and unchanged), explicitly **non-commercial/research-use
   only** per its own disclaimer, and has had only 4 commits total. Not usable as a dependency.
4. **`cihai`** (Python, MIT, actively maintained - 2,391 commits) wraps Unicode's Unihan database
   and does expose Korean per-character data (`kHangul`, `kKorean` fields) with zero registration.
   Real and usable, but only for the same character-level reading data described above - useful as
   a *secondary*, honestly-labeled "possible reading" tier, not a primary confirmed-word source.
5. **Wiktionary, via `kaikki.org`'s pre-extracted Korean dataset - the recommended primary
   source.** English Wiktionary documents Sino-Korean etymology directly on each word's page (the
   same way it documents Japanese/Chinese cognates), and `kaikki.org` runs the open-source
   `wiktextract` tool to publish this as ready-to-use structured JSONL, refreshed periodically,
   **no registration, no API key, no phone verification** - freely downloadable like
   `kanjidic2.xml`/`JMdict_e`/`cedict_1_0_ts_utf-8_mdbg.txt` already are in this project.
   **Verified directly, not just from the landing page**: downloaded the actual file
   (`kaikki.org-dictionary-Korean.jsonl`, ~201MB, 57,252 distinct Korean words) and confirmed real
   structured entries for every word tested - 학교 → `Sino-Korean word from 學校, from 學 ("learn")
   + 校 ("school")` via a machine-parseable `ko-etym-Sino`/`ko-etym-sino` template (args give the
   Hanja string and per-character gloss directly); same confirmed for 시간→時間, 감사→感謝,
   가족→家族, 공부→工夫, 친구→親舊. **17,333 of the 57,252 words (~30%) carry a confirmed
   Sino-Korean etymology template** - real, substantial coverage. **Homograph coverage confirmed
   real, not hypothetical**: 화 has exactly the six separate entries listed above, each its own
   JSONL record with its own Hanja and gloss - this dataset already represents Korean's homophone
   ambiguity honestly, the same way CC-CEDICT/JMdict already do for Chinese/Japanese homographs.
   License: Wiktionary content is CC BY-SA (+ GFDL) - same license family already handled in this
   project (KANJIDIC2/JMdict are also CC BY-SA), so no new legal territory.

## Recommended approach

Primary: a derived index built from the `kaikki.org` Korean Wiktionary extraction, following the
**exact same pattern** already used three times in this project (`data/kanjidic2/`,
`data/cedict/`, `data/jmdict/`) - download the raw source once (not committed, per this project's
existing `.gitignore`/`NOTICE.md` convention), parse it with a small one-time build script into a
compact derived JSON that *is* committed, and never re-touch the 201MB raw file again unless
refreshing the data later.

- New `data/ko_wiktionary/build_index.py`: parses `kaikki.org-dictionary-Korean.jsonl`, keeping
  only `lang_code == "ko"` entries whose `etymology_templates` include a `ko-etym-sino`/
  `ko-etym-Sino` (or similarly-named) template, extracting `{hanja, gloss, pos}` per entry.
  Index keyed by the Hangul `word` field, one entry per homograph (mirrors `data/cedict/index.json`'s
  own `{headword: [entries]}` shape exactly - no new data-modeling pattern needed).
- New `data/ko_wiktionary/NOTICE.md`: same format as the other three `NOTICE.md` files - source
  URL, license (CC BY-SA/GFDL via Wiktionary, extracted by `wiktextract`), and what's derived.
- New `core/korean_hanja.py` (kept separate from `core/kanji_reference.py`, same reasoning as
  `core/grammar_breakdown.py` - a different script/language axis, not a per-Kanji-word extension):
  `lookupHanja(hangulWord) -> list[dict]`, returning every homograph candidate
  `{"hanja": ..., "gloss": ..., "pos": ...}` for a given Hangul spelling - **always a list**, since
  showing all real candidates (as CC-CEDICT/JMdict-style multi-sense entries already do elsewhere
  in this project) is the honest answer when a word is genuinely ambiguous (화's six candidates),
  not something to silently guess down to one. An unambiguous word (only one Sino-Korean entry, or
  a native word with none) naturally returns a one-item or empty list - no special-casing needed.
- Secondary/fallback tier (lower priority, optional): `cihai`'s Unihan wrapper for a
  character-by-character "possible reading" hint when a whole word isn't in the Wiktionary
  extraction at all - clearly labeled as unconfirmed, same honesty posture as
  `_fallbackPinyin()`'s "constructed, not real" pinyin for wasei-kango.

### UI wiring

Confirmed directly in `gui/lyrics_editor.py`: there is currently **no Korean-specific tools row**
at all - only `japaneseToolsFrame` exists, shown/hidden by `switchLanguage()` based on
`langVar.get() == "Japanese"` (around line 348-362). Plan: add a parallel `koreanToolsFrame`,
shown when `langVar.get() == "Korean"`, holding one button - **"Show Hanja"** - that reuses the
exact same highlighted-text handling `addKanjiToDocument()` already has (the `Text.count()`
`None`-guard for a highlight starting at the very first character, etc.) and calls
`lookupHanja(selectedText)`, showing results in a `messagebox.showinfo()` the same way the
existing two Japanese buttons do - e.g. for 화 highlighted alone:
```
화 — possible Hanja:
  火 (fire)
  禍 (misfortune)
  和 (harmony)
  化 (suffix)
  畫 (picture)
  靴 (shoe)
```
No persistence in Phase 1 - same bootstrapping-as-a-test-harness pattern Milestone 1 and the
Grammar Breakdown plan both used before any persistence was added.

## Milestones

- **Phase 0** (this doc) - done, 2026-09-08. Includes a real dead end (government API phone
  verification) documented so it isn't re-investigated without new information.
- **Phase 1** (not built) - the minimum real, testable version:
  - Download `kaikki.org-dictionary-Korean.jsonl` (not committed), build
    `data/ko_wiktionary/build_index.py` → `data/ko_wiktionary/index.json`.
  - `core/korean_hanja.py: lookupHanja()`.
  - "Show Hanja" button + `koreanToolsFrame` in `gui/lyrics_editor.py`.
  - `core/test_korean_hanja.py` (stdlib `unittest`, matching this project's existing test style) -
    at minimum the worked examples confirmed above (학교→學校, 화's six-way homograph list, a
    native word with zero Sino-Korean entries returning an empty list correctly).
  - Verification step before calling Phase 1 done: run `lookupHanja()` against real Korean lines
    in `saved_labels/*/*_lyrics.json` and spot-check the results look right, same discipline as
    every other milestone in this project.
- **Phase 2+** (not scoped in detail yet):
  - The `cihai`/Unihan fallback tier for words missing from the Wiktionary extraction.
  - Context-based disambiguation (pick the most likely candidate instead of listing all of them) -
    deliberately deferred; listing all real candidates is already useful and honest, and smarter
    disambiguation is a strictly harder NLP problem worth revisiting only if the plain list proves
    too noisy in practice.
  - Persistence (a per-song Hanja vocab file, mirroring Milestone 4's `*_kanji_vocab.json`) once
    the display-only version has been used for real.
  - Periodically re-downloading and rebuilding the index as Wiktionary/kaikki.org's data improves.

## Known limits (flagged up front, same posture as every other tier in this project)

- **Historical-drift words are a real trap.** 사랑 ("love") is etymologically linked to Sino-Korean
  사량 (思量) per Wiktionary, but the modern word is fully nativized and no fluent speaker thinks
  of it as a Chinese-character word today - showing "사랑 = 思量" without qualification would be
  misleading in the same way `氣分`'s "confirmed cognate" needed the false-friend caveat on the
  Japanese side. Worth surfacing a similar caveat in the UI once this is built, not just trusting
  every etymology template at face value.
- **~30% real coverage, not exhaustive.** 17,333 of 57,252 words - substantial, but
  community-maintained Wiktionary content will have real gaps compared to a professional
  dictionary; a missing word means "not yet documented," not "definitely not Sino-Korean."
- **Homophone ambiguity is shown, not resolved, in Phase 1.** 화's six-way candidate list is the
  honest answer given only a bare highlighted word with no sentence context - same category of
  simplification as showing one representative particle gloss per lemma in the Grammar Breakdown
  plan.
- **The raw source file is ~201MB and not committed**, same convention as `kanjidic2.xml`/
  `JMdict_e`/`cedict_1_0_ts_utf-8_mdbg.txt` - rebuilding `index.json` later requires
  re-downloading it fresh from `kaikki.org`.
