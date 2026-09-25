# Korean Grammar Breakdown Layer — Technical Plan

Design doc for the Korean sibling of `.claude/GRAMMAR_BREAKDOWN_PLAN.md`: tokenize a highlighted
Korean lyric line into a stem + particle + ending chain, the same way the Japanese version uses
`fugashi`. This one uses **Kiwi** (`kiwipiepy`, https://github.com/bab2min/kiwipiepy) instead.
Verified directly against a real user-supplied lyric line before writing anything here - not
speculative, same discipline as every other plan doc in this project.

## The goal, pinned down by the user's own worked example

Line supplied for testing:
```
목소릴 따라 너의 호흡을 따라
다 전해져 어떤 아픔 어떤 슬픔
```

Real Kiwi output (installed and run directly, not simulated):

| Surface | Tag | Meaning | Lemma |
|---|---|---|---|
| 목소리 | NNG | noun, "voice" | 목소리 |
| ㄹ | JKO | object particle (contracted 를) | ㄹ |
| 따르 | VV | verb stem, "to follow" | 따르다 |
| 어 | EC | connective ending | 어 |
| 너 | NP | pronoun, "you" | 너 |
| 의 | JKG | genitive particle | 의 |
| 호흡 | NNG | noun, "breath" | 호흡 |
| 을 | JKO | object particle | 을 |
| 다 | MAG | adverb, "all/completely" | 다 |
| 전하 | VV | verb stem, "to convey" | 전하다 |
| 어 | EC | connective ending | 어 |
| 지 | VX | auxiliary verb, "to become/passive" | 지다 |
| 어 | EC | connective ending | 어 |
| 어떤 | MM | determiner, "what kind of" | 어떤 |
| 아픔 | NNG | noun, "pain" | 아픔 |
| 슬픔 | NNG | noun, "sadness" | 슬픔 |

Two results here are worth pinning down as *why this is exactly the target feature*, not just a
plausible-looking tokenization:

- **전해져** decomposed into 전하(다) ["to convey"] + 어 [connective] + 지(다) [passive/inchoative
  auxiliary] + 어 [connective] - a real four-morpheme agglutinative chain ("comes to be conveyed" →
  "comes through"), the same shape of thing 食べたくない (食べ+たく+ない) is for Japanese. This is
  direct proof Kiwi exposes real conjugation structure, not just word-segmentation.
- **목소릴** is a casual, contracted spoken form (목소리+를, object marker shortened to ㄹ) - real
  lyric-register Korean, not textbook Korean - and Kiwi parsed it correctly with zero
  special-casing. Lyrics are full of exactly this kind of contraction, so this was a meaningful
  real-world robustness check, not a toy example.

## Direct comparison to fugashi (the reference point requested)

| | `fugashi` (Japanese, already in use) | `kiwipiepy`/Kiwi (Korean) |
|---|---|---|
| Dictionary/citation form | `word.feature.lemma` | `token.lemma` - confirmed real above (따르다, 전하다, 지다) |
| POS/grammatical role | `word.feature.pos1` (動詞/助詞/助動詞/...) | `token.tag` - the standard **Sejong tagset** (VV/VA verb/adjective, JK*/JX particles, E* endings, VX auxiliary verb, NN* nouns, MM/MAG modifiers/adverbs, X* affixes, S* punctuation) |
| Conjugation detail | `word.feature.cType`/`cForm` | Not separately exposed the same way - conjugation is instead *visible directly in the token chain itself* (전해져's 4-token split already shows the auxiliary structure, without needing a separate conjugation-form field) |
| Character offsets | **Not built in** - `resolveContextReading()` had to reconstruct offsets manually by summing `word.surface`/`word.white_space`, which is exactly the bug this project found and fixed once already (a dropped space/newline silently desynced the count) | **Built in**: `token.start` / `token.len` are explicit fields on every token. A Korean equivalent of `resolveContextReading()` would not need to rebuild that fragile offset-tracking logic at all - a real, concrete robustness advantage, not just a style difference |
| Underlying engine | MeCab + unidic-lite (statistical, dictionary-based) | Kiwi's own statistical+dictionary hybrid model, actively maintained (bab2min) | 
| License | unidic-lite: BSD-ish/permissive | **LGPL v3** - meaningfully different from `gukhanmun`'s GPL-3 (flagged as risky in the Hanja plan): LGPL is specifically designed to be safe to depend on as a library regardless of this project's own license, as long as Kiwi's own source isn't modified |
| Install | already a dependency | `pip install kiwipiepy` (pulls a ~90MB bundled model, `kiwipiepy_model`) - confirmed installs cleanly in this project's venv |

Net assessment, matching the user's own read: **this actually is easier than the Japanese case**,
for one concrete, structural reason - Kiwi's explicit token offsets remove an entire category of
bug (the whitespace-drift class) that had to be discovered and fixed by hand for the Japanese side.

## What's reused vs. what's new (mirrors the Japanese plan's own honesty about this)

1. **Tokenization** - new, but a thin wrapper: `core/korean_utils.py: getKiwi()`, a singleton
   accessor exactly mirroring `core/japanese_utils.py: getTagger()`'s role for `fugashi`.
2. **English meaning for content words** (nouns/verbs/adjectives) - **already available without
   writing a new parser**: confirmed directly that `kaikki.org`'s Korean Wiktionary JSONL (the
   same 201MB file already downloaded for `.claude/KOREAN_HANJA_PLAN.md`) has full `senses`/
   `glosses` data for ordinary native words too, not just the Sino-Korean-tagged ones - 목소리→
   "voice", 아픔→"pain", 슬픔→"sadness", 따르다→"to follow", 전하다→"to pass along; to convey",
   지다→"to fall; to sink; to set", 예쁘다→"to be pretty, lovely, beautiful" all confirmed live.
   **This means the Hanja plan's `data/ko_wiktionary/build_index.py` should be widened up front**
   to keep a `gloss`/`pos` field for *every* Korean word it processes, not just the ones with a
   Sino-Korean etymology template - one parse of the 201MB file then serves both plans, instead of
   parsing it twice. Worth doing before/alongside Phase 1 of the Hanja plan, not after.
3. **What's actually new**: glosses for the *function words* - particles (JK*/JX) and endings
   (E*/VX) - same shape of gap as the Japanese plan found in JMdict, and the same fix: a small
   hand-curated dictionary, since Korean function words are also a small, closed class.

## Why a hand-curated dictionary is right here too (same reasoning, new real numbers)

Scanned every Korean-tagged, Hangul-containing line in `saved_labels/*/*_lyrics.json` with Kiwi,
counting every non-content-word token by `(tag, lemma)`:

```
277 EC 어      190 SP ,       106 ETM ᆫ      85 JKS 이      84 JKO ᆯ
83  JKG 의     80 JX ᆫ        72 SSC ’       72 EC 게        70 ETM ᆯ
68  EF 어      66 SL I        56 VCP 이다     54 ETM 는       54 JKB 에
50  JKO 을     50 EC 고       47 XSA 하       46 XSV 하       45 EC 지
40  ETM 은     39 NNB 거      34 JX 은        33 JKS 가       32 EC 어도
32  JKO 를     30 SL that/it/my/t/s (English words mixed into lyrics)
28  JX 는      28 VX 하다     27 EC 어야      26 SS '
```

Same shape of result as the Japanese scan: a handful of endings/particles (어, ᆫ/은/는, 이/가,
을/를, 의, 게, 고, 지, 하다, 이다) account for the overwhelming majority of every function-word
token in the real corpus. `SP`/`SS`/`SSC` (punctuation) and `SL` (foreign-script words, mostly
English lyric fragments like "I"/"that"/"my") are skipped entirely from the breakdown, same
treatment as `補助記号` in the Japanese plan.

## Algorithm design

New `core/korean_grammar_breakdown.py` (kept separate from `core/korean_hanja.py`, same reasoning
as keeping `core/grammar_breakdown.py` separate from `core/kanji_reference.py` - a whole-sentence
structural view, not a per-word dictionary lookup, even though it imports that module's meaning
lookup):

1. Tokenize the full highlighted text with `core.korean_utils.getKiwi()`.
2. For each token, route by `token.tag`:
   - `NNG`/`NNP`/`NNB`/`NP`/`NR`/`VV`/`VA`/`VA-I`/`MAG`/`MM`/`IC` (content) → look up
     `lookupKoreanMeaning(token.lemma)` (new function in `core/korean_hanja.py` or a shared
     `core/korean_dictionary.py` - see the shared-infrastructure note above) for its gloss.
   - `JK*`/`JX`/`E*`/`VX`/`VCP` (function) → look up `token.lemma` in new curated
     `_PARTICLE_GLOSSES`/`_ENDING_GLOSSES` dicts, seeded from the real frequency list above.
     Missing lemma → show the bare lemma, never fabricate a gloss - same discipline as every other
     tier in this project.
   - `SP`/`SS`/`SSC`/`SF`/`SE`/`SO`/`SW` (punctuation) → skipped entirely.
   - `SL`/`SH`/`SN` (foreign script/hanja-in-text/numerals actually appearing in the lyric, e.g.
     mixed-in English words) → shown as-is, no gloss lookup attempted.
3. Return one entry per token (skipping punctuation): `{"surface", "lemma", "tag", "role", "gloss"}`,
   `role` being `"content"` or `"function"` - same output shape as the Japanese plan's
   `breakdownLine()`, so the eventual GUI rendering code can be close to identical between the two.

No manual offset-tracking code needed (see the comparison table) - `token.start`/`token.len` are
used directly if a highlight-overlap version is ever wanted; for the whole-line "Break Down
Grammar" button itself (mirroring the Japanese one), the whole highlighted string is just handed
to Kiwi directly, same as `breakdownLine()` does for Japanese.

## UI wiring

Reuses the `koreanToolsFrame` already planned in `.claude/KOREAN_HANJA_PLAN.md` (shown when
`langVar.get() == "Korean"`) - add a second button, **"Break Down Grammar"**, right next to "Show
Hanja". Same selection-handling code, same `messagebox.showinfo()` rendering pattern as every
other button in this row across both languages, e.g. for the reference line's second half:
```
다 (content, adverb) — all / completely
전하 (content, verb) — to convey
  어 (function) — connective
지 (function, auxiliary) — passive/inchoative ("comes to be")
  어 (function) — connective
어떤 (content, determiner) — what kind of
아픔 (content, noun) — pain
```
No persistence in Phase 1, same bootstrapping-as-test-harness pattern used everywhere else in this
project before a feature earns real persistence.

## Milestones

- **Phase 0** (this doc) - done, 2026-09-08. Grounded in a real user-supplied lyric line, a real
  frequency scan of the whole corpus, and a real installed-and-tested library - not speculation.
- **Phase 1** (not built) - the minimum real, testable version:
  - Widen `.claude/KOREAN_HANJA_PLAN.md`'s Phase 1 build script (`data/ko_wiktionary/build_index.py`)
    to also keep a `gloss`/`pos` field for every word, not just Sino-Korean ones - do this once,
    shared by both plans, rather than parsing the 201MB file twice.
  - `core/korean_utils.py: getKiwi()`.
  - `core/korean_grammar_breakdown.py`: `_PARTICLE_GLOSSES`, `_ENDING_GLOSSES` (seeded from the
    real frequency list above), `_classifyToken(token)`, `breakdownLine(text) -> list[dict]`.
  - "Break Down Grammar" button in the shared `koreanToolsFrame` from the Hanja plan.
  - `core/test_korean_grammar_breakdown.py` (stdlib `unittest`, matching this project's style) -
    at minimum the reference line's exact worked breakdown above (particularly 전해져's 4-token
    chain), a couple of the real frequent particles/endings from the scan, and a mixed-script line
    (real lyrics contain English words, per the `SL` findings) to confirm foreign-script tokens
    pass through cleanly instead of crashing or getting a nonsense gloss lookup.
  - Verification step before calling Phase 1 done: run `breakdownLine()` across every real
    Korean-tagged line in `saved_labels/` and report what fraction of function-word tokens already
    have a curated gloss - same discipline as the Japanese plan's own verification step.
- **Phase 2+** (not scoped in detail) - candidates raised while designing this, deliberately left
  open: persistence (fold into a per-song vocab file, maybe shared with the Hanja converter's
  eventual persistence tier); expanding curated coverage as new songs surface new endings/particles;
  a `resolveContextReading()`-style highlight-overlap mode using `token.start`/`token.len` directly
  (would be simpler to build than the Japanese version, per the comparison table above) if a
  single-word (rather than whole-line) breakdown mode is ever wanted.

## Known limits

- **One gloss per particle/ending is a real simplification**, same caveat as the Japanese plan -
  Korean endings are genuinely polysemous (e.g. 는/은 as topic marker vs. contrast marker depends
  on context) and Phase 1 shows one representative gloss per lemma, not a sense-disambiguated one.
- **Content-word coverage depends on the same `kaikki.org` Wiktionary extraction as the Hanja
  plan** - real but not exhaustive (community-maintained), and a missing word means "not yet
  documented," not "not a real word." Same posture as JMdict's coverage caveat on the Japanese
  side.
- **Kiwi's own tag granularity occasionally splits a single conceptual ending into multiple tokens**
  (전해져's 어+지+어 is linguistically real, but a learner glancing at four separate tokens for one
  auxiliary construction may want them visually grouped rather than listed flat) - Phase 1 lists
  tokens flat, same as the Japanese plan's own MVP scope; grouping related tokens into one visual
  "chain" is a Phase 2+ polish item, not a Phase 1 requirement.
