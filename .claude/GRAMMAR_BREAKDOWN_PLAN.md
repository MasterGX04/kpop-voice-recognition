# Grammar Breakdown Layer — Technical Plan

Design doc for the "grammar breakdown" idea named in `.claude/JAPANESE_LEARNING_PLAN.md`'s "What's
next" section: tokenize a highlighted Japanese lyric line into a stem + particle + conjugation
chain (e.g. 食べたくない = 食べ[eat] + たく[want] + ない[negative]), using the same `fugashi`
tagger the Kanji Reference Collector already uses - so agglutinative conjugation is visible
instead of something you have to reverse-engineer by eye. This is the durable version of the
planning conversation that scoped it, kept here the same way `.claude/KANJI_REFERENCE_PLAN.md`
preserved that project's design reasoning.

## The goal, pinned down by a worked example

- `食べたくない` (tabetakunai, "don't want to eat") → three pieces, each independently glossed:
  - `食べ` — the verb stem, dictionary form 食べる, "to eat"
  - `たく` — the conjunctive form of auxiliary たい, "want to"
  - `ない` — the negative auxiliary, "not" / negation
- The point isn't to auto-generate a polished English sentence ("I don't want to eat") - it's to
  make the *pieces* visible, the same way `resolveContextReading()` already makes a Kanji word's
  in-context reading visible instead of something you'd have to guess. Composing the final meaning
  from the pieces is left to the reader, same spirit as the existing "Convert Kanji → Reading"
  button just shows the reading rather than a translation.

## What's already built and reusable vs. what's actually new

This is a smaller lift than the Kanji Reference Collector was, because two of its three pieces
already exist:

1. **Tokenizing a whole line with `fugashi`** - already built (`core/japanese_utils.py: getTagger()`,
   already reused by `core/kanji_reference.py: resolveContextReading()`). No new tokenization code
   needed, same as Milestone 5 needed none for its full-line scanning.
2. **English meaning for the content words** (verbs, adjectives, nouns) - already built. Confirmed
   directly: `lookupJapaneseMeaning(lemma, reading)` (Milestone 6, JMdict) already resolves 食べる
   → "to eat", 行く → "to go", 見る → "to see", 話す → "to speak", 居る → "to be" correctly, with
   zero new code. The content-word half of a grammar breakdown is free.
3. **What's actually new**: English glosses for the *function words* - particles (助詞) and
   auxiliary verbs (助動詞) - which JMdict's existing index in this project **cannot** currently
   answer. Confirmed why: `data/jmdict/build_index.py` only indexes entries that have a Kanji
   spelling (`if not kebs: continue` - see its own docstring), because every previous lookup in
   this project only ever queried Kanji-containing words. Particles and auxiliary verbs are almost
   always pure kana (て, は, が, ない, た, ます, たい, ...) - they were never indexed, and even if
   they were, JMdict's own glosses for pure-grammar entries are written for lexicographers, not
   as a quick "one phrase per token" mnemonic (e.g. its real entry for た runs several dense
   grammatical-note lines, not a short "past tense" gloss). This needs its own small, hand-curated
   dictionary - not a bigger download.

## Why a hand-curated dictionary is the right call here (not another JMdict-style bulk import)

Unlike Kanji vocabulary (tens of thousands of words, genuinely open-ended - hence needing
KANJIDIC2/CC-CEDICT/JMdict), Japanese function words are a **small, closed class**. Confirmed by
scanning the real corpus this tool already has - every `助詞`/`助動詞` token across every
Japanese-detected line in `saved_labels/*/*_lyrics.json`, counted by lemma:

```
59 の(助詞)   35 て(助詞)   33 だ(助動詞)  30 に(助詞)   24 は(助詞)
22 で(助詞)   22 も(助詞)   18 が(助詞)   17 ない(助動詞) 17 を(助詞)
16 た(助動詞) 15 てる(助動詞) 8 よ(助詞)    6 から(助詞)   6 と(助詞)
4 れる(助動詞) 4 だけ(助詞)  4 か(助詞)    3 へ(助詞)     3 ね(助詞)
3 まで(助詞)  3 たい(助動詞) 2 ば(助詞)    2 られる(助動詞) 2 ほど(助詞)
2 ず(助動詞)  1 しか/より/じゃん/な/さ(助詞) 1 てく/ます/ちゃう(助動詞)
```

The top ~20 lemmas already account for the overwhelming majority of every particle/auxiliary
token in the whole real library. This is exactly the same shape of problem Milestone 1 solved
with a small pure-Python lookup table instead of a heavyweight dependency (`toTraditional()`
against ~2 small `.txt` files) - a hand-curated `{lemma: gloss}` dict, seeded from this real
frequency list and expanded opportunistically as new songs surface new particles, is simpler,
more reliable, and more pedagogically legible (a clean "want to" beats a paraphrased dictionary
entry) than trying to repurpose JMdict for something it was never indexed for.

## Algorithm design

New module `core/grammar_breakdown.py` (kept separate from `core/kanji_reference.py` - this is a
whole-sentence structural view, not a per-Kanji-word vocab lookup, even though it reuses pieces of
that module's infrastructure):

1. Tokenize the full highlighted text with the shared tagger (`core.japanese_utils.getTagger()`) -
   same reuse pattern as `resolveContextReading()`.
2. For each token, route by `word.feature.pos1`:
   - `動詞`/`形容詞`/`名詞`/`形状詞` (verb/adjective/noun/adjectival-noun) → **content word**.
     Look up `lookupJapaneseMeaning(lemma, lemmaReading)` (reusing Milestone 6 directly, imported
     from `core.kanji_reference`) for its gloss. A kana-only content word (adverbs like とても) will
     come back `not_found` since JMdict is Kanji-keyed - a known gap, see Known limits.
   - `助詞` (particle) / `助動詞` (auxiliary verb) → **function word**. Look up `word.feature.lemma`
     in the new curated `_PARTICLE_GLOSSES` / `_AUXILIARY_GLOSSES` dicts (seeded from the real
     frequency list above). Missing lemma → fall back to showing the bare lemma with no gloss,
     never fabricate one - same "don't guess" discipline as the rest of this project.
   - `補助記号`/`記号` (punctuation) → skipped entirely from the breakdown output (nothing to show).
   - Anything else (副詞 adverbs, 接頭辞 prefixes, 接続詞 conjunctions, etc.) → shown with its
     lemma but no gloss lookup in Phase 1 (falls in the "known gap" bucket below - expand
     opportunistically once real examples show it matters, same as the particle dict).
3. Return one entry per token (skipping punctuation): `{"surface", "lemma", "pos1", "role", "gloss"}`
   where `role` is `"content"` or `"function"`, and `gloss` is `None` when nothing was found (the
   caller decides how to render a missing gloss, never silently drops the token).

No new tokenization/offset logic needed, same as Milestone 5 - `breakdownLine(text)` just runs the
tagger over the whole highlighted string directly (this feature always wants the *whole* line's
structure, not a single highlighted word, so it doesn't need `resolveContextReading()`'s
highlight-overlap logic at all - simpler than the Kanji Reference Collector in this one respect).

## UI wiring (using the current Lyrics Editor UI, no new dialog needed)

Mirrors Milestone 1's own bootstrapping exactly: add one more button to the same
`japaneseToolsFrame` row in `gui/lyrics_editor.py` that already holds "Convert Kanji → Reading" and
"Add Kanji to Document" (`gui/lyrics_editor.py` around line 432-498) - call it **"Break Down
Grammar"**. Phase 1 wiring, as a test harness (same pattern Milestone 1 used before Milestone 4
added real persistence):

- Reuse the exact same selected-text-or-whole-field handling `addKanjiToDocument()` already has
  (`koreanEntry.get("1.0", "end-1c")`, the `Text.count()` `None`-guard for a highlight starting at
  the very first character - already fixed once this session, no need to rediscover it).
- Call `breakdownLine(selectedText)`, then show the per-token results in a `messagebox.showinfo()`,
  same UX as the existing two buttons - one line per token, e.g.:
  ```
  食べ (content, verb) — to eat
  たく (function) — want to
  ない (function) — not / negative
  ```
- No persistence in Phase 1 (matches how the Kanji Reference Collector didn't get real persistence
  until Milestone 4, three milestones after its first display-only button).

## Milestones

- **Phase 0** (this doc) — done, 2026-09-08. Scoped and grounded in real data (the frequency
  scan above) before writing any code, same discipline as every Kanji Reference milestone.
- **Phase 1** (not built) — the minimum real, testable version:
  - `core/grammar_breakdown.py`: `_PARTICLE_GLOSSES`, `_AUXILIARY_GLOSSES` (seeded from the top
    ~20-30 real lemmas above), `_classifyToken(word)`, `breakdownLine(text) -> list[dict]`.
  - "Break Down Grammar" button in `gui/lyrics_editor.py`'s Japanese tools row, wired as a display
    test harness per the UI section above.
  - Tests in a new `core/test_grammar_breakdown.py` (stdlib `unittest`, matching
    `core/test_kanji_reference.py`'s own style) - at minimum the worked example (食べたくない),
    a couple of the real frequent particles/auxiliaries from the scan above, and a sentence with
    at least one gap (an unglossed particle/auxiliary or a kana-only adverb) to confirm the
    "show the bare lemma, never fabricate a gloss" fallback behaves correctly instead of crashing
    or silently dropping the token.
  - Verification step before calling Phase 1 done (same pattern as every milestone before it):
    run `breakdownLine()` across every real Japanese-detected line in `saved_labels/` (reusing
    `core.kanji_reference._isJapaneseLyricEntry`/`scanLyricsForKanjiVocab`'s own scanning loop as
    a template) and report what fraction of function-word tokens across the *whole* real library
    already have a curated gloss vs. fall back to a bare lemma - use that gap list to decide
    whether the seed dictionary needs a quick top-up before Phase 1 is genuinely usable end to end.
- **Phase 2+** (not scoped in detail yet - revisit once Phase 1 is used for real) — candidates
  raised while designing this, deliberately left open rather than committed to:
  - Persist breakdown results somewhere (their own file? folded into the existing per-song vocab
    JSON from Milestone 4?) instead of a one-off messagebox, the same jump the Kanji Reference
    Collector made from Milestone 1 to Milestone 4.
  - Surface `cType`/`cForm` (already available from `fugashi`, confirmed via testing - e.g.
    `連用形-一般`, `未然形-一般`) as an explicit conjugation-form label per token, not just the
    plain-English gloss - useful once the basic per-token gloss chain has been used for a while and
    proven the simpler version is genuinely too shallow on its own.
  - Expand `_PARTICLE_GLOSSES`/`_AUXILIARY_GLOSSES` coverage opportunistically as new songs are
    added and produce unglossed function words - re-run the same frequency scan periodically
    rather than trying to hand-curate all of Japanese grammar up front.
  - A curated fallback list for the highest-frequency kana-only *content* words (adverbs like
    とても, もう) that JMdict's Kanji-keyed index can't answer - only worth doing if the
    verification step in Phase 1 shows this is actually a common gap in practice, not a
    hypothetical one.

## Known limits (to flag up front, same posture as every other tier in this project)

- **A single gloss per particle is a real simplification.** Japanese particles are genuinely
  polysemous depending on context - に alone can mark a location, a time, an indirect object, or a
  purpose depending on the sentence. Phase 1 shows one representative gloss per lemma, not a
  context-sensitive sense-disambiguated one (the same category of simplification KANJIDIC2's
  `_matchSegmentation()` already accepts for on'yomi/kun'yomi matching - a strong draft, not a
  contextual final answer).
- **Kana-only content words (adverbs, etc.) get no gloss in Phase 1** - JMdict's index here is
  Kanji-keyed only (see above), so a token classified "content" but written in pure kana will
  report `not_found` from `lookupJapaneseMeaning()`, same as any kana-only word already does
  elsewhere in this project.
- **Auxiliary-verb chains can be genuinely ambiguous in fugashi's own segmentation** the same way
  Kanji reading resolution can be (see `KANJI_REFERENCE_PLAN.md`'s 出そう/出る-vs-出す finding) -
  this plan doesn't attempt to fix tokenizer-level ambiguity, only to gloss whatever `fugashi`
  actually returns.
