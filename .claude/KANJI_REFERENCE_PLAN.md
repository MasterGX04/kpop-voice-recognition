# Kanji Reference Collector — Technical Plan

Design reference for a feature that lets you highlight a Kanji word in the Lyrics Editor's Japanese
text field and save it into a personal vocab reference, tagged with on'yomi/kun'yomi and (when real)
its Chinese cognate. This doc is the durable version of the Claude Code planning session that
designed it — kept here so the reasoning survives even if that ephemeral plan file doesn't.

## The goal, pinned down by two worked examples

- `時間` (jikan) — a real Sino-Japanese compound → show Chinese equivalent `時間 / shí jiān`, tag "on'yomi".
- `出会う` (deau) — a native Japanese verb (kun'yomi) → **no** Chinese equivalent shown. A naive
  romaji→pinyin guess ("chu1hui4") would not be a real Chinese word, so the tool must recognize that
  and just tag "kun'yomi" with nothing else.

The whole design exists to get that distinction right automatically, without ever fabricating a fake
Chinese reading for a native Japanese word.

## Where to actually find a Shinjitai↔Traditional dictionary

The 楽/樂/乐 problem is real: Japanese shinjitai, Chinese Traditional, and Chinese Simplified are three
separate character-form standards that only sometimes agree. No need to hand-curate a table — concrete,
named, already-maintained resources exist:

1. **OpenCC's `JPShinjitai*` dictionaries (what's actually implemented)** — the [OpenCC project](https://github.com/BYVoid/OpenCC)
   (Open Chinese Convert) ships a `jp2t.json` config: "New Japanese Kanji (Shinjitai) → Traditional
   Chinese Characters." That config is backed by two plain tab-separated text files,
   `data/dictionary/JPShinjitaiCharacters.txt` and `JPShinjitaiPhrases.txt` (Apache-2.0 licensed).
   **Important finding from testing:** the pure-Python `opencc-python-reimplemented` pip package (the
   obvious lightweight choice) does **not** bundle `jp2t.json` at all — it only ships the standard
   Simplified/Traditional/HK/TW regional configs, so `OpenCC('jp2t')` raises `FileNotFoundError` with
   that package. The real `opencc` package does bundle it, but requires compiling a C++ extension via
   CMake — too heavy for one static lookup table. So instead: the two `JPShinjitai*.txt` files were
   copied directly into `data/jp2t/` (with `LICENSE-OpenCC` and `NOTICE.md` for attribution) and are
   parsed by a small pure-Python longest-match function, `core/kanji_reference.py: toTraditional()` —
   no OpenCC package dependency at all, verified against 楽→樂, 会→會, 学→學, 気→氣, 国→國, 予定→預定,
   一獲千金→一攫千金.
2. **[`cjkvi/cjkvi-variants`](https://github.com/cjkvi/cjkvi-variants)** — a structured, freely licensed
   variant database with `joyo-variants.txt` (Jōyō-kanji shinjitai/kyūjitai pairs) and
   `cjkvi-simplified.txt`. Good as a cross-check/fallback source for characters where `jp2t.json` seems
   wrong or incomplete.
3. **Wikipedia's "Shinjitai" / "Kyūjitai" / "Extended shinjitai" articles** — human-curated reference
   tables, not machine-parseable, but useful for manually eyeballing whether a specific character's
   conversion looks right.
4. Unicode's own **Unihan database** (`kTraditionalVariant`/`kSimplifiedVariant`/`kZVariant` fields) also
   encodes some of these links, but it's general-purpose Han-unification variant data (not Japan-specific)
   and noisier for this exact purpose — not the primary recommendation, listed only as the "official"
   tiebreaker if the sources above disagree.

## How the on'yomi/kun'yomi + Chinese-cognate detection works (algorithm)

**Critical finding from building this (not obvious up front): a single Kanji has no one "true"
reading outside of context.** Tested directly: a bare, context-free `時` tokenizes as **ジ**
(an on'yomi reading) with `fugashi`, but the same character in an actual sentence (`時が止まる`)
correctly resolves to **とき** (toki, kun'yomi) — because the tokenizer has nothing else to
disambiguate on when given an isolated character, so it just falls back to some dictionary
sense. This means classification **must never tokenize the highlighted substring in isolation**
— it has to tokenize the whole surrounding lyric line and read off whichever token(s) the
highlighted range actually overlaps, so the in-context reading fugashi already resolves
correctly gets used. This also means "how much you highlight" changes results at **token**
granularity, not character granularity: highlighting just the first character of a two-Kanji
word that the tokenizer treats as a single token (e.g. selecting only "時" inside "時間が
止まる", where 時間 is one token) still resolves to the whole token's reading (時間/jikan) — to
get 時 read alone as "toki" it has to actually be tokenized as its own word (e.g. select it
in "時が止まる", where 時 and 間 aren't adjacent).

**Second finding: classify the lemma, not the raw conjugated surface.** A highlighted verb like
出会った ("deatta", the た-form of 出会う) has surface okurigana that doesn't match KANJIDIC2's
dictionary-form kun'yomi entry for 会 (`あ.う`) at all — conjugation changes う to っ/わ/え/etc.
depending on form. `fugashi` already exposes each token's `lemma` (dictionary/citation form,
e.g. 出会う) and `lForm` (that lemma's reading, デアウ) — classifying against the **lemma**
instead of the raw surface sidesteps conjugation entirely, since a lemma is always in citation
form. Nouns/particles are unaffected (their lemma equals their surface already).

1. Tokenize the *entire* lyric line/field with the existing `fugashi` tagger (already in
   `core/japanese_utils.py`), then collect whichever tokens overlap the highlighted character
   range (`core/kanji_reference.py: resolveContextReading()`).
2. For each Kanji character in the matched tokens' **lemma**, pull on'yomi/kun'yomi candidates
   from **KANJIDIC2** (EDRDG, CC BY-SA, attribution required) — downloaded `kanjidic2.xml.gz`,
   parsed with stdlib `xml.etree.ElementTree` into `data/kanjidic2/readings.json` via
   `data/kanjidic2/build_readings.py`. Each `<character>` has `<reading r_type="ja_on">`
   (katakana) and `<reading r_type="ja_kun">` (hiragana, okurigana marked with `.`, e.g. `あ.う`
   for 会う; a leading/trailing `-` marks a reading that only occurs as a compound's first/last
   element - stripped during parsing since this build's matcher doesn't need positional info).
3. A small backtracking matcher (`classifyReading()`) tries to fully account for the lemma's
   reading using only on'yomi candidates, then only kun'yomi candidates:
   - Full match, on'yomi only → **"onyomi"** (時間 = 時[ジ]+間[カン] → jikan ✓).
   - Full match, kun'yomi only → **"kunyomi"** (出会う = 出[で]+会[あ.う] → deau ✓).
   - Full match only via mixing on+kun across characters → **"mixed"** (jūbako/yutō).
   - No clean decomposition at all → **"jukujigo"** (今日=kyō, 大人=otona — idiomatic, not
     decomposable per character; okurigana that doesn't literally match any candidate also lands
     here, which is the correct/expected fallback, not a bug).
4. Chinese-equivalent lookup (Milestone 3, done) — **only for "onyomi"/"mixed" results, never for
   "kunyomi"/"jukujigo"**:
   - Convert the word to Traditional Chinese forms via `core/kanji_reference.py: toTraditional()`
     (already built in Milestone 1 — NOT literally `jp2t.json`, see the corrected note above on why).
   - Look it up in **CC-CEDICT** (MDBG, CC BY-SA 4.0, attribution required; parsed into
     `data/cedict/index.json` by `data/cedict/build_index.py`).
     - Found → "confirmed cognate": CC-CEDICT's real pinyin + gloss (the 時間 → shí jiān case).
       Confirms the character string is a real Chinese word — not a guarantee the *meaning*
       matches the Japanese sense (false friends like 大丈夫/勉強 exist, see Milestone 3 notes below).
     - Not found → "Japan-coined term (和製漢語), not attested in Chinese": fallback pinyin built by
       `_fallbackPinyin()`, concatenating each character's first KANJIDIC2 `r_type="pinyin"`
       reading, explicitly labeled as constructed, not a real word. The `pypinyin` dependency was
       never needed — KANJIDIC2's own per-character pinyin field turned out to be sufficient.
   - kun'yomi/jukujigo → skip entirely, no Chinese section rendered (the 出会う case — never fabricate
     "chu1hui4"). Enforced by `analyzeSelection()` only calling `lookupChineseCognate()` when
     `category` is `"onyomi"`/`"mixed"`.

## Implementation milestones

Built as small, independently testable pieces rather than one big change.

- **Milestone 0** (this doc + the study plan) — done.
- **Milestone 1** (done) — Shinjitai→Traditional conversion via OpenCC's `JPShinjitaiCharacters.txt`/
  `JPShinjitaiPhrases.txt` dictionaries, copied into `data/jp2t/` and parsed by a pure-Python
  longest-match function (`core/kanji_reference.py: toTraditional()` — no OpenCC package dependency,
  see the note above on why), plus a GUI test button ("Add Kanji to Document") in the Lyrics Editor's
  Japanese tools row that shows the conversion for a highlighted word. This button is a test harness for
  now — it doesn't persist anything yet.
- **Milestone 2** (done) — on'yomi/kun'yomi classifier: `data/kanjidic2/build_readings.py` parses
  KANJIDIC2 into `data/kanjidic2/readings.json` (12,356 characters); `core/kanji_reference.py` adds
  `resolveContextReading()` (full-line tokenization + overlap matching), `classifyReading()` (the
  backtracking matcher), and `analyzeSelection()` (combines both, classifying against each matched
  token's lemma per the finding above). `core/japanese_utils.py` gained two small public accessors
  (`getTagger()`, `tokenReading()`) so `kanji_reference.py` can reuse the shared tagger singleton
  instead of constructing its own. `analyzeSelection()` returns a **list**, one entry per
  Kanji-containing word overlapping the highlight (non-Kanji tokens like bare particles are
  skipped) - highlighting one word gives a one-item list, highlighting a whole line gives a
  per-word breakdown of every Kanji word in it, so the same button works for both a targeted
  single-word check and a "read the whole line to me" pass. The "Add Kanji to Document" GUI
  button (still a test harness, no persistence yet) shows each word's surface/reading, dictionary
  form when different, and category — verified live against 時間 (onyomi), 時 in context (kunyomi),
  出会った conjugated (kunyomi via lemma 出会う), 今日 (jukujigo), and a full multi-word line
  (時間が経つのは早い、出会った瞬間から → 5 correctly-classified entries in one pass).
  - **Bug found and fixed post-Milestone-3, via a real saved lyric**: `resolveContextReading()`
    tracked each token's position by naively summing `len(word.surface)`, but `fugashi`/MeCab never
    emits whitespace (spaces, newlines) as its own token — it's dropped from `surface` entirely and
    only reported via the *next* token's `white_space` attribute. This let the tracked position
    silently drift out of sync with the real string offset after every space/newline in the text,
    corrupting the highlight-overlap check for everything that followed. Reproduced exactly with
    `saved_labels/TWICE/Funny Valentine_lyrics.json`'s two-line field `"情熱で溶ける甘い愛の
    Chocolate Ah\n衝撃が走る"`: highlighting exactly "衝撃が走る" returned only 走る — 衝撃 (a real
    CC-CEDICT cognate, "impact/shock", same meaning as Japanese) was silently dropped because the
    accumulated 2-character drift (1 dropped space + 1 dropped newline) pushed its computed range
    just outside the highlight. Fixed by adding `pos += len(word.white_space)` before computing each
    token's range. Also added a try/except around the "Add Kanji to Document" button's analysis call
    (`gui/lyrics_editor.py`) — there was no error handling at all before, so any unexpected exception
    would vanish into the console instead of surfacing to the user. Regression tests in
    `core/test_kanji_reference.py: WhitespaceOffsetDriftTests`.
  - **Third bug found and fixed, via that same try/except surfacing it**: highlighting from the very
    first character of the field ("1.0") crashed with `TypeError: 'NoneType' object is not
    subscriptable`. Cause: Tkinter's `Text.count(a, b, "chars")` returns `None` instead of `(0,)`
    whenever `a == b` (a documented Tcl/Tk quirk, not specific to this app) - which happens exactly
    when `startOffset = koreanEntry.count("1.0", selStart, "chars")[0]` compares "1.0" against
    itself, i.e. whenever the highlight starts at the field's very first character. Fixed with
    `(koreanEntry.count(...) or (0,))[0]` for both `startOffset` and `endOffset`.
- **Milestone 3** (done) — Chinese-cognate tiering. The `pypinyin` shortcut check panned out:
  KANJIDIC2's per-character `r_type="pinyin"` field is populated for ~12.4k of 12.8k characters
  (confirmed via a fresh `kanjidic2.xml` download, e.g. 愛→`ai4`, 会→`hui4`/`kuai4` in on'yomi-
  frequency order), so `data/kanjidic2/build_readings.py` now also captures a `"pinyin"` list per
  character in `readings.json`, and no `pypinyin` dependency was added.
  - `data/cedict/` — CC-CEDICT (MDBG, CC BY-SA 4.0) parsed by `data/cedict/build_index.py` into
    `data/cedict/index.json` (122,445 Traditional headwords, 125,025 entries total — matches
    CC-CEDICT's own declared entry count exactly, confirming the line parser dropped nothing). Raw
    `cedict_1_0_ts_utf-8_mdbg.txt` is not committed, same as `kanjidic2.xml` (see `NOTICE.md`).
  - `core/kanji_reference.py: lookupChineseCognate()` — converts the lemma via `toTraditional()`
    then looks it up in the CC-CEDICT index. Found → `{"status": "confirmed", "traditional",
    "pinyin", "gloss"}` (first-listed CC-CEDICT sense). Not found → `{"status": "not_attested",
    "traditional", "pinyinFallback"}`, built by `_fallbackPinyin()` concatenating each character's
    first KANJIDIC2 pinyin candidate — explicitly labelled as constructed, never presented as real.
    Wired into `analyzeSelection()`'s per-word result as a `"chineseCognate"` key, computed only
    when `category` is `"onyomi"`/`"mixed"` (`None` for `"kunyomi"`/`"jukujigo"` — never called at
    all for those, so a native Japanese word can't accidentally get a fabricated Chinese reading).
  - Verified against the plan's two worked examples plus new cases found while building: 時間 →
    confirmed (`shi2 jian1`, "time"); 出会う → no cognate section (kunyomi); 天気 → confirmed via
    `天氣` (proves the Shinjitai→Traditional conversion step runs before lookup, since Shinjitai
    `天気` itself isn't in CC-CEDICT — only Traditional `天氣` is); 残業 → not attested (`殘業`,
    genuine wasei-kango, fallback `can2 ye4`); 今日 → no cognate section (jukujigo).
  - **New known limitation found during testing**: CC-CEDICT containing a word's character string
    doesn't mean the meaning matches the Japanese sense — false friends exist (`大丈夫`/大丈夫 is
    "a manly man" in Chinese, not "it's okay" as in Japanese; `勉強`/勉强 is "to force sb"/"reluctant"
    in Chinese, not "study" as in Japanese). This is correctly reported as `"confirmed"` (the string
    is a real Chinese word) — the lookup can't and doesn't attempt semantic validation, only
    character-string attestation. Worth surfacing to the user as "confirmed, but double-check the
    meaning" rather than treating it as fully validated, when the real (non-test-harness) UI is
    built in Milestone 6.
  - Automated tests: `core/test_kanji_reference.py` (stdlib `unittest`, no new dependency — no test
    framework existed in this project yet) covers confirmed/not-attested/false-friend lookups, the
    Shinjitai-conversion-matters case, the fallback-pinyin construction, and the onyomi/mixed-only
    gating in `analyzeSelection()`. Run with `python -m unittest core.test_kanji_reference -v`.
  - The GUI test harness ("Add Kanji to Document" button, still just a display harness — no
    persistence until Milestone 4) now shows the real Chinese-cognate line instead of the old
    Milestone-1-only "not yet CC-CEDICT-validated" placeholder.
- **Milestone 4** (done, 2026-09-08) — personal per-song vocab persistence.
  - File path convention confirmed from existing code: `gui/lyrics_editor.py: LyricsEditor._lyricsJsonPath()`
    already builds `./saved_labels/{self.app.selectedGroup}/{self.app.songName}_lyrics.json` from
    `app.selectedGroup`/`app.songName`, which are in scope everywhere `addKanjiToDocument()` is
    defined (it's a closure inside a `LyricsEditor` method). Mirror that exactly:
    `saved_labels/<group>/<song>_kanji_vocab.json` (JSON, source of truth) and
    `saved_labels/<group>/<song>_kanji_reference.html` (regenerated fully from the JSON on every
    save - HTML is a derived view, never hand-edited or incrementally patched).
  - New `core/kanji_reference.py` functions: `_vocabJsonPath(group, song)`, `loadSongVocab(group,
    song)`, `addWordToSongReference(group, song, entry)` (upserts by **lemma**, same dedupe key
    philosophy as the on'yomi/kun'yomi classifier - re-highlighting 出会った and 出会う should hit
    the same saved entry, not create two), `renderSongReferenceHtml(group, song, vocabEntries)`.
  - Vocab entry schema (one `analyzeSelection()` result plus save metadata): `surface`, `lemma`,
    `lemmaReading`, `category`, `chineseCognate`, `dateAdded` (ISO date), `sourceLine` (the lyric
    excerpt it was found in, for context when reviewing later - `saved_labels` JSON's existing
    per-entry `korean`/`memberName` fields are the natural source for this).
  - GUI wiring: `addKanjiToDocument()` in `gui/lyrics_editor.py` calls `addWordToSongReference()`
    for every result in the highlighted selection (one click = save everything just analyzed,
    matching the button's existing "Add Kanji to Document" wording) after showing the analysis
    messagebox, inside the same try/except added for the two bugs found this session.
  - Built exactly as planned, plus one small addition: `sourceLine` is written as
    `"<memberLabel>: <fullText>"` (member dropdown value(s), joined with `/` for multi-member
    lines, then a colon, then the *entire* Japanese lyric field text - not just the highlighted
    substring) rather than just the bare `korean` field text, since the plan's own justification
    for choosing `korean`/`memberName` as the source ("for context when reviewing later") is
    better served by having both in one string.
  - Tests: `SongVocabPersistenceTests` in `core/test_kanji_reference.py` - new-entry write,
    lemma-based upsert/dedupe (re-saving 出会う after 出会った replaces rather than duplicates),
    loading a nonexistent song, and HTML rendering (cognate line present for onyomi, absent for
    kunyomi) - run inside a throwaway temp directory (`chdir`'d into) so tests never touch the
    real `saved_labels/` tree.
  - **Live smoke test against real data (done)**: ran `analyzeSelection()` +
    `addWordToSongReference()` end-to-end on `saved_labels/TWICE/Doughnut_lyrics.json`'s actual
    first lyric entry ("手を振って\n背を向けた瞬間に\nすぐにさみしさにやられた") into a throwaway
    song name, confirming all 5 words classify correctly (手/振っ/背/向け as kunyomi, 瞬間 as
    onyomi) and both `_kanji_vocab.json`/`_kanji_reference.html` are written and readable -
    exercising the exact function-call chain the GUI button now runs. Cleaned up the throwaway
    output files afterward (never committed). **Caveat**: this verified the underlying persist
    path this session's environment could actually exercise (non-interactive, no confirmed
    desktop session for driving the real Tkinter window) - it is not a literal click on the "Add
    Kanji to Document" button in a live GUI window. `gui/lyrics_editor.py`'s import chain and
    `python -m core.kanji_reference` were both confirmed to run cleanly, but an actual interactive
    click-through is still worth doing once a windowed session is available, per the project's
    established "verify against real data, not just synthetic strings" pattern.
- **Milestone 5 — redesigned per 2026-09-08 conversation** (done, 2026-09-08) —
  **automatic** cross-song shared-vocabulary index, scanned directly from every
  `saved_labels/<group>/<song>_lyrics.json`, not from Milestone 4's manually-curated
  `*_kanji_vocab.json` files. This decouples discovery from curation: it finds every repeated Kanji
  word across the whole library with zero manual highlighting required, using the data that's
  already there.
  - **Data-quality finding while scoping this**: filtering strictly by each lyric entry's stored
    `"language": "Japanese"` field is not reliable enough - `saved_labels/TWICE/Do Not Touch_lyrics.json`
    has a real Japanese verse (`"待つほど　甘み増すじゃん\n恵の雨降る瞬間\n..."`, containing 瞬間) tagged
    `"language": "Korean"`, presumably written before Phase 1 added proper Japanese-mode support.
    A strict-tag filter would silently miss it. Fix: detect Japanese lines by scanning the `korean`
    field's text for hiragana/katakana characters directly (Unicode ranges already available via
    `_KATAKANA_TO_HIRAGANA`'s range in `core/kanji_reference.py`/`core/japanese_utils.py`) rather
    than trusting the tag - kana is a reliable, content-based signal Korean Hangul/English text can
    never produce, so this works regardless of how the entry happens to be labeled. Report (print or
    log) any mismatch found between kana-detection and the stored tag, so mislabeled entries surface
    for the user to optionally fix, without blocking the scan on them.
  - **Confirmed with real data, grounding the user's own example**: 瞬間 already appears in both
    `saved_labels/TWICE/Doughnut_lyrics.json` and the mislabeled `Do Not Touch_lyrics.json` entry
    above - a real, present-day cross-song match, not a hypothetical.
  - **Data cleanup done (2026-09-08), and a real false-positive it caught**: swept
    `saved_labels/TWICE/{Do Not Touch, Doughnut, Funny Valentine, Marshmallow}_lyrics.json` - all
    four are full Japanese-version TWICE songs (confirmed zero Hangul anywhere in any of them) whose
    Japanese lines were left tagged `"language": "Korean"` because Japanese mode didn't exist yet
    when they were transcribed (per the user - Korean/Japanese reused identical fields, "korean" /
    "romanization" / "english", so there was no functional reason to go back and fix the tag).
    Relabeled 73 entries total (71 via kana-detection + 2 manually-confirmed kanji-only fragments,
    準備 and 時, in Do Not Touch) to `"language": "Japanese"`. Verified via `git diff` that only the
    `language` field changed (plus unrelated pre-existing uncommitted edits already in the working
    tree from before this cleanup - romanization spacing tweaks, `lyricId`/`linkedLabel` migration
    backfill - none of it caused by this sweep), entry counts unchanged, JSON still valid.
    **Important, project-wide finding from scoping this**: a broader "any Kanji, not just kana"
    version of the sweep produced a real false positive - `saved_labels/WJSN/Secret (是秘密呀)_lyrics.json`
    is genuine **Mandarin Chinese** (WJSN's Chinese-language members, Traditional characters, zero
    kana, zero Hangul), tagged `"language": "Korean"` for the same "no dedicated option existed"
    reason - relabeling it "Japanese" would have been wrong. Left untouched; this is a separate,
    unrelated data-quality issue (no "Chinese" language mode exists in the tool) and out of scope
    here. This is exactly why Milestone 5's scanner (below) must keep using **kana**-detection
    specifically, never bare kanji-presence, as its Japanese-line signal - kanji alone is ambiguous
    between Chinese and Japanese, kana is not. Kana-detection remains worth keeping in the scanner
    even after this cleanup, as a safety net against the same mistake happening again on new entries.
  - **Design correction found while building this (2026-09-08): kana-detection alone
    under-includes.** The plan's own cleanup notes above record that `Do Not Touch_lyrics.json`
    has two entries that are genuinely Japanese but pure Kanji with **zero kana at all**
    ("準備" and "時") - manually relabeled to `"language": "Japanese"` precisely *because*
    kana-detection can't see them. A scanner using kana-detection as a strict *replacement* for
    the tag (as originally phrased above) would therefore silently drop those two real entries -
    the exact opposite failure from the one this milestone set out to fix. Verified directly:
    confirmed both entries exist, are tagged `"language": "Japanese"`, and contain no
    U+3040-30FF characters. Fixed by making `_isJapaneseLyricEntry()` a **union** of both
    signals - `language == "Japanese"` OR kana-detected - not kana-detection alone. This keeps
    both real cases working: a kana-containing entry mistagged non-Japanese is still caught (the
    original 瞬間/Do Not Touch finding), and a correctly-tagged pure-Kanji entry is no longer
    silently dropped.
  - **No new tokenization/offset code needed** — `analyzeSelection(fullText, 0, len(fullText))`
    already returns a full per-word breakdown of an entire line (this is exactly what the "Add Kanji
    to Document" button does when you highlight a whole line), and already correctly handles
    multi-line text and embedded whitespace after this session's whitespace-drift fix. Milestone 5
    reuses it as-is, one call per qualifying lyric entry's `korean` field - no separate scanning
    algorithm to build.
  - New function `scanLyricsForKanjiVocab()`: walks `saved_labels/*/*_lyrics.json` (all groups),
    filters to Japanese-detected entries (see above), runs `analyzeSelection()` on each entry's full
    `korean` text, and aggregates results into a dict keyed by **lemma** — `{reading, category,
    chineseCognate, occurrences: [{group, song, memberName, lyricId}, ...]}` (dedupe repeats within
    the *same* song, but keep every distinct song).
  - Two outputs: `kanji_reference/word_index.json` (the full aggregated index, every word regardless
    of how many songs it appears in — a complete, always-current cross-song reference) and a
    rendered HTML view (see the 2026-09-08 UI redesign below). `kanji_reference/` is gitignored
    (fully regeneratable from `saved_labels/`, so treated like `dist/`/`build/` rather than
    committed, unlike the `data/jp2t`/`data/kanjidic2`/`data/cedict` derived dictionaries, which
    are external-source-derived and committed since they can't be regenerated without a network
    fetch).
  - Runnable standalone via `python -m core.kanji_reference` (as originally planned), and exposed as
    a plain function so a future GUI button (Milestone 6+) can trigger a rebuild on demand.
  - Built as planned: `scanLyricsForKanjiVocab()`, `_isJapaneseLyricEntry()` (the union filter,
    see finding above), `_containsKana()`, `buildKanjiReferenceIndex()` (writes both outputs), and
    a `kanji_reference/` `.gitignore` entry.
  - **HTML output redesigned per user request (2026-09-08), after the first version shipped**:
    the original plan/first build rendered a single flat table of only the 2+-song words
    (`shared_words.html`, "瞬間 — found in: TWICE / Doughnut, TWICE / Do Not Touch"). The user
    asked for a per-song browsing view instead: split by song (grouped by artist group, both
    sorted alphabetically), with a way to switch between songs, and each word's "Found In" column
    listing the *other* songs it recurs in as clickable links that jump to that song's list. Two
    open design questions were resolved by asking the user directly rather than guessing:
    **(a) word scope** — the user chose to show **every** word in a
    song's own list (not just words it shares with another song), so a word unique to one song
    just gets an empty "Found In" cell, rather than being hidden entirely; **(b) navigation** —
    the user chose a sidebar/dropdown that swaps the visible table via inline JS (feels like a
    single-page app), over a long scrollable page with jump-to-anchor sections.
    - Replaced `renderSharedWordsHtml()` with `renderWordIndexBySongHtml()`, writing
      `kanji_reference/by_song.html` (renamed from `shared_words.html` since the page is no
      longer "just the shared subset" — it's every word, organized by song).
    - `_slugify()`/`_songKey()` build stable per-song HTML anchor ids (`#song-<group>--<song>`)
      used by both the sidebar links and every "Found In" cross-link, so clicking either jumps to
      the same place.
    - The page is a single self-contained static file (opened directly from disk, not served) —
      all CSS/JS is inline, no external dependency, using one delegated `click` listener
      (`_BY_SONG_SCRIPT`) that shows/hides `<section class="song-panel" hidden>` elements via the
      native `hidden` attribute, keeps the sidebar's active-song highlight in sync, and updates
      `location.hash` so a link (or a bookmark) can deep-link straight to one song.
    - Tests: `WordIndexBySongHtmlTests` in `core/test_kanji_reference.py` - every song gets its
      own panel, a shared word's "Found In" link points at the *other* song and never at itself
      (checked by slicing out each panel's HTML and asserting no self-referential `data-song`),
      a word unique to one song renders with no found-in list at all, and songs are grouped/sorted
      group-first-then-song (TWICE before WJSN). `CrossSongIndexTests`'s
      `test_build_kanji_reference_index_writes_both_outputs` was updated for the new scope: a
      single-song word (残業) now appears in `by_song.html` too, just under its own song's panel
      with no found-in link, instead of being excluded from the HTML entirely.
    - **Verified against real data**: regenerated `kanji_reference/by_song.html` from the actual
      `saved_labels/` tree (5 real songs across 2 groups: `BTS/Let Go`,
      `TWICE/{Do Not Touch, Doughnut, Funny Valentine, Marshmallow}` — every other saved song has
      no Japanese/Kanji content to index) and confirmed structurally: 5 `<section class="song-panel">`
      blocks, groups rendered in the correct sorted order (BTS before TWICE), and — scanning every
      panel's HTML — **zero** self-referential found-in links anywhere in the real dataset (a
      song never lists itself as a place the word is "also found").
  - Original tests unaffected: `CrossSongIndexTests`'s kana-detection/union-filter/same-song-dedup
    coverage (described above) still applies unchanged - only the two output-format tests moved to
    reflect the new HTML shape. 29 tests total in `core/test_kanji_reference.py` as of this
    redesign.
- **Milestone 6 — English meaning via JMdict, for every word (done, 2026-09-08)**. The
  originally-planned Milestone 6 ("upgrade the test button into the real feature") turned out
  already done in substance by the time Milestone 4 shipped - the button already classifies,
  looks up the Chinese cognate, and persists. The actual open gap, surfaced by the user asking
  how a kunyomi word like 離す ("to separate/let go of" - easily and wrongly guessed as "to
  leave," which is really 離れる) would ever get an English translation: **kunyomi words get no
  help at all from the Chinese-cognate tier by design** (it only ever runs for onyomi/mixed), so
  they had no meaning shown anywhere. Also directly motivated by the user's own example: 気分
  (Japanese "mood") converts correctly to 氣分 but isn't a real Chinese word (see the accuracy
  limit below) - the tool needs the word's *own* Japanese meaning, independent of any Chinese
  connection at all, to be useful for every word, not just onyomi/mixed ones.
  - **Data source**: **JMdict** (EDRDG, same publisher/license - CC BY-SA 4.0 - as KANJIDIC2),
    specifically the `JMdict_e` English-glosses-only subset from
    http://ftp.edrdg.org/pub/Nihongo/JMdict_e.gz. Confirmed live before committing to it (same
    "verify library/API claims empirically" pattern as every other data source this project
    uses): downloaded and parsed with stdlib `xml.etree.ElementTree` - JMdict's XML declares all
    its part-of-speech/field abbreviation entities (`&v5r;`, etc.) in its own inline DOCTYPE
    internal subset, which `ElementTree`/expat resolves natively, no external DTD fetch or
    hand-written entity table needed. 218,740 raw `<entry>` elements.
  - `data/jmdict/build_index.py` parses `JMdict_e` into `data/jmdict/index.json` (228,974 kanji
    headwords, 44MB - the raw source file, like `kanjidic2.xml`/`cedict_1_0_ts_utf-8_mdbg.txt`,
    is not committed, only the derived index is; see `data/jmdict/NOTICE.md`). Indexed by
    **kanji spelling only** (every word this tool ever looks up already contains Kanji), mapping
    each spelling to a *list* of JMdict entries - necessary because one kanji spelling can carry
    multiple, unrelated readings/meanings (see the 湯 finding below).
  - `core/kanji_reference.py: lookupJapaneseMeaning(kanji, reading)` - looks up by the
    **(kanji, reading) pair together**, picking whichever JMdict entry's reading list contains
    the word's real in-context reading (falling back to the first entry if none match exactly,
    e.g. a small `fugashi` reading discrepancy). Returns `{"status": "found", "pos": [...],
    "gloss": [...]}` (first-listed sense) or `{"status": "not_found"}`.
  - Wired into `_analyzeToken()`/`analyzeSelection()` as a new `"japaneseMeaning"` key,
    populated **unconditionally for every word** regardless of on'yomi/kun'yomi category -
    unlike `chineseCognate`, which stays gated to onyomi/mixed only. This is the field that
    finally gives kunyomi words (and jukujigo) a real meaning.
  - **Two real disambiguation cases confirmed live against JMdict**, both illustrating exactly
    why lookup must key on (kanji, reading) together rather than either alone:
    - **Same reading, different kanji (homophones)**: 離す and 話す are both read はなす but mean
      "to separate" and "to speak" respectively - JMdict keys them as fully separate entries;
      looking up by kanji spelling first correctly returns the right one for each.
    - **Same kanji, different reading, unrelated meaning**: 湯 read ゆ means "hot water" (the
      everyday native-Japanese sense, kun'yomi); the *same* kanji read タン means "soup" - a
      reading borrowed specifically for the Chinese sense of the character. This is the direct
      answer to the user's own 気分/氣氛 question from a different angle: a shared character
      doesn't imply a shared meaning across languages, and 湯 shows it doesn't even imply a
      shared meaning across *readings of the same character* within Japanese itself.
  - **New "Japanese Meaning" column added to both existing HTML views** (Milestone 4's
    per-song `<song>_kanji_reference.html` and Milestone 5's `kanji_reference/by_song.html`),
    per direct user request - placed between Category and Chinese Cognate in both. Also added to
    the "Add Kanji to Document" messagebox text in `gui/lyrics_editor.py`, shown for every word
    (not gated), right after the dictionary-form line and before the Chinese-cognate line.
  - **Real bug found and fixed while verifying against real saved_labels data (not synthetic
    strings)**: two personal pronouns, 私 (BTS/Let Go) and 君 (TWICE/Funny Valentine), were
    silently misclassified as "jukujigo" instead of kunyomi, and their JMdict lookups failed
    outright. Root cause: `unidic-lite` bakes a `"-<POS category>"` disambiguator suffix directly
    into the `lemma` field for some common homograph-prone words - `word.feature.lemma` for 私
    literally comes back as `"私-代名詞"` ("私-pronoun"), not bare `"私"` (confirmed `lForm`, the
    reading field, is unaffected - only `.lemma` carries the pollution). This is a different
    manifestation of the same underlying `unidic-lite` unreliability already documented for 東京's
    broken katakana-only lemma (see Known accuracy limits) - a third, previously-undiscovered
    failure mode of the same root cause. Fixed in `_analyzeToken()`: since a literal ASCII hyphen
    never appears in a genuine Japanese lemma, stripping everything from the first `"-"` onward
    recovers the real dictionary form. Verified fix against real data: 私/君 now correctly
    classify as kunyomi with real JMdict meanings ("I, me" / "you, buddy, pal"), and the fix also
    correctly merged what had been two spuriously-separate `word_index.json` entries for the same
    word (151 -> 150 distinct words after the fix, since 私-代名詞/君-代名詞 had been polluting
    the Milestone 5 index as separate lemma keys from any cleanly-tokenized occurrence of the same
    pronoun elsewhere).
  - **Third lemma issue found, documented but not fixed**: `歩き出そう` ("let's start walking",
    volitional form of 歩き出す) in `saved_labels/TWICE/Marshmallow_lyrics.json` resolves to
    lemma `歩き出る` instead of the correct `歩き出す` - `unidic-lite` picked the wrong (but also
    real) dictionary verb for this conjugated compound. Unlike the pronoun-suffix bug, there's no
    clean string-level signal to detect this generally (both 出る and 出す are legitimate verbs,
    so nothing marks the resolution as "wrong" the way a literal hyphen did) - noted as a known
    accuracy limit below, same category as the existing わたくし/わたし uncommon-reading issue.
  - Tests: `JapaneseMeaningLookupTests` (kunyomi word gets a real meaning, the 離す/話す homophone
    case, the 湯 same-kanji-different-reading case, onyomi words get a meaning *in addition to*
    their Chinese cognate, unknown-spelling and reading-mismatch fallback behavior, and
    `analyzeSelection()` populating `japaneseMeaning` for kunyomi/jukujigo words specifically)
    and `LemmaPosSuffixBugTests` (regression coverage for the 私/君 fix) added to
    `core/test_kanji_reference.py` - 38 tests total.
  - **Verified against real data**: regenerated `kanji_reference/by_song.html` and a per-song
    `_kanji_reference.html` from actual `saved_labels/` lyric lines after the pronoun fix - of
    150 distinct words in the current library, only the one `歩き出る` mis-lemmatization above
    comes back `"not_found"` in JMdict; every other word (including all 5 words in
    `Doughnut_lyrics.json`'s first entry - 手, 振る, 背, 向ける, all kunyomi, plus 瞬間, onyomi)
    resolved to a real English meaning.
- **Milestone 7 — Mandarin pinyin for every word, as a memorization aid (done, 2026-09-08)**.
  Motivated directly by the user's own study technique: recalling a kun'yomi word's meaning by
  visualizing its Kyuujitai/Traditional form and reading it with Mandarin pronunciation "through
  their Chinese brain," even when the word has no real Chinese-cognate relationship at all - e.g.
  夢 is kun'yomi ("yume") in Japanese, but recalling Mandarin "meng4" via the shared character is
  a genuinely useful personal memory hook. `chineseCognate` can never serve this need because it's
  deliberately gated to onyomi/mixed only (the project's founding rule: never imply a kunyomi word
  is a real Chinese word - see the 出会う worked example at the top of this doc).
  - **This does not weaken that founding rule - it's a narrower, always-true claim next to it.**
    Reporting "here's how each character, standing alone, sounds in Mandarin" is true regardless
    of whether the compound is an attested Chinese word; it never says "this word is Chinese," it
    only says "these characters have Mandarin readings," which is a fact about the characters, not
    a claim about the word. `chineseCognate` still answers "is this word real in Chinese" and stays
    gated exactly as before.
  - New `core/kanji_reference.py: getMandarinPinyin(lemma, chineseCognate)`, called
    **unconditionally** for every word in `_analyzeToken()`. Reuses `chineseCognate`'s own
    already-computed traditional form and pinyin when one exists (onyomi/mixed - both the
    "confirmed" real CC-CEDICT pinyin and the "not_attested" constructed-fallback pinyin already
    carry exactly this data, so there's no duplicate work); for kunyomi/jukujigo (`chineseCognate`
    is always `None`), computes fresh via `toTraditional()` + `_fallbackPinyin()` - the Kanji
    characters are extracted from the lemma first, since kunyomi verbs commonly carry real
    okurigana kana (履く, not just 履) that `_fallbackPinyin()` has no pinyin for and would
    otherwise paste in literally (confirmed via testing: 履く naively produced `"lü3 く"` before
    this filter). Returns `{"traditional": ..., "pinyin": ...}`, always populated.
  - New `"Mandarin Pinyin"` column added to both HTML views (Milestone 4's per-song page,
    Milestone 5's `by_song.html`, positioned right after Category) and the "Add Kanji to
    Document" messagebox, shown for every word.
  - **Real pre-existing bug found and fixed while verifying against the user's own 履 example**:
    both KANJIDIC2 and CC-CEDICT encode the "ü" vowel as a literal ASCII `"u:"` in their raw
    pinyin fields (e.g. 履 -> `"lu:3"`, 女 -> `"nu:3"`) - a plain-text convention neither source
    ever expands, present since Milestone 3 but never visible before because no earlier example
    happened to contain a "ü" character. New `_normalizePinyin()` replaces `"u:"` with the actual
    `"ü"` character, applied everywhere pinyin is emitted (CC-CEDICT's confirmed pinyin,
    `_fallbackPinyin()`'s constructed pinyin, and therefore `getMandarinPinyin()` too) - 履 now
    correctly renders `"lü3"`.
  - Verified against the user's own two worked examples plus the real saved_labels library:
    夢 -> `meng4` (single kunyomi character, matches the user's example exactly); 履く -> `lü3`
    (the Kanji-only-filtered lemma, matching "shoe/to tread on" in CC-CEDICT and "to put on
    footwear" in JMdict - a real, concrete "Japanese preserved archaic/literary Chinese
    vocabulary" case, same category as the user's 赤/紅 observation: modern Mandarin's everyday
    word for "shoe" is 鞋, not 履, but 履 survives in literary/compound Chinese - 履行 "to carry
    out" - and as the ordinary Japanese kun'yomi verb 履く). Regenerated `kanji_reference/by_song.html`
    from the real library (150 words) and spot-checked several kunyomi words that previously had
    no Chinese-related field at all (手/shou3, 私/si1, 甘い/gan1, 靴/xue1) now show a Mandarin
    pinyin cell with an empty Chinese Cognate cell beside it - exactly the intended distinction.
  - Tests: `MandarinPinyinMnemonicTests` in `core/test_kanji_reference.py` (kunyomi word gets
    pinyin despite no cognate; onyomi word's pinyin is reused verbatim from a confirmed cognate,
    not recomputed; a not_attested onyomi word reuses its constructed fallback; the u:/ü
    normalization fix; 履く's Kanji-only filtering; `getMandarinPinyin()`'s two branches directly)
    - 45 tests total in `core/test_kanji_reference.py` as of this milestone.
  - **Second real bug found via a follow-up user question, same day**: 言葉 (kotoba, "language")
    returned `"yan2 xie2"` instead of the expected `"yan2 ye4"`. Root cause confirmed directly
    against a fresh `kanjidic2.xml` download: 葉's own `<reading r_type="pinyin">` elements are
    listed `xie2`, `ye4`, `she4`, **in that literal document order** - not a parsing bug, KANJIDIC2's
    own source data genuinely lists the rare/classical reading "xie2" first (葉's historical
    connection to 叶, a separate ancient character later reused as 葉's Simplified form, whose own
    native reading is "xié," meaning "to harmonize/rhyme" - unrelated to "leaf"), ahead of the
    everyday "leaf" reading "ye4" and the surname/place-name reading "she4" (as in 葉公 Shè Gōng).
    `_fallbackPinyin()`'s "take KANJIDIC2 index 0" heuristic had held for every character this
    project had tested before (殘/業/時/間/気/會/出/夢/履/甘/手/私 all already agreed with index 0),
    making 葉 the first real miss, not a sign the whole approach was broken.
    - **Fix**: new `_charPinyin(ch)` prefers CC-CEDICT's own single-character entry (curated for
      actual Chinese usage) over KANJIDIC2's `pinyin` field whenever the character is a CC-CEDICT
      headword in its own right, falling back to KANJIDIC2's index-0 pick only when it isn't
      (confirmed this still happens for at least 気). `_fallbackPinyin()` now calls `_charPinyin()`
      per character instead of reading KANJIDIC2 directly.
    - **A second, smaller CC-CEDICT quirk surfaced while fixing this**: CC-CEDICT itself lists
      葉's entries as `["Ye4" (surname Ye), "ye4" (leaf)]` - the surname reading first,
      capitalized. 業 has the identical pattern (`"Ye4"` surname before `"ye4"` occupation). This
      is a real, intentional MDBG convention (capitalizing a pinyin syllable marks a proper-noun
      reading), not a defect - `_charPinyin()` prefers the first non-capitalized CC-CEDICT entry,
      falling back to the first entry outright only if every one is a surname reading.
    - Verified the fix changes nothing for the 12 characters already exercised elsewhere in this
      project (all continue to agree with CC-CEDICT), and fixes 言葉 -> `"yan2 ye4"` and bare
      葉 -> `"ye4"` (not "Ye4" or "xie2"). Tests: two new cases in `FallbackPinyinTests` -
      47 tests total in `core/test_kanji_reference.py` as of this fix.
  - **Third real bug found via a follow-up user report, same day**: 切ない (setsunai,
    "heartrending/painful") was showing "not attested (Japan-coined term) — constructed pinyin:
    qie1 な い" for its Chinese Cognate cell - kana literally leaking into a supposedly-Chinese
    pinyin string, and `chineseCognate` being populated at all for what is really a native
    Japanese adjective. Root cause traced to `classifyReading()`/`_matchSegmentation()`, not the
    pinyin-building code: 切's on'yomi せつ matches the start of the reading せつない, and the
    matcher's top-level "non-Kanji character" branch let the leftover な/い pass through via a
    blanket literal-identity check - with **no requirement that this kana be legitimate okurigana
    for anything**. That check tautologically always succeeds for a word's own genuine kana
    (since the tested `reading` is that exact word's own pronunciation), so it silently classified
    切ない as pure "onyomi", which then made `lookupChineseCognate()` (and Milestone 7's
    `getMandarinPinyin()`) run on the full lemma "切ない" including the kana - neither function
    filters kana out of its input, since a real onyomi/mixed lemma had never contained any before
    (real Sino-Japanese compounds - 時間, 瞬間, 残業 - are always 100% Kanji).
    - **Fix**: removed the blanket kana pass-through from `_matchSegmentation()` entirely - a
      non-Kanji character at the top level of the recursion now always fails the match outright.
      Verified this loses no legitimate word: genuine kun'yomi-with-okurigana words (出会う,
      危ない, 少ない) already fully consume their own trailing kana through the function's
      existing, more specific "okuri" branch (which advances past the Kanji *and* its declared
      okurigana together in one step) - they never reach the blanket branch pointed at a bare
      kana character in the first place, confirmed directly before and after the fix.
    - A narrower fix (only blocking the pass-through for the pure-onyomi pass specifically) was
      considered and rejected: 切ない would have simply reclassified as "mixed" instead of
      "onyomi" - `chineseCognate` stays populated for "mixed" too, so the exact same kana-leak
      bug would have persisted under a different category label. The full removal was needed.
    - 切ない now correctly classifies as `"jukujigo"` (no clean on'yomi/kun'yomi/mixed
      decomposition - exactly the right fallback per the algorithm's own definition), with
      `chineseCognate: null` and a clean `japaneseMeaning` from JMdict ("painful", "heartrending",
      "trying") unaffected (that lookup is unconditional, Milestone 6). `mandarinPinyin` now shows
      `切 (qie1)` - the Kanji-only-filtered mnemonic pinyin from Milestone 7, with no kana leak.
    - Verified against the real `saved_labels/` library: regenerating the full index produced the
      exact same word count and shared-word count as before the fix (150 words, 21 shared) -
      confirming this was a narrow, previously-latent bug that hadn't silently corrupted any word
      already in the corpus, and a full sweep of the regenerated `word_index.json` found zero
      remaining words with kana leaking into `mandarinPinyin`.
    - Tests: `OnyomiRootPlusNativeSuffixBugTests` in `core/test_kanji_reference.py` (切ない
      classifies as jukujigo with no cognate; its pinyin has no leaked kana; its Japanese meaning
      is unaffected; 出会う/危ない/少ない - genuine kun'yomi-okurigana words - still classify
      correctly as kunyomi after the fix) - 51 tests total in `core/test_kanji_reference.py` as
      of this fix.

## Known accuracy limits

- `fugashi`/unidic occasionally picks an uncommon reading (e.g. わたくし vs わたし).
- **`unidic-lite` can resolve a conjugated compound verb to the wrong (but still real) dictionary
  form, with no clean way to detect it** — found via Milestone 6 verification: 歩き出そう ("let's
  start walking," volitional of 歩き出す) in `saved_labels/TWICE/Marshmallow_lyrics.json`
  resolves to lemma 歩き出る instead of 歩き出す. Both 出る and 出す are legitimate verbs, so
  nothing marks this resolution as wrong the way a literal hyphen did for the 私/君 lemma-suffix
  bug (Milestone 6) — this one is a plain tokenizer misjudgment, same category as the
  わたくし/わたし issue above, not something fixable with a string-level heuristic.
- The classifier can't resolve true jukujigo automatically — that's by design (see algorithm step 3).
- CC-CEDICT coverage of wasei-kango (Japan-coined Sino-style compounds) is inherently incomplete — a
  real word simply not being in CC-CEDICT yet will read as "not attested," which is a limitation of the
  dictionary, not proof the word doesn't exist in Chinese.
- **"Not attested" can also mean a real Chinese equivalent exists but is spelled with a different,
  merely-homophonous character — found via user question on 気分 (kibun, "mood/feeling").**
  `toTraditional("気分")` correctly produces `氣分` (気→氣 is a genuine Shinjitai/Traditional
  variant of the *same* character), but `氣分` itself isn't a real Chinese word — Mandarin spells
  "mood/atmosphere" as `氣氛`/`气氛` (qìfēn), using `氛` ("atmosphere"), not `分` ("portion/minute").
  `分` and `氛` are two unrelated characters that both happen to be read `fēn` in Mandarin - not a
  script-variant pair `toTraditional()` could ever bridge, since that table only maps *the same*
  character across script standards (楽/樂, 会/會), never *different* characters that merely
  rhyme. So 気分 reports "not attested" correctly for the literal string 氣分, even though a
  Chinese speaker would recognize a homophonous, near-synonymous word one character-swap away.
  Distinct from the existing false-friend limitation above (same string, different meaning) - this
  is the reverse: different string, same meaning, related by sound rather than by spelling. No
  general fix exists without a curated homophone-substitution list (out of scope, same reasoning
  as the long-vowel-romanization limitation below); worth surfacing as "not attested, but note a
  similar-sounding Chinese word may exist" rather than an outright "not Chinese" if a real
  (non-test-harness) UI is ever built for this tier.
- **"Confirmed cognate" (Milestone 3) means the character string is attested in Chinese, not that
  the meaning matches.** False friends exist: 大丈夫/大丈夫 means "a manly man" in Chinese, not "it's
  okay" as in Japanese; 勉強/勉强 means "to force sb"/"reluctant" in Chinese, not "study" as in
  Japanese. `lookupChineseCognate()` can't do semantic validation, only string lookup — worth
  surfacing as "confirmed, but check the meaning" rather than a flat green light, once a real
  (non-test-harness) UI exists.
- `jp2t.json` is explicitly flagged upstream as exploratory — treat its output as a strong draft, same
  as the existing reading auto-fill.
- **Long-vowel romanization is a genuine, unresolvable ambiguity for a small set of words.**
  `fugashi`'s `.pron` field marks long vowels with a bare chōonpu (ー) rather than spelling them out
  (学校 → ガッコー, not ガッコウ), and feeding that straight to `pykakasi` produced doubled-vowel
  romaji ("gakkoo") instead of the conventional spelling ("gakkou") - found via user testing on
  そう→"soo" and 重要→"juuyoo". Fixed in `core/japanese_utils.py: _expandLongVowelMark()`, which
  expands ー back to a real kana based on the preceding vowel, defaulting o-row/e-row to ウ/イ (the
  overwhelmingly common spelling, especially for Sino-Japanese on'yomi compounds like 重要). This is
  right far more often than not, but there's no way to recover the original spelling from a
  pronunciation-only ー mark alone - words that are genuinely spelled with a literal doubled vowel
  (大阪=Ōsaka, おおきい, ねえさん) will come out wrong (大阪 → "おうさか" instead of "おおさか").
  Fixing that fully would need a curated list of these exceptions; not done, since it's a small,
  known category and the existing "auto-generate then hand-correct" pattern already covers it.
- **A token's `lemma` field can be broken for some proper nouns in `unidic-lite`** — found while
  testing the fix above: 東京's lemma came back as katakana "トウキョウ" instead of kanji "東京"
  (confirmed not universal: 日本's lemma is correctly "日本"), which broke on'yomi/kun'yomi
  classification entirely for those words (nothing in a katakana string is recognized as Kanji, so
  the matcher fails and falls back to "jukujigo"). This is a `unidic-lite` data gap, not something
  fixable by better matching logic - worked around defensively in `_analyzeToken()`: if a token's
  lemma lost the kanji its surface has, that lemma can't be a real citation form, so fall back to
  classifying the surface directly instead of trusting it.
