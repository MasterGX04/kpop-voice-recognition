# Vocab / Flashcard SQLite Feature - Implementation Record

Status: **built and working**, actively used, has already been through one real-world bug-fix
cycle post-ship. This doc is the detailed context for anyone (human or Claude) extending it next -
read it before adding features so you don't reintroduce a bug that was already found and fixed once.

## Why this exists

The project already had a mature Japanese Kanji analysis pipeline (`core/kanji_reference.py`) that
persisted curated vocab per-song as JSON + a generated HTML report. It had no mastery-tracking
(no SRS), and Korean had no persistence at all (`core/korean_hanja.py`/`korean_grammar_breakdown.py`
were display-only). The user wanted a proper SQLite-backed flashcard system covering **both**
languages, inspired by a draft schema in `Study_tool_ideas.txt` (a `vocab`/`srs_card`/
`song_occurrences` sketch) - built out into something that actually fits both languages' real
shapes rather than that draft's single unified table.

## Key design decisions (and why)

- **Separate `vocab_ja` / `vocab_ko` tables, not one polymorphic `vocab` table.** Japanese has a
  single on'yomi/kun'yomi/mixed/jukujigo `category` and at most one Chinese cognate (0/1). A Korean
  Hangul spelling can map to **several unrelated real Hanja at once** (e.g. 화 → 火/禍/和/化/畫/靴 -
  `core/korean_hanja.py: lookupHanja()`), so it needs a child table (`vocab_ko_hanja`), not a
  scalar column. Forcing both into one shape (as `Study_tool_ideas.txt`'s draft did) breaks for one
  side or the other.
- **Cross-language linking via `cognate_link`, not baked into the vocab tables.** Both
  `kanji_reference.lookupChineseCognate()` (Japanese) and `korean_hanja.py` (which directly reuses
  that same function for pinyin) resolve against the same CC-CEDICT traditional-form space, so a
  Japanese word and Korean word sharing a root can be linked by exact string match - no new lookup
  needed, just a join over already-computed data. `core/vocab_link.py: syncCognateLinks()`.
  Linking by shared English gloss too is a deferred idea, not built.
- **"Compile Vocab" scan is the primary ingestion mechanism, not a per-word manual save button.**
  Originally Japanese had a per-selection "Add Kanji to Document" button (kept, still works, now
  writes to the DB instead of JSON). But the user explicitly wanted a **scan** function you run
  per-song or for the whole library, so you control which songs get mined rather than being nagged
  on every edit. `core/vocab_sync.py: scanSongForVocab()` / `scanAllSongsForVocab()` is that
  mechanism, used for both the one-time backfill of pre-existing vocab and ongoing re-syncs.
- **Dual independent SRS tracks per word: `meaning` and `reading`.** Each has its own Leitner
  box/ease/due-timestamp columns (`meaning_box`/`meaning_ease`/`meaning_due_ts` and the `reading_`
  equivalents), not one shared set - a word can be "known" for meaning (e.g. a confirmed Sino-
  cognate) while still needing active recall for its reading.

## Schema (`core/vocab_db.py`, SQLite at `data/vocab_srs.db`)

```
vocab_ja(id, lemma UNIQUE, lemma_reading, surface, category, meaning_json, meaning_locked,
         cognate_form, cognate_status, cognate_pinyin, cognate_gloss_json, mnemonic_pinyin_json,
         created_at, updated_at)
srs_card_ja(vocab_ja_id PK, meaning_state, reading_state,
            meaning_box/ease/due_ts/last_reviewed_ts, reading_box/ease/due_ts/last_reviewed_ts)
vocab_occurrence_ja(id, vocab_ja_id, group_name, song_title, singer_names_json, lyric_line,
                    lyric_id, start_chunk, end_chunk, added_at)  -- UNIQUE(vocab_ja_id, lyric_id)

vocab_ko(id, lemma UNIQUE, surface, meaning_json, meaning_locked, hanja_locked,
         created_at, updated_at)
vocab_ko_hanja(id, vocab_ko_id, hanja_form, pinyin, pos, gloss_json)  -- 0..N rows per word
srs_card_ko(...)                    -- same shape as srs_card_ja
vocab_occurrence_ko(...)            -- same shape as vocab_occurrence_ja

cognate_link(id, vocab_ja_id, vocab_ko_id, cognate_form)  -- UNIQUE(vocab_ja_id, vocab_ko_id)
```

`meaning_locked` (both tables) and `hanja_locked` (`vocab_ko` only) were **added after initial
ship** to fix a real data-loss bug - see "Bug history" below. `core/vocab_db.py` migrates any
pre-existing database file in place via `PRAGMA table_info` + `ALTER TABLE ADD COLUMN` (see
`_migrateColumns`), since `CREATE TABLE IF NOT EXISTS` is a no-op once a table already exists - this
is exactly what happened to a real, already-populated `data/vocab_srs.db` and had to be patched in
place without losing the 863 words already in it.

`getConnection()` only runs the schema script once per absolute DB path per process (a global
`_initializedDbPaths` set) - re-running 15 DDL statements on every call was measurably expensive
once something opens thousands of connections in a loop (see perf bug below). `timeout=30` on the
connection gives headroom for a long bulk write to not collide with a quick interactive call.

## Module map

- **`core/vocab_db.py`** - schema + `getConnection()` + column migration.
- **`core/vocab_store.py`** - language-agnostic SRS math only (`computeNextReview(box, ease,
  rating)`, Leitner-ish: again→box 1 + ease penalty, good→+1 box, easy→+2 box + ease bonus; box→days
  via `[1,3,7,14,30,90]`). No table access.
- **`core/vocab_store_ja.py`** / **`core/vocab_store_ko.py`** - per-language CRUD + SRS:
  `upsertVocab(entry, conn=None) -> (vocabId, isNew)`, `addOccurrence(...) -> bool`,
  `submitReview(vocabId, track, rating)`, `getDueCards(track, limit)`, `listAllVocab(limit)`,
  `getOccurrences(vocabId, limit)`, `updateMeaning(vocabId, gloss, pos=None)`,
  `deleteVocab(vocabId)`. Korean-only: `keepOnlyHanjaCandidate(vocabId, hanjaId)`,
  `clearHanjaCandidates(vocabId)`. Every write function accepts an optional shared `conn` - pass
  one when calling in a loop (own-connection-per-call was the perf bug, see below).
- **`core/vocab_link.py`** - `syncCognateLinks()` (recomputes `cognate_link` from current
  `vocab_ja`/`vocab_ko_hanja` rows), `getLinkedWords(language, vocabId)`.
- **`core/korean_vocab.py`** - `analyzeKoreanSelection(fullText) -> list`, Korean's parity with
  `kanji_reference.analyzeSelection()`. Built on top of `korean_grammar_breakdown.breakdownLine()`
  (reuses its Hanja-gating/XR+하다-composition/merge logic rather than re-tokenizing), filtered to
  `role == "content"` entries, with `meaning` re-derived via `lookupKoreanMeaning()` as a structured
  dict (not `breakdownLine()`'s already-flattened display string).
- **`core/vocab_sync.py`** - the compile/scan entry points:
  - `scanSongForVocab(group, song) -> {"jaAdded", "koAdded", "occurrencesLinked"}` - one connection,
    one commit for the whole song; regenerates that song's `_kanji_reference.html` from the DB.
  - `scanAllSongsForVocab(labelsGlob=...) -> {"<group>/<song>": {...}, ...}` - one connection, one
    commit for the **entire** library; `syncCognateLinks()` and the cross-song `by_song.html`/
    `word_index.json` only regenerate once at the end, not once per song.
  - Both idempotent (safe to re-run) via `UNIQUE(lemma)` upserts and `UNIQUE(..., lyric_id)`
    occurrence dedup - this is also why the locking columns above matter (idempotent-by-default
    would otherwise mean "re-run" == "reset any manual edits").
- **`gui/vocab_review.py`** - the "Vocab Review" Toplevel: browse/rate/edit/delete/resolve-Hanja.
  See "GUI" section below for its exact controls.
- **GUI wiring** in `gui/audio_tester.py`: a "Vocab" menu (Compile This Song's Vocab / Compile All
  Songs' Vocab / Review Vocab…) added to `VoiceDetectionApp`'s menu bar, plus `startLayout()`'s
  unrelated `targetLyricsX` fix (see bug history). `gui/lyrics_editor.py`'s "Add Kanji to Document"
  button now calls `vocab_store_ja.upsertVocab()`/`addOccurrence()` instead of the old
  `kanji_reference.addWordToSongReference()` JSON writer (that function still exists, still passes
  its own tests, just isn't called by anything anymore - deliberately left alone rather than gutted,
  to avoid touching a well-tested 1000+ line module for no functional gain).

## The Vocab Review window (`gui/vocab_review.py`) - current controls

- **Language**: `tk.Radiobutton` pair (Japanese/Korean) - **not** `ttk.Combobox** (see bug history:
  a themed combobox rendered completely invisible on a real user's system).
- **Track**: `tk.Radiobutton` pair (Reading/Meaning).
- Both reload the list **immediately** on change (`command=loadQueue`, not a deferred event).
- **Filters**: "Show all words (ignore due date)" (switches `getDueCards()` → `listAllVocab()`),
  "Only missing meaning", "Only ambiguous Hanja (2+)" (Korean-only, client-side filtered).
- **Prev/Next** over a stable in-memory list + index (not a destructively-popped queue) - plus
  Left/Right arrow key bindings on the window.
- Body area always shows reading/meaning/cognate-or-Hanja-candidates/one example lyric occurrence/
  cross-language cognate cousins (via `vocab_link.getLinkedWords`) - no hidden "reveal" step.
- A red "⚠ No meaning found" flag when `meaning.status != "found"`.
- **Edit Meaning** (`simpledialog.askstring`, `;`-separated senses) → `updateMeaning()`.
- **Delete Word** (confirm dialog) → `deleteVocab()`.
- Korean, 2+ Hanja candidates: one "Keep only this" button per candidate → `keepOnlyHanjaCandidate()`.
- Korean, **any** ≥1 candidates: "Remove Hanja entirely (native word)" → `clearHanjaCandidates()` -
  needed because a *wrong single* candidate isn't "ambiguous" so the keep-only picker has nothing
  to contrast against (see the 해/害 bug below).
- **Again/Good/Easy** rating buttons → `submitReview()`, then auto-advance.
- "Reload" button = re-run `loadQueue()` with current filters. **It is not a save button** - every
  action above commits to SQLite the instant you click it. This confused a user once (see below).

## Bug history (real bugs found after this feature shipped - read before touching related code)

1. **`targetLyricsX` crash on lyric load** - unrelated to the vocab feature itself, found while
   testing it, but fixed in the same session. `LyricBox._baseAnchor()` depends on
   `self.parent.targetLyricsX`, only set inside `_applyCanvasResize()` behind a **120ms** resize
   debounce. Lyrics load on a flat **50ms** `after()` timer scheduled in `__init__` - 50 < 120, so
   it always lost the race. Fixed by having `startLayout()` (scheduled via `after_idle`, which
   reliably beats a fixed-delay timer) call `addBackgroundImage()` once up front.
2. **"Compile All Songs' Vocab" froze the app for 60-85 seconds.** Root cause: every single
   word/occurrence write opened a **fresh SQLite connection**, re-ran the **entire 8-table schema
   script**, and **committed individually** (~4,262 times for a 35-song library) - all synchronously
   on the Tkinter main thread, with zero progress feedback, so the window looked hung/"(Not
   Responding)". Fixed: schema-init-once-per-path (`_initializedDbPaths`), `upsertVocab`/
   `addOccurrence` accept an optional shared `conn`, and `vocab_sync` now holds **one connection,
   one commit** per song (or per entire library). Measured: 64.5s→4.7s fresh compile, 85.7s→0.4s
   re-scan. No background thread was needed after this fix.
3. **Review screen UX bugs** (all fixed in one pass): no way to browse backward (only auto-advance
   forward after rating) → added stable-list Prev/Next; switching the language dropdown didn't
   reload, so `reveal()`/`rate()` kept using the *old* card's `vocabId` against the *new* language's
   table, hitting an unrelated word by coincidental row-id match → dropdown change now reloads
   immediately; no way to flag/fix/delete bad entries → added the flag + Edit Meaning + Delete Word
   + Hanja-candidate resolution described above.
4. **Notifications appearing on the wrong window.** `messagebox.showinfo(...)` calls in
   `compileThisSongVocab`/`compileAllSongsVocab` had no `parent=`, so Tkinter defaulted to
   `_default_root` (the **main** app window) instead of the Audio Tester `Toplevel` they were
   triggered from. Fixed by adding `parent=self.root` explicitly - this project's established
   convention everywhere else (`gui/voice_recognition_gui.py`, `gui/thumbnail_functions.py`) already
   did this; the new vocab code just missed it. Also added `win.transient(parent)` to
   `openVocabReviewWindow()`, which was missing entirely.
5. **Language/Track `ttk.Combobox` rendered completely invisible** on a real user's system (no box,
   no dropdown arrow, blank space) while adjacent plain `tk.Checkbutton`s on the very next row
   rendered fine - a ttk-theme rendering quirk. Fixed by replacing both comboboxes with plain
   `tk.Radiobutton` pairs (same unthemed widget family as the working Checkbuttons); this also
   incidentally satisfied "make it feel like an actual toggle."
6. **화 vs 害-style false Hanja matches with only one candidate.** 해 ("sun/day", native Korean) got
   matched against 害 ("harm") - a real Sino-Korean reading, but only for compounds like 재해/유해,
   not bare native 해. Not "ambiguous" (only one candidate exists), so `keepOnlyHanjaCandidate()`
   has nothing to contrast against. Added `clearHanjaCandidates()` + an always-shown (≥1 candidate)
   "Remove Hanja entirely" button.
7. **The big one: manual curation silently reverted by the next Compile run.** A user resolved
   several ambiguous Korean words via "Keep only this"/hand-edited meanings, then re-ran "Compile"
   (very easy to do without realizing the consequence), and **every edit reverted** - confirmed
   directly against their real database (a word they'd supposedly fixed still showed 2 unresolved
   Hanja candidates). Root cause: `upsertVocab()` always unconditionally deleted+reinserted the
   Hanja candidate list and overwrote `meaning_json` on every call, with no concept of "a human
   already decided this." Fixed with `meaning_locked`/`hanja_locked` columns, set by
   `updateMeaning()`/`keepOnlyHanjaCandidate()`/`clearHanjaCandidates()`, and respected by
   `upsertVocab()` (an unlocked word still refreshes normally on rescan - locking is opt-in per
   word, not a blanket freeze). **Their specific prior picks for already-corrupted words could not
   be recovered** (no history/undo log existed) - they had to redo those, but the fix is durable
   going forward. Migrated their live database in place with zero data loss (863 words intact).

## Known limitations / deliberately not built

- **Korean vowel-contraction fusion artifacts can pollute the vocab list** - e.g. a noun+particle
  contraction can surface as a lemma like `"세계+ᆯ"` alongside the clean `"세계"` entry
  (`core/korean_grammar_breakdown.py: _mergeGroup()` joins sub-entry lemmas with `+` when a content
  word fuses with a following particle in the same syllable span). Not fixed - use "Delete Word" on
  these when you spot them; a real fix would need `analyzeKoreanSelection()` to detect a `+` in the
  lemma and re-derive just the content sub-entry's own lemma, which `breakdownLine()`'s current
  return shape doesn't expose separately.
- **No "jump to a specific word" search** in the Review screen - you can only page through the
  filtered list. Would be a natural, low-risk follow-up (client-side substring filter on `lemma`/
  `surface`, no schema change needed).
- **No audio playback from the Review screen**, though `vocab_occurrence_{ja,ko}.start_chunk`/
  `end_chunk` are captured and ready for it (`Study_tool_ideas.txt`'s Idea 2).
- **No lock/unlock toggle exposed in the UI** - once a word is `hanja_locked`/`meaning_locked`,
  there's no button to go back to "let rescans auto-update this again." Would need one new
  store function + one button per lock type.
- **Cross-language linking is cognate-form-only** - linking by shared English gloss too was
  discussed and deliberately deferred.
- **No three-way JA/KO/ZH comparison view** (`Study_tool_ideas.txt`'s "Tri-Lingual Ideographic
  Bridge" idea) - out of scope so far.
- **`core/kanji_reference.py`'s old JSON+HTML persistence path still exists, unused.** Fine to
  ignore; don't feel obligated to route new work through it.

## Testing

`core/test_vocab_db.py`, `test_vocab_store_ja.py`, `test_vocab_store_ko.py`, `test_vocab_link.py`,
`test_vocab_sync.py` - 43 tests, all `unittest`, all tempdir-isolated. Includes explicit regression
tests for bug #7 above (`test_rescan_does_not_undo_keep_only_hanja_candidate`, etc.) and a schema
migration test that simulates a pre-fix on-disk database. Run everything relevant with:

```
python -m unittest core.test_vocab_db core.test_vocab_store_ja core.test_vocab_store_ko core.test_vocab_link core.test_vocab_sync core.test_kanji_reference -v
```

97 tests, all green as of this writing (54 pre-existing `test_kanji_reference` + 43 new).

## Process notes for whoever extends this next

- This project's `core/test_*.py` files are the **permanent** regression suite - always add to
  them for new logic. Separately, verify tricky behavior empirically with a throwaway script
  first (matches this project's established "verify before build" habit) - but **delete the
  throwaway script once you're done**, it's scratch, not a deliverable. Don't confuse the two.
  Every bug fix above was verified two ways before being called done: the permanent unittest suite,
  *and* a live Tkinter smoke test that actually clicks the real widgets against seeded data (not
  just backend function calls) - this caught things a unit test alone wouldn't (the invisible
  combobox, the wrong-parent messagebox).
- Any time you add a **new user-visible way to persist a change to `vocab_ja`/`vocab_ko`** (a new
  "edit" action, say), ask whether it should also set a `*_locked` flag so a future rescan doesn't
  quietly undo it - bug #7 above is the exact shape of mistake to avoid repeating.
- If you add a write path that runs in a loop over many words (another bulk operation), thread a
  shared `conn` through it like `vocab_sync.py` does - don't reintroduce the connection-per-row
  performance bug.
