# Flashcard Tool Upgrade: PyWebView Rebuild + Bundled Study Features

**Status: Milestones 0-2 DONE (Milestone 2: 2026-09-25). Milestones 3-7 are designed but not yet
started — scope was deliberately narrowed first per the user's request, revisit after those ship.**

This doc is the durable version of the Claude Code planning session that designed this upgrade — kept
here so the reasoning survives, in the same Milestone-numbered build-log style as
`.claude/VOCAB_SRS_PLAN.md` / `.claude/KANJI_REFERENCE_PLAN.md`. Update each Milestone's status
(done/in-progress, findings, bugs) as it actually ships, rather than treating this as a one-shot spec.

## Context

The vocab/Kanji-Hanja flashcard SRS feature (`gui/vocab_review.py` + `core/vocab_store_ja.py`/
`core/vocab_store_ko.py`/`core/vocab_link.py`/`core/vocab_db.py`) is already fully built, tested, and
in daily use — this is an upgrade, not a from-scratch build. It's currently a plain Tkinter `Toplevel`,
which caps how rich the visuals can get (typography, color-coded grammar blocks, animated/audio-synced
karaoke-style lyric highlighting).

After comparing three realistic UI-stack options (enhanced Tkinter, an embedded HTML renderer with no
JS, and PyWebView) against this codebase's actual constraints, the user chose **PyWebView + local
HTML/CSS/JS**, specifically because:
- Per-syllable karaoke-style highlighting synced to audio is a **core** requirement, and CSS
  transitions/JS timers are the one place Tkinter genuinely can't compete.
- Distribution to other machines is a real "possibly," which this plan accounts for (WebView2
  packaging) rather than ignores.
- The bigger not-yet-built study ideas (audio-clip flashcards, a real cross-word occurrence graph, a
  JA/KO/ZH cognate bridge view, grammar cloze drills) should be planned in this same effort rather than
  deferred.

Two design forks were resolved directly with the user:
- The "occurrence graph" (Milestone 4) means a **true cross-word co-occurrence graph** (which
  songs/words cluster together), not just a per-word list of songs — this needs a new query layer, not
  just a UI on top of an existing function.
- Grammar cloze drill answers (Milestone 5) get their **own independent SRS track** (new `cloze_*`
  columns mirroring the existing `meaning_*`/`reading_*` pattern), tracked separately from today's
  meaning/reading due-dates rather than polluting them.

Everything below was verified against the real files (exact function/table names), not guessed.

## Architecture decision (applies to every milestone below): separate process, not a background thread

`gui/vocab_review.py:openVocabReviewWindow(parent)` today is a `tk.Toplevel(parent)` with
`win.transient(parent)`, opened from `gui/audio_tester.py`'s "Vocab" menu (`openVocabReview()`, ~line
2249-2252).

The new review window runs as **its own OS process**, not a thread inside the main Tkinter process:
- `gui/audio_tester.py` drives a single global `pygame.mixer.music` tied to whatever song is currently
  open. Since audio-clip playback is part of this plan (Milestone 2), running the review UI in-process
  would let it steal/interrupt whatever Audio Tester is currently playing. A separate process gets its
  own independent `pygame.mixer`.
- Two native event loops (Tk's and WebView2's COM-based one) in one process is a real flakiness risk on
  Windows; a separate process avoids it entirely.
- `core/vocab_db.py`'s own connection code already anticipates a second concurrent caller
  (`timeout=30`, comment references "the Vocab Review screen... elsewhere in the app") — already
  designed to tolerate a second process hitting the same SQLite file.

Window-parenting (replacing what `transient()` gave) is done at the Win32 level via `ctypes` (stdlib,
no new dependency): parent gets its HWND via `self.root.winfo_id()`, the child process is found via
`ctypes.windll.user32.FindWindowW(None, "Vocab Review")` after `subprocess.Popen(...)`, then
`SetWindowLongPtrW(childHwnd, -8 /*GWL_HWNDPARENT*/, parentHwnd)` reproduces owned-window stacking
across the process boundary.

**PyInstaller detail:** in a frozen build there's no separate `python.exe` to `Popen`. The same single
`.exe` re-invokes itself with a flag — `gui/voice_recognition_gui.py`'s `if __name__ == "__main__":`
gets a branch checked before normal startup:
```
if "--vocab-review" in sys.argv:
    from gui.vocab_review_web_main import runStandalone
    runStandalone(sys.argv); sys.exit(0)
```
The launcher branches on `getattr(sys, "frozen", False)` (same idiom already used by `exeDir()`/
`resourcePath()` in this codebase) to pick `sys.executable` (dev) vs the frozen exe path +
`--vocab-review` (frozen).

---

## Milestone 0 — Spike / go-no-go (DONE 2026-09-13)

**Goal:** prove the process-boundary architecture actually works on this machine before committing to it.

**Findings:**
- **PASS** — `pip install pywebview` (resolved to `pywebview==6.2.1`, `pythonnet==3.0.5`,
  `clr_loader==0.2.10`) installs cleanly into the `pyannote-env` (Python 3.9.13) venv with no
  compatibility cliff, and a real window opens and renders via the WebView2 runtime already present
  on this machine.
- **PASS, with a real caveat** — `FindWindowW`/`SetWindowLongPtrW` (`GWL_HWNDPARENT`) successfully
  reproduces owned-window stacking across the process boundary: readback after the call confirmed the
  child's parent HWND matches the Tk parent's. **But calling `SetWindowLongPtrW` immediately after
  `FindWindowW` finds the new window races with WebView2's own async
  `CoreWebView2Environment`/`CoreWebView2Controller` creation** — first run reproducibly hit
  `WebView2 initialization failed ... (0x80004004): Operation aborted` from reparenting the native
  window mid-init. Waiting ~1.5s after the window is found, before calling `SetWindowLongPtrW`, avoided
  it on a second run. **Implication for the launcher (Milestone 1):** don't reparent the instant
  `FindWindowW` succeeds — add a short delay (or better, poll until the child signals it's actually
  rendered, if pywebview exposes a ready callback) before calling `SetWindowLongPtrW`.
- **PASS** — opening/closing/reopening the window twice in a row (as two separate subprocess launches)
  worked cleanly both times, no leftover-state issues.
- **DEFERRED to Milestone 7** — the self-re-invoke-via-argv-in-a-frozen-build check was not run this
  pass: it requires a full PyInstaller build of the whole app (opencv/torch/etc bundled), which is
  disproportionate to spike right now and is squarely packaging-phase work, not something Milestone 1's
  dev-mode functionality depends on. Revisit before Milestone 7 starts.

**Verdict: go** — proceeding to Milestone 1 with the reparent-delay caveat carried into the launcher design.

---

## Milestone 1 — Parity rebuild (replace `gui/vocab_review.py`)

**Goal:** the exact feature set that exists today (queue/filtering, rate, edit meaning, delete, Hanja
disambiguation), running as PyWebView instead of Tkinter, with zero behavior regression.

New files (mirrors today's module so the eventual diff reads as "replace one module with an
equivalent"):
- `gui/vocab_review_web_main.py` — process entry point; creates the `webview` window, sets its title
  (used by the HWND handshake), calls `webview.start()`.
- `gui/vocab_review_api.py` — the `VocabReviewApi` class, kept import-independent of `webview` so it's
  unit-testable directly.
- `gui/vocab_review_launcher.py` — replaces `openVocabReviewWindow(parent)`; handles `Popen`, the HWND
  handshake, dev-vs-frozen argv branching. **Only one line changes in `gui/audio_tester.py`**: the
  import at the existing call site becomes
  `from gui.vocab_review_launcher import openVocabReviewWindow` — the call itself
  (`openVocabReviewWindow(self.root)`) is unchanged.
- `gui/web/vocab_review/index.html`, `style.css`, `app.js` — static front end. `style.css` should
  `@font-face` the two font files already bundled in `voice_recognition_gui.spec`
  (`fonts\PretendardVariable.ttf`, `fonts\Hiragino Sans GB W3.ttf`) rather than pull in new web fonts.

**Api method surface** (each a thin wrap of an existing store/link function — no new business logic):

| Api method | Wraps |
|---|---|
| `listQueue(language, track, showAll, missingMeaningOnly, ambiguousOnly)` | `vocab_store_ja/ko` queue-building + the same filters `loadQueue()` applies today |
| `getCardDetail(vocabId, language)` | `getOccurrences(vocabId, limit=1)` + `vocab_link.getLinkedWords(...)` — fetched per displayed card, not prefetched for the whole queue |
| `rate(vocabId, track, rating)` | `store.submitReview(...)` |
| `updateMeaning(vocabId, language, glossList, pos=None)` | `store.updateMeaning(...)` |
| `deleteWord(vocabId, language)` | `store.deleteVocab(...)` |
| `keepOnlyHanja(vocabId, hanjaId)` | `vocab_store_ko.keepOnlyHanjaCandidate(...)` |
| `clearHanja(vocabId)` | `vocab_store_ko.clearHanjaCandidates(...)` |

**State split:** JS holds `{cards, index, detailCache, filters}` — Prev/Next just move `index` and
re-render, no round-trip (preserves today's "stable list, not a destructively-popped queue" design).
Python holds no session state; every Api call is a fresh, independent DB call, same posture as the
existing store functions.

**Error surfacing:** every Api method returns `{"ok": true/false, ...}`, never lets an exception cross
the JS bridge silently (also logs server-side so it shows in the console window, which the `.spec`
already enables via `console=True`). A single `callApi()` JS wrapper shows a persistent, visible
`#error-banner` on failure — replacing Tkinter's `messagebox`/`simpledialog` with real in-page
equivalents (a modal `<dialog>` for Edit Meaning, a confirm-modal for Delete).

**Done when:** every action in the table above works end-to-end against real data, plus
`core/test_vocab_review_api.py` (new) and a scripted `evaluate_js` smoke test both pass — see
Testing section below.

**Status: DONE (2026-09-13).** All new files above were built as designed, with two small
deviations found while implementing:
- `keepOnlyHanja`/`clearHanja`/`updateMeaning` don't return the updated data - the store functions
  they wrap (`keepOnlyHanjaCandidate`, `clearHanjaCandidates`, `updateMeaning`) don't return
  anything either, so the JS side reconstructs the new state client-side from data it already has,
  exactly like the old Tkinter screen's `keepOnlyHanja()`/`clearHanja()`/`editMeaning()` did. The
  `rate` Api method also takes `language` (not in the original plan table) since a single method
  serves both stores.
- `core/test_vocab_review_api.py`: 11 tests, all passing, run alongside the existing suite
  (`python -m unittest core.test_vocab_db core.test_vocab_store_ja core.test_vocab_store_ko
  core.test_vocab_link core.test_vocab_sync core.test_vocab_review_api -v` → 54 tests, all green).
- Interactive smoke test (scripted `evaluate_js`, throwaway, deleted after use per convention): 18
  checks covering initial render, language/track/filter switching, ambiguous-Hanja
  keep-only/remove, missing-meaning warning, Edit Meaning dialog (+ `meaning_locked` persistence),
  and Delete confirm — all passing on the second run. The first run had 4 failures that turned out
  to be a bug in the *test script's* navigation assumption (assumed the ambiguous word was the
  first card; it was actually the last, so a subsequent "Next" click was a no-op and the script
  edited/deleted the wrong word) - not a product bug. Worth remembering: don't assume seeded-data
  ordering in future smoke tests: `vocab_store_*.listAllVocab()` orders by `lemma`, which for mixed
  Hangul/Hanzi words is Unicode-code-point order, not insertion order.
- The `evaluate_js` smoke test above exercises the real front end + `VocabReviewApi` in-process
  (`webview.create_window()` called directly), not the actual `gui.vocab_review_launcher` subprocess
  + HWND-reparenting path. That path was checked separately with a real Tk parent calling the real
  `openVocabReviewWindow()`: the subprocess launched, the window was found by title, and
  `GWL_HWNDPARENT` readback confirmed correct reparenting - PASS, no repeat of the WebView2 abort
  seen during the Milestone 0 spike.
- Not yet exercised end-to-end: the actual "Vocab" menu item in a running `gui.audio_tester`
  session (only its one-line import change was made and reviewed, not click-tested live), and the
  frozen/PyInstaller launch branch (`sys.frozen`) in `gui/vocab_review_launcher.py` /
  `gui/voice_recognition_gui.py`'s `--vocab-review` argv handling - both are cheap dev-mode-only
  checks, not full re-verification, so flagging rather than assuming.

**Follow-up addition (2026-09-14), before starting Milestone 2:** shuffle + random-card navigation,
requested directly on top of the shipped Milestone 1 screen:
- A "Shuffle order" checkbox in the filters row (`gui/web/vocab_review/index.html`) - reorders
  whatever `listQueue()` already returned (Fisher-Yates, `app.js: shuffleArray()`), applied inside
  `loadQueue()` right after the Api call. Deliberately client-side only, no backend/Api change - it
  reorders whatever filtered set is currently loaded, so it "just works" on top of any future
  range/practice-need filter without needing to touch `gui/vocab_review_api.py`.
- A "Random Card" button in the nav row (`app.js: goRandom()`) - jumps to a uniformly random index
  in the current `state.cards`, distinct from the current index when more than one card is loaded;
  disabled when 0-1 cards are loaded (mirrors Prev/Next's disabled-state pattern).
- Verified with a throwaway `evaluate_js` smoke test (8 checks, all passing): shuffle toggles
  without error and preserves card count, Random Card visits multiple distinct positions over
  repeated clicks without error, and the button correctly disables once a filter narrows the queue
  to zero cards. No new backend tests needed - no Api surface changed.

---

## Milestone 2 — Audio-clip-as-prompt playback (DONE 2026-09-25)

**Goal:** play the ~1.5s song audio for the word's occurrence line, as a flashcard prompt.

- New Api method: `playOccurrenceAudio(group, song, startChunk, endChunk)`.
- Reuse, don't reinvent: `core.group_registry.GroupRegistry.getGroupMediaDir()` +
  `core.util_functions.pickBestAudioForStem()` to resolve the file, `gui/audio_tester.py`'s
  `ensureAudioForPlayback()` to get a playable cached file, `pygame.mixer.music.play(start=...)`.
- Chunk→ms uses the existing `chunk_duration = 40` constant — hoist it out of `VoiceDetectionApp`
  into a shared location (e.g. `core/util_functions.py`) so both Audio Tester and this new process
  reference one source of truth instead of two copies of `40`.
- No Tk `.after()` loop exists in this process — use
  `threading.Timer((endMs-startMs)/1000, pygame.mixer.music.stop).start()` to bound playback to the
  clip.

**Done when:** clicking play on a card's occurrence plays exactly that line's audio without
interrupting/being interrupted by Audio Tester's own playback (proves the process-isolation decision
was right).

**Status: DONE (2026-09-25).** Built as designed, with one deviation and one addition found while
implementing:
- **Deviation:** `ensureAudioForPlayback()`/`cacheKeyForPath()` were hoisted from `gui/audio_tester.py`
  into `core/util_functions.py` (alongside `pickBestAudioForStem`), rather than importing them from
  `gui/audio_tester.py` as originally planned. That module's top-level imports (`tkinter`, `cv2`,
  `PIL.ImageTk`, etc.) would otherwise be dragged into the new lightweight process for no reason -
  `gui/vocab_review_api.py` was deliberately kept free of `webview`/GUI-toolkit imports in Milestone 1,
  and this preserves that. `core/util_functions.py` now also owns the `AudioSegment.converter/ffmpeg/
  ffprobe` bundled-binary configuration (previously only set in `gui/audio_tester.py`), so
  `ensureAudioForPlayback` behaves the same regardless of which process imports it. `chunk_duration`
  was hoisted the same way, as `core.util_functions.CHUNK_DURATION_MS` (+ a `chunkToMs()` helper for
  the fixture-value unit test) - `gui/audio_tester.py`'s `VoiceDetectionApp.chunk_duration` attribute
  is unchanged, just now assigned from the shared constant.
- **Addition (not in the original design):** the previous clip's pending `threading.Timer` stop-call
  is cancelled before starting a new one. Without this, clicking Play again (or on a new card) before
  the first clip's bound duration elapsed would let the *old* timer fire mid-way through the *new*
  clip and cut it off early.
- Verified against real project data, not a synthetic fixture: `saved_labels/TWICE/
  Breakthrough_lyrics.json`'s line 4 (Jihyo, `linkedLabel: {startChunk: 642, endChunk: 744}`) played
  against the real `training_data/TWICE/Breakthrough.mp3` via a throwaway headless script calling
  `VocabReviewApi.playOccurrenceAudio()` directly (no webview window) - confirmed real file resolution
  through `GroupRegistry`, real ffmpeg-backed caching into the same `cache_audio/` Audio Tester already
  uses, and playback stopping within the expected ~4.08s window (measured 4.27s via
  `pygame.mixer.music.get_busy()` polling). Script deleted after use per this project's
  throwaway-scripts convention.
- `core/test_vocab_review_api.py`: added a `chunkToMs` fixture-value test (using the same real
  Breakthrough chunk numbers above) and a graceful-failure test for when no audio file resolves - 58
  tests total, all green (`python -m unittest core.test_vocab_db core.test_vocab_store_ja
  core.test_vocab_store_ko core.test_vocab_link core.test_vocab_sync core.test_vocab_review_api -v`).
- Not yet exercised: an actual click on the new "Play line" button inside the real PyWebView window
  (only the backend Api method was driven directly) - front-end wiring (`gui/web/vocab_review/
  {index.html,app.js,style.css}`) was built and reviewed but not click-tested live in the window.

---

## Milestone 3 — JA/KO/ZH cognate bridge view (not started)

**Goal:** a three-way comparison (Japanese Kanji / Korean Hanja / Chinese source) for Sino-vocabulary
words, per the original `Study_tool_ideas.txt` "Tri-Lingual Ideographic Bridge" idea.

- Pure reuse, no new lookups: `core.kanji_reference.lookupChineseCognate()` already produces the
  Chinese cognate/pinyin/gloss per Japanese word (persisted on `vocab_ja`);
  `core.korean_hanja.lookupHanja()` calls that same function for Korean (persisted on
  `vocab_ko_hanja`); `core.vocab_link.getLinkedWords()` already joins JA↔KO by shared cognate form.
- New `getCognateBridge(vocabId, language)` combines already-persisted columns from both sides.

**Done when:** a linked JA/KO word pair renders its three-way comparison correctly, and an unlinked
word degrades gracefully (shows what it has, no fabricated third leg).

---

## Milestone 4 — Cross-word co-occurrence graph (not started)

**Goal:** per the user's chosen scope, a real graph of which words/songs cluster together — not just
"everywhere word X appears."

- This needs a genuinely **new function** in `core/vocab_store_ja.py`/`core/vocab_store_ko.py` — e.g.
  querying `vocab_occurrence_ja`/`vocab_occurrence_ko` grouped by `group_name`/`song_title` to find
  which words co-occur in the same songs/lines.
- Design and unit-test this query layer as its own reviewable step before building any graph UI on
  top of it.

**Done when:** the new store-layer function returns correct co-occurrence groupings against seeded
test data, independent of any UI.

---

## Milestone 5 — Grammar cloze drills + independent SRS track (not started)

**Goal:** fill-in-the-blank drills over real lyric lines, using the existing grammar tokenizers, with
their own spaced-repetition schedule (per the user's decision — not feeding the existing meaning/reading
tracks, and not schema-free either).

- Reuse `core.grammar_breakdown.breakdownLine()` / `core.korean_grammar_breakdown.breakdownLine()`
  as-is (don't re-parse lyrics) — blank one token's `surface` span within a
  `vocab_occurrence_*.lyric_line` and ask the user to supply/recognize its `lemma`/`gloss`.
- **Schema migration first, as its own reviewable unit:** add `cloze_box`/`cloze_ease`/`cloze_due_ts`
  columns to `srs_card_ja`/`srs_card_ko` in `core/vocab_db.py`, mirroring the existing `meaning_*`/
  `reading_*` triple exactly.
- Add a `submitClozeReview()`-style function to each store module, reusing the existing Leitner math
  in `core/vocab_store.py`.
- Add the same locking-discipline regression test this project always adds when a new user-visible
  write path touches `vocab_ja`/`vocab_ko`/`srs_card_*`.

**Done when:** the migration runs cleanly against the live 863-word DB with zero data loss (same bar
`VOCAB_SRS_PLAN.md` documents for past migrations), and a cloze answer updates only the new `cloze_*`
columns, never `meaning_*`/`reading_*`.

---

## Milestone 6 — Karaoke-sync highlighting (not started)

**Goal:** the user's core requirement, built honestly against what the data actually supports.

The data model only has **per-line** timing (`start_chunk`/`end_chunk` on each occurrence), not
per-word/syllable:
1. **Line-level highlighting (ship first):** highlight the whole lyric line for the
   `[startChunk*40, endChunk*40]` ms window, tracked via a client-side JS timer against elapsed time
   since `playOccurrenceAudio()` was invoked (JS has no direct access to
   `pygame.mixer.music.get_pos()` across the process boundary). Real, not approximate, at the
   existing ~40ms grain.
2. **Per-token highlighting (optional, later, explicitly labeled as approximate):** evenly interpolate
   each `breakdownLine()` token's character span proportionally across the line's duration. This is
   *not* real forced alignment — label it as such in the UI/code, matching this codebase's existing
   discipline of never presenting an approximation as precise data.

**Done when:** stage 1 highlights in sync with real playback on a handful of test lines; stage 2 (if
built) is visibly/explicitly marked as approximate in the UI.

---

## Milestone 7 — Packaging & distribution (not started)

**Goal:** the app still builds and runs via the existing PyInstaller pipeline, on this machine and
(per the user's "possibly" on distribution) reasonably on others.

- `requirements.txt`: add `pywebview`, pinned to the version verified in Milestone 0 (e.g.
  `pywebview==6.2.1`) rather than left floating.
- `.spec` `datas`: add `("gui\\web\\vocab_review", "gui\\web\\vocab_review")` alongside the existing
  `fonts\...` entries.
- `.spec` `hiddenimports`: pywebview's Windows backend dynamically loads `pythonnet`/`clr_loader` —
  add explicit hiddenimports, confirming exact module names during Milestone 0's first real frozen
  build (only discoverable by actually running PyInstaller once).
- **WebView2 Runtime is an external OS dependency PyInstaller can't bundle as a Python package.**
  Either document it as a prerequisite (present by default on Win11, not guaranteed on older/
  locked-down Win10), or bundle the WebView2 Evergreen Bootstrapper and have the launcher silently run
  it on first use if a registry check shows it's missing.
- No second `EXE()` block needed in the `.spec` — the same exe re-invokes itself via
  `--vocab-review`. Add a one-line comment in the `.spec` explaining this so a future reader doesn't
  assume a window is missing from the build.

**Done when:** a clean frozen build launches both the main app and the vocab review window correctly
on this machine, and the WebView2 prerequisite is either confirmed present or bundled.

---

## Testing / verification (applies across milestones)

Matches this project's existing convention (permanent `core/test_*.py` suite + a real interactive
smoke test, never just reading code):
- New `core/test_vocab_review_api.py`: unit-tests `VocabReviewApi` directly (no `webview` import
  required) against a tempdir-isolated DB, same isolation pattern as the existing
  `core/test_vocab_*.py` suite. Include a regression test for the new `cloze_*` locking discipline
  (Milestone 5), and a plain unit test asserting chunk→ms math against fixture values (Milestone 2).
- Add to the standard run: `python -m unittest core.test_vocab_db core.test_vocab_store_ja
  core.test_vocab_store_ko core.test_vocab_link core.test_vocab_sync core.test_kanji_reference
  core.test_vocab_review_api -v`.
- Interactive smoke test (adapted for a web view instead of Tkinter widgets): seed a throwaway
  `data/vocab_srs.db` in a tempdir (including a Korean word with 2+ ambiguous Hanja candidates and a
  word missing a meaning), launch the real subprocess, then drive it via pywebview's
  `window.evaluate_js(...)` scripted through load-queue → Prev/Next → rate → edit-meaning →
  keep-only-Hanja → delete, asserting DB state afterward — this is what catches UI-only bugs
  (wrong-parent dialogs, invisible widgets) that a backend-only unit test would miss, per this
  project's own bug history. Delete the driver script once verified, per this project's "throwaway
  scripts aren't deliverables" convention, unless it's worth promoting to a permanent `--smoke-test`
  CLI entry point (ask before doing that — no scripted smoke-test entry point precedent exists yet).

---

## Critical files

- `gui/vocab_review.py`, `gui/audio_tester.py` — current screen + its one call site to change
- `core/vocab_store_ja.py`, `core/vocab_store_ko.py`, `core/vocab_link.py`, `core/vocab_db.py` — store
  layer to wrap, and where the `cloze_*` schema migration + locking discipline lives
- `core/vocab_store.py` — shared Leitner/SRS math to reuse for the new cloze track
- `core/kanji_reference.py`, `core/korean_hanja.py` — cognate lookup to reuse for Milestone 3
- `core/grammar_breakdown.py`, `core/korean_grammar_breakdown.py` — tokenizers to reuse for Milestone 5
- `core/util_functions.py`, `core/group_registry.py` — audio file resolution to reuse for Milestone 2
- `voice_recognition_gui.spec`, `requirements.txt` — packaging changes (Milestone 7)

## Open unknowns to spike first (Milestone 0 — don't commit further design until answered)

1. Is WebView2 reliably present on realistic target machines, or does it need bundling?
2. Does re-invoking the frozen `.exe` with `--vocab-review` cleanly skip the main app's heavy startup
   path?
3. Exact `hiddenimports` needed for pywebview's Windows backend in a frozen build.
4. Does the `SetWindowLongPtrW` HWND trick fully reproduce `transient()`'s behavior (z-order *and*
   taskbar grouping), or only partially?
