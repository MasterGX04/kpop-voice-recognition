# Flashcard scheduling: replace the Leitner boxes with FSRS + Anki-style queueing

Status: plan only, nothing built yet (revised 2026-09-30 after the Python 3.13 migration; the
`venv/` on Python 3.13 with `fsrs` 6.x already exists and all 283 `core/` tests pass in it).

## Why (diagnosis, verified by reading the code)

`core/vocab_store.py::computeNextReview` is a 6-box Leitner ladder `[1, 3, 7, 14, 30, 90]` days
scaled by a barely-used ease. Consequences:

- "Easy" only skips one box, so 私 / 僕 / 本当 come back every <=90 days forever (hard cap).
- Every new card is created with `due_ts = now`, `getDueCards` sorts by `due_ts ASC LIMIT 200`, so
  on first load *all* words are due at once and the order is effectively insertion order. The
  Shuffle checkbox is client-side only; "Show all" bypasses the SRS and sorts by lemma.
- "Again" just resets to box 1 (due tomorrow, not minutes), so a failed word isn't re-tested in the
  same session - there is no in-session relearning at all.
- No new-cards-per-day limit, no known-word handling, no review history.

Goal: maximise long-term retention (not simplicity), i.e. what modern Anki does: FSRS scheduling,
learning/relearning steps, daily new limit, review log.

## What FSRS is (one paragraph, for the record)

Per card it tracks **stability** (days until recall probability falls to the target retention),
**difficulty** (how hard *you* find it), and derives **retrievability** (recall probability right
now). You choose a target retention (default 90%); the scheduler picks the due date that hits it.
Four ratings: Again / Hard / Good / Easy. Intervals grow without a practical cap, so easy words
drift to years. The weights are fitted to millions of real reviews and can later be re-fitted to
your own log.

## Environment and library (verified empirically, Python 3.13.2 venv, 2026-09-30)

Environment: `venv/` (Python 3.13, gitignored) built from `requirements.txt`; the old Python 3.9
`pyannote-env` is untouched and remains the ML fallback. `fsrs` (6.3.2, FSRS-6, 21 weights,
pure Python, only dependency `typing-extensions`) is in `requirements.txt`.

API facts, each checked by running it:

- `from fsrs import Scheduler, Card, Rating, State, ReviewLog`.
  `Scheduler(parameters=<21 defaults>, desired_retention=0.9, learning_steps=(1 min, 10 min),
  relearning_steps=(10 min,), maximum_interval=36500, enable_fuzzing=True)`.
- `scheduler.review_card(card, rating, review_datetime=None, review_duration=None)` returns
  `(new_card, ReviewLog)` and **does not mutate the input card**. So an interval preview for the
  four buttons is just four calls on the same card - no separate preview API needed.
- **There is no "New" state.** `State` is only `Learning=1, Review=2, Relearning=3`, and a fresh
  `Card()` is already `Learning` with `stability=None`, `difficulty=None`, `last_review=None`.
  "New" is *our* concept: a card that has never been reviewed (`stability IS NULL`). Our own DB
  state column therefore uses 0 = New plus the library's 1/2/3.
- `Card(card_id=..., state, step, stability, difficulty, due, last_review)`; `card_id` is just an
  int we can set. `to_dict()`/`from_dict()` round-trip exactly (datetimes as ISO strings, tz-aware
  UTC); we store our own int timestamps and rebuild `Card` objects at the wrapper boundary.
- `scheduler.get_card_retrievability(card, datetime)` exists (0.909 at due time for a card that
  came due after 2 Goods, decaying to 0.757 ten days later - matches the 90% target).
- `ReviewLog` is `card_id, rating, review_datetime, review_duration` - our own `review_log` table
  is a superset of that.
- `scheduler.reschedule_card(card, review_logs)` rebuilds a card from its review history - this is
  what lets us change weights/retention later without losing progress (see M6).
- Optimizer is the `fsrs[optimizer]` extra: needs `torch`, `numpy`, `pandas`, `tqdm`.
- Fuzzing is **on by default** and does change the due dates run to run (verified) - tests must
  construct `Scheduler(enable_fuzzing=False)`.

Verified behaviour with fuzzing off (defaults, new card at t0):

| Rating on a new card | Result |
|---|---|
| Again | Learning step 0, due in 1 min |
| Hard | Learning step 0, due in 5.5 min |
| Good | Learning step 1, due in 10 min; a second Good graduates to Review, due in 2 days |
| Easy | graduates straight to Review, due in 8 days |

Again on a Review card -> Relearning step 0, due in 10 min. Good streak (fuzzed run): 2, 12, 48,
162, 527, 1375, 3272 days. Easy streak: 11, 76, 433, 2052 days. After a lapse the difficulty jumps
(~2.1 -> ~7.4) and recovery intervals are 4 then 12 days.

Wrapper module: `core/srs_fsrs.py` is still the *only* file that imports `fsrs` (keeps library
types out of the SQLite layer and makes a future FSRS-7 swap a one-file change). Exposes plain
int/dict functions: `review(cardState, rating, nowTs)`, `previewIntervals(cardState, nowTs)` ->
`{again, hard, good, easy}` seconds, `retrievability(cardState, nowTs)`.
`core/vocab_store.py` keeps `now()` and drops `computeNextReview`.

## Data model

Each of the 3 tracks (meaning, reading, cloze) x 2 languages currently has `*_box, *_ease,
*_due_ts, *_last_reviewed_ts` in `srs_card_ja` / `srs_card_ko` (core/vocab_db.py). Add per track,
using the existing `ALTER TABLE ADD COLUMN` migration loop in vocab_db.py:

`*_state` (0 New, 1 Learning, 2 Review, 3 Relearning - see note above), `*_step`, `*_stability`
(NULL while New), `*_difficulty` (NULL while New), `*_reps`, `*_lapses`, `*_suspended`. Keep
`*_due_ts` / `*_last_reviewed_ts` (already indexed). Leave old `*_box` / `*_ease` columns in place
(harmless, lets us roll back).

New table `review_log` (both languages, one table): `id, language, vocab_id, track, rating,
state_before, stability_before, difficulty_before, due_before_ts, reviewed_ts, elapsed_days,
duration_ms`. Required for (a) the daily new-card count, (b) session stats, (c) fitting FSRS
weights to your own reviews and `reschedule_card`. Cheap to write, impossible to backfill, so add
it from day one.

Migration of existing cards: never-reviewed (`last_reviewed_ts` NULL) -> New (state 0, stability
NULL). Reviewed cards -> Review state with `stability = box interval days`, `difficulty` mapped
from ease (2.5 -> ~5; 1.3 -> ~9; higher ease -> lower, clamped 1-10), `due_ts` unchanged.
Approximate is fine: FSRS self-corrects within a couple of reviews.

## Milestones (each small, each with a live test hook, per the verify-then-build habit)

### M0 - Wrapper + pure-logic tests (no UI, no DB)
`core/srs_fsrs.py` + `core/test_srs_fsrs.py`, run with `venv\Scripts\python -m unittest`. Tests:
a never-reviewed card + Good walks Learning steps then graduates; Easy on a new card graduates
immediately; Again on a Review card -> Relearning, difficulty up; intervals grow on Good streaks
with no cap below 10 years; `previewIntervals` matches what `review` then actually produces;
card state round-trips through plain ints; retrievability is ~0.9 at the due time. Use
`Scheduler(enable_fuzzing=False)` in tests. (`fsrs` is already in requirements.txt.)

### M1 - Schema + migration + review log
vocab_db.py columns/table above; migration function with a test that builds an old-schema DB
(existing tests in core/test_vocab_db.py show the pattern) and asserts New/Review mapping and that
re-running the migration is a no-op (same trap the cloze_due_ts backfill comment describes).

### M2 - Rating with four buttons and real interval preview
- `submitReview` (vocab_store_ja.py, vocab_store_ko.py) -> load state, call wrapper, write columns +
  `review_log` row in one transaction. Rating values become `again|hard|good|easy`.
- gui/vocab_review_api.py `rate()` unchanged shape, plus new `previewIntervals(vocabId, language,
  track)` for the current card only (like `getCardDetail`).
- index.html / app.js: add **Hard**; label each button with its outcome ("Again <1m", "Hard 5m",
  "Good 10m", "Easy 8d"). This is what makes "Easy" mean something visible.
- Test: rate a card through all four ratings on a temp DB, assert due/stability/log row.

### M3 - Anki-style queue (fixes the "order never changes" problem)
Replace static `getDueCards(limit=200)` list-and-index with a scheduler-driven pull:
1. **Learning/relearning** cards due now (minute-scale steps) - come first.
2. **Review** cards due, ordered by lowest retrievability first (most-forgotten first), random
   tiebreak, so order changes every session.
3. **New** cards, capped at N/day (default 15, setting), ordered by how often the word occurs in
   your saved lyrics (`vocab_occurrence_*` count) so useful words arrive first.
Daily new count is derived from `review_log` (first review of a card today) - no separate counter.
**Bury siblings:** after reviewing one track of a word, hold its other tracks until tomorrow so
the cloze card cannot leak the meaning answer.
Frontend: study mode calls `getNextCard()` after every rating instead of `state.index += 1`, so
a card rated Again reappears ~1-10 minutes later in the same session. Keep the existing
prev/next/random/"Show all" flow as a separate **Browse** mode (it is editing/browsing, not study).
The Shuffle checkbox is then redundant in study mode. Test: seed 30 new + 10 due + 2 learning
cards, assert ordering and the daily cap, and that a second call after an Again returns the card
once its step elapses (inject `nowTs`).

### M4 - Known-word handling (the 私 / 僕 / 本当 problem)
- **Suspend** button (`*_suspended`): never shown until un-suspended.
- **"I already know this"**: seeds the card directly into Review with a long stability (e.g. 60
  days) rather than rating Easy on New (verified: Easy on a new card only yields 8 days). It still
  gets tested once later, at your target retention, instead of never.
- **Bulk triage screen**: list the words with the highest occurrence counts in your library, with a
  checkbox each -> "mark known / suspend". Do not hard-code a stoplist; you judge what you know.
- Test: known card does not appear in the new queue; seeded stability yields the expected first due.

### M5 - Session/deck counters + retention target setting
Header shows New / Learning / Review counts remaining today (Anki-style). Settings: target
retention (default 0.90; higher = more reviews for fewer lapses; 0.95 is reasonable if you accept
the workload), new cards/day, learning steps. Persist next to existing app settings. Test: counts
from a seeded DB.

### M6 - Optional, after ~1,000 logged reviews: fit weights to your own history
Install `fsrs[optimizer]` (torch/numpy/pandas) into the ML environment (`requirements-ml.txt`,
where torch already lives) and run the optimizer over `review_log`; paste the resulting 21-weight
tuple into the wrapper. Because the log stores every rating, `reschedule_card` can then rebuild
each card's state under the new weights (or a new retention target) without losing progress. Not
verified yet: the optimizer's exact call signature and input format - check when reaching this.

## Beyond scheduling: making "recalled" real

FSRS only trusts your self-rating. Cheap upgrades, only after M0-M5:
- Reading track: type the reading (kana / hangul) and auto-suggest Again vs Good from the match.
- Meaning track: reveal the gloss only after you commit to a spoken/typed answer.
Not planned in detail yet.

## Risks / open questions

- The app itself has not been launched on Python 3.13 (only imports and the `core/` tests were
  run), so open the flashcard window and the main GUI once before starting M1.
- `run.bat` and `voice_recognition_gui.spec` still point at the old `pyannote-env` / 3.9; switch
  them when you commit to the 3.13 venv. The exe build on 3.13 has not been tried.
- Working tree has many uncommitted changes in exactly these files (vocab_db.py, vocab_store_*.py,
  vocab_review_api.py, app.js, test files). Commit or checkpoint before starting M1.
