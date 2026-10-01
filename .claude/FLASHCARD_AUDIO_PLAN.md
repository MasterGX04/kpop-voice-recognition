# Flashcard "Play audio" - bug analysis + per-session song variety

Status: analysis + plan only, nothing built yet (2026-09-29).

## Part 1 - Bug: "Play audio" fails for most Korean words

### Diagnosis (verified empirically, corrects the commands.txt note)

The commands.txt note says ".wav-only songs break". That is **not** the cause:

- `pickBestAudioForStem()` (core/util_functions.py) scores `.wav` *highest* and handles every
  extension; it does not require an mp3.
- Ran resolve -> `ensureAudioForPlayback` -> `pygame.mixer.music.load/play` for every song that has
  a stored vocab occurrence (31 songs, wav / mp3 / flac mix). All 31 succeed.
- `VocabReviewApi.playOccurrenceAudio("BTS", "Aliens", 79, 200)` (a wav-only song) -> `ok: True`.

Real cause: `end_chunk` is NULL for most occurrences, and `playOccurrenceAudio` does
`chunkToMs(endChunk)` -> `None * int` -> `TypeError`.

```
playOccurrenceAudio("BTS","Aliens",79,None)
  -> TypeError: unsupported operand type(s) for *: 'NoneType' and 'int'
```

Why NULL: `core/vocab_sync.py::_resolveChunks()` only knows an end when the lyric has a linked label
(`linkedLabel`) that `core.label_runs.resolveLyricSpans` can resolve. Otherwise it stores
`(lyric.startChunk, None)`. Korean lyrics are almost never linked yet; Japanese ones mostly are.

| DB table | occurrences with NULL end | words with >=1 playable occurrence |
|---|---|---|
| vocab_occurrence_ko | 9534 / 9556 | 20 / 702 |
| vocab_occurrence_ja | 146 / 538 | 222 / 290 |

Japanese failures also include mp3 songs (e.g. TWICE "Do Not Touch"), which is what disproves the
wav theory. "Korean songs = wav" is a coincidence of the same songs being the unlinked ones.

Secondary issue: `app.js::playOccurrenceAudio` ignores the `{ok:false}` result, so the button just
does nothing with no message.

### Fix (two layers)

1. **Runtime fallback (fixes it immediately, no rescan).** In `playOccurrenceAudio`, if
   `endChunk` is None, play `[startChunk, startChunk + DEFAULT_CLIP_CHUNKS)`. Pick the constant to
   match roughly one sung line (start ~8s; confirm by ear on a few Korean songs).
2. **Better data (proper clip length).** In `_resolveChunks`, when there is no linked label, use the
   *next lyric's startChunk* in the same song as the end, capped at the same max. Needs the sorted
   lyric list passed in (it already has `lyricEntries` in scope at vocab_sync.py:172). Re-running the
   scan refreshes existing rows already (`addOccurrence` updates spans in place). Linking lyrics to
   labels in the lyrics editor still gives the most accurate span and overrides this.
3. **Surface errors in the UI.** `app.js::playOccurrenceAudio`: on `!ok` show the message near the
   button instead of failing silently.

Tests (unittest, next to core/test_vocab_review_api.py): None-end plays with default duration (mock
pygame); `_resolveChunks` next-lyric-start + cap; last lyric in a song (no next) uses the cap.

## Part 2 - Feature: random song per word, chosen once per app load

Goal: a word that appears in several songs is shown/played from a different song each time the
flashcard window is opened, not always the same first row.

Current behaviour: `getCardDetail` calls `store.getOccurrences(vocabId, limit=1)`, an unordered
`SELECT ... LIMIT 1`, so it is effectively always the same row. The data supports the feature:
203 Korean and 63 Japanese words occur in more than one song.

### Design

- `VocabReviewApi.__init__` creates a per-process `self._sessionSeed` (the flashcard window is its
  own process, so "app load" == process start).
- New helper `_pickOccurrence(store, vocabId)`:
  1. fetch all occurrences for the word (raise/bypass the `limit`),
  2. group by `(group, song)`; prefer songs that have a *playable* occurrence, but do not exclude
     the rest (Part 1's fallback makes all of them playable, so after Part 1 this is moot),
  3. `rng = random.Random(f"{seed}:{vocabId}")`; choose a song, then an occurrence within it.
  Seeding on `(session, vocabId)` means Prev/Next on the same card shows the same song for the whole
  session (no flicker between the text line and the audio that plays), and only changes on reload.
- `getCardDetail`: use `_pickOccurrence`.
- `getClozeCardDetail`: keep its "shorter lines first" ordering
  (`orderOccurrencesForCloze`), but move the session-chosen song's occurrences to the front so the
  cloze also varies by session; fall through to the other songs if that song yields no card. Note:
  the cloze deliberately re-rolls per review (Milestone 5 decision); this change makes only the
  *song* per-session, so confirm with the user whether cloze should stay per-review-random.
- Store layer: add `getOccurrences(vocab_id, limit=5, song=None)` or a new `getAllOccurrences`
  (both vocab_store_ja/ko) instead of pulling 20 rows and filtering in Python. Optional
  `ORDER BY` is not needed since selection is by seeded RNG.

Tests: same seed -> same pick; different seeds over many trials hit >1 song for a multi-song word;
single-song word is unaffected; word with no occurrences returns None.

## Order of work (small, testable milestones)

1. Runtime None-end fallback + UI error message + tests (unblocks the reported bug).
2. `_resolveChunks` next-lyric-start end + rescan + tests.
3. `_pickOccurrence` + `getCardDetail` + tests.
4. Cloze song preference (pending the question above).

## Open questions

- Default/max clip length in chunks (how long is one CHUNK_DURATION_MS, and what feels right for a
  line)?
- Cloze: keep per-review random line, or per-session song too?
