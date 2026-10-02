# Karaoke Highlighting + Flashcard Audio Fixes (Flashcard Web View + Tk LyricBox)

**Status: PLAN ONLY (written 2026-09-30). Nothing built yet.** Update each Milestone's status as it ships,
in the same build-log style as `.claude/FLASHCARD_WEB_UPGRADE_PLAN.md` (this extends its Milestone 6).

## Context

Two surfaces need karaoke-style highlighting, in this priority order:
1. **Flashcard web view** (`gui/web/vocab_review/`, `gui/vocab_review_api.py`) - priority, and it has
   audio/display bugs to fix first.
2. **Tk lyric box** (`gui/lyrics_box.py`, driven by `renderLyrics()` in `gui/audio_tester.py`).

Data reality: a lyric only has **per-line** timing (`startChunk`/`endChunk`, 1 chunk = 40 ms, see
`core.util_functions.CHUNK_DURATION_MS`). There is no per-word/syllable timing, so anything finer than a
line is a proportional estimate and must be labeled approximate in the UI/code.

### Terminology: "per-syllable pop"
- Each syllable/token is in one of two states: **dim** (not sung yet) or **lit** (member's full color).
  It flips to lit the instant its estimated time slice begins. Like a bouncing ball.
- **Pop** = that instant flip. One canvas item / one `<span>` per token, no geometry tricks; cheap and
  deterministic (same result in playback, seeking, export).
- **Wipe** = gradual left-to-right fill within a syllable. Smoother, but needs a curtain rectangle (Tk)
  or `background-clip: text` gradient (web). Optional later.
- Adds over line-level highlight: you can see *where in the line* the singer is; also powers
  "click a word to play from there".
- Does NOT add: real timing. Token boundaries are estimated (weighted by char/mora count). Real word
  timestamps (forced alignment) would swap in later with no renderer change.

### Why not Gemini's "Expanding Window Mask" (nested tk.Canvas) for Tk
- `canvas.create_window` widgets always draw above all canvas items; `tag_raise` / `enforceCanvasLayering`
  can't layer them. A nested canvas is opaque.
- Lyric text is many items (one per `|` colored segment) moved via `textItemOffsets` every tick;
  a native child window per card adds flicker, extra move/hide/rebuild/destroy bookkeeping.
- Export captures the real window via `PrintWindow`/`BitBlt` (`gui/video_record.py`) on a fixed export
  timeline - highlight must be a pure function of `chunkIndex`, not a wall-clock width tween.
- The per-chunk canvas update is not a CPU problem; 40 ms grain is fine for karaoke.
- If a smooth wipe is wanted: use a white canvas *rectangle* curtain item (layers, moves, captures fine).

---

## Part 1 - Flashcard audio/display bug fixes (do first; karaoke timing builds on the same span)

Reported bugs: (a) some lines get cut off, (b) words at the end of a line are cut off by audio playback,
(c) no way to start playback mid-lyric (e.g. just hear the last word).

Current code: `VocabReviewApi.playOccurrenceAudio(group, song, startChunk, endChunk)`
(`gui/vocab_review_api.py:215`) plays `[start, end)` and stops via `threading.Timer(duration)`.
Spans come from `core/label_runs.py` (`resolveLyricSpans` / `inferLyricSpans`) via `core/vocab_sync.py`,
stored as `start_chunk`/`end_chunk` on `vocab_occurrence_*`. The card renders
`From {group} — {song}: {lyricLine}` as a single text node (`app.js:359`).

**A1 FINDING (2026-09-30, verified against real data): root cause is a hard 250-chunk (10 s) cap, not
the label resolver.**
- User-reported lines: Doughnut/Tzuyu (lyric card starts 2445, `linkedLabel` 2456-2473 but her rows run
  2456-2473, 2478-2579, 2605-2828) and Doughnut/Sana (911; rows 922-936, 941-1045, 1070-1233, ...).
- `resolveLyricSpans` already merges these correctly: Tzuyu -> (2456, 2828), Sana -> (922, 1289).
- But `core/vocab_sync.py:_resolveChunks` (line ~77) does `end = min(end, start + MAX_CLIP_CHUNKS)` and
  `MAX_CLIP_CHUNKS = 250` (`core/label_runs.py:156`), so the stored DB spans are (2456, 2706) and
  (922, 1172) - i.e. ~120 and ~117 chunks (4.8-4.7 s) of real audio chopped off each. The cap was added
  to stop the *last* linked lyric of a song merging same-member rows forward for minutes; it also
  truncates legitimately long lines (slow ballads, continuous BTS raps).
- Scope: 186 of 763 JA occurrences and 297 of 2416 KO occurrences sit exactly at the cap (JA worst: BTS
  Let Go 12 lines, TWICE Marshmallow 8, BTS Crystal Snow 7, BTS Stay Gold 5, TWICE Doughnut 4).
  `inferLyricSpans` (lines ~210/265) applies the same cap to unlinked lyrics.
- Secondary: Tzuyu's resolved end (2828) overlaps the next lyric's linked start (2806) by ~22 chunks
  (Chaeyoung's row starts inside Tzuyu's last row) - the resolver only bounds by row *index*, so a
  same-member row can straddle the next card's start. Decide in A3 whether to clip to the next lyric's
  `linkedLabel.startChunk`.
- Not yet checked: the playback/timer hypothesis (A2) - the cap fully explains the reported cases, so
  treat padding as a small separate polish item, not the cause.

Other hypotheses (still unverified): MP3 seek/decode latency; the "cut off" *text* may be a CSS
overflow/multi-line display issue (A4).

### Milestone A1 - DONE (see finding above). No code changed.

### Milestone A2 - Fix the cap (this is the real fix; do first)

**Status: DONE (2026-09-30).** Deviations from the design below, found while implementing:
- Final rule: bounded spans (a next linked lyric exists) are never capped. Only a span with no next
  boundary is capped, and the cap limits the *merged-forward extension* only - never the lyric's own
  anchor row (`max(anchorRowEnd, start + cap)`), in both `resolveLyricSpans` and `inferLyricSpans`.
- New `MAX_UNBOUNDED_CLIP_CHUNKS = 250` in `core/label_runs.py` (kept 10 s, NOT raised): a first try at
  30 s over-extended the three real last-lyric cases (BTS Film out V, BTS Let Go Jungkook/Jimin, WJSN
  Save Me Bona). The blanket cap in `core/vocab_sync.py:_resolveChunks` is removed. `MAX_CLIP_CHUNKS`
  now only caps an ad-lib's stated duration.
- Did NOT clip a resolved end to the next lyric's linked start (the Tzuyu 2828 vs Chaeyoung 2806
  overlap): that would re-truncate the tail, which is the reported bug. Left as-is.
- Result on live `data/vocab_srs.db` (backup: `data/vocab_srs.db.bak-before-A2-20260930-142500`): Doughnut
  Sana now 922-1289, Tzuyu 2456-2828. JA >250-chunk occurrences: 191 of 789; KO: 286 of 2462; longest
  span 403 (JA) / 371 (KO) chunks, i.e. ~16 s / ~15 s; none remain capped-and-truncated except 11 JA / 18
  KO exact-250 unbounded tails. (Row counts rose 763->789 JA, 2416->2462 KO because the recompile also
  picked up earlier intake fixes, not this change.)
- Tests: new cases in `core/test_label_runs.py` (long bounded span, unbounded cap, anchor row never cut,
  override beats cap) and updated `core/test_vocab_sync.py`; 200 tests green via `venv/Scripts/python.exe`.
- Note: default `python` on PATH is 3.9 without `fsrs`; run the suite with `venv/Scripts/python.exe`.

(Original design, for reference:)
- Stop applying a blanket 250-chunk cap to *bounded* spans. New rule: a span's end is bounded by the
  next linked lyric's row (already done by `resolveLyricSpans`) and, for the final lyric of a song only
  (no next bound), by a cap. Move the cap so it only applies in that unbounded case (consider a larger
  cap, e.g. 20-30 s, since the 10 s value was a guess).
- Keep `inferLyricSpans`'s fallback cap for lyrics with no label match at all (there it is a genuine
  guess).
- Optionally clip a resolved end to the next lyric's `linkedLabel.startChunk` (overlap case above).
- Tests: fixture values from real Doughnut Tzuyu (2456 -> 2828) and Sana (922 -> 1289); a regression
  test that the final-lyric case is still bounded; BTS Let Go / Crystal Snow real spans.
- Re-run "Compile Vocab for all songs" (upsert refreshes `start_chunk`/`end_chunk` in place) and confirm
  the capped-at-250 counts drop (186 JA / 297 KO today). Back up `data/vocab_srs.db` first.
- Consequence: clips become long (10-12 s). That makes A5 (play from the middle / per-word start) and
  K1/K2 (highlight showing where you are) matter more, not less.

### Milestone A2b - Playback padding (small polish, only if a residual tail cut remains after A2)
- Trailing pad (~300-500 ms) and small leading pad in `playOccurrenceAudio` as constants next to
  `CHUNK_DURATION_MS`; do not alter stored chunks. Unit test with the Breakthrough fixture values.

### Milestone A3 - Verify remaining span quality
- Spot-check the previously-capped lines after A2 (esp. BTS Suga/J-Hope Japanese rap lines) for
  over-extension into the next member's part; use `endChunkOverride` for hand fixes. Label-run logic has
  known traps (duets, nested rows, co-sung credit mismatch; see `core/label_runs.py` docstring).

### Milestone A4 - Display fix

**Status: DONE (2026-10-01).** Cause found: 691/881 JA and 2169/2462 KO stored lyric lines contain
newlines, and `textContent` into a plain span collapsed them, so multi-line lyrics ran together; a few
lines (4 JA / 20 KO) also contain `|` Tk colour-change markers that should never display. Fixed with
`white-space: pre-wrap; overflow-wrap: anywhere` on a dedicated `#occurrenceLine` block, `|` stripped in
`karaoke.js:cleanLine`, the source label ("From group - song") split onto its own line, and the controls
column no longer squeezes the text (`min-width: 0`).
- Wrap, never truncate; show the full multi-line lyric. Test with the longest real lines.

### Milestone A5 - Play from the middle

**Status: DONE (2026-10-01).** Deviations: (1) tokenizing/timing is client-side JS
(`gui/web/vocab_review/karaoke.js`, `Intl.Segmenter`; tests: `node gui/web/vocab_review/karaoke.test.js`),
not `breakdownLine()` - that function drops tokens and exposes no offsets, so it can't reproduce the exact
displayed text. The Tk lyric box (T1+) needs its own port of this weighting. (2) The front end sends a
`startFraction` (0..1), not ms; `core.util_functions.clipStartOffsetMs` (WORD_LEAD_MS=300,
MIN_TAIL_MS=600) turns it into the real offset, so the 40 ms constant stays single-sourced. (3) UI: every
word clickable + a "Last word" button; small kana add no weight; labelled as an estimate. Live-tested in a
real PyWebView window (12 checks) against the real Doughnut/Tzuyu line.
- `playOccurrenceAudio(group, song, startChunk, endChunk, offsetMs=0)`.
- Front end: each token in the lyric line is clickable -> plays from that token's estimated time to the
  clip end (+ trailing pad). Start slightly early (~-200 ms) so the word isn't clipped.
- "Play last word" shortcut (seek to ~last 20% of span); "replay from here" key.
- Estimated token time = same proportional table as Milestone K2 (build once, share).

---

### Milestone B1 - Span correctness (DONE 2026-10-01; found via BTS Stay Gold cards 20/21)
- **Field bug:** label rows are `[member, start, end, isBacking, isAdLib]` (`gui/audio_tester.py:55`).
  `core/label_runs.py` read `row[3]` (isBacking) as "isCut": it dropped real backing vocals and let ad-lib
  rows through. Now `_isAdLibRow` (row[4]); backing counts as the member's singing.
- **Broken links:** `core.label_runs.resolveAllSpans` (used by `vocab_sync`) uses the lead-in-corrected
  inferred span (`startChunk + 11`) for a hand link that matches no row, instead of the stale snapshot, and
  clips a resolved span at the next unresolved card's sung start - but never through a row the card owns.
  Also bounds merging by time (a backing row tied with the next card's start). 24 of 1184 spans changed.
- **Needs a "Compile Vocab for all songs"** to refresh stored spans (back up `data/vocab_srs.db` first).
- **Mid-row card starts are deliberate (author-confirmed):** Stay Gold card 21 (`startChunk` 2787) is NOT a
  data error - 君を is sung at 2798 = 2787 + 11, at the end of the 2727-2832 row, because the Add Lyric menu
  only auto-maps a lyric to a label *start*; for a mid-label lyric the author adds the marker by hand and
  edits `startChunk`. The 11-chunk rule holds, just against the sung start, not a row start. So a span is clipped
  at the next card's sung start ONLY when that card includes this card's singer (J-Hope 20 -> 21, SWIM Suga ->
  Suga+Jungkook). A different singer starting over the end of a line never cuts it (author's rule: play the
  whole line, overlap included - Stay Gold card 21 plays to 2996 though Jungkook's card starts at 2992; ~25 K-pop
  cards now run 20-85 chunks into another singer's start). A tail pad (A2b) is still unbuilt.
- **Start = `startChunk + lead` for every non-ad-lib card** (author rule: the 11 chunks are the lyric box
  animating in; the singer starts at +11). Hand-linked cards used to start at the linked row's start, which
  disagreed for 8 cards (Crystal Snow 3/21/22/32, Let Go 6/15, Stay Gold 36, Save Me 0) - those links likely
  point at the wrong row and are worth re-linking. The real defect is only the stale `linkedLabel` (2727-2996 matches no row; rows were split
  after linking - same on cards 2 and 4), which inference now covers.

### Milestone B2 - Pause-aware word timing (DONE 2026-10-01)
- `core/karaoke_timing.py` (tests: `core/test_karaoke_timing.py`): label rows -> "stretches" of
  continuous singing (ad-libs ignored, overlaps fused, only genuinely OVERLAPPING rows fused - every gap is a boundary, see B3); words weighted by
  kana-reading morae (fugashi) / English syllables / Hangul blocks, not characters; words assigned to
  stretches by a small DP (text share vs time share, preferring cuts at line breaks, then after a
  particle). `VocabReviewApi.getLineTiming` feeds `app.js` (redraws the line after the first paint).
- Known limit: when a pause falls *inside* a written line (Stay Gold card 21: pause after 君を), the DP
  would prefer the line break at the original 0.02 particle-cut penalty. Lowered to 0.01 (calibrated on this
  one author-confirmed card; 9 other of 128 multi-row JA cards shift cuts - spot-check by ear). A manual
  `timingAnchors` marker is still the planned escape hatch (not built).
- Tk lyric box (T1+) can reuse `timeLine` directly instead of porting the JS weighting.

---

## Part 2 - Flashcard karaoke (priority)

### Milestone K1 - Line-level highlight (real timing at 40 ms grain)
- JS elapsed timer (`performance.now()`) started when `playOccurrenceAudio` returns; JS can't read
  `pygame.mixer.music.get_pos()` across the process boundary. Have the Api return `{startedAtMs}` so the
  front end can compensate for bridge latency (expect tens of ms drift; fine for line level).
- Highlight the whole line for the padded span. Clear on `stopAudio`, card change, clip end.

### Milestone K2 - Per-syllable pop (approximate, labeled as such)
- Tokenize with existing `breakdownLine()` (JA and KO); fall back to characters.
- Weight tokens by char/mora count; compute (tokenStartMs, tokenEndMs) over the padded span.
- One `<span>` per token, `.dim` / `.lit` classes; `requestAnimationFrame` loop toggles classes only when
  a token's state changes. UI note: "approximate timing".
- Shares the token/time table with A5.

### Milestone K3 - Optional wipe
- `background-clip: text` gradient on the active token only. Skip unless pop feels choppy.

**Done when:** replay a known-problem line, highlight follows the audio, clicking the last word plays it
fully.

---

## Part 3 - Tk LyricBox karaoke (after flashcard)

Structure today: all card content are items on the shared canvas; text is one `create_text` per `|`
segment (`_createColorCodedText`); `setPosition` moves everything via `textItemOffsets`;
`rebuildForResize` deletes and recreates all items; `renderLyrics(chunkIndex)` runs from the
`updateChunk` `after(40ms)` loop; export captures the real window per frame on a fixed timeline.

### Milestone T1 - Data plumbing
- Each `LyricBox` gets its span: `linkedLabel` via `core.label_runs` when linked, else up to next lyric's
  start. Pass in where lyrics are loaded (`gui/lyrics_editor.py`, `gui/audio_tester.py`). No stored-format
  change.

### Milestone T2 - Per-token dim/lit items
- In `createLyricDisplay`, build per-token dim+lit items (dim = member color blended toward card white),
  still registered in `textItemOffsets`/`textOnlyItems` so background fit keeps working.
- Ad-lib boxes take the separate `createAdLibDisplay` path - decide separately whether they get karaoke.

### Milestone T3 - `LyricBox.updateKaraokeForChunk(chunk)`
- Called from `renderLyrics` right after `setPosition`; cache last state, skip no-ops (same idea as
  `updateAdLibForChunk`). Pure function of `chunkIndex` -> seeking backwards and export need no state.

### Milestone T4 - Resize + export
- `rebuildForResize` rebuilds spans and reapplies current progress. Verify ~5 exported frames against known
  chunks (PrintWindow capture).

### Milestone T5 - Optional wipe
- Canvas curtain rectangle per line (never an embedded widget).

**Done when:** a karaoke-enabled card plays, seeks, resizes, and exports with the highlight on the right
syllable.

---

## Part 4 - Manual pause marker (M1-M4 DONE 2026-10-01)

**Why:** `core/karaoke_timing.py` places a card's words using the label rows as pauses. It cannot know a pause
that falls *inside* a written line when the text alone doesn't hint at it (Stay Gold card 21's 君を|優しく
only came out right after tuning one penalty on that one card). The marker is the author saying "the singer
pauses here" - the escape hatch for the rare card the automatic split gets wrong. Not the default path.

### The character: U+2063 INVISIBLE SEPARATOR (`PAUSE_MARK = "\u2063"`)
Verified empirically (2026-10-01), not assumed:
- Width 0 in Tk in the real lyric font (Pretendard Variable, installed) and in Segoe UI / Malgun / Yu Gothic /
  Arial, both `font.measure` and canvas `bbox`. Contrast: private-use U+E000 adds 21 px (tofu box) and soft
  hyphen U+00AD adds 9-11 px - both rejected. `|` itself adds 6 px (and is taken: per-member colour split).
- Zero hits for it (or any other Cf/Co/Cc char) in all 36 lyric files.
- Rejected: U+200B ZWSP (very common in text pasted from lyric sites - exactly the collision to avoid) and
  U+FEFF (JS `trim()` eats it; BOM semantics). U+2060 / U+200C / U+200D would also work but U+2063 is the one
  whose *meaning* ("invisible comma") matches and that nobody pastes by accident.
- Not whitespace to Python/JS: `"\u2063".isspace()` is False and `/\s/` doesn't match, so an unstripped marker
  survives `.strip()` and `if not text.strip()` checks, and **fugashi and `Intl.Segmenter` both emit it as its
  own token** (君|を|U+2063|優しく). So it MUST be stripped before any analysis - hence the choke point below.

### Semantics
`君を<M>優しく` = "the singer pauses between 君を and 優しく". In the timing DP the marker is a strong
*attractor* for a stretch cut at that word boundary (a large negative cut cost, so it beats the proportion and
particle/line-break priors), not a hard constraint, so it degrades gracefully:
- stretches >= markers + 1: every marker becomes a cut; remaining cuts chosen as today.
- fewer stretches than markers + 1 (a breath too short/unlabelled to be its own row): the unused markers
  become a small *rest* inside the stretch (words after it start later; `REST_BEATS` ~1.5, tunable).
- Words are split at a marker even mid-word (the marker is a hard word break; the word before it gets a
  `pauseAfter` flag).
This also lets the 0.01 particle-cut tuning be reverted to the principled 0.02 once the cards the author
cares about carry markers (9 other multi-row JA cards shift under 0.01 - see Milestone B2).

### Where the text flows (so the marker never leaks)
The raw lyric (with `|` and now the marker) lives in `saved_labels/<g>/<song>_lyrics.json` key `"korean"`
(all languages). Readers found:
- `gui/lyrics_box.py` draws it (` split("\n")` at ~344/674, `_createColorCodedText` splits on `|` at 760) and
  `gui/lyrics_editor.py:191` **saves `lb.koreanLyric` back to JSON** - so LyricBox must KEEP the raw text and
  draw from a stripped copy (`displayKorean`).
- `core/vocab_sync.py:46,181` (analysis + stored `lyric_line`), `core/kanji_reference.py:800,839`,
  `core/japanese_utils.kanjiLineToReading` (already strips `|`), editor tools: Convert Kanji -> Reading,
  Add Kanji to Document, Grammar buttons (read `koreanEntry` directly).
- Flashcard: reads the DB `lyric_line`; `getLineTiming` currently re-strips `|`.
Rule: one module, `core/lyric_text.py` - `PAUSE_MARK`, `stripForDisplay(text)` (marker only; keeps `|` for the
Tk colour split), `stripAll(text)` (marker and `|`, for analysis/DB/flashcard), `toEditorText` / `fromEditorText`.
Every reader above calls it; none strips inline.

### DB / flashcard decision
Keep the DB `lyric_line` CLEAN (stripAll at sync): zero blast radius for cloze, kanji reference, search, and the
clean text the card displays. The timing API then needs the raw text: add `lyric_id` to the occurrence query
and have `getLineTiming` read the raw lyric by `lyricId` from the lyrics file (fallback when there is no id:
match the lyric whose `stripAll(korean)` equals the stored line). Rejected: storing the marker in the DB (every
DB reader would have to strip) and a separate `timingAnchors` offset list (offsets silently break on any text
edit; the marker moves with the text).

### Editing UX (the marker is invisible, so the editor must show it)
In the Add/Edit Lyric dialog (`gui/lyrics_editor.py`, `koreanEntry`): a visible stand-in glyph (proposed `▾`,
needs a glance in Pretendard - it is 15 px wide there, fine in an editor) is what the author types/sees;
`fromEditorText` converts it to U+2063 on save and `toEditorText` converts back on load (lines ~749, 808, 838).
Button "Insert pause" + hotkey inserts it at the cursor. The stored/drawn text never contains `▾`. (If `▾` ever
appeared in a real lyric it would be eaten - corpus has none; pick another glyph if that changes.)
Later option: a tap-along tool in the Audio Tester (press a key at the pause while the line plays) that inserts
the marker at the matching word - the same data, a nicer input.
Note: lyrics are saved `ensure_ascii=False`, so the marker is a raw invisible character in the JSON and in git
diffs (reviewable only via the editor's `▾`). Accepted.

### Milestones (small, each with a live check, per the project convention)
- **M1** `core/lyric_text.py` + tests: strip/convert, round trip, `|` untouched by `stripForDisplay`, line that is
  only a marker counts as empty after stripping; corpus scan test that no real lyric already contains the marker.
- **M2** Wire the readers (vocab_sync, kanji_reference, japanese_utils, editor tools, LyricBox draw sites with
  raw text kept for save). Tests: identical analysis output with and without markers. Live: open a song with a
  marked card in the Tk lyric box and confirm no extra width/box and Save keeps the marker.
- **M3** `karaoke_timing`: split at markers, `pauseAfter`, attractor bonus, rest weight; tests incl. Stay Gold
  card 21 with the marker placed after 君を AT the 0.02 penalty. API: occurrence `lyricId`, `getLineTiming` reads
  the raw lyric. Live: flashcard click-to-play on a marked card.
- **M4** Editor UX (`▾` stand-in, Insert-pause button/hotkey, conversion on load/save). Live in the dialog.
- **M5 (optional)** Tap-along tool; revert the 0.01 penalty to 0.02 after the author has marked the cards that
  need it.

### Build log (2026-10-01)
Decisions (author): `▾` is the editor stand-in; the marker is allowed in the native-script line ONLY; the
0.01 particle-cut tuning was removed (back to the principled 0.02) now that markers exist.
- **M1** `core/lyric_text.py` (+ `core/test_lyric_text.py`, 17 tests). `PAUSE_MARK`/`EDITOR_PAUSE_GLYPH` are
  `chr(0x2063)`/`chr(0x25BE)` on purpose: the Write tool turned a `⁣` escape into the raw invisible
  character in source, so never write the escape in a file - use `chr()`.
- **M2** Readers wired: `vocab_sync` (analysis + stored `lyric_line` now clean of `|` and the marker),
  `kanji_reference`, `japanese_utils.kanjiLineToReading`, editor tools (Convert Kanji -> Reading, Add Kanji,
  both Grammar buttons - the Add-Kanji selection is re-mapped past stripped characters), `LyricBox` draws from
  `stripForDisplay` while `koreanLyric` stays RAW (it is saved back to JSON), flashcard `karaoke.js` cleanLine.
  Live check (real `LyricBox`, real Tk canvas): a marked card draws identically to an unmarked one, no marker
  ever drawn, raw text kept; `|` colour split + marker still gives two coloured segments.
- **M3** `core/karaoke_timing.py`: split at markers (a hard word break), `pauseAfter` flag, `_CUT_AT_MARKER`
  = -1.0 attractor in the stretch DP (never farmed by empty stretches), unused markers become a
  `REST_BEATS` = 1.5 rest inside a stretch. Verified a no-op for unmarked text (0 of 647 card assignments
  changed, at both 0.01 and the restored 0.02). Occurrences now carry `lyricId`; `getLineTiming` recovers the
  raw marker-bearing text from the lyrics file by `lyricId`, else by matching the clean line.
- **M4** Editor: "Insert pause ▾ (Ctrl+Space)" button + hotkey under the native-script field; `toEditorText` on
  load (also the list preview), `fromEditorText` on save. Live check (real dialog, mocked app): loads with ▾
  visible, Ctrl+Space and the button both insert, Save writes real markers and no stand-ins.
- **Author action needed:** Stay Gold card 21 now needs its marker (open it, put the cursor after 君を, press
  Ctrl+Space, Save), then re-run Compile Vocab. Until then it times as the unmarked 君を優しく|... split.
- Known: `core/test_vocab_store_ja.py::test_lapse_is_counted_and_log_records_prior_state` is flaky on its own
  (6 of 20 single runs fail: it asserts an FSRS interval > 7 and scheduler fuzzing sometimes lands on exactly
  7). Unrelated to Part 4; it is the one-off full-suite failure seen during this work.
- **Fix B3 (Stay Gold V, card 1122):** the markers were stored and parsed fine (the chat paste is what strips
  invisible characters - read the file, not the paste). The bug was `MIN_PAUSE_CHUNKS = 4`: it fused rows
  less than 4 chunks apart, but 2-5 chunk gaps are all equally common in the data (~750-900 each; only 46 at
  0-1), i.e. ordinary spacing between phrase rows, not slop. V's 5 rows (3 chunks apart) became 2 stretches,
  so the 2 markers had nothing to land on and 針さえ/動きを were ~35-50 chunks late. The author's rows are
  EXACT (they label singer timing as precisely as possible), so no gap is slop: now `MIN_PAUSE_CHUNKS = 0`,
  only overlapping (duplicate/nested) rows fuse. V's card places
  every phrase on its row (1133/1182/1229/1276/1367). Side effect: 156 of 647 cards' word timings shift
  (rows 2-3 chunks apart are now separate stretches) - spot-check by ear.
- **M6 Edit pauses from the flashcard (2026-10-01):** an "Edit pauses" button on the occurrence row opens a
  dialog (textarea with the `▾` stand-in, Insert pause button, Ctrl+Space, Play line, Save). It writes ONLY
  the lyric's pause markers into that song's `_lyrics.json` by `lyricId` (`core/lyric_file.py`), so it works
  for ANY song (e.g. Doughnut while Stay Gold is open in the Tk app) and the card re-times at once - no
  Compile Vocab needed, since the DB line is clean and timing reads the raw lyric at request time. Why not
  literally open the Tk Lyric Editor: it is welded to the app's one loaded song (save path from
  `app.selectedGroup`/`songName`; saving rebuilds canvas LyricBoxes), so another song would mean switching
  the whole app's song (audio, labels, video) from a different process - risking unsaved work. Wording changes
  are refused (the vocab DB keeps a clean copy; edit wording in the Lyric Editor + Compile Vocab). The writer
  keeps each file's own line endings (8 of 36 files are CRLF; all 36 round-trip byte-identical) and writes via
  temp file + replace. Live-checked in a real pywebview window against temp copies (16 checks incl. editing a
  second song). Caveat: if the Tk app has that same song open, its in-memory copy is stale until the song is
  reloaded - re-saving that same lyric from Tk would overwrite the marker.
- **Fix B4 (Doughnut, Nayeon's first card: clicking line N played line N-1):** every word click started
  playback `WORD_LEAD_MS` = 300 ms early, a cushion for rough onset estimates. With exact row starts and
  rows only ~0.3 s apart (row 1 is just 0.44 s) the lead reached back into the previous line (line 2 started
  at chunk 641.5, before row 1 ends at 642; line 4 at 827.5 before 830). Now `timeLine` emits `playFraction`
  = onset minus the lead, clamped to the word's OWN stretch start, so a row's first word plays from exactly
  that row start, and `playOccurrenceAudio(..., exact=True)` adds no further lead. Mid-row words keep the
  lead (their onset is an estimate). Live-checked in a real window (8 checks). NOT verified: pygame's MP3
  `start=` seek accuracy (an audio-capture probe was started and declined). If a click still lands in the
  previous line, that is the next suspect - fix by slicing the audio exactly (decode once, play the slice)
  instead of seeking an MP3.
- **Fix B5 (Doughnut, Nayeon's first card, after the author added 手▾を振って):** clicking 背 played from を振って.
  The card has 4 phrases (手 | を振って | 背を向けた瞬間に | すぐにさみしさにやられた) and 4 rows (11 / 105 / 68 / 68
  chunks): the second phrase is a slow held one (4 morae in 105 chunks = ~1 s per mora). The word-to-row search
  matches text share to time share, i.e. assumes one even tempo, so it crammed を振って AND 背を向けた瞬間に into the
  105-chunk row and guessed 背 at 679. Now, when a card carries a pause marker and (line breaks + markers) + 1 ==
  rows, each phrase is exactly one row (`assignWordsToStretches`). 背を -> 762, すぐに -> 835. Scoped to MARKED
  cards on purpose: tried on all cards it changed 41 and made the speed between rows absurd on a dozen (up to 87x:
  ATTITUDE#26, Aliens#1 22x) where line count == row count by coincidence. Only Doughnut#0 changed (8 marked cards
  exist). Live-checked in a real window (8 checks). The author's earlier tempo remark was right: it matters here.
- **Fix B6 (Doughnut Sana + Tzuyu: a held syllable, a pause, the rest of line 1, then lines 2+3 sung as ONE long
  row):** same root as B5 - the word-to-row search is tempo-blind, and nothing in the text says a line break is
  NOT a pause (Tzuyu: cut mid-line after 切れずに into the 101-chunk row; Sana, 5 rows: all four "Na" crammed
  into row 4, row 5 left EMPTY, 君に…Mind put in the slow をしてから row). What the program knows exactly is the ROW
  COUNT, so the contract is: mark every pause (rows - 1 markers) and the markers are the row boundaries, line breaks
  ignored (`assignWordsToStretches`, checked before the marked-card phrase rule). To make that quick, the Edit
  pauses dialog now has a LIVE PREVIEW (`previewPauseEdit`): for the text as typed, which words land in which row
  (chunk range, seconds), a "N of rows-1 pauses marked" counter ("GUESSED" until complete, then "exact"), and empty
  rows flagged red. Live-checked on Sana's real card (12 checks). Tzuyu needs 1 more marker (after ばにいなくても);
  Sana needs 3 more (after してから, after 忙しいな, after the second Na). Reading note: Sana sings こ not こい for 恋;
  harmless once 恋 sits alone in its own row (exact), it only skews words that SHARE a row.
- **B6 addendum (empty rows):** the author's screenshot of Sana's dialog (1 of 4 pauses marked) showed row 5 "(no words
  in this row)". The unmarked guess weighs several competing preferences (line breaks as cheap cuts, text length vs row
  length, mid-line cuts penalised, empty rows tolerated) and when a card breaks all of them (merged lines + a held first
  syllable) it lands on an odd compromise. One of them was plainly too weak: `_EMPTY_STRETCH` 0.03 left an empty row on
  82 of 647 cards; now 1.0 (only the 33 cards with more rows than words keep one; 55 cards re-assigned). Sana's default
  guess now has no empty row but is STILL wrong without markers (忙しいな alone in the Na row) - no default can know two
  lines share a row; that is what the markers + live preview are for.
- Tempo (author's observation): per-row speed varies ~8x within one card (Doughnut Nayeon: 88 / 420 / 340 /
  680 ms per mora), so the in-row proportional estimate is rough. It cannot affect row-start clicks (exact),
  only mid-row words; marker + row split (exact boundaries) is the current remedy. A tempo-aware in-row model
  would need ground truth to calibrate against - not built.
- Still open (M5, optional): the tap-along tool.
- **K1 + K2 (2026-10-02) DONE (flashcard only; Tk untouched).** `playOccurrenceAudio` now returns `{clipStartChunk,
  offsetMs, clipMs, playMs, chunkMs, latencyMs}`; the page runs its own clock (`Karaoke.chunkAt`, started at call return
  minus half the bridge round trip) and a `requestAnimationFrame` loop lights each word once `chunk >= startChunk`
  (`Karaoke.isLit`; stays lit to clip end; words without chunk info never light). Cleared via `stopPlayback()` on card
  change, rating, opening the pause editor, and at clip end (window close already shuts the mixer down). While a clip
  runs unsung words dim (`.playing`). `KARAOKE_LATENCY_MS` (core/util_functions.py, 0) is the single tuning knob.
  `getPlaybackPositionMs()` exists (pygame `get_pos`) but is NOT used to correct the clock - unverified how it behaves
  after `start=`. `?debug` on the page URL shows a live chunk readout. Live-checked in a real pywebview window on
  temp copies (Doughnut Nayeon + Sana): the lit set matched the clock on every sample (0 mismatches), cleared on stop.
  NOT verified: that the highlight matches what is HEARD (I did not listen) - audio seek accuracy is still the first
  suspect if it is off; then tune `KARAOKE_LATENCY_MS`, then consider decode-once exact slices.
- **Split kanji -> hiragana (author request, 2026-10-02).** A pause can now sit INSIDE a kanji word (Doughnut, Sana's 恋
  = こ | い, sung three times in that song). Saved IN THE LYRIC TEXT as an invisible reading annotation right after the
  kanji: `恋<U+2064>こ<U+2063>い<U+2061>` (editor shows `恋《こ▾い》`). Chosen over a side field because the Tk Lyric
  Editor rebuilds each entry from its LyricBox and would drop unknown keys, and over offsets because the annotation
  moves with the text. `core/lyric_text.py`: every strip (`stripAll`, `stripForDisplay`, `stripAllWithSelection`)
  removes the whole annotation, so display/DB/analysis still see plain 恋; `toEditorText`/`fromEditorText` convert the
  brackets. `core/karaoke_timing.py`: the annotated kanji run is ONE word timed from the annotation's kana; each kana
  part is a DP unit (part break = pause, so it counts as a marker), then folded back into the word, which carries
  `parts: [{reading, startChunk, endChunk}]` (word start = first part, so 恋 lights at こ's row start and stays lit over
  い; the particle after it is its own word). Mid-word highlighting by part is NOT built (data is ready). Pause editor:
  "Split kanji -> hiragana" button (select the kanji or put the cursor on it; `getKanjiReading` fills in `《こい》`,
  cursor parked after the first kana), then Ctrl+Space between the kana. Corpus had none of the new characters.
  `core/lyric_file` accepts it (wording check goes through `stripForDisplay`). Author action: re-open Doughnut's
  Sana/Tzuyu cards, split the kanji, add the pauses.

---

## Part 5 - Tap-along word timing (SCOPED 2026-10-02, nothing built)

**Why:** the in-row word times are an even-tempo guess (tempo varies ~8x within a card) and the only way to fix them
today is typing pause markers. Tapping a key as each word is sung is faster and more fun than editing numbers, and it
gives the missing ground truth. Taps become per-word ANCHORS that override the estimate; untapped words keep the
estimate between the anchors. The author's label rows stay exact and always win at a row start.

### Key design points (decided in scoping)
- **Same clock as the highlight.** A tap is recorded as `Karaoke.chunkAt(play, now)` - the page clock from K1. Because
  highlight replay uses that same clock, any constant clock-vs-audio offset (bridge latency, MP3 seek error) cancels
  out in the flashcard, so taps stay in sync even though audio seek accuracy is unverified. It does NOT cancel for the
  Tk lyric box / video export, which run on true audio time - measuring that offset is a separate later step.
- **Human lag is auto-calibrated from data we already have.** A tapper hits ~100-200 ms after the sound. The first word
  of every label row has a KNOWN exact time (the row start), so `lag = median(tap - rowStart)` over those words in the
  take, applied to every tap (fall back to a stored global `TAP_LAG_MS` when the card has < 2 rows). The take reports
  the lag it found and the spread of row-start errors, so you can see how good the take was.
- **Storage: a sidecar, not the lyrics file.** `saved_labels/<group>/<song>_taps.json`, `{lyricId: {textHash, words:
  [{i, text, startChunk}], lag, takenAt}}`. The Tk Lyric Editor rebuilds each lyric entry from its LyricBox and would
  drop unknown keys, so tap data cannot live in `_lyrics.json`; a sidecar is also untouched by Compile Vocab. `i` is the
  word index and `text` is checked on load; if the lyric's words changed (`textHash` mismatch) the card shows "taps out
  of date" and falls back to the estimate - never applies stale times. Absolute chunks, so editing the card's own
  startChunk does not invalidate them. Written via temp file + replace like `core/lyric_file.py`.
- **How taps feed timing.** `timeLine(..., anchors={wordIndex: chunk})` (pure, `core/karaoke_timing.py`): an anchored
  word's `startChunk` is the tap; words between two anchors are re-spread by mora weight over just that gap (so a
  slow held phrase fixed by two taps stops distorting its neighbours); a word at a row start snaps to the row start
  (labels win, shown as such). An annotated split kanji (恋《こ▾い》) contributes one tap per kana part.
  Pieces gain `source: "tapped" | "row" | "estimated"` so the UI can say which is which (honest-UI rule).

### The fun UI (a "Tap along" dialog next to Edit pauses)
- Big lyric line, words as chips; the NEXT word to tap pulses; each tap lights that word instantly in the singer's
  colour (reuses `--singer-color`) with a small pop, so it feels like a rhythm game. 3-2-1 count-in, then the clip plays.
- Keys: Space (or any key) = tap this word; Backspace = undo last tap; R = restart the take; Esc = cancel. Skipping a
  word you missed: Tab (it stays estimated).
- End of take: automatic replay with the karaoke highlight driven by the NEW times, the old estimate shown as a faint
  ghost marker per word so you see what changed; buttons Keep / Retake. Words show a dot: tapped vs estimated.
- Fine-tune pass (fixes a single sloppy tap without retaking): click a word, Left/Right = +/-1 chunk (40 ms), each
  nudge auditions from 300 ms before that word.
- Optional, only if tapping fast lines is too hard: slow playback (0.5x). pygame cannot change speed; would need an
  ffmpeg `atempo` slice (ffmpeg.exe is in the repo) with the clock scaled to match. Not part of the first pass.

### Milestones (each with a live check in a real pywebview window on temp copies; unit tests in core/test_*.py)
- **T1** `core/tap_store.py`: read/write the sidecar, textHash staleness, round trip + CRLF-irrelevant (own file),
  tests incl. stale text and a missing file. Pure Python.
- **T2** `timeLine(anchors=...)` + `source` per piece + lag calibration function (`calibrateLag(taps, rows)`); tests:
  anchors override, gap re-spread, row-start snap, split-kanji parts, no-row fallback, lag median.
- **T3** API: `getTaps`, `saveTaps` (lag applied server-side so the file holds corrected chunks), `getLineTiming`
  merges taps automatically (so highlight + click-to-play both use them at once). Live: save a take, restart, card
  re-times from the file.
- **T4** Tap dialog UI: count-in, record, undo/restart/skip, instant lighting. Live check by injecting synthetic key
  events at known clock times and asserting the saved chunks and the computed lag.
- **T5** Replay with ghost estimate + Keep/Retake; tapped-vs-estimated dots.
- **T6** Fine-tune nudge pass. Optional later: slow-down playback; Tk lyric box consuming taps; per-take averaging.

### Decisions (author, 2026-10-02)
1. One tap per word (a split kanji still takes one per kana part).
2. Sidecar is fine, but in a subfolder: `saved_labels/<group>/taps/<song>_taps.json` (the existing
   `saved_labels/*/*_lyrics.json` globs do not see it).
3. Backspace + restart is enough. Keys: **Space = tap**, Backspace = undo last tap, R = restart, Tab = skip a word.
4. Normal speed first; slow-down only if fast lines prove too hard.

### Build status
- **T1 DONE (2026-10-02)** `core/tap_store.py` (+ `core/test_tap_store.py`, 9 tests): `getTake` (status none/stale/ok,
  integer-indexed anchors), `saveTake` (per-lyric, empty clears, temp file + replace, unreadable file refused),
  `wordsHash`. Taps/ subfolder created on first save.
- **T2 DONE (2026-10-02)** `timeLine(..., anchors=)`, per-word `source` ("tapped"/"row"/"estimated"), result keys
  `words` (what one tap is taken for) and `rowStarts`, `calibrateLag(taps, rowStarts) -> (lag, spread)`;
  7 new tests in `TapAnchorTests`. No-anchor output unchanged (all earlier timing tests still pass).
- **T3 DONE (2026-10-02)** `VocabReviewApi.getLineTiming` applies a saved take automatically (so highlight AND
  click-to-play use it) and adds `tapStatus` (none/ok/stale), `tapLag`, resolved `lyricId`, `words`, `rowStarts`. New
  `saveTaps(group, song, startChunk, endChunk, lyricLine, singers, language, lyricId, taps)`: `taps` = {wordIndex:
  chunk} as the page clock read it; the server measures the lag (`calibrateLag`; fallback `TAP_LAG_MS` = 150 ms, an
  UNMEASURED typical value in core/util_functions.py, used when fewer than two row-start words were tapped), removes
  it from every tap, saves, and returns `{lag, spread, measured, timing}`; empty `taps` clears the take. A card
  whose occurrence has no lyricId still finds its take through the matched lyric entry. 5 new API tests; live-checked
  in a real pywebview window on temp copies (JS string keys accepted; lag 5 measured, row starts stayed exact, the
  mid-row word moved to its tap, file written under the temp `taps/`, real `saved_labels` untouched).
- **T4 DONE (2026-10-02)** "Tap along" button on the occurrence row opens a dialog: chips laid out like the lyric (a
  split kanji gets a chip per kana part), Start take -> 3-2-1 count-in -> clip plays; Space taps the pulsing next word
  (it pops into the singer's colour), Backspace undoes, Tab skips, R restarts; the take ends when every word has a tap
  or skip, or the clip runs out. Then Save take (shows the measured lag + steadiness, re-times the card behind at once),
  Retake, or Clear saved take; a stale take is explained. Pure take logic is `Karaoke.newTake` (node-tested). Live-checked
  in a real window on temp copies (Doughnut Nayeon): synthetic Space presses timed against the real clock, 200 ms late
  -> measured 202 ms, mid-row words landed on the true onsets, undo and skip worked, a screenshot shows the pulsing
  next word. NOT verified by a human tapping by ear.
- **S1-S4 DONE (2026-10-02): syllable taps + slow-down (author request, decided before T5).** Author's rules: English =
  one tap per word (drawn as a wide dashed pill so it never reads as a syllable; round chips are syllables), ー and っ
  each count as a beat of their own like ひ・と・つ, start at 1x with a suggestion.
  - S1 `core/karaoke_timing.py`: `splitMorae`, `splitUnits` (Japanese: one unit per beat of the word's kana reading,
    labelled hiragana/katakana; the particle is as pronounced, を = お; Korean: one per Hangul block, an English run =
    one; anything that does not add up to the word's weight stays ONE unit). Row assignment still runs per word;
    syllables are split out afterwards. Known cosmetic: ー shows as the vowel kana the reading expands it to (ら あ め ん).
  - S2 `timeLine(..., unit="word"|"syllable")`: anchors are per unit; a word starts at its first syllable; word pieces
    get `syllables`; result adds `unit` and `pace` {fastest taps/s averaged per row, suggestedRate}. Takes are stored
    per unit (word under the lyricId as before, syllable under `<lyricId>#syllable`); `getLineTiming(unit=None)` applies
    the best saved take (syllable, else word), `unit="word"|"syllable"` asks for one view. `saveTaps(..., unit, rate)`.
  - S3 slow-down: `makeSlowClip` (ffmpeg `atempo`, chained below 0.5x, cut from the same cached playback MP3 the 1x path
    uses, cached as `cache_audio/slow_*.wav`), `playOccurrenceAudio(..., rate)` plays it from its own start (no MP3 seek),
    returns `rate`; `Karaoke.chunkAt` advances song time at `rate`; `prepareSlowClip` builds it before the count-in. The
    default tap lag is REAL time (150 ms), so on a slowed clip it is 150 ms * rate of song time; a measured lag needs no
    scaling (taps and row starts are both song time).
  - S4 dialog: syllable/word toggle, speed 1/0.75/0.5/0.35x, pace hint + suggestion (never auto-selected), legend,
    "n tapped of N", kanji small above its kana, buttons pinned under a scrolling chip area.
  - Live-checked (real window, temp copies, Doughnut Nayeon, 0.5x): 27 syllable chips, real ffmpeg clip (playMs
    21760 = 10880/0.5), 27 synthetic taps 200 ms late in real time -> measured 203 ms, saved at 2.54 song-time chunks,
    auto mode then applies the syllable take. Pace for that line: ~4.4 taps/s at 1x (suggested 0.75x). NOT verified by
    a human tapping by ear.
- **ー / っ / ん handling (2026-10-02, author: っ is sometimes sung "futte" as one word, sometimes "fu-ute"; ん likewise
  "dan" vs "da-n" - ん joined the same switch right after).** They stay beats of
  their own (one chip each, drawn smaller, flagged `hold`; a real ー is now kept as ー via the new public
  `japanese_utils.rawTokenReading` instead of the vowel it expands to). A tap on one means "the beat starts here" - a
  silent stop or a held "-uu-". Dialog checkbox "tap ー っ ん" (default on): off = the take steps over them
  automatically (they keep their estimate; Backspace also steps back over beats skipped for you), for the quick
  "futte" reading. Tab still skips a single one by hand. Live-checked in the real dialog (Nayeon line: 27 taps on,
  26 off, Backspace returns to the state before the tap).
- **T5 DONE (2026-10-02)** After a take the dialog asks the server what the take WOULD do (`previewTaps`: the card timed
  with the taps, lag removed, plus the untouched estimate; nothing is written), replays the clip at the tapped speed
  with the chips lighting at the NEW times (`replay-lit`, driven by a new `karaokeOnFrame` hook on the shared clock),
  and marks every tapped chip with how far it moved from the estimate in ms (+ later / - earlier; red at >= 200 ms).
  "Replay take" repeats it; "Keep & save take" saves (Retake discards; nothing is saved before Keep). `Karaoke.unitStarts`
  (node-tested) is the one place that walks pieces in chip order. Live-checked (real window, temp copies): a stretched
  "fu-ute" showed +400/+800 ms on the two shifted beats, nothing saved before Keep, saved after, lit chips cleared at the
  end; 145 of 148 replay samples matched the expected lit set exactly (the 3 off were sampling between a clock tick
  and the next animation frame). NOT verified: how the replay sounds/looks to a human; a screenshot attempt only caught
  the author's own app window.
- **Fix F1 (TWICE Funny Valentine, Momo's first card: a well-tapped line came out seriously misaligned, found from the
  author's screenshot + saved take).** Cause was NOT the 曖昧 line: the label rows say "Moon night あなたが" | "くれた" |
  "曖昧なこの感情は" but the author's ▾ markers sit after "night" and after あなた, and 2 markers + 1 == 3 rows made the
  markers the row boundaries (B6 rule), so (a) correct taps were clamped into the wrong rows and (b) `calibrateLag`, which
  assumed the text's row-start units, compared taps to the wrong row starts and measured a 26-chunk (1 s) "lag" that was
  then subtracted from every tap. Now: taps decide the row (a unit tapped inside another row than the text chose moves
  there; untapped units stay between the rows of the taps around them; `reassigned` counts the moves and the dialog says
  "your taps put N beats in a different label row than your ▾ markers say - the taps win"), and `calibrateLag` trusts the
  by-unit gaps only while all are plausible (-3..30 chunks), else matches each row start to the first tap at least
  `MIN_REACTION_MS` (120 ms, scaled by playback speed) after it - that floor is needed because rows 4 chunks apart plus a
  5 chunk lag make the previous row's last tap land after the next row start. Regression tests use the real rows and lyric.
  Re-checked through the real API on temp copies: 21 units, lag 6 measured (true 5), every unit within ~1 chunk of the
  true onset. Existing saved takes made before this fix keep the bad lag baked in (only corrected chunks are stored) -
  clear and retake them. The unmarked estimate for this card is still wrong while the markers disagree with the rows
  (Edit pauses preview shows it); taps now override it.
- S5 (syllable-by-syllable highlight in the main flashcard line, kana shown above kanji) still unbuilt: the flashcard
  lights whole words; the tap dialog's replay is where syllables are visible today.
- **Row check + exact audio start (2026-10-02, after the author reported Funny Valentine "a second behind").** Measured, not
  guessed: pygame's MP3 `start=` seek lands 76-99 ms EARLY at five positions in that song (recorded via SDL's disk audio
  driver and cross-correlated against the decoded song), and the bridge round trip is 6-32 ms, so neither explains a second.
  `playOccurrenceAudio` now returns `startedAtMs` (epoch ms when play() ran) and the page clock starts exactly there
  instead of "half the round trip ago". Takes now also report a **row check** (`rowCheck`: the first lag-corrected tap near
  each label row start vs that start, in ms) so "does the take agree with my labels" is a number. The red +/- ms under
  chips are vs the ESTIMATE (badly off on a card whose markers disagree with its rows) - the dialog now says so. The
  author's last report ("Moon night counted as one word / a second behind") could NOT be reproduced: they are two words, two
  units and two taps in every layer (regression test `test_two_english_words...`). UNRESOLVED: whether anything is still
  off in the replay or the main line after Keep; needs the author's direction (early/late), where (dialog replay or main
  line) and the new "Row check" line.
- S5 (syllable-by-syllable highlight in the main flashcard line, kana shown above kanji) still unbuilt: the flashcard
  lights whole words; the tap dialog's replay is where syllables are visible today.

### Session 2026-10-02 (c): countdown popup + clicks, ad-lib gap fix, T6 nudge pass - all DONE
- **Count-in popup + metronome.** `#tapCount` is now `position: fixed`, centred in the window (scroll-independent, no pointer
  events, so Space still taps), 6em on a dark pill, pops on each beat. 4 clicks (3 x 880 Hz, then 1320 Hz on GO) are scheduled on the
  WebAudio clock in one go, 650 ms apart (`COUNT_BEAT_MS`); the clip starts on the 4th. Checkbox "count-in clicks" (default on).
  Interpretation: the metronome is the COUNT-IN only - no click track during the take (it would fight the song); say so if a
  steady click while tapping was meant.
- **Funny Valentine Sana card cut off by Mina's "tick tock".** Cause (measured, not guessed): card 2 (Sana, startChunk 518) resolved to
  529-671. Her last phrase is the row 698-754, 27 chunks after her 640-671 row, and `_MAX_MERGE_GAP_CHUNKS` = 25 stopped the merge.
  Mina's tick-tock sits in that gap (ad-lib CARD 655, `adLibDuration` 50 -> 655-705, though her row 666-698 is NOT flagged ad-lib).
  Fix: `label_runs._gapIsInterrupted` - a gap over the limit is still bridged when ONE ad-lib (an ad-lib label row, or an ad-lib
  card with a stated duration) covers the whole silence. NOT "any other singer's row": that version also fired on trading lines
  (SWIM, Flu, Do Not Touch) and gang vocals (Blue Valentine) and was rejected. Blast radius over all 1184 spans: 7 change
  (Funny Valentine #2 529-671 -> 529-754, Like Animals #13, Merry Go Round #0/#11, SWIM #30, Flu #30, Do Not Touch #16), all
  with an ad-lib marked in the gap; the author should spot-check the other 6. Needs "Compile Vocab for all songs" to refresh
  the stored span. The card's pause markers (2 + 1 line break) now meet 5 rows, so it still wants a marker per row boundary.
  Tests: `AdLibInterruptionTests` (3).
- **T6 fine-tune pass.** `core/karaoke_timing.nudgeAnchor(timing, anchors, index, delta)` (pure) + `timeLine` result key `slots`
  (per unit: label, start/end, source, row, hold). Rules: first beat of a label row = "locked"; a beat stays inside its own row and
  never passes the nearest tapped/row-start beat on either side (estimated beats between re-spread, so they never block); a
  nudged estimated beat becomes "tapped"; delta None = reset to the estimate. API: `nudgeTake` (stateless: page sends the
  current corrected anchors, nothing saved), `saveCorrectedTake` (chunks written AS GIVEN - no second lag subtraction; keeps
  or stores raw taps/rate/lag), `previewTaps` now returns `anchors`, `getLineTiming` returns `anchors`/`raw`/`tapRate` for an
  applied take. `tap_store` stores `raw` + `rate` beside the corrected `words` (old takes without them still load); `saveTaps`
  now stores raw + rate too. UI: after a take, and on any saved take of the viewed unit, click a chip to select it; Left/Right
  = 40 ms, Shift = 200 ms, Backspace = reset, Space = hear again, Esc = deselect (then close); every nudge re-times the card and,
  once the keys settle (300 ms), auditions from 300 ms (real time) before the beat at the chosen speed with the chips lighting at
  the new times. "Keep & save take" (fresh take) / "Save nudges" (saved take). Also fixed: Left/Right in ANY open dialog used to
  change the card behind it (global handler).
  Tests: 9 `NudgeTests`, 2 tap_store, 5 API; node tests unchanged. Live check (real pywebview window, temp copies, Doughnut
  Nayeon card, 27 syllable beats, synthetic Space presses 5 chunks late against the real clock): 38 of 39 checks passed - the
  count-in showed 3/2/1/GO, popup centred and fixed with the dialog scrolled to the bottom, 4 clicks 650 ms apart; lag measured
  5.06; +1 / +5 / -1 / -5 chunk nudges exact; 60 big nudges stopped exactly on the neighbouring tapped beat with the reason shown;
  Backspace reset; row-start beat locked; arrows did not change the card; the audition request fraction matched the formula
  (0.14596 vs 0.1460, exact=true); saved chunks == the nudged anchors (lag not subtracted twice), raw taps + rate + lag stored;
  the saved take reopened editable and "Save nudges" kept the raw taps; the author's real taps files untouched. The 1 "fail"
  was my test expecting a single audition (selecting a beat also auditions it, so 2). NOT verified: how the audition SOUNDS (I did
  not listen), audition at a slowed speed (builds an ffmpeg slice per audition - each is a new slice, so expect a short wait),
  a human nudging by ear.
- Later options: drag a chip on a timeline strip; per-take averaging; Tk lyric box consuming takes; S5 (syllable highlight in the main line).

---

## Shared code

Put proportional-weight/token-timing/padded-span math in one `core/` module (e.g. `core/karaoke_timing.py`)
used by both surfaces and by flashcard A5, unit-tested once against fixture values.

## Suggested order

A1 (done) -> A2 (cap fix) -> A4 -> A5 -> K1 -> K2 -> A3/A2b (verify, polish) -> T1..T4 -> optional K3/T5.

## Open questions

- ANSWERED (2026-09-30): audio tails cut off is the worst bug (BTS Suga/J-Hope JA rap lines, Doughnut ballad lines).
- Any way to get real word timing (forced alignment) for some songs? Would make K2/A5 exact, not estimated.
- Should ad-lib boxes get Tk karaoke?
- Verify-then-build: each milestone needs a live test hook (throwaway driver, deleted after use, per project
  convention) before moving on.

### Held vowels + hold defaults (2026-10-02, author: Momo sings 感情 as "gan jyou", 重要性 as "juu you sei")
- Dialog switches: "tap っ ん" now OFF by default (it used to be "tap ー っ ん", on); new "held vowels count as one" ON by default
  covers ー, and the う of じょう/じゅう, い of せい (`karaoke_timing.isHeldVowel`; units carry `held`, slots too). A skipped beat keeps
  its estimate. The reading already spells long vowels as ー (fugashi), so in practice ー is what gets folded; the う/い rule covers
  readings that keep them. Verified on the real Funny Valentine lyric+labels (Momo card 0, real window, temp copy): with the new
  defaults 感情は is か・じょ・わ (3 taps, was 5); tapping ん back on gives か・ん・じょ・わ; both off gives the old 5. Card 2 (重要性)
  was only run after the switches had been flipped by the card-0 run (page state carried over), so its default sequence was NOT
  separately confirmed; unit labels for it come out じゅ ー よ ー せ ー, i.e. じゅ・よ・せ with ー folded. 3 new tests (HeldVowelTests).
