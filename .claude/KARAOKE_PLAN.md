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
- Still open (M5, optional): the tap-along tool.

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
