# Prompt: karaoke tap-along, next steps (handoff written 2026-10-02)

Paste everything below the line into a fresh session. Verify any file/function name against the current code first.

---

You are continuing the karaoke / tap-along work in `k:\Python Projects\Voice Recognition` (Python 3.13 venv at `venv/`,
PyWebView flashcard window in `gui/web/vocab_review/`). Read `.claude/KARAOKE_PLAN.md` first: Parts 1-4 plus the
**Part 5 build log** (T1-T5, S1-S4, fix F1, row check, the T6 spec at the end). Then the memory notes
`feedback_label_data_is_precise.md`, `feedback_verify_then_build.md` and `project_karaoke_tap_along.md`.

## What exists (done, tested, live-checked in a real window)
- Word highlighting on the flashcard line (K1/K2): page clock from `playOccurrenceAudio`'s return (`clipStartChunk`,
  `offsetMs`, `playMs`, `rate`, `startedAtMs`), `Karaoke.chunkAt`, rAF loop, singer colour from groups.json.
- Split kanji (`恋《こ▾い》`, invisible U+2064/U+2061 in the file) and pause markers (U+2063) via `core/lyric_text.py`.
- Tap-along dialog ("Tap along" button): one tap per syllable (every beat; ー っ ん are their own beats, a switch skips
  them) or per word; English word = one tap (dashed pill); Space taps, Backspace undoes, Tab skips, R restarts; speeds
  1/0.75/0.5/0.35x via ffmpeg slice; lag measured per take (`calibrateLag`), takes saved per unit in
  `saved_labels/<group>/taps/<song>_taps.json`; after a take: preview, auto-replay with chips lighting at the new times,
  red +/- ms vs the estimate, a **row check** line, "Keep & save take".
- Taps decide which label row a beat belongs to (they override a pause marker that disagrees with the rows, fix F1).
- Also the countdown before tapping should be a popup instead of at the bottom of the lyrics because longer verses can push the timer out of view. Also add a metronome beat for audio feedback.
- Slight bug regarding a line getting cut off as I was too lazy to label (Tonight▾　二人の▾時間の重要性
分かってないのね) gets cut off by "tick tock tick tock" ad-lib by Mina/Momo. Analyze the lyrics for that 

## Open items, in this order
2. **T6 nudge pass** - spec at the end of Part 5 (click a beat, Left/Right = 40 ms, audition by ear, save corrected chunks
   without re-subtracting lag, also store raw taps so takes can be re-derived).
3. S5 (optional): light the flashcard line syllable by syllable (kana above kanji words).
4. Later: Tk lyric box consuming takes (`timeLine` is shared); tap-along for other units.

## Working rules (this project)
- Small testable milestones, each with a live check in a REAL pywebview window against TEMP COPIES of saved_labels (never
  write the author's real files; their real `saved_labels/*/taps/*.json` are THEIR takes - never touch them). Delete
  throwaway drivers afterwards. Junctions into `training_data` / `group_icons` / `cache_audio` are fine for reading.
- Tests: `venv/Scripts/python.exe -m unittest discover -s core -p "test_*.py"` and
  `node gui/web/vocab_review/karaoke.test.js`. `core/test_vocab_store_ja.py::test_lapse_is_counted_and_log_records_prior_state`
  is a known flaky FSRS test; ignore it.
- The pause marker is U+2063 and the reading brackets U+2064/U+2061: never write those escapes or raw characters into a source
  file; use the constants in `core/lyric_text.py` or `chr()`. In shell heredocs a `\n` inside a Python string becomes a real
  newline and breaks string literals: write scripts with the Write tool, or fix with Edit. Set `PYTHONIOENCODING=utf-8`
  when printing Japanese from the shell.
- Do not take full-screen screenshots (`ImageGrab.grab()` captures the author's other windows). Prove UI behaviour from the
  DOM (class/attribute state, measured rects) instead, and say what was NOT verified (e.g. "I did not listen").
- Measure before explaining: when the author reports an offset, measure it (SDL `disk` audio driver + cross-correlation worked
  for audio position) before theorising. Report what was verified and what was not.
- Update `.claude/KARAOKE_PLAN.md` and the memory notes as you go.
