# Consolidate playback position/pause/seek into one PlaybackClock

## Status (2026-09-23) — read this first before continuing

**Code migration is DONE; the manual UI checklist below has NOT been run yet.**

Run first:
```
python -m unittest core.test_playback_clock gui.test_playback_seek_sync -v
```
Expect 26/26 green (20 clock tests, 6 app integration tests).

**Done:**
- `PlaybackClock` wired into `VoiceDetectionApp` (`self.clock`). `playbackOffset`/`isPaused`/
  `isPlaying` are now read-only properties over the clock, except `isPaused` keeps a setter (see
  "Remaining" below).
- New clock methods added during migration (all tested): `start(ms)` (unconditional play, raises
  pygame.error; used by `playWithSavedResults`), `hasEnded()` (replaces `updateChunk`'s
  `get_pos() == -1` end-of-song check), `suspendForScrub()` + `isScrubbing` (silent live-drag state:
  snapshots position, `currentMs()` returns the preview target, cleared by the release `seekTo()`).
- Migrated: `getPlaybackTimeMs`, `updateProgressBar`, `updateChunk`, `playWithSavedResults` start
  branch, `toggleAudioMode`, `switchAudioPathPreserveTime`, `_restartAudioAtTime`, `jumpToMs`,
  arrow nudges (now one `_nudgeByMs` helper), `onDragHandle`, `updateCurrentTime`, `onMarkerDrag` +
  `onMarkerRelease`, `pause`, `play`, `onClose`; `ZoomManager.updateZoomLevel`;
  `VideoTrackItem._sampleAudioClock` and `_resetExportUiToChunk`; `shutdownVoiceApp`.
- Bugs #1–#5 all addressed. Extra bugs found and fixed along the way:
  - Arrow nudges computed ±5s from `playbackOffset` (the last seek point, which doesn't advance while
    playing), so a nudge while playing jumped relative to a stale position. Now uses `currentMs()`.
  - Toggling vocals/mix (or lead/back) while **paused** didn't reload the file, so resuming played
    the old source. `switchSource` now loads while paused too.
  - Marker drag: audio now goes silent during the drag and restarts at the dropped spot on release
    (previously audio kept playing while the UI loop froze until some other action restarted it).
- No `pygame.mixer.music.play/stop/pause/get_pos/load` calls remain outside `core/playback_clock.py`
  (only `set_volume` and `onClose`'s `unload()`/`quit()` teardown, which are out of scope).

**Remaining:**
- `gui/VideoTrack.py:873` and `:1245` (video export) still write `self.parent.isPaused = False`
  through the shim setter. Not in the original audit. Leaving it in place since export's interaction
  with live playback wasn't in scope; migrate it once export behavior is understood, then delete the
  setter.
- **Next concrete step:** the user runs the manual checklist below in the real app (Claude can't see
  the Tkinter UI). Rows 6 (nudges), 9/10 (source toggles — also try them while paused), 11 (clip-skip
  while paused), and 12 (marker drag while playing: audio should go silent during the drag and resume
  at the drop point) exercise the changed behavior most directly.

**To resume in a new chat:** paste this file's path and ask Claude to read it in full, then continue
from "Next concrete step" above. No separate handoff-prompt file was made for this — this file's
Status section IS the handoff.

---

## Context

Across several rounds of debugging, every sync bug (video lag, paused-seek being dropped, pause/resume
snapping back, progress bar jumping ahead) traced back to the same root cause: **one undocumented
invariant — `playbackOffset + pygame.mixer.music.get_pos() = current position, but only between one
real `play(start=X)` call and the next`** — reimplemented independently, slightly differently, in many
places. Fixing each symptom as reported only touched the one call site in front of me, so it kept
resurfacing elsewhere.

A full audit (Explore agent, cross-checked directly) found this is bigger than the 3-4 spots already
patched. Confirmed duplicate/inconsistent implementations, all keyed off the same
`self.playbackOffset`/`self.isPaused`/`self.isPlaying` state on `VoiceDetectionApp` (gui/audio_tester.py):

**Position-reading duplicates** (each hand-rolls `playbackOffset + get_pos()`, or a chunk-quantized
fallback, independently):
`getPlaybackTimeMs()`, `updateProgressBar()`, `updateChunk()` (inside `playWithSavedResults`),
`toggleAudioMode()`, `ZoomManager.updateZoomLevel()` (gui/zoom_functions.py),
`VideoTrackItem._sampleAudioClock()` (gui/VideoTrack.py).

**Seek/restart duplicates** (each hand-rolls "stop, play(start=X), update playbackOffset", with
different guards):
`_restartAudioAtTime()`, `jumpToMs()`, `moveBackwardByChunks()`/`moveForwardByChunks()`,
`onDragHandle()`, `updateCurrentTime()`, `onMarkerDrag()`, `switchAudioPathPreserveTime()`.

**Pause/resume:** `pause()`, `play()`.

This also turned up real, currently-live bugs distinct from the invariant issue, caused by the same
copy-paste-and-diverge pattern:

1. **`updateProgressBar()` takes zero arguments, but `moveBackwardByChunks`/`moveForwardByChunks` call
   it with one** (`self.updateProgressBar(newPlaybackTime)`) — confirmed by reading the signature
   directly. This raises `TypeError` every time an arrow-key nudge fires; Tk swallows callback
   exceptions silently, so the rest of that handler (handle redraw, canvas update, and the actual audio
   restart) never runs. The arrow-key nudge feature is effectively broken right now.
2. `jumpToMs()` (used by clip-skip) never checks `isPaused` before calling `pygame.mixer.music.play()`
   — a clip-skip while paused would start audio playing unexpectedly.
3. `moveBackwardByChunks`/`moveForwardByChunks` never call `videoTrackItem.seek()` — the video preview
   doesn't follow arrow-key nudges.
4. `onMarkerDrag()` (dragging a label boundary) seeks the video only — audio/`playbackOffset` never
   move.
5. `play()`'s own resume branch is internally inconsistent: it restarts real pygame audio at the exact
   `playbackOffset`, but three lines later kicks off the UI tick loop from the coarser
   `currentChunkIndex * chunk_duration` instead.

Goal: one object owns this state and is the only thing allowed to touch `pygame.mixer.music`, with
everything else reading from it — so this class of bug becomes structurally impossible to reintroduce,
and it's commented well enough to read without re-deriving the invariant from scratch.

## Design: `PlaybackClock` (built — see core/playback_clock.py)

Lives in `core/playback_clock.py` (paired with `core/test_playback_clock.py`, matching this repo's
existing `core/test_*.py` convention — this class needs nothing Tkinter-specific, only
`pygame.mixer.music`, so it belongs with the other testable core logic, not in `gui/`).

Public surface, as built:

- `currentMs() -> int` — the one correct "what time is it right now" implementation (playing:
  `offset + get_pos()`; paused: `offset` directly, no `get_pos()` involved).
- `load(path)` — load a file once; separate from seeking, since pygame keeps a stopped file's buffer
  loaded and reloading on every restart would be wasted work.
- `seekTo(ms, resync=True)` — replaces every hand-rolled restart. Always updates the tracked offset;
  only touches real pygame audio if not paused AND `resync` is True. Resync is gated on "not paused",
  not "is playing" — deliberately preserves the original app's behavior where the very first scrub
  before Play has ever been pressed still starts real audio.
- `pause()` — snapshots true position into the offset, then pauses pygame.
- `resume()` — restarts pygame explicitly at the tracked offset (never blind `unpause()` — a seek
  during pause must be honored, which `unpause()` alone would silently ignore).
- `switchSource(path, keepMs=None)` — for the vocals/lead-back submix toggles, "same position,
  different file."
- `stop()` — full teardown (`pygame.mixer.music.stop()`), idempotent.
- Read-only properties: `isPaused`, `isPlaying`, `offsetMs`, `audioPath`.

## Integration strategy: delegate, don't rewrite every call site (NOT DONE YET)

`VoiceDetectionApp.__init__` should create `self.clock = PlaybackClock()`. `self.playbackOffset`,
`self.isPaused`, `self.isPlaying` should become `@property` pass-throughs to `self.clock` instead of
plain attributes.

This matters because the audit found ~10 more *read-only* consumers of this state beyond the
duplicated-formula sites (e.g. `TrackItem.updateAndDrawTimer` reading `self.parent.isPaused`,
`VideoTrackItem._sampleAudioClock` reading `self.parent.playbackOffset`). Making them properties means
those keep working with **zero changes** — only the functions that duplicate the formula or hand-roll
a restart need to be touched. This is the difference between a large mechanical migration and a full
rewrite.

## Migration (function → clock method) — NOT DONE YET

- **Position readers**, replace inline formula with `self.clock.currentMs()`: `getPlaybackTimeMs`,
  `updateProgressBar`, `updateChunk`, `toggleAudioMode`, `ZoomManager.updateZoomLevel`
  (gui/zoom_functions.py), `VideoTrackItem._sampleAudioClock` (gui/VideoTrack.py, via
  `self.parent.clock.currentMs()`).
- **Seek/restart writers**, replace with `self.clock.seekTo(ms, resync=...)`: `_restartAudioAtTime`,
  `jumpToMs` (fixes bug #2), `moveBackwardByChunks`/`moveForwardByChunks` (fixes bug #1 by no longer
  calling the broken `updateProgressBar(arg)`, and fixes bug #3 by adding the `videoTrackItem.seek()`
  call it was missing), `onDragHandle`, `updateCurrentTime`, `onMarkerDrag` (fixes bug #4 by also
  moving audio, not just video).
- **Pause/resume**: `pause()` → `self.clock.pause()`; `play()`'s resume branch → `self.clock.resume()`
  (fixes bug #5 since there's only one position value left to pass to the UI loop).
- **Source switching**: `switchAudioPathPreserveTime`, `toggleAudioMode` → `self.clock.switchSource(...)`.

## Explicitly out of scope

- Merging `VoiceDetectionApp.onClose` and gui/voice_recognition_gui.py's `shutdownVoiceApp` —
  different scopes (feature teardown vs. whole-app exit). Both should call `self.clock.stop()` for the
  mixer part, but stay separate functions.
- `CutClipManager.maybeSkip` (confirmed unused/dead — only `maybeSkipNext` is called) — not touched.
- Any zoom/layout/visual code beyond the position-formula duplication in `updateZoomLevel`.

## Verification

### Automated

1. `core/test_playback_clock.py` — done, see Status above.
2. `gui/test_playback_seek_sync.py` — kept as a thinner integration check that `VoiceDetectionApp`'s
   migrated methods correctly delegate to `self.clock` (re-run after migration, updated only where
   method internals changed enough to need it). Not yet touched.

### Manual test checklist (required once migration is done — Claude cannot see the Tkinter UI)

Run through this exact list after implementing, before calling it done. For each row: do the action,
check the expected result.

| # | Steps | Expected result |
|---|-------|------------------|
| 1 | Press Play, let it run a few seconds | Audio, video, and progress bar handle all move together, no lag |
| 2 | While playing, drag the progress bar handle to a new spot and release | Audio and video both jump to the new spot immediately, handle doesn't drift ahead or lag |
| 3 | Press Pause, then drag the progress bar handle, then press Play | Resumes from the scrubbed-to spot, not the pre-scrub spot |
| 4 | Press Pause (no scrub), then Play, let it play a few seconds, Pause again, then Play again | Second resume continues from where it was actually paused, does **not** snap back to an earlier position |
| 5 | Press Pause, wait ~1 second doing nothing, watch the progress bar handle | Handle does not jump or drift while paused |
| 6 | Press the arrow-key nudge (forward and backward) a few times, both while playing and while paused | Handle, canvas, **and video** all visibly move by the nudge amount; audio audibly jumps too (previously silently broken by the `updateProgressBar` argument-count crash) |
| 7 | While playing, drag the zoom slider | Section index doesn't jump to the wrong page; timeline scroll matches where audio actually is |
| 8 | While **paused**, drag the zoom slider | No crash, section index reflects the paused position, not a stale/advancing one |
| 9 | Press 'V' to toggle vocals-only / full-mix mode while playing | Audio switches source but keeps playing from the same position, no restart-from-0 or silent gap |
| 10 | Toggle lead/backing vocals (if applicable to current song) while playing | Same as above — position preserved across the source switch |
| 11 | With clip-cut mode enabled and a Cut label ahead, let playback reach it | Auto-skips past the cut smoothly; if currently paused when a skip would occur, it should not start audio playing unexpectedly (bug #2 fix) |
| 12 | Drag a label's start/end boundary marker on the canvas | Video preview follows the drag. While playing: audio goes silent during the drag and resumes from the drop point on release. While paused: pressing Play afterwards resumes from the drop point |
| 13 | Close the window (X button) while a song is playing | Audio stops cleanly, no orphaned pygame process/error on next launch |

## Files touched

- New, done: `core/playback_clock.py`, `core/test_playback_clock.py`
- To edit: `gui/audio_tester.py` (wire in `self.clock`, migrate the ~12 functions above),
  `gui/VideoTrack.py` (`_sampleAudioClock`), `gui/zoom_functions.py` (`updateZoomLevel`),
  `gui/voice_recognition_gui.py` (`shutdownVoiceApp` calls `self.clock.stop()`),
  `gui/test_playback_seek_sync.py` (updated as needed)
- This file (`.claude/PLAYBACK_CLOCK_PLAN.md`) — keep the Status section at the top updated as work
  continues, same convention as this project's other `.claude/*_PLAN.md` docs.
