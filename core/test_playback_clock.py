"""
Tests for core/playback_clock.py's PlaybackClock - the single source of truth
for playback position/pause/seek that replaced ~13 independent hand-rolled
copies of the same "offsetMs + pygame.mixer.music.get_pos()" formula across
gui/audio_tester.py, gui/VideoTrack.py, and gui/zoom_functions.py. See
.claude/PLAYBACK_CLOCK_PLAN.md for the debugging story that led here.

Uses a fake pygame.mixer.music driven by an explicit fake clock instead of
real wall-clock time, so tests are deterministic and need no real audio
device. The fake's pause behavior (get_pos() freezes at whatever it read when
pause() was called, does NOT keep advancing, does NOT reset to 0) was
verified empirically against the real pygame.mixer.music before being encoded
here - not assumed:

    pygame.mixer.init()
    pygame.mixer.music.load(...); pygame.mixer.music.play(start=0.0)
    time.sleep(0.3); pygame.mixer.music.get_pos()   -> 309
    pygame.mixer.music.pause(); pygame.mixer.music.get_pos()  -> 301
    time.sleep(0.5); pygame.mixer.music.get_pos()   -> 301 (unchanged)
    pygame.mixer.music.unpause(); time.sleep(0.2); get_pos()  -> 508

Stdlib unittest only, matching core/test_kanji_reference.py's style. Run with:
    python -m unittest core.test_playback_clock -v
"""

import unittest
from unittest import mock

from core.playback_clock import PlaybackClock


class FakePygameMixerMusic:
    """See module docstring - mirrors real pygame.mixer.music's verified
    pause/get_pos behavior against an explicit fake clock instead of real
    wall-clock time."""

    def __init__(self):
        self.clock = 0
        self.playing = False
        self.paused = False
        self.startMs = None
        self.loadedPath = None
        self._playedAtClock = 0
        self._pausedAtClock = None

    def load(self, path):
        self.loadedPath = path

    def stop(self):
        self.playing = False
        self.paused = False
        self._pausedAtClock = None

    def play(self, start=0.0):
        self.startMs = int(round(start * 1000))
        self._playedAtClock = self.clock
        self._pausedAtClock = None
        self.playing = True
        self.paused = False

    def pause(self):
        self.paused = True
        self._pausedAtClock = self.clock

    def unpause(self):
        self.paused = False
        self._pausedAtClock = None

    def get_pos(self):
        if not self.playing:
            return -1
        if self.paused and self._pausedAtClock is not None:
            return self._pausedAtClock - self._playedAtClock
        return self.clock - self._playedAtClock


class PlaybackClockTests(unittest.TestCase):
    def setUp(self):
        self.music = FakePygameMixerMusic()
        patcher = mock.patch("core.playback_clock.pygame.mixer.music", self.music)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.clock = PlaybackClock()
        self.clock.load("song.mp3")

    # -- the core invariant ---------------------------------------------

    def test_current_ms_while_playing_adds_get_pos(self):
        self.clock.seekTo(1000)
        self.music.clock += 500
        self.assertEqual(self.clock.currentMs(), 1500)

    def test_current_ms_while_paused_is_exact_no_double_counting(self):
        """The bug this whole class exists to prevent: while paused,
        currentMs() must be offsetMs ALONE, never offsetMs + get_pos() again
        - get_pos() is frozen but non-zero, so adding it a second time on
        top of an offset that already accounts for it would overshoot."""
        self.clock.seekTo(1000)
        self.music.clock += 500  # real position 1500ms
        self.clock.pause()
        self.assertEqual(self.clock.currentMs(), 1500)

        self.music.clock += 2000  # time passing while paused must not matter
        self.assertEqual(self.clock.currentMs(), 1500)

    def test_seek_while_paused_is_honored_on_resume(self):
        self.clock.seekTo(1000)
        self.music.clock += 5000
        self.clock.pause()

        self.clock.seekTo(400)  # scrub while paused
        self.assertEqual(self.clock.offsetMs, 400)
        self.assertTrue(self.music.paused, "must stay silent while still paused")

        self.clock.resume()
        self.assertFalse(self.clock.isPaused)
        self.assertEqual(self.music.startMs, 400, "must honor the seek made while paused")

    def test_pause_then_resume_with_no_further_seek_stays_put(self):
        self.clock.seekTo(400)
        self.music.clock += 2000  # played forward 2s -> really at 2400ms
        self.clock.pause()

        self.clock.resume()

        self.assertEqual(self.music.startMs, 2400, "must resume where it was, not snap back to 400")

    def test_multiple_pause_resume_cycles_never_drift_to_a_stale_seek(self):
        self.clock.seekTo(1000)
        self.music.clock += 3000
        self.clock.pause()
        self.clock.seekTo(400)
        self.clock.resume()
        self.assertEqual(self.music.startMs, 400)

        self.music.clock += 1200
        self.clock.pause()
        self.clock.resume()
        self.assertEqual(self.music.startMs, 1600, "must resume at 400 + 1200 played")

    # -- resync flag (live drag preview vs authoritative jump) ------------

    def test_seek_with_resync_false_updates_offset_but_not_real_audio(self):
        self.clock.seekTo(1000)
        priorStartMs = self.music.startMs

        self.clock.seekTo(4000, resync=False)

        self.assertEqual(self.clock.offsetMs, 4000, "offset must track the preview target")
        self.assertEqual(self.music.startMs, priorStartMs, "real audio must not restart mid-drag")

    def test_seek_with_resync_false_while_paused_stays_silent_and_is_honored_later(self):
        self.clock.seekTo(1000)
        self.clock.pause()

        self.clock.seekTo(700, resync=False)  # equivalent to resync=True here since paused
        self.assertEqual(self.clock.offsetMs, 700)
        self.assertTrue(self.music.paused)

        self.clock.resume()
        self.assertEqual(self.music.startMs, 700)

    def test_seek_before_play_was_ever_pressed_still_starts_real_audio(self):
        """Matches the original app's deliberate behavior: scrubbing a freshly
        opened song (isPlaying/isPaused both still False/default) should
        start real audio immediately, not silently do nothing until Play is
        pressed separately."""
        self.assertFalse(self.clock.isPlaying)
        self.clock.seekTo(2500)
        self.assertTrue(self.clock.isPlaying)
        self.assertEqual(self.music.startMs, 2500)

    # -- source switching ---------------------------------------------------

    def test_switch_source_preserves_position_while_playing(self):
        self.clock.seekTo(1000)
        self.music.clock += 800  # real position 1800ms

        self.clock.switchSource("vocals_only.mp3")

        self.assertEqual(self.music.loadedPath, "vocals_only.mp3")
        self.assertEqual(self.music.startMs, 1800, "must resume the new source at the same position")
        self.assertTrue(self.clock.isPlaying)

    def test_switch_source_while_paused_does_not_start_audible_playback(self):
        self.clock.seekTo(1000)
        self.clock.pause()
        priorStartMs = self.music.startMs

        self.clock.switchSource("vocals_only.mp3")

        self.assertEqual(self.music.loadedPath, "vocals_only.mp3")
        self.assertEqual(
            self.music.startMs, priorStartMs,
            "switching source while paused must not issue a new play() call",
        )
        self.assertTrue(self.music.paused, "must remain paused/silent")
        self.assertEqual(self.clock.offsetMs, 1000, "position must be preserved for the eventual resume")

    def test_switch_source_to_explicit_ms_overrides_current_position(self):
        self.clock.seekTo(1000)
        self.music.clock += 500

        self.clock.switchSource("vocals_only.mp3", keepMs=9000)

        self.assertEqual(self.music.startMs, 9000)

    # -- start / hasEnded / suspendForScrub --------------------------------

    def test_start_clears_pause_and_plays_even_while_paused(self):
        self.clock.seekTo(1000)
        self.clock.pause()
        self.clock.start(3000)
        self.assertFalse(self.clock.isPaused)
        self.assertEqual(self.music.startMs, 3000)
        self.assertEqual(self.clock.currentMs(), 3000)

    def test_has_ended_only_when_pygame_stopped_on_its_own(self):
        self.clock.seekTo(1000)
        self.assertFalse(self.clock.hasEnded())
        self.music.stop()  # song ran out
        self.assertTrue(self.clock.hasEnded())

    def test_has_ended_is_false_while_paused(self):
        self.clock.seekTo(1000)
        self.clock.pause()
        self.assertFalse(self.clock.hasEnded())

    def test_suspend_for_scrub_silences_without_marking_paused(self):
        self.clock.seekTo(1000)
        self.clock.suspendForScrub()
        self.assertTrue(self.music.paused)
        self.assertFalse(self.clock.isPaused)
        self.clock.seekTo(5000)  # drag release
        self.assertEqual(self.music.startMs, 5000)
        self.assertFalse(self.music.paused)
        self.assertFalse(self.clock.isScrubbing)

    def test_current_ms_during_scrub_is_the_preview_target_not_offset_plus_stale_pos(self):
        self.clock.seekTo(1000)
        self.music.clock += 700  # really at 1700
        self.clock.suspendForScrub()
        self.assertEqual(self.clock.currentMs(), 1700)
        self.clock.seekTo(4000, resync=False)  # mid-drag preview
        self.assertEqual(self.clock.currentMs(), 4000)
        self.music.clock += 900  # time passing mid-drag must not matter
        self.assertEqual(self.clock.currentMs(), 4000)

    def test_pause_during_scrub_does_not_double_count(self):
        self.clock.seekTo(1000)
        self.music.clock += 700
        self.clock.suspendForScrub()
        self.clock.pause()
        self.assertEqual(self.clock.currentMs(), 1700)
        self.clock.resume()
        self.assertEqual(self.music.startMs, 1700)

    # -- stop / idempotency -----------------------------------------------

    def test_stop_is_safe_when_nothing_is_playing(self):
        self.clock.stop()  # must not raise
        self.assertFalse(self.clock.isPlaying)
        self.assertFalse(self.clock.isPaused)

    def test_pause_is_a_noop_if_not_playing(self):
        self.clock.pause()
        self.assertFalse(self.clock.isPaused)

    def test_resume_is_a_noop_if_not_paused(self):
        self.clock.seekTo(1000)
        priorStartMs = self.music.startMs
        self.clock.resume()  # not paused - should do nothing
        self.assertEqual(self.music.startMs, priorStartMs)


if __name__ == "__main__":
    unittest.main()
