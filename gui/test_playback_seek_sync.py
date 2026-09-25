"""
Regression tests for the play()/pause()/_restartAudioAtTime() audio position
state machine in gui/audio_tester.py (VoiceDetectionApp).

This exists because that state machine was broken twice in a row by seek/sync
fixes that each looked correct in isolation but didn't account for how
self.playbackOffset is actually maintained:

  - self.playbackOffset is ONLY the timestamp passed to the last
    pygame.mixer.music.play(start=...) call. It does NOT advance while audio
    is playing (updateChunk() derives currentChunkIndex from
    playbackOffset + get_pos() every tick, but never writes back into
    playbackOffset).
  - A seek while paused must update playbackOffset (so resuming honors it)
    without touching the real pygame audio position (so it doesn't audibly
    jump while still paused).
  - pause() must snapshot the TRUE current position (playbackOffset +
    get_pos() at the moment of pausing) back into playbackOffset, or a second
    pause/resume with no seek in between resumes from the stale timestamp of
    whatever play(start=...) call happened last, not from where playback
    actually was.

These tests exercise the real VoiceDetectionApp.pause/play/_restartAudioAtTime
methods (not a reimplementation) against a fake pygame.mixer.music driven by
an explicit fake clock, so they're deterministic and need no real audio
device, Tk window, or video file.

Run with:
    python -m unittest gui.test_playback_seek_sync -v
"""

import unittest
from unittest import mock

from gui.audio_tester import VoiceDetectionApp
from core.playback_clock import PlaybackClock


class FakePygameMixerMusic:
    """Stands in for pygame.mixer.music. Position is driven by an explicit
    fake clock (self.clock, in ms) that the test advances directly, instead
    of real wall-clock time.

    get_pos() freezes at whatever it read when pause() was called and does
    NOT keep advancing with self.clock while paused - this was verified
    empirically against the real pygame.mixer.music (play, sleep, pause,
    sleep longer, check get_pos() is unchanged, unpause, check it resumes
    counting from where it left off). Getting this right in the fake matters:
    the bug this file guards against (a stale tick re-adding get_pos() on
    top of an already-corrected playbackOffset) only reproduces if get_pos()
    behaves like the real thing during a pause.
    """

    def __init__(self):
        self.clock = 0
        self.playing = False
        self.paused = False
        self.startMs = None       # the `start=` value of the last play() call
        self._playedAtClock = 0
        self._pausedAtClock = None

    def get_init(self):
        return True

    def load(self, path):
        pass

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


class SilentApp(VoiceDetectionApp):
    """VoiceDetectionApp stand-in that skips __init__ (no Tk root, canvas, or
    real audio device needed) and stubs out the UI-facing methods that
    pause()/play()/_restartAudioAtTime() call into. pause, play, and
    _restartAudioAtTime themselves are the REAL, unmodified methods inherited
    from VoiceDetectionApp - only unrelated UI plumbing is stubbed."""

    def __init__(self):
        self.clock = PlaybackClock()
        self.isManualUpdate = False
        self.currentChunkIndex = 0
        self.chunk_duration = 40
        self.chunks = list(range(100000))

    def playWithSavedResults(self, startTimeMs):
        pass

    def syncVisualsToTime(self, timeMs):
        pass

    def updateChunkIndexDisplay(self, *args, **kwargs):
        pass

    def updateCurrentTime(self, *args, **kwargs):
        pass

    def seekToChunk(self, newChunkIndex):
        self.currentChunkIndex = newChunkIndex


class FakeAfterRoot:
    """Minimal stand-in for the Tk root's .after(), which updateChunk() (the
    inner closure of playWithSavedResults) uses to reschedule itself.
    Scheduled callbacks are queued, not auto-invoked - call .tick() to run
    the next one, simulating one time step under test control."""

    def __init__(self):
        self.scheduled = []

    def after(self, delayMs, fn):
        self.scheduled.append(fn)
        return len(self.scheduled)

    def tick(self):
        if not self.scheduled:
            return
        fn = self.scheduled.pop(0)
        fn()


class TickingApp(SilentApp):
    """Like SilentApp, but keeps the REAL playWithSavedResults()/updateChunk()
    ticking loop instead of SilentApp's no-op stub, so tests can exercise
    its interaction with pause() directly."""

    def __init__(self):
        super().__init__()
        self.root = FakeAfterRoot()
        self.detectionResults = []
        self.currentAudioPath = "unused.wav"

    def playWithSavedResults(self, startTimeMs):
        return VoiceDetectionApp.playWithSavedResults(self, startTimeMs)


class PlaybackSeekSyncTests(unittest.TestCase):
    def setUp(self):
        self.music = FakePygameMixerMusic()
        musicPatcher = mock.patch("gui.audio_tester.pygame.mixer.music", self.music)
        musicPatcher.start()
        self.addCleanup(musicPatcher.stop)

        # _restartAudioAtTime() gates the real pygame restart on
        # pygame.mixer.get_init() - the real mixer is never pygame.mixer.init()'d
        # in this test, so it would otherwise report "not initialized" and the
        # method would silently skip touching self.music.
        getInitPatcher = mock.patch("gui.audio_tester.pygame.mixer.get_init", return_value=True)
        getInitPatcher.start()
        self.addCleanup(getInitPatcher.stop)

        self.app = SilentApp()

    def _startPlayingAt(self, ms):
        self.app.clock.start(ms)

    def test_seek_while_paused_is_honored_on_resume(self):
        """A seek made while paused must not be silently dropped: resuming
        should restart audio at the seek target, not wherever it was paused."""
        self._startPlayingAt(1000)
        self.music.clock += 5000  # played forward 5s
        self.app.pause()

        self.app._restartAudioAtTime(400)
        self.assertEqual(self.app.playbackOffset, 400)
        self.assertTrue(self.music.paused, "audio must stay paused/silent while still paused")

        self.app.play()
        self.assertFalse(self.app.isPaused)
        self.assertEqual(
            self.music.startMs, 400,
            "resume must restart audio at the seek made while paused, not the pre-seek position",
        )

    def test_pause_then_resume_with_no_further_seek_stays_put(self):
        """Regression for the follow-up bug: after fixing the above, pausing
        a SECOND time with no further seek must resume from where playback
        actually was, not snap back to an earlier seek's timestamp."""
        self._startPlayingAt(400)  # e.g. resumed at a seek target of 400ms
        self.music.clock += 2000   # played forward 2s -> really at 2400ms now
        self.app.pause()

        self.app.play()

        self.assertEqual(
            self.music.startMs, 2400,
            "resume must continue from the true paused position, not snap back to 400",
        )

    def test_multiple_pause_resume_cycles_never_drift_to_a_stale_seek(self):
        """The exact scenario reported: seek to 400, resume, play forward,
        pause with no seek, resume again - must land at the played-forward
        position both times, never back at 400."""
        self._startPlayingAt(1000)
        self.music.clock += 3000
        self.app.pause()
        self.app._restartAudioAtTime(400)  # scrub to 400 while paused
        self.app.play()
        self.assertEqual(self.music.startMs, 400)

        self.music.clock += 1200  # play forward 1.2s from the 400ms resume
        self.app.pause()
        self.app.play()
        self.assertEqual(
            self.music.startMs, 1600,
            "must resume at 400 + 1200 played, not snap back to the 400 seek target",
        )

    def test_seek_while_playing_restarts_audio_immediately(self):
        """Unchanged existing behavior: a seek while NOT paused should
        restart audio right away, not defer it."""
        self._startPlayingAt(1000)
        self.app._restartAudioAtTime(5000)
        self.assertEqual(self.music.startMs, 5000)
        self.assertTrue(self.music.playing)


class UpdateChunkPauseTests(unittest.TestCase):
    """Regression coverage for the progress-bar-handle-jumps-on-pause bug:
    pause() snapshotting the true position into playbackOffset (see above)
    is only correct if nothing else ALSO re-derives position from
    playbackOffset + get_pos() while paused - that formula is only valid
    between a real play(start=...) call and the next one, which pause()'s
    snapshot necessarily breaks."""

    def setUp(self):
        self.music = FakePygameMixerMusic()
        musicPatcher = mock.patch("gui.audio_tester.pygame.mixer.music", self.music)
        musicPatcher.start()
        self.addCleanup(musicPatcher.stop)

        getInitPatcher = mock.patch("gui.audio_tester.pygame.mixer.get_init", return_value=True)
        getInitPatcher.start()
        self.addCleanup(getInitPatcher.stop)

        self.app = TickingApp()

    def test_a_tick_scheduled_before_pause_does_not_jump_the_displayed_position(self):
        self.app.clock.start(1000)

        self.music.clock += 500  # 500ms into playback -> real position 1500ms
        self.app.playWithSavedResults(1000)  # kicks off the first real tick
        self.assertEqual(self.app.currentChunkIndex, 1500 // self.app.chunk_duration)

        self.music.clock += 300  # another 300ms passes -> real position 1800ms
        self.app.pause()
        self.assertEqual(self.app.playbackOffset, 1800)
        chunkIndexAtPause = self.app.currentChunkIndex

        # The tick queued by the call above (scheduled BEFORE pause()) fires
        # anyway shortly after, exactly as it does in the real app - nothing
        # cancels it on pause.
        self.app.root.tick()

        self.assertEqual(
            self.app.currentChunkIndex, chunkIndexAtPause,
            "a tick firing right after pause() must not move the displayed "
            "position further ahead of the real, unmoving paused audio",
        )

    def test_resuming_restarts_the_ticking_loop(self):
        """The loop that stops on pause must come back to life on resume,
        or the progress bar would freeze forever after the first pause."""
        self.app.clock.start(1000)

        self.music.clock += 500  # real position now 1500ms
        self.app.playWithSavedResults(1000)
        self.app.pause()  # snapshots playbackOffset to 1500
        self.assertEqual(self.app.playbackOffset, 1500)
        self.app.root.tick()  # stale tick: must no-op, not crash or reschedule

        self.music.clock += 100  # time passing while paused must not matter
        self.app.play()  # resumes at the paused position (playbackOffset: 1500ms)
        self.assertEqual(self.music.startMs, 1500)

        self.music.clock += 200  # 200ms further playback since resuming
        self.assertTrue(self.app.root.scheduled, "resume must queue a fresh tick")
        self.app.root.tick()
        self.assertEqual(self.app.currentChunkIndex, 1700 // self.app.chunk_duration)


if __name__ == "__main__":
    unittest.main()
