"""
Single source of truth for "where is playback right now" - position, pause
state, and the loaded audio file - for the whole Voice Recognition GUI.

WHY THIS EXISTS (read this before touching pygame.mixer.music anywhere else)
------------------------------------------------------------------------------
pygame.mixer.music has exactly one piece of state you can read back: get_pos(),
"milliseconds since the last play(start=X) call". It does NOT tell you the
absolute song position - you have to add it to X yourself:

    absolute position = offsetMs + pygame.mixer.music.get_pos()

where offsetMs is whatever `start=` value you last passed to play(). This
formula is only true while that exact play() call is still the active one.
The instant you stop()/play() again, get_pos() resets to 0 and offsetMs must
change to match, or the formula lies.

Historically (see the project's git history / .claude/PLAYBACK_CLOCK_PLAN.md
for the debugging story) this formula got hand-copied into six-plus different
places across gui/audio_tester.py, gui/VideoTrack.py, and gui/zoom_functions.py
- the progress bar, the video sync clock, the zoom slider, the vocals/lead-
back toggle, etc. Every one of those copies had to independently get the
pause-time edge case right (see below), and they kept NOT agreeing with each
other, which is what caused a string of "the handle jumped", "the video is
lagging", "pausing snapped back to an old position" bugs.

The pause-time edge case, explained:
    get_pos() FREEZES the instant pygame.mixer.music.pause() is called (this
    was verified empirically against real pygame/SDL, not assumed - see
    core/test_playback_clock.py's docstring). It does NOT reset to 0, and it
    does NOT keep counting. So while paused, `offsetMs + get_pos()` is a
    FIXED number - fine to read, but if anything mutates offsetMs while
    paused (which pause() below does, to remember the exact pause position)
    the "+ get_pos()" formula becomes double-counting: get_pos() still holds
    the elapsed time up to the pause, and adding it again on top of an
    offsetMs that ALREADY includes that elapsed time overshoots by that same
    amount. This is exactly the "progress bar jumps ahead the instant you
    pause" bug. The fix is structural, not a guard you can bolt on in six
    places: while paused, current position is offsetMs ALONE, never combined
    with get_pos() again, until a real play(start=...) call resets the
    baseline.

THE RULE THIS CLASS ENFORCES
------------------------------------------------------------------------------
Nothing outside this class calls pygame.mixer.music.play/stop/pause/get_pos/
load directly. Everything else - the progress bar, the video decode thread's
audio clock, the zoom slider, arrow-key nudges, label-marker drags, the
vocals/lead-back toggle - asks THIS object what time it is or tells THIS
object to seek/pause/resume, and never touches pygame.mixer.music itself.
That's what makes the invariant unbreakable instead of "correct until the
next person copies the formula wrong."

THREADING NOTE
------------------------------------------------------------------------------
pygame's mixer.music API is not thread-safe. Every method on this class must
only ever be called from the Tk main thread (exactly the same constraint
VideoTrackItem's decode thread already works around by caching a value on the
main thread instead of calling into pygame directly - see
VideoTrackItem._sampleAudioClock). This class does not add any new thread
-safety; it just gives the existing single-main-thread-caller rule one place
to live instead of six.
"""

import pygame


class PlaybackClock:
    """Owns pygame.mixer.music. See module docstring for why."""

    def __init__(self):
        # The `start=` value (ms) of the last real pygame.mixer.music.play()
        # call. Exact and authoritative while paused or never-started; while
        # actively playing, the true position is this PLUS get_pos().
        self._offsetMs = 0
        # True once pause() has captured the true position and told pygame to
        # go silent. False the rest of the time (including "never started").
        self._paused = False
        # True once any real pygame.mixer.music.play() has happened and
        # hasn't been followed by stop(). Distinct from "not paused": a
        # paused clock is still "playing" in the sense that resume() should
        # un-pause it rather than start a brand new session.
        self._playing = False
        # Path last passed to load(); avoids reloading the same file.
        self._audioPath = None
        # True between suspendForScrub() and the next real restart.
        self._scrubbing = False

    # -- read-only state -----------------------------------------------

    @property
    def isPaused(self) -> bool:
        return self._paused

    @property
    def isPlaying(self) -> bool:
        return self._playing

    @property
    def offsetMs(self) -> int:
        return self._offsetMs

    @property
    def audioPath(self):
        return self._audioPath

    def currentMs(self) -> int:
        """The one correct implementation of "what time is it right now".
        Every other place in the app that used to compute this itself
        (progress bar, zoom slider, video sync clock, vocals toggle) should
        call this instead."""
        if self._playing and not self._paused and not self._scrubbing:
            try:
                pos = pygame.mixer.music.get_pos()
            except Exception:
                pos = -1
            if pos < 0:
                pos = 0
            return self._offsetMs + pos
        # Paused, or never started: offsetMs alone is exact (pause() already
        # snapshotted the true position into it - see pause() below). No
        # get_pos() involved here, deliberately - see module docstring.
        return self._offsetMs

    # -- loading ----------------------------------------------------------

    def load(self, path: str):
        """Load an audio file. Call once when opening a song, or when
        switching source files (see switchSource) - NOT on every seek/pause/
        resume; pygame keeps a stopped file's buffer loaded, so re-loading
        the same path on every restart would be wasted work. Does not start
        playback. Propagates pygame.error so the caller (which owns the UI
        status bar) can decide how to report a load failure - unlike the
        methods below, this isn't swallowed."""
        pygame.mixer.music.load(path)
        self._audioPath = path

    def start(self, ms):
        """Begin real playback at `ms` unconditionally, clearing any pause.
        Propagates pygame.error like load(), so the caller can report it."""
        ms = max(0, int(ms))
        pygame.mixer.music.stop()
        pygame.mixer.music.play(start=ms / 1000.0)
        self._offsetMs = ms
        self._playing = True
        self._paused = False
        self._scrubbing = False

    def hasEnded(self) -> bool:
        """True if pygame stopped on its own (song reached its end)."""
        if not (self._playing and not self._paused) or self._scrubbing:
            return False
        try:
            return pygame.mixer.music.get_pos() == -1
        except Exception:
            return False

    @property
    def isScrubbing(self) -> bool:
        return self._scrubbing

    def suspendForScrub(self):
        """Silence audio for a live drag (progress handle or label marker).
        Like pause(), the true position is snapshotted into offsetMs and
        currentMs() stops adding get_pos() - otherwise seekTo(resync=False)
        calls during the drag would be offset by the stale pygame baseline.
        Unlike pause(), isPaused stays False: the drag's release calls
        seekTo(), which restarts audio at the dropped position."""
        if not (self._playing and not self._paused) or self._scrubbing:
            return
        try:
            pos = pygame.mixer.music.get_pos()
            if pos and pos > 0:
                self._offsetMs = int(self._offsetMs) + pos
            pygame.mixer.music.pause()
        except Exception:
            pass
        self._scrubbing = True

    # -- seeking ------------------------------------------------------------

    def seekTo(self, ms, resync: bool = True):
        """Move to `ms`. Always updates the tracked position, whether or not
        we're paused - a seek made while paused must be remembered and
        honored on resume, not silently dropped (this was a real bug: see
        .claude/PLAYBACK_CLOCK_PLAN.md).

        resync: whether to also restart the REAL pygame audio right now.
        True (default) for an authoritative jump - progress bar release,
        arrow-key nudge, jumpToMs, a marker drag that should move playback.
        False for a live scrub-drag preview, where audio is intentionally
        left silent/paused mid-drag and restarting it on every pixel of
        mouse movement would be both wrong (audio would stutter through
        every intermediate position) and slow (each seek can hit a non-O(1)
        keyframe hunt).

        Resyncing is gated on "not paused", NOT on "is playing" - this
        matches the original app's behavior deliberately: seeking before
        Play has ever been pressed (isPlaying False, isPaused False, its
        default) should still start real audio, so the very first scrub on
        a freshly opened song works instead of doing nothing until Play is
        pressed separately.
        """
        ms = max(0, int(ms))
        self._offsetMs = ms
        if not resync or self._paused:
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.play(start=ms / 1000.0)
            self._playing = True
            self._scrubbing = False
        except Exception:
            pass

    # -- pause / resume -------------------------------------------------

    def pause(self):
        """Snapshot the TRUE current position into offsetMs, then silence
        pygame. The snapshot matters: without it, offsetMs would still hold
        whatever the last real play(start=) call used, so a second
        pause/resume with no seek in between would resume from that stale
        old timestamp instead of where playback actually was - the exact
        "pausing twice snaps back to an old position" bug this class exists
        to make impossible."""
        if not (self._playing and not self._paused):
            return
        if not self._scrubbing:  # suspendForScrub already snapshotted
            try:
                pos = pygame.mixer.music.get_pos()
                if pos and pos > 0:
                    self._offsetMs = int(self._offsetMs) + pos
            except Exception:
                pass
        self._scrubbing = False
        self._paused = True
        try:
            pygame.mixer.music.pause()
        except Exception:
            pass

    def resume(self):
        """Restart pygame explicitly at offsetMs - deliberately NOT
        pygame.mixer.music.unpause(). unpause() resumes wherever pygame's
        OWN internal cursor was last left, which is stale if a seek()
        happened while paused (seekTo() above defers the real pygame
        restart while paused, by design, so it doesn't audibly jump mid-
        pause). Restarting at offsetMs instead means a seek made while
        paused is always honored on resume."""
        if not (self._playing and self._paused):
            return
        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.play(start=self._offsetMs / 1000.0)
        except Exception:
            pass
        self._paused = False

    # -- source switching (vocals-only / lead-backing toggles) ------------

    def switchSource(self, path: str, keepMs=None):
        """Change the loaded file while preserving playback position (or
        jump to keepMs if given instead of "wherever we are now"). Used by
        the vocals-only/full-mix and lead/backing submix toggles, which need
        to swap audio files without the song appearing to restart or jump."""
        if keepMs is None:
            keepMs = self.currentMs()
        keepMs = max(0, int(keepMs))
        wasActivelyPlaying = self._playing and not self._paused

        self.load(path)
        self._offsetMs = keepMs

        if wasActivelyPlaying:
            try:
                pygame.mixer.music.play(start=keepMs / 1000.0)
                self._playing = True
                self._scrubbing = False
            except Exception:
                pass

    # -- teardown -----------------------------------------------------------

    def stop(self):
        """Full stop, e.g. on window/app close. Safe to call more than once
        or when nothing is playing."""
        try:
            pygame.mixer.music.stop()
        except Exception:
            pass
        self._playing = False
        self._paused = False
        self._scrubbing = False
