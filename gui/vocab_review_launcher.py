"""
Launches the PyWebView vocab review screen as its own OS process (replaces the old
gui.vocab_review.openVocabReviewWindow(parent) as the "Vocab" menu's call target - same signature,
so gui/audio_tester.py's call site is unchanged, only the import).

Why a separate process, not a thread in the main Tkinter process: gui/audio_tester.py drives a
single global pygame.mixer.music tied to whatever song it currently has open. A future
audio-clip-playback feature in the review screen (Milestone 2) would steal/interrupt that if it
ran in-process; a separate process gets its own independent pygame.mixer. It also avoids mixing
Tk's and WebView2's native event loops in one process. See
.claude/FLASHCARD_WEB_UPGRADE_PLAN.md's "Architecture decision" section for the full reasoning.

Window-parenting (replacing what the old screen's win.transient(parent) gave) is done at the Win32
level via ctypes, since the child is a separate process, not a child widget: FindWindowW locates
the new window by its title, SetWindowLongPtrW(GWL_HWNDPARENT) reproduces owned-window stacking.

Milestone 0 spike finding (2026-09-13): reparenting immediately after FindWindowW finds the window
races with WebView2's own async CoreWebView2Environment creation and can abort it (HRESULT
0x80004004, "Operation aborted"). A short delay before SetWindowLongPtrW avoided it in testing -
see .claude/FLASHCARD_WEB_UPGRADE_PLAN.md Milestone 0 for the reproduction.
"""

import ctypes
import os
import subprocess
import sys
import threading
import time

CHILD_WINDOW_TITLE = "Vocab Review"
_GWL_HWNDPARENT = -8
_REPARENT_DELAY_SEC = 1.5

_user32 = ctypes.windll.user32
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _findChildHwnd(timeoutSec=5.0):
    deadline = time.time() + timeoutSec
    while time.time() < deadline:
        hwnd = _user32.FindWindowW(None, CHILD_WINDOW_TITLE)
        if hwnd:
            return hwnd
        time.sleep(0.1)
    return None


def _reparentInBackground(parentHwnd):
    childHwnd = _findChildHwnd()
    if not childHwnd:
        return
    time.sleep(_REPARENT_DELAY_SEC)
    _user32.SetWindowLongPtrW(childHwnd, _GWL_HWNDPARENT, parentHwnd)


def openVocabReviewWindow(parent):
    """`parent` is the Tk root/Toplevel this was opened from (gui/audio_tester.py's self.root) -
    same argument the old gui.vocab_review.openVocabReviewWindow(parent) took."""
    if getattr(sys, "frozen", False):
        # Re-invoke the same frozen exe with a flag - there's no separate python.exe to launch in
        # a frozen build. NOTE: this branch is untested pending Milestone 7's own frozen-build
        # spike (deferred from Milestone 0 - see the plan doc); revisit before relying on it.
        args = [sys.executable, "--vocab-review"]
        cwd = os.path.dirname(sys.executable)
    else:
        args = [sys.executable, "-m", "gui.vocab_review_web_main"]
        cwd = _PROJECT_ROOT

    subprocess.Popen(args, cwd=cwd)

    parentHwnd = parent.winfo_id()
    threading.Thread(target=_reparentInBackground, args=(parentHwnd,), daemon=True).start()
