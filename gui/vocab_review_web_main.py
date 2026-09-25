"""
Process entry point for the PyWebView vocab review window (Milestone 1 of
.claude/FLASHCARD_WEB_UPGRADE_PLAN.md). Runs as its own OS process, launched by
gui.vocab_review_launcher.openVocabReviewWindow() - see that module's docstring for why this isn't
a thread in the main Tkinter process.

Dev: `python -m gui.vocab_review_web_main`. Frozen: the packaged exe re-invoked with
`--vocab-review` (see gui/voice_recognition_gui.py's __main__ block and
gui.vocab_review_launcher).
"""

import os
import sys

import webview

from gui.vocab_review_api import VocabReviewApi
from gui.vocab_review_launcher import CHILD_WINDOW_TITLE


def _resourcePath(*parts):
    # Same idiom as gui/voice_recognition_gui.py's resourcePath()/gui/audio_tester.py's
    # resourcePath() - dev falls back to the project root two levels up from this file.
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(base, *parts)


def runStandalone(argv=None):
    indexPath = _resourcePath("gui", "web", "vocab_review", "index.html")
    webview.create_window(
        CHILD_WINDOW_TITLE,
        indexPath,
        js_api=VocabReviewApi(),
        width=760,
        height=680,
        min_size=(560, 420),
    )
    webview.start()


if __name__ == "__main__":
    runStandalone(sys.argv)
