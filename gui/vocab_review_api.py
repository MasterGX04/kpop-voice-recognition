"""
Api surface for the PyWebView vocab review screen (Milestone 1 of
.claude/FLASHCARD_WEB_UPGRADE_PLAN.md). Thin wrapper over core.vocab_store_ja/_ko/vocab_link - same
"no business logic of its own" posture as the old gui/vocab_review.py, just reachable over the JS
bridge instead of direct Python calls. Kept free of any `webview` import so it's unit-testable on
its own (see core/test_vocab_review_api.py) without needing a real window.

Every method returns {"ok": True, "data": ...} or {"ok": False, "error": str}. Never let an
exception cross the JS bridge silently (it would otherwise vanish into a browser-console line the
user never sees) - see FLASHCARD_WEB_UPGRADE_PLAN.md's "Error surfacing" note.
"""

import threading
import traceback

import pygame

from core import vocab_store_ja, vocab_store_ko, vocab_link
from core.group_registry import GroupRegistry
from core.util_functions import pickBestAudioForStem, ensureAudioForPlayback, chunkToMs

_STORES = {"Japanese": vocab_store_ja, "Korean": vocab_store_ko}
_LINK_LANGUAGE = {"Japanese": "ja", "Korean": "ko"}
_groupRegistry = GroupRegistry()


def _ok(data=None):
    return {"ok": True, "data": data}


def _err(exc):
    traceback.print_exc()
    return {"ok": False, "error": str(exc)}


def _hasNoMeaning(card):
    meaning = card.get("meaning") or {}
    return meaning.get("status") != "found"


class VocabReviewApi:
    def __init__(self):
        # This process has no Tk/Toplevel driving pygame like gui/audio_tester.py's
        # VoiceDetectionApp does - init the mixer lazily on first playback request instead.
        self._mixerReady = False
        self._stopTimer = None

    def _ensureMixer(self):
        if not self._mixerReady:
            pygame.mixer.init()
            self._mixerReady = True

    def listQueue(self, language, track, showAll=False, missingMeaningOnly=False, ambiguousOnly=False):
        """Mirrors gui/vocab_review.py's old loadQueue(): due cards (or every word, if showAll)
        filtered by missing-meaning / ambiguous-Hanja, same as today."""
        try:
            store = _STORES[language]
            cards = store.listAllVocab() if showAll else store.getDueCards(track, limit=200)
            if missingMeaningOnly:
                cards = [c for c in cards if _hasNoMeaning(c)]
            if ambiguousOnly and language == "Korean":
                cards = [c for c in cards if len(c.get("hanjaCandidates") or []) >= 2]
            return _ok(cards)
        except Exception as exc:
            return _err(exc)

    def getCardDetail(self, vocabId, language):
        """Occurrence line + cross-language cognate cousins for one card - fetched only for the
        currently displayed card, not prefetched for the whole queue (mirrors _formatBody())."""
        try:
            store = _STORES[language]
            occurrences = store.getOccurrences(vocabId, limit=1)
            occurrence = occurrences[0] if occurrences else None
            linked = vocab_link.getLinkedWords(_LINK_LANGUAGE[language], vocabId)
            return _ok({"occurrence": occurrence, "linked": linked})
        except Exception as exc:
            return _err(exc)

    def rate(self, vocabId, language, track, rating):
        try:
            _STORES[language].submitReview(vocabId, track, rating)
            return _ok()
        except Exception as exc:
            return _err(exc)

    def updateMeaning(self, vocabId, language, glossList, pos=None):
        try:
            _STORES[language].updateMeaning(vocabId, glossList, pos)
            return _ok()
        except Exception as exc:
            return _err(exc)

    def deleteWord(self, vocabId, language):
        try:
            _STORES[language].deleteVocab(vocabId)
            return _ok()
        except Exception as exc:
            return _err(exc)

    def keepOnlyHanja(self, vocabId, hanjaId):
        """Korean-only. No return payload - the caller already has the full candidate list
        client-side and just filters it down to the kept one, same as the old Tkinter screen did."""
        try:
            vocab_store_ko.keepOnlyHanjaCandidate(vocabId, hanjaId)
            return _ok()
        except Exception as exc:
            return _err(exc)

    def clearHanja(self, vocabId):
        """Korean-only."""
        try:
            vocab_store_ko.clearHanjaCandidates(vocabId)
            return _ok()
        except Exception as exc:
            return _err(exc)

    def playOccurrenceAudio(self, group, song, startChunk, endChunk):
        """Plays the song audio for one occurrence's [startChunk, endChunk) window - the flashcard
        audio-clip prompt (Milestone 2). Reuses the same file-resolution/caching path
        gui/voice_recognition_gui.py's selectSong() uses (getGroupMediaDir + pickBestAudioForStem +
        ensureAudioForPlayback), against this process's own independent pygame.mixer so it can't
        steal/interrupt Audio Tester's playback (see the plan's "separate process" architecture
        decision)."""
        try:
            songDir = _groupRegistry.getGroupMediaDir(group)
            audioPath = pickBestAudioForStem(songDir, song)
            if not audioPath:
                raise FileNotFoundError(f"No audio file found for {group}/{song}")

            cachedPath, _ = ensureAudioForPlayback(audioPath)

            # Cancel any previous clip's pending stop-timer first - otherwise a stop scheduled for
            # the last-played clip can fire after this one starts and cut it off early.
            if self._stopTimer is not None:
                self._stopTimer.cancel()

            self._ensureMixer()
            startSec = chunkToMs(startChunk) / 1000
            durationSec = max(0, chunkToMs(endChunk) - chunkToMs(startChunk)) / 1000

            pygame.mixer.music.load(cachedPath)
            pygame.mixer.music.play(start=startSec)

            self._stopTimer = threading.Timer(durationSec, pygame.mixer.music.stop)
            self._stopTimer.daemon = True
            self._stopTimer.start()
            return _ok()
        except Exception as exc:
            return _err(exc)
