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
from core.cloze import buildClozeCard, orderOccurrencesForCloze
from core.group_registry import GroupRegistry
from core.karaoke_timing import timeLine
from core import lyric_file
from core.lyric_text import findRawLyricText, stripAll, toEditorText, fromEditorText
from core.song_stats import loadRawLabels
from core.srs_fsrs import formatInterval
from core.util_functions import pickBestAudioForStem, ensureAudioForPlayback, chunkToMs, clipStartOffsetMs

# Higher than getOccurrences()'s own default of 5 - that default was tuned for "show a couple of
# example sentences" in the occurrence row, not for maximizing the odds a cloze card can be built
# from at least one of them (see getClozeCardDetail()).
_CLOZE_OCCURRENCE_LIMIT = 20

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


def _songOccurrences(store, vocabId, limit, song):
    """Occurrences from the selected song first; falls back to any song when the word has none
    there (or no song is selected)."""
    if song:
        found = store.getOccurrences(vocabId, limit=limit, group=song["group"], song=song["song"])
        if found:
            return found
    return store.getOccurrences(vocabId, limit=limit)


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

    def listSongs(self, language):
        """[{group, song, wordCount}] sorted Group -> Song, for the song picker."""
        try:
            return _ok(_STORES[language].listSongs())
        except Exception as exc:
            return _err(exc)

    def listQueue(self, language, track, showAll=False, missingMeaningOnly=False, ambiguousOnly=False,
                  song=None):
        """Mirrors gui/vocab_review.py's old loadQueue(): due cards (or every word, if showAll)
        filtered by missing-meaning / ambiguous-Hanja, same as today. `song` ({group, song}) lists
        that song's words in lyric order instead - all of them, since the point is lookup."""
        try:
            store = _STORES[language]
            if song:
                cards = store.listSongVocab(song["group"], song["song"])
            else:
                cards = store.listAllVocab() if showAll else store.getDueCards(track, limit=200)
            if missingMeaningOnly:
                cards = [c for c in cards if _hasNoMeaning(c)]
            if ambiguousOnly and language == "Korean":
                cards = [c for c in cards if len(c.get("hanjaCandidates") or []) >= 2]
            return _ok(cards)
        except Exception as exc:
            return _err(exc)

    def getNextCard(self, language, track, song=None, shuffleNew=False):
        """Study mode: the single next card from the FSRS queue (learning -> due reviews ->
        capped new cards), or None when the session is done. Called again after every rating, so
        an "Again" card can come back within the same session."""
        try:
            songKey = (song["group"], song["song"]) if song else None
            return _ok(_STORES[language].getNextCard(track, song=songKey, shuffleNew=shuffleNew))
        except Exception as exc:
            return _err(exc)

    def getCardDetail(self, vocabId, language, song=None):
        """Occurrence line + cross-language cognate cousins for one card - fetched only for the
        currently displayed card, not prefetched for the whole queue (mirrors _formatBody())."""
        try:
            store = _STORES[language]
            occurrences = _songOccurrences(store, vocabId, 1, song)
            occurrence = occurrences[0] if occurrences else None
            linked = vocab_link.getLinkedWords(_LINK_LANGUAGE[language], vocabId)
            return _ok({"occurrence": occurrence, "linked": linked})
        except Exception as exc:
            return _err(exc)

    def getClozeCardDetail(self, vocabId, language, lemma, song=None):
        """
        Grammar cloze drill (Milestone 5): blank `lemma`'s own surface span out of one of its real
        lyric-line occurrences. `lemma` is supplied by the caller (already held by the loaded queue
        card) rather than re-derived here - mirrors getCardDetail()'s own "queue already has
        lemma/surface, only occurrence/linked data is fetched per-card" design.

        Tries every real occurrence (up to _CLOZE_OCCURRENCE_LIMIT), shorter lines first - see
        core.cloze.orderOccurrencesForCloze()'s own docstring for why: a uniformly random line
        made this feel like "which song is this from" rather than "do you know this word",
        real user feedback. core.cloze.buildClozeCard() returns None when a given line's
        tokenization doesn't turn up a content entry matching this lemma (rare, e.g. a stale/
        edited lyric line) - falls through to the next candidate rather than giving up. Returns
        _ok(None) rather than an error when NO occurrence works - a real, expected "can't quiz this
        word right now" case, not a failure.
        """
        try:
            store = _STORES[language]
            occurrences = _songOccurrences(store, vocabId, _CLOZE_OCCURRENCE_LIMIT, song)
            for occurrence in orderOccurrencesForCloze(occurrences):
                card = buildClozeCard(language, lemma, occurrence["lyricLine"])
                if card:
                    card["group"] = occurrence["group"]
                    card["song"] = occurrence["song"]
                    return _ok(card)
            return _ok(None)
        except Exception as exc:
            return _err(exc)

    def getCognateBridge(self, vocabId, language):
        """JA/KO/Chinese-source three-way comparison (Milestone 3) - see
        core.vocab_link.getCognateBridge for the full shape/degrade-gracefully rules."""
        try:
            return _ok(vocab_link.getCognateBridge(_LINK_LANGUAGE[language], vocabId))
        except Exception as exc:
            return _err(exc)

    def previewIntervals(self, vocabId, language, track):
        """Seconds until due for each of again/hard/good/easy, plus ready-made button labels
        ("10m", "8d") - current card only, like getCardDetail()."""
        try:
            seconds = _STORES[language].previewIntervals(vocabId, track)
            if seconds is None:
                raise KeyError(f"no SRS card for word {vocabId}")
            labels = {name: formatInterval(s) for name, s in seconds.items()}
            return _ok({"seconds": seconds, "labels": labels})
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

    def updateSurface(self, vocabId, language, surface):
        """Japanese-only: hand-set the displayed spelling (blank resets it)."""
        try:
            if language != "Japanese":
                raise ValueError("Editing the spelling is only supported for Japanese words")
            vocab_store_ja.updateSurface(vocabId, surface)
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

    def setSuspended(self, vocabIds, language, suspended):
        try:
            _STORES[language].setSuspended(vocabIds, bool(suspended))
            return _ok()
        except Exception as exc:
            return _err(exc)

    def markKnown(self, vocabIds, language):
        try:
            _STORES[language].markKnown(vocabIds)
            return _ok()
        except Exception as exc:
            return _err(exc)

    def listTriageCandidates(self, language, limit=100):
        """Untouched words, most frequent in your lyrics first - for the bulk known/suspend screen."""
        try:
            return _ok(_STORES[language].listTriageCandidates(limit))
        except Exception as exc:
            return _err(exc)

    def stopAudio(self):
        """Stops any occurrence clip currently playing (called when the flashcard changes)."""
        try:
            if self._stopTimer is not None:
                self._stopTimer.cancel()
                self._stopTimer = None
            if self._mixerReady:
                pygame.mixer.music.stop()
            return _ok()
        except Exception as exc:
            return _err(exc)

    def shutdown(self):
        """Stops any clip and releases the mixer. Called when the review window closes: this runs in
        its own process (see gui/vocab_review_launcher.py), and without it a clip keeps playing
        until the process finally exits."""
        try:
            self.stopAudio()
            if self._mixerReady:
                pygame.mixer.quit()
                self._mixerReady = False
            return _ok()
        except Exception as exc:
            return _err(exc)

    def playOccurrenceAudio(self, group, song, startChunk, endChunk, startFraction=0):
        """Plays the song audio for one occurrence's [startChunk, endChunk) window - the flashcard
        audio-clip prompt (Milestone 2). Reuses the same file-resolution/caching path
        gui/voice_recognition_gui.py's selectSong() uses (getGroupMediaDir + pickBestAudioForStem +
        ensureAudioForPlayback), against this process's own independent pygame.mixer so it can't
        steal/interrupt Audio Tester's playback (see the plan's "separate process" architecture
        decision).

        `startFraction` (0..1, default 0 = whole clip): begin partway through the clip, for "play
        from this word" - the front end estimates a word's onset as a fraction of the line (see
        gui/web/vocab_review/karaoke.js), and core.util_functions.clipStartOffsetMs turns that into
        a safe offset (small lead-in, never leaving less than a short tail)."""
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
            clipMs = max(0, chunkToMs(endChunk) - chunkToMs(startChunk))
            offsetMs = clipStartOffsetMs(clipMs, startFraction)
            startSec = (chunkToMs(startChunk) + offsetMs) / 1000
            durationSec = (clipMs - offsetMs) / 1000

            pygame.mixer.music.load(cachedPath)
            pygame.mixer.music.play(start=startSec)

            self._stopTimer = threading.Timer(durationSec, pygame.mixer.music.stop)
            self._stopTimer.daemon = True
            self._stopTimer.start()
            return _ok()
        except Exception as exc:
            return _err(exc)

    def getLineTiming(self, group, song, startChunk, endChunk, lyricLine, singers, language, lyricId=None):
        """Word-by-word position estimates for one occurrence's lyric card, so the front end can
        make each word clickable ("play from here") at a sensible spot. Splits the line at the
        pauses marked in the song's label rows and weights words by their reading - see
        core.karaoke_timing. Returns {"stretches": n, "pieces": [...]}; pieces reassemble to the
        lyric text (with the Tk-only "|" colour markers and the invisible pause marker removed) and
        each word has a `fraction` (0..1 of the way through [startChunk, endChunk)). Returns None
        when there is no span.

        The stored occurrence line is clean, so the author's pause markers (core.lyric_text) are
        recovered from the song's lyrics file: by `lyricId` when the occurrence has one, else by
        matching the clean text. No match just means timing without markers."""
        try:
            if startChunk is None or endChunk is None or endChunk <= startChunk:
                return _ok(None)
            from core.vocab_sync import _readLyricEntries, _lyricsPath

            text = stripAll(lyricLine)
            raw = findRawLyricText(_readLyricEntries(_lyricsPath(group, song)), lyricId, text)
            # Keep the pause markers (the raw text has them); drop only the colour-split "|".
            timingText = (text if raw is None else raw).replace("|", "")
            return _ok(timeLine(timingText, language, startChunk, endChunk, loadRawLabels(group, song), singers))
        except Exception as exc:
            return _err(exc)

    def getLyricForPauseEdit(self, group, song, lyricId, lyricLine):
        """The lyric text an occurrence came from, as the pause editor shows it (pause markers made visible as
        the editor stand-in), read from that song's own lyrics file - works for any song, not just the one
        the Tk app has open. See core.lyric_file for why this does not go through the Tk Lyric Editor."""
        try:
            resolvedId, raw = lyric_file.getRawLyric(group, song, lyricId, stripAll(lyricLine))
            return _ok({"lyricId": resolvedId, "text": toEditorText(raw)})
        except lyric_file.LyricEditError as exc:
            return {"ok": False, "error": str(exc)}      # a message for the user, not a bug: no traceback
        except Exception as exc:
            return _err(exc)

    def savePauseMarkers(self, group, song, lyricId, lyricLine, editorText):
        """Writes the edited text's pause markers into the song's lyrics file. Only markers may change (the
        wording is refused - see core.lyric_file). Takes the editor form (stand-in glyph) and returns it back."""
        try:
            written = lyric_file.setPauseMarkers(group, song, lyricId, stripAll(lyricLine), fromEditorText(editorText))
            return _ok({"text": toEditorText(written)})
        except lyric_file.LyricEditError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return _err(exc)
