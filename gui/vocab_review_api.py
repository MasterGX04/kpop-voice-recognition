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
import time
import traceback

import pygame

from core import vocab_store_ja, vocab_store_ko, vocab_link
from core.cloze import buildClozeCard, orderOccurrencesForCloze
from core.group_registry import GroupRegistry
from core.karaoke_timing import timeLine, calibrateLag, rowCheck, nudgeAnchor
from core import tap_store
from core import lyric_file
from core.lyric_text import findRawLyricText, stripAll, toEditorText, fromEditorText
from core.song_stats import loadRawLabels
from core.srs_fsrs import formatInterval
from core.util_functions import (makeSlowClip, pickBestAudioForStem, ensureAudioForPlayback, chunkToMs, clipStartOffsetMs,
                                 WORD_LEAD_MS, CHUNK_DURATION_MS, KARAOKE_LATENCY_MS, TAP_LAG_MS, MIN_REACTION_MS)

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

    def playOccurrenceAudio(self, group, song, startChunk, endChunk, startFraction=0, exact=False, rate=1.0):
        """Plays the song audio for one occurrence's [startChunk, endChunk) window - the flashcard
        audio-clip prompt (Milestone 2). Reuses the same file-resolution/caching path
        gui/voice_recognition_gui.py's selectSong() uses (getGroupMediaDir + pickBestAudioForStem +
        ensureAudioForPlayback), against this process's own independent pygame.mixer so it can't
        steal/interrupt Audio Tester's playback (see the plan's "separate process" architecture
        decision).

        `startFraction` (0..1, default 0 = whole clip): begin partway through the clip, for "play
        from this word" - the front end estimates a word's onset as a fraction of the line (see
        gui/web/vocab_review/karaoke.js), and core.util_functions.clipStartOffsetMs turns that into
        a safe offset (small lead-in, never leaving less than a short tail).

        `rate` < 1 plays a pitch-preserving slowed-down copy (core.util_functions.makeSlowClip) for the tap tool: the
        clip is cut from the decoded audio and played from its own start, so no MP3 seeking is involved. The page
        clock then advances `rate` song-seconds per real second (see the returned `rate`)."""
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
            # exact=True: `startFraction` is already a lead-adjusted start (core.karaoke_timing playFraction)
            offsetMs = clipStartOffsetMs(clipMs, startFraction, 0 if exact else WORD_LEAD_MS)
            startSec = (chunkToMs(startChunk) + offsetMs) / 1000
            durationSec = (clipMs - offsetMs) / 1000

            rate = float(rate or 1.0)
            if rate < 1.0:
                pygame.mixer.music.load(makeSlowClip(cachedPath, startSec, durationSec, rate))
                pygame.mixer.music.play()
                durationSec = durationSec / rate
            else:
                rate = 1.0
                pygame.mixer.music.load(cachedPath)
                pygame.mixer.music.play(start=startSec)

            startedAtMs = time.time() * 1000           # the moment play() returned: audio is running from here
            self._stopTimer = threading.Timer(durationSec, pygame.mixer.music.stop)
            self._stopTimer.daemon = True
            self._stopTimer.start()
            # What the page needs to run its own karaoke clock (the audio lives in this process, JS cannot read
            # its position): the clip's absolute start chunk, how far into the clip playback actually began, and
            # the clip length. chunk now = clipStartChunk + (offsetMs + (elapsed ms - latencyMs) * rate) / CHUNK_MS.
            return _ok({"clipStartChunk": startChunk, "offsetMs": offsetMs, "clipMs": clipMs, "rate": rate,
                        "playMs": (clipMs - offsetMs) / rate, "chunkMs": CHUNK_DURATION_MS,
                        "latencyMs": KARAOKE_LATENCY_MS, "startedAtMs": startedAtMs})
        except Exception as exc:
            return _err(exc)

    def prepareSlowClip(self, group, song, startChunk, endChunk, rate):
        """Builds (and caches) the slowed-down copy of a whole occurrence clip ahead of time, so the tap tool's
        count-in is not followed by a pause while ffmpeg runs. No-op for rate 1."""
        try:
            if float(rate or 1.0) < 1.0:
                audioPath = pickBestAudioForStem(_groupRegistry.getGroupMediaDir(group), song)
                if not audioPath:
                    raise FileNotFoundError(f"No audio file found for {group}/{song}")
                cachedPath, _ = ensureAudioForPlayback(audioPath)
                clipMs = max(0, chunkToMs(endChunk) - chunkToMs(startChunk))
                makeSlowClip(cachedPath, chunkToMs(startChunk) / 1000, clipMs / 1000, float(rate))
            return _ok()
        except Exception as exc:
            return _err(exc)

    def getPlaybackPositionMs(self):
        """How long the current clip has been playing, in ms (pygame's own clock, not counting the seek offset), or
        None when nothing is playing. The page polls this now and then to correct drift in its own clock."""
        try:
            if not self._mixerReady or not pygame.mixer.music.get_busy():
                return _ok(None)
            pos = pygame.mixer.music.get_pos()
            return _ok(None if pos < 0 else pos)
        except Exception as exc:
            return _err(exc)

    def getMemberColor(self, group, singers):
        """Colour (groups.json: group name -> members -> color) of the card's first singer, or None. Only the first
        singer for now; duets/backing/ad-libs are not handled yet."""
        try:
            first = (singers or [None])[0]
            for member in (_groupRegistry.groups.get(group) or {}).get("members", []):
                if member.get("name") == first and member.get("color"):
                    return _ok(member["color"])
            return _ok(None)
        except Exception as exc:
            return _err(exc)

    def getKanjiReading(self, text):
        """Hiragana reading of a (kanji) word, for the pause editor's "split kanji" button. None if it has no
        kana reading or is not Japanese script."""
        try:
            from core.japanese_utils import kanjiLineToReading
            reading = kanjiLineToReading(stripAll(text), "hiragana").replace(" ", "")
            return _ok(reading or None)
        except Exception as exc:
            return _err(exc)

    def _timingText(self, group, song, lyricLine, lyricId):
        """(text for timeLine, lyricId): the raw lyric (pause markers and split-kanji readings kept, "|" dropped)
        recovered from the song's lyrics file by `lyricId`, else by matching the clean line; the clean line itself
        when no match. The resolved id is what taps are stored under, so a card without one still finds its take."""
        from core.vocab_sync import _readLyricEntries, _lyricsPath
        from core.lyric_text import findRawLyricEntry

        text = stripAll(lyricLine)
        entry = findRawLyricEntry(_readLyricEntries(_lyricsPath(group, song)), lyricId, text)
        if entry is None:
            return text, lyricId
        return (entry.get("korean") or "").replace("|", ""), entry.get("lyricId") or lyricId

    @staticmethod
    def _tapKey(lyricId, unit):
        """Takes are kept per unit: word takes under the lyric id itself (as first saved), syllable takes beside it."""
        return lyricId if unit == "word" else f"{lyricId}#{unit}"

    def _timed(self, group, song, startChunk, endChunk, timingText, lyricId, rows, singers, language, unit):
        """The card's timing with its saved take applied. `unit` None = automatic: a syllable take if one applies,
        else a word take, else the plain estimate (word units). Adds tapStatus ("none" | "ok" | "stale"), tapLag,
        tapUnit (which take is applied) and the resolved lyricId."""
        def run(u, anchors=None):
            return timeLine(timingText, language, startChunk, endChunk, rows, singers, anchors=anchors, unit=u)

        stale = False
        for u in ([unit] if unit else ["syllable", "word"]):
            result = run(u)
            take = tap_store.getTake(group, song, self._tapKey(lyricId, u), result["words"])
            if take["status"] == "ok" and take["anchors"]:
                result = run(u, take["anchors"])
                result.update(tapStatus="ok", tapLag=take["lag"], tapUnit=u, lyricId=lyricId, anchors=take["anchors"],
                              raw=take["raw"], tapRate=take["rate"])
                return result
            stale = stale or take["status"] == "stale"
        shown = run(unit or "word")
        shown.update(tapStatus="stale" if stale else "none", tapLag=None, tapUnit=None, lyricId=lyricId)
        return shown

    def _timingText(self, group, song, lyricLine, lyricId):
        """(text for timeLine, lyricId): the raw lyric (pause markers and split-kanji readings kept, "|" dropped)
        recovered from the song's lyrics file by `lyricId`, else by matching the clean line; the clean line itself
        when no match. The resolved id is what taps are stored under, so a card without one still finds its take."""
        from core.vocab_sync import _readLyricEntries, _lyricsPath
        from core.lyric_text import findRawLyricEntry

        text = stripAll(lyricLine)
        entry = findRawLyricEntry(_readLyricEntries(_lyricsPath(group, song)), lyricId, text)
        if entry is None:
            return text, lyricId
        return (entry.get("korean") or "").replace("|", ""), entry.get("lyricId") or lyricId

    def getLineTiming(self, group, song, startChunk, endChunk, lyricLine, singers, language, lyricId=None, unit=None):
        """Word-by-word position estimates for one occurrence's lyric card, so the front end can
        make each word clickable ("play from here") at a sensible spot. Splits the line at the
        pauses marked in the song's label rows and weights words by their reading - see
        core.karaoke_timing. Returns {"stretches": n, "pieces": [...]}; pieces reassemble to the
        lyric text (with the Tk-only "|" colour markers and the invisible pause marker removed) and
        each word has a `fraction` (0..1 of the way through [startChunk, endChunk)). Returns None
        when there is no span.

        The stored occurrence line is clean, so the author's pause markers (core.lyric_text) are
        recovered from the song's lyrics file: by `lyricId` when the occurrence has one, else by
        matching the clean text. No match just means timing without markers.

        A saved tap-along take (core.tap_store) is applied automatically, so the highlight and click-to-play both
        use it at once. `unit` ("word" | "syllable") asks for one tap unit's view (the tap dialog); None picks the
        best saved take. Extra keys: `tapStatus` ("none" | "ok" | "stale": taps exist but the words changed since),
        `tapLag` (chunks removed from the taps), `tapUnit`, `lyricId` (resolved), plus the timeLine `words` /
        `rowStarts` / `pace`."""
        try:
            if startChunk is None or endChunk is None or endChunk <= startChunk:
                return _ok(None)
            timingText, resolvedId = self._timingText(group, song, lyricLine, lyricId)
            return _ok(self._timed(group, song, startChunk, endChunk, timingText, resolvedId,
                                   loadRawLabels(group, song), singers, language, unit))
        except Exception as exc:
            return _err(exc)

    def _prepareTaps(self, group, song, startChunk, endChunk, lyricLine, singers, language, lyricId, taps, unit, rate):
        """Shared by saveTaps / previewTaps: the base (estimate-only) timing for `unit`, the taps with the human lag
        removed, and the lag itself. See saveTaps for the lag rules."""
        if startChunk is None or endChunk is None or endChunk <= startChunk:
            raise ValueError("This card has no audio span to tap along to.")
        timingText, resolvedId = self._timingText(group, song, lyricLine, lyricId)
        rows = loadRawLabels(group, song)
        base = timeLine(timingText, language, startChunk, endChunk, rows, singers, unit=unit)
        raw = {int(i): float(chunk) for i, chunk in (taps or {}).items() if 0 <= int(i) < len(base["words"])}
        lag, spread = calibrateLag(raw, base["rowStarts"], base["stretchStarts"],
                                   minLag=MIN_REACTION_MS * float(rate or 1.0) / CHUNK_DURATION_MS)
        measured = lag is not None
        if not measured:
            lag = TAP_LAG_MS * float(rate or 1.0) / CHUNK_DURATION_MS
        corrected = {i: max(float(startChunk), chunk - lag) for i, chunk in raw.items()}
        return {"text": timingText, "id": resolvedId, "rows": rows, "base": base, "corrected": corrected, "raw": raw,
                "lag": lag, "spread": spread, "measured": measured}

    def saveTaps(self, group, song, startChunk, endChunk, lyricLine, singers, language, lyricId, taps,
                 unit="word", rate=1.0):
        """Save one tap-along take. `taps` is {unitIndex: chunk} exactly as the page clock read it (index = position
        in getLineTiming's `words` for that `unit`). The human lag is measured here from the taps on exact
        row-start units (core.karaoke_timing.calibrateLag; when fewer than two, TAP_LAG_MS in real time, which is
        TAP_LAG_MS * `rate` of song time on a slowed clip) and removed from every tap before saving, so the file holds
        corrected chunks. An empty `taps` clears that unit's take.
        Returns {lag, spread, measured, timing}: `timing` is the card re-timed with the saved take."""
        try:
            t = self._prepareTaps(group, song, startChunk, endChunk, lyricLine, singers, language, lyricId, taps,
                                  unit, rate)
            tap_store.saveTake(group, song, self._tapKey(t["id"], unit), t["base"]["words"], t["corrected"],
                               lag=round(t["lag"], 2), raw=t["raw"], rate=float(rate or 1.0))
            timing = self._timed(group, song, startChunk, endChunk, t["text"], t["id"], t["rows"], singers, language, unit)
            return _ok({"lag": round(t["lag"], 2), "spread": t["spread"], "measured": t["measured"], "timing": timing,
                        "rowCheck": rowCheck(t["corrected"], t["base"]["stretchStarts"])})
        except tap_store.TapStoreError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return _err(exc)

    def previewTaps(self, group, song, startChunk, endChunk, lyricLine, singers, language, lyricId, taps,
                    unit="word", rate=1.0):
        """What a take WOULD do, without saving it (the tap dialog's replay): {lag, spread, measured, timing,
        estimate}: `timing` is the card timed with these taps (lag removed), `estimate` the same card with no taps,
        so the page can show how far each tapped unit moved."""
        try:
            t = self._prepareTaps(group, song, startChunk, endChunk, lyricLine, singers, language, lyricId, taps,
                                  unit, rate)
            timing = timeLine(t["text"], language, startChunk, endChunk, t["rows"], singers,
                              anchors=t["corrected"], unit=unit)
            return _ok({"lag": round(t["lag"], 2), "spread": t["spread"], "measured": t["measured"],
                        "timing": timing, "estimate": t["base"], "anchors": t["corrected"],
                        "rowCheck": rowCheck(t["corrected"], t["base"]["stretchStarts"])})
        except Exception as exc:
            return _err(exc)

    def nudgeTake(self, group, song, startChunk, endChunk, lyricLine, singers, language, lyricId, anchors, unit,
                  index, deltaChunks):
        """Fine-tune pass: move beat `index` of a take by `deltaChunks` (None = back to the estimate) and re-time the
        card. `anchors` are the take's CURRENT lag-corrected chunks ({unitIndex: chunk}); nothing is saved and no lag
        is subtracted. Returns {status, anchors, timing, beatStart}: status per core.karaoke_timing.nudgeAnchor
        ("moved" | "reset" | "edge" | "locked"), `anchors` the new set, `timing` the card timed with them."""
        try:
            if startChunk is None or endChunk is None or endChunk <= startChunk:
                raise ValueError("This card has no audio span to tap along to.")
            text, _ = self._timingText(group, song, lyricLine, lyricId)
            rows = loadRawLabels(group, song)
            current = {int(i): float(chunk) for i, chunk in (anchors or {}).items()}
            timing = timeLine(text, language, startChunk, endChunk, rows, singers, anchors=current, unit=unit)
            moved, status = nudgeAnchor(timing, current, int(index), deltaChunks)
            if status in ("moved", "reset"):
                timing = timeLine(text, language, startChunk, endChunk, rows, singers, anchors=moved, unit=unit)
            return _ok({"status": status, "anchors": moved, "timing": timing,
                        "beatStart": timing["slots"][int(index)]["startChunk"]})
        except Exception as exc:
            return _err(exc)

    def saveCorrectedTake(self, group, song, startChunk, endChunk, lyricLine, singers, language, lyricId, anchors,
                          unit="word", rate=None, raw=None, lag=None):
        """Save a take whose chunks are ALREADY corrected (a fine-tuned take): written as given, no lag measured or
        removed - saveTaps would subtract it a second time. `raw` / `rate` / `lag` (the uncorrected taps this take
        came from, their playback speed, the lag that was removed) are kept beside it; None keeps what the lyric's
        saved take already has. Returns {timing}."""
        try:
            if startChunk is None or endChunk is None or endChunk <= startChunk:
                raise ValueError("This card has no audio span to tap along to.")
            text, resolvedId = self._timingText(group, song, lyricLine, lyricId)
            rows = loadRawLabels(group, song)
            base = timeLine(text, language, startChunk, endChunk, rows, singers, unit=unit)
            count = len(base["words"])
            corrected = {int(i): float(chunk) for i, chunk in (anchors or {}).items() if 0 <= int(i) < count}
            key = self._tapKey(resolvedId, unit)
            old = tap_store.getTake(group, song, key, base["words"])
            if raw is None:
                raw = old["raw"]
            else:
                raw = {int(i): float(chunk) for i, chunk in raw.items() if 0 <= int(i) < count}
            if lag is None and old["status"] == "ok":
                lag = old["lag"]
            if rate is None and old["status"] == "ok":
                rate = old["rate"]
            tap_store.saveTake(group, song, key, base["words"], corrected, lag=lag, raw=raw,
                               rate=None if rate is None else float(rate))
            return _ok({"timing": self._timed(group, song, startChunk, endChunk, text, resolvedId, rows, singers,
                                              language, unit)})
        except tap_store.TapStoreError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:
            return _err(exc)

    def previewPauseEdit(self, group, song, startChunk, endChunk, singers, language, editorText):
        """Live preview for the pause editor: for the text AS EDITED (not saved), which words land in which of the
        card's label rows. Lets the author see a wrong guess (a word in the wrong row, an empty row) and add a
        marker where the singer pauses, instead of saving and listening. Returns None when there is no span."""
        try:
            if startChunk is None or endChunk is None or endChunk <= startChunk:
                return _ok(None)
            text = fromEditorText(editorText).replace("|", "")
            result = timeLine(text, language, startChunk, endChunk, loadRawLabels(group, song), singers)
            rows = [{"start": a, "end": b, "seconds": round((b - a) * CHUNK_DURATION_MS / 1000, 1), "words": " ".join(words)}
                    for a, b, words in result["rows"]]
            return _ok({"rows": rows, "markers": result["markers"], "rowCount": result["stretches"]})
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
