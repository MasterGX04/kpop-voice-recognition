"""
Tests for gui/vocab_review_api.py's VocabReviewApi - the PyWebView Api class from Milestone 1 of
.claude/FLASHCARD_WEB_UPGRADE_PLAN.md. Exercises it directly as a plain Python object (no `webview`
import needed, per its own docstring), against a tempdir-isolated DB, same isolation pattern as
core/test_vocab_store_ja.py / core/test_vocab_store_ko.py. Run with:
    python -m unittest core.test_vocab_review_api -v
"""

import os
import shutil
import tempfile
import time
import unittest

from core import vocab_store_ja, vocab_store_ko, vocab_link
from core.util_functions import chunkToMs
from gui.vocab_review_api import VocabReviewApi


def _jaEntry(lemma, cognate=None):
    return {
        "surface": lemma, "reading": "じかん", "lemma": lemma, "lemmaReading": "じかん",
        "category": "onyomi", "chineseCognate": cognate,
        "japaneseMeaning": {"status": "found", "pos": ["n"], "gloss": ["time"]},
        "mandarinPinyin": {"traditional": lemma, "pinyin": "shi2 jian1"},
    }


def _koEntry(lemma, hanjaCandidates=None, meaning=None):
    return {
        "surface": lemma, "lemma": lemma,
        "meaning": meaning or {"status": "found", "pos": "noun", "gloss": ["fire"]},
        "hanjaCandidates": hanjaCandidates or [],
    }


class VocabReviewApiTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)
        self.api = VocabReviewApi()

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_list_queue_returns_due_japanese_cards(self):
        vocab_store_ja.upsertVocab(_jaEntry("時間"))
        res = self.api.listQueue("Japanese", "reading")
        self.assertTrue(res["ok"])
        self.assertEqual(len(res["data"]), 1)
        self.assertEqual(res["data"][0]["lemma"], "時間")

    def _seedSongs(self):
        ids = {}
        for lemma, group, song, chunk in [("一", "BTS", "Stay Gold", 30), ("二", "BTS", "Stay Gold", 10),
                                          ("三", "BTS", "Let Go", 5), ("四", "TWICE", "Doughnut", 1)]:
            ids[lemma], _ = vocab_store_ja.upsertVocab(_jaEntry(lemma))
            vocab_store_ja.addOccurrence(ids[lemma], group, song, [], f"line {lemma}", f"l-{lemma}", chunk, chunk + 1)
        # 三 also appears in Stay Gold, later than its Let Go line
        vocab_store_ja.addOccurrence(ids["三"], "BTS", "Stay Gold", [], "line 三 again", "l-三b", 50, 51)
        return ids

    def test_list_songs_sorted_group_then_song_with_word_counts(self):
        self._seedSongs()
        res = self.api.listSongs("Japanese")
        self.assertEqual(
            [(s["group"], s["song"], s["wordCount"]) for s in res["data"]],
            [("BTS", "Let Go", 1), ("BTS", "Stay Gold", 3), ("TWICE", "Doughnut", 1)])

    def test_list_queue_with_song_returns_only_that_songs_words_in_lyric_order(self):
        self._seedSongs()
        res = self.api.listQueue("Japanese", "reading", song={"group": "BTS", "song": "Stay Gold"})
        self.assertEqual([c["lemma"] for c in res["data"]], ["二", "一", "三"])

    def test_get_next_card_with_song_stays_inside_the_song(self):
        self._seedSongs()
        song = {"group": "TWICE", "song": "Doughnut"}
        self.assertEqual(self.api.getNextCard("Japanese", "reading", song)["data"]["lemma"], "四")
        for shuffle in (False, True):
            lemma = self.api.getNextCard("Japanese", "reading", {"group": "BTS", "song": "Let Go"}, shuffle)
            self.assertEqual(lemma["data"]["lemma"], "三")

    def test_card_detail_prefers_occurrence_from_selected_song(self):
        ids = self._seedSongs()
        song = {"group": "BTS", "song": "Stay Gold"}
        occ = self.api.getCardDetail(ids["三"], "Japanese", song)["data"]["occurrence"]
        self.assertEqual((occ["song"], occ["lyricLine"]), ("Stay Gold", "line 三 again"))

    def test_card_detail_falls_back_when_word_not_in_selected_song(self):
        ids = self._seedSongs()
        occ = self.api.getCardDetail(ids["四"], "Japanese", {"group": "BTS", "song": "Stay Gold"})["data"]["occurrence"]
        self.assertEqual(occ["song"], "Doughnut")

    def test_list_queue_missing_meaning_only_filters_korean_cards(self):
        vocab_store_ko.upsertVocab(_koEntry("불", meaning={"status": "not_found", "pos": "", "gloss": []}))
        vocab_store_ko.upsertVocab(_koEntry("물"))
        res = self.api.listQueue("Korean", "reading", showAll=True, missingMeaningOnly=True)
        self.assertTrue(res["ok"])
        self.assertEqual([c["lemma"] for c in res["data"]], ["불"])

    def test_list_queue_ambiguous_only_requires_two_plus_hanja_candidates(self):
        ambiguous = [
            {"hanja": "火", "gloss": ["fire"], "pos": "noun", "pinyin": "huo3"},
            {"hanja": "和", "gloss": ["harmony"], "pos": "noun", "pinyin": "he2"},
        ]
        vocab_store_ko.upsertVocab(_koEntry("화", hanjaCandidates=ambiguous))
        vocab_store_ko.upsertVocab(_koEntry("물", hanjaCandidates=[
            {"hanja": "水", "gloss": ["water"], "pos": "noun", "pinyin": "shui3"}
        ]))
        res = self.api.listQueue("Korean", "reading", showAll=True, ambiguousOnly=True)
        self.assertTrue(res["ok"])
        self.assertEqual([c["lemma"] for c in res["data"]], ["화"])

    def test_get_card_detail_includes_occurrence_and_linked_words(self):
        cognate = {"status": "confirmed", "traditional": "時間", "pinyin": "shi2 jian1", "gloss": ["time"]}
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間", cognate=cognate))
        vocab_store_ja.addOccurrence(
            jaId, "BTS", "Let Go", ["Jungkook"], "時間がない", "line1", 10, 20
        )
        koId, _ = vocab_store_ko.upsertVocab(_koEntry("시간", hanjaCandidates=[
            {"hanja": "時間", "gloss": ["time"], "pos": "noun", "pinyin": "shi2 jian1"}
        ]))
        vocab_link.syncCognateLinks()

        res = self.api.getCardDetail(jaId, "Japanese")
        self.assertTrue(res["ok"])
        self.assertEqual(res["data"]["occurrence"]["group"], "BTS")
        self.assertEqual(res["data"]["occurrence"]["song"], "Let Go")
        self.assertEqual([w["vocabId"] for w in res["data"]["linked"]], [koId])

    def test_get_cognate_bridge_wraps_vocab_link_for_both_languages(self):
        cognate = {"status": "confirmed", "traditional": "時間", "pinyin": "shi2 jian1", "gloss": ["time"]}
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間", cognate=cognate))
        koId, _ = vocab_store_ko.upsertVocab(_koEntry("시간", hanjaCandidates=[
            {"hanja": "時間", "gloss": ["time"], "pos": "noun", "pinyin": "shi2 jian1"}
        ]))
        vocab_link.syncCognateLinks()

        jaRes = self.api.getCognateBridge(jaId, "Japanese")
        self.assertTrue(jaRes["ok"])
        self.assertEqual(jaRes["data"]["cognateForm"], "時間")
        self.assertEqual(jaRes["data"]["korean"]["vocabId"], koId)

        koRes = self.api.getCognateBridge(koId, "Korean")
        self.assertTrue(koRes["ok"])
        self.assertEqual(koRes["data"]["japanese"]["vocabId"], jaId)

    def test_get_card_detail_with_no_occurrence_returns_none(self):
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間"))
        res = self.api.getCardDetail(jaId, "Japanese")
        self.assertTrue(res["ok"])
        self.assertIsNone(res["data"]["occurrence"])
        self.assertEqual(res["data"]["linked"], [])

    def test_rate_advances_due_date_so_card_no_longer_in_due_queue(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間"))
        res = self.api.rate(vocabId, "Japanese", "reading", "easy")
        self.assertTrue(res["ok"])
        due = self.api.listQueue("Japanese", "reading")["data"]
        self.assertEqual(due, [])

    def test_get_next_card_returns_queue_type_and_none_when_done(self):
        vocab_store_ja.upsertVocab(_jaEntry("時間"))
        res = self.api.getNextCard("Japanese", "reading")
        self.assertTrue(res["ok"])
        self.assertEqual((res["data"]["lemma"], res["data"]["queueType"]), ("時間", "new"))
        self.api.rate(res["data"]["vocabId"], "Japanese", "reading", "easy")
        self.assertIsNone(self.api.getNextCard("Japanese", "reading")["data"])
        self.assertFalse(self.api.getNextCard("Klingon", "reading")["ok"])

    def test_stop_audio_is_safe_with_nothing_playing_and_cancels_pending_timer(self):
        self.assertTrue(self.api.stopAudio()["ok"])  # mixer never initialised - must not raise

        class FakeTimer:
            cancelled = False
            def cancel(self):
                self.cancelled = True
        timer = FakeTimer()
        self.api._stopTimer = timer
        self.assertTrue(self.api.stopAudio()["ok"])
        self.assertTrue(timer.cancelled)
        self.assertIsNone(self.api._stopTimer)

    def test_suspend_known_and_triage_through_the_api(self):
        a, _ = vocab_store_ja.upsertVocab(_jaEntry("私"))
        b, _ = vocab_store_ja.upsertVocab(_jaEntry("僕"))
        c, _ = vocab_store_ja.upsertVocab(_jaEntry("君"))
        rows = self.api.listTriageCandidates("Japanese")["data"]
        self.assertEqual({r["lemma"] for r in rows}, {"私", "僕", "君"})

        self.assertTrue(self.api.setSuspended([a], "Japanese", True)["ok"])
        self.assertTrue(self.api.markKnown([b], "Japanese")["ok"])
        self.assertEqual([r["lemma"] for r in self.api.listTriageCandidates("Japanese")["data"]], ["君"])
        self.assertEqual(self.api.getNextCard("Japanese", "reading")["data"]["lemma"], "君")

        self.assertTrue(self.api.setSuspended([a], "Japanese", False)["ok"])
        self.assertFalse(self.api.setSuspended([a], "Klingon", True)["ok"])
        self.assertFalse(self.api.listTriageCandidates("Klingon")["ok"])

    def test_all_four_ratings_are_accepted(self):
        for rating in ("again", "hard", "good", "easy"):
            vocabId, _ = vocab_store_ja.upsertVocab(_jaEntry(f"語{rating}"))
            self.assertTrue(self.api.rate(vocabId, "Japanese", "meaning", rating)["ok"])
        self.assertFalse(self.api.rate(vocabId, "Japanese", "meaning", "perfect")["ok"])

    def test_preview_intervals_returns_seconds_and_button_labels(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間"))
        res = self.api.previewIntervals(vocabId, "Japanese", "reading")
        self.assertTrue(res["ok"])
        self.assertEqual(set(res["data"]["labels"]), {"again", "hard", "good", "easy"})
        self.assertEqual(res["data"]["labels"]["again"], "1m")
        self.assertTrue(res["data"]["labels"]["easy"].endswith("d"))
        self.assertLess(res["data"]["seconds"]["again"], res["data"]["seconds"]["easy"])

    def test_preview_intervals_errors_gracefully_for_unknown_word(self):
        self.assertFalse(self.api.previewIntervals(9999, "Korean", "reading")["ok"])

    def test_update_meaning_sets_meaning_locked_so_a_rescan_does_not_revert_it(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間"))
        res = self.api.updateMeaning(vocabId, "Japanese", ["custom meaning"])
        self.assertTrue(res["ok"])

        # Re-run the exact scan a "Compile" would do - the manual edit must survive.
        vocab_store_ja.upsertVocab(_jaEntry("時間"))
        card = self.api.listQueue("Japanese", "reading", showAll=True)["data"][0]
        self.assertEqual(card["meaning"]["gloss"], ["custom meaning"])

    def test_delete_word_removes_it_from_queue(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間"))
        res = self.api.deleteWord(vocabId, "Japanese")
        self.assertTrue(res["ok"])
        self.assertEqual(self.api.listQueue("Japanese", "reading", showAll=True)["data"], [])

    def test_keep_only_hanja_leaves_a_single_locked_candidate(self):
        candidates = [
            {"hanja": "火", "gloss": ["fire"], "pos": "noun", "pinyin": "huo3"},
            {"hanja": "和", "gloss": ["harmony"], "pos": "noun", "pinyin": "he2"},
        ]
        vocabId, _ = vocab_store_ko.upsertVocab(_koEntry("화", hanjaCandidates=candidates))
        card = self.api.listQueue("Korean", "reading", showAll=True)["data"][0]
        keepId = card["hanjaCandidates"][0]["hanjaId"]

        res = self.api.keepOnlyHanja(vocabId, keepId)
        self.assertTrue(res["ok"])

        # Locked, so a rescan must not re-add the deleted candidates back.
        vocab_store_ko.upsertVocab(_koEntry("화", hanjaCandidates=candidates))
        card = self.api.listQueue("Korean", "reading", showAll=True)["data"][0]
        self.assertEqual(len(card["hanjaCandidates"]), 1)
        self.assertEqual(card["hanjaCandidates"][0]["hanjaId"], keepId)

    def test_clear_hanja_removes_all_candidates_and_locks_it(self):
        vocabId, _ = vocab_store_ko.upsertVocab(_koEntry("해", hanjaCandidates=[
            {"hanja": "害", "gloss": ["harm"], "pos": "noun", "pinyin": "hai4"}
        ]))
        res = self.api.clearHanja(vocabId)
        self.assertTrue(res["ok"])

        # Locked, so a rescan must not silently re-add the removed candidate.
        vocab_store_ko.upsertVocab(_koEntry("해", hanjaCandidates=[
            {"hanja": "害", "gloss": ["harm"], "pos": "noun", "pinyin": "hai4"}
        ]))
        card = self.api.listQueue("Korean", "reading", showAll=True)["data"][0]
        self.assertEqual(card["hanjaCandidates"], [])

    def test_get_cloze_card_detail_blanks_the_word_in_a_real_occurrence(self):
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間"))
        vocab_store_ja.addOccurrence(
            jaId, "BTS", "Let Go", ["Jungkook"], "時間がない", "line1", 10, 20
        )
        res = self.api.getClozeCardDetail(jaId, "Japanese", "時間")
        self.assertTrue(res["ok"])
        self.assertIsNotNone(res["data"])
        self.assertEqual(res["data"]["answerSurface"], "時間")
        self.assertNotIn("時間", res["data"]["blankedLine"])
        self.assertEqual(res["data"]["group"], "BTS")
        self.assertEqual(res["data"]["song"], "Let Go")

    def test_get_cloze_card_detail_with_no_occurrence_returns_ok_none(self):
        # A real, expected "can't quiz this word right now" case - not an error.
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間"))
        res = self.api.getClozeCardDetail(jaId, "Japanese", "時間")
        self.assertTrue(res["ok"])
        self.assertIsNone(res["data"])

    def test_get_cloze_card_detail_errors_gracefully_on_bad_language(self):
        res = self.api.getClozeCardDetail(1, "Klingon", "時間")
        self.assertFalse(res["ok"])
        self.assertIn("Klingon", res["error"])

    def test_errors_are_caught_and_surfaced_not_raised(self):
        # An unknown language key would otherwise raise a bare KeyError across the JS bridge.
        res = self.api.listQueue("Klingon", "reading")
        self.assertFalse(res["ok"])
        self.assertIn("Klingon", res["error"])

    def test_chunk_to_ms_matches_fixture_values(self):
        # Real fixture values from saved_labels/TWICE/Breakthrough_lyrics.json's linkedLabel spans
        # (chunk_duration = 40ms), not made-up numbers.
        self.assertEqual(chunkToMs(642), 25680)
        self.assertEqual(chunkToMs(744), 29760)
        self.assertEqual(chunkToMs(0), 0)

    def test_play_occurrence_audio_errors_gracefully_when_no_audio_file_resolves(self):
        # No group_icons/groups.json in this tempdir, so getGroupMediaDir() falls back to a
        # nonexistent ./training_data/{group} - this must surface as {"ok": False}, not crash.
        res = self.api.playOccurrenceAudio("NoSuchGroup", "NoSuchSong", 0, 40)
        self.assertFalse(res["ok"])
        self.assertIn("NoSuchGroup", res["error"])

    def test_clip_start_offset_ms(self):
        from core.util_functions import clipStartOffsetMs, WORD_LEAD_MS, MIN_TAIL_MS
        d = 12000  # a ~12 s line like the long Doughnut/BTS spans
        self.assertEqual(clipStartOffsetMs(d, 0), 0)                       # whole clip
        self.assertEqual(clipStartOffsetMs(d, 0.5), 6000 - WORD_LEAD_MS)   # lead-in before the word
        self.assertEqual(clipStartOffsetMs(d, 0.01), 0)                    # lead-in never goes negative
        self.assertEqual(clipStartOffsetMs(d, 1.0), d - MIN_TAIL_MS)       # last word keeps a tail
        self.assertEqual(clipStartOffsetMs(d, 7), d - MIN_TAIL_MS)         # out-of-range clamps
        self.assertEqual(clipStartOffsetMs(400, 0.9), 0)                   # clip shorter than the tail

    def test_play_occurrence_audio_starts_partway_and_bounds_remaining_duration(self):
        from unittest import mock
        from core.util_functions import clipStartOffsetMs
        timers = []

        class FakeTimer:
            def __init__(self, interval, fn):
                self.interval = interval
                self.daemon = False
                timers.append(self)
            def start(self): pass
            def cancel(self): pass

        # Real Doughnut/Tzuyu span after the A2 fix: chunks 2456-2828 = 14880 ms.
        start, end = 2456, 2828
        clipMs = chunkToMs(end) - chunkToMs(start)
        offsetMs = clipStartOffsetMs(clipMs, 0.8)
        with mock.patch("gui.vocab_review_api.pickBestAudioForStem", return_value="x.mp3"),              mock.patch("gui.vocab_review_api.ensureAudioForPlayback", return_value=("x.wav", None)),              mock.patch("gui.vocab_review_api.threading.Timer", FakeTimer),              mock.patch("gui.vocab_review_api.pygame.mixer.music") as music,              mock.patch.object(self.api, "_ensureMixer"):
            res = self.api.playOccurrenceAudio("TWICE", "Doughnut", start, end, 0.8)
            self.assertTrue(res["ok"], res)
            music.play.assert_called_once_with(start=(chunkToMs(start) + offsetMs) / 1000)
            self.assertAlmostEqual(timers[-1].interval, (clipMs - offsetMs) / 1000)

            music.reset_mock()
            self.api.playOccurrenceAudio("TWICE", "Doughnut", start, end)  # default = whole clip
            music.play.assert_called_once_with(start=chunkToMs(start) / 1000)
            self.assertAlmostEqual(timers[-1].interval, clipMs / 1000)

    def test_get_line_timing_splits_at_label_row_pauses(self):
        from unittest import mock
        rows = [["J-Hope", 2838, 2880, False, False], ["J-Hope", 2886, 2996, False, False]]
        with mock.patch("gui.vocab_review_api.loadRawLabels", return_value=rows), \
                mock.patch("core.vocab_sync._readLyricEntries", return_value=[]):
            res = self.api.getLineTiming("BTS", "Stay Gold", 2838, 2996, "君を\n優しくいただく|のさ", ["J-Hope"], "Japanese")
        self.assertTrue(res["ok"], res)
        self.assertEqual(res["data"]["stretches"], 2)
        self.assertEqual("".join(p["text"] for p in res["data"]["pieces"]), "君を\n優しくいただくのさ")  # "|" stripped
        words = [p for p in res["data"]["pieces"] if p["isWord"]]
        self.assertEqual(words[0]["fraction"], 0.0)
        self.assertEqual(words[1]["startChunk"], 2886)          # first word after the pause starts at the next row

    def test_play_occurrence_audio_returns_what_the_page_clock_needs(self):
        from unittest import mock
        from core.util_functions import clipStartOffsetMs, KARAOKE_LATENCY_MS

        class FakeTimer:
            def __init__(self, interval, fn): self.daemon = False
            def start(self): pass
            def cancel(self): pass

        start, end = 649, 754
        clipMs = chunkToMs(end) - chunkToMs(start)
        with mock.patch("gui.vocab_review_api.pickBestAudioForStem", return_value="x.mp3"), \
                mock.patch("gui.vocab_review_api.ensureAudioForPlayback", return_value=("x.wav", None)), \
                mock.patch("gui.vocab_review_api.threading.Timer", FakeTimer), \
                mock.patch("gui.vocab_review_api.pygame.mixer.music"), \
                mock.patch.object(self.api, "_ensureMixer"):
            before = time.time() * 1000
            data = self.api.playOccurrenceAudio("TWICE", "Doughnut", start, end, 0.5, True)["data"]
            after = time.time() * 1000
        self.assertTrue(before <= data.pop("startedAtMs") <= after)     # stamped when play() ran, for the page clock
        offsetMs = clipStartOffsetMs(clipMs, 0.5, 0)
        self.assertEqual(data, {"clipStartChunk": start, "offsetMs": offsetMs, "clipMs": clipMs,
                                "playMs": clipMs - offsetMs, "chunkMs": 40, "latencyMs": KARAOKE_LATENCY_MS,
                                "rate": 1.0})

    def test_playback_position_is_none_when_idle_and_pygames_clock_when_playing(self):
        from unittest import mock
        self.assertEqual(self.api.getPlaybackPositionMs(), {"ok": True, "data": None})   # mixer never started
        self.api._mixerReady = True
        with mock.patch("gui.vocab_review_api.pygame.mixer.music") as music:
            music.get_busy.return_value = False
            self.assertIsNone(self.api.getPlaybackPositionMs()["data"])
            music.get_busy.return_value = True
            music.get_pos.return_value = 1234
            self.assertEqual(self.api.getPlaybackPositionMs()["data"], 1234)
            music.get_pos.return_value = -1
            self.assertIsNone(self.api.getPlaybackPositionMs()["data"])

    def test_get_member_color_uses_the_first_singer_from_groups_json(self):
        from unittest import mock
        groups = {"TWICE": {"members": [{"name": "Nayeon", "color": "#ff0000"}, {"name": "Sana", "color": "#00ff00"}]}}
        with mock.patch("gui.vocab_review_api._groupRegistry") as registry:
            registry.groups = groups
            self.assertEqual(self.api.getMemberColor("TWICE", ["Sana", "Nayeon"])["data"], "#00ff00")
            self.assertIsNone(self.api.getMemberColor("TWICE", ["Nobody"])["data"])
            self.assertIsNone(self.api.getMemberColor("TWICE", [])["data"])
            self.assertIsNone(self.api.getMemberColor("Other", ["Sana"])["data"])

    def _tapCard(self, korean=None):
        """A real-shaped Sana card (split 恋, two rows) with the lyric file and labels mocked."""
        from unittest import mock
        from core.lyric_text import READING_OPEN as RO, READING_CLOSE as RC, PAUSE_MARK as M
        rows = [["Sana", 922, 936, False, False], ["Sana", 941, 1045, False, False]]
        raw = korean or ("恋" + RO + "こ" + M + "い" + RC + "をしてから")
        entries = [{"lyricId": "L1", "korean": raw}]
        patches = [mock.patch("gui.vocab_review_api.loadRawLabels", return_value=rows),
                   mock.patch("core.vocab_sync._readLyricEntries", return_value=entries)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        return lambda **kw: self.api.getLineTiming("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"],
                                                   "Japanese", kw.get("lyricId", "L1"), kw.get("unit"))

    def _save(self, taps, lyricId="L1"):
        return self.api.saveTaps("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", lyricId, taps)

    @staticmethod
    def _starts(res):
        return [w["startChunk"] for w in res["data"]["pieces"] if w["isWord"]]

    def test_a_saved_take_is_applied_to_the_line_timing_with_the_measured_lag_removed(self):
        timing = self._tapCard()
        before = timing()["data"]
        self.assertEqual((before["tapStatus"], before["words"]), ("none", ["恋(こ)", "恋(い)", "を", "してから"]))
        self.assertEqual(before["rowStarts"], {0: 922, 1: 941})

        # taps on both row starts were 10 chunks late -> lag 10; the tap on してから (index 3) is corrected by it
        saved = self._save({0: 932, 1: 951, 3: 1010})
        self.assertTrue(saved["ok"], saved)
        self.assertEqual((saved["data"]["lag"], saved["data"]["measured"], saved["data"]["spread"]), (10, True, 0))

        after = timing()["data"]
        self.assertEqual((after["tapStatus"], after["tapLag"]), ("ok", 10))
        starts = {w["text"]: w["startChunk"] for w in after["pieces"] if w["isWord"]}
        self.assertEqual(starts["してから"], 1000)
        self.assertEqual(starts["恋"], 922)                       # row start stays exact
        self.assertTrue(os.path.exists(os.path.join("saved_labels", "TWICE", "taps", "Doughnut_taps.json")))

    def test_without_two_row_start_taps_the_default_lag_is_used_and_reported_as_not_measured(self):
        self._tapCard()
        saved = self._save({3: 1000})
        self.assertFalse(saved["data"]["measured"])
        self.assertEqual(saved["data"]["lag"], 3.75)              # TAP_LAG_MS 150 / 40 ms per chunk
        self.assertEqual(saved["data"]["timing"]["pieces"][-1]["startChunk"], 996.2)

    def test_a_take_goes_stale_when_the_lyric_words_change_and_clearing_removes_it(self):
        from unittest import mock
        timing = self._tapCard()
        self._save({0: 932, 1: 951, 3: 1010})
        with mock.patch("core.vocab_sync._readLyricEntries", return_value=[{"lyricId": "L1", "korean": "恋をしてから"}]):
            stale = timing()["data"]
        self.assertEqual(stale["tapStatus"], "stale")              # the split-kanji annotation was removed
        self.assertNotEqual(stale["pieces"][-1]["startChunk"], 1000)
        self.assertEqual(self._save({})["data"]["timing"]["tapStatus"], "none")

    def test_a_card_with_no_lyric_id_finds_its_take_through_the_matched_lyric(self):
        timing = self._tapCard()
        self._save({0: 932, 1: 951, 3: 1010}, lyricId=None)        # the page had no id; the file lookup supplies L1
        self.assertEqual(timing(lyricId=None)["data"]["tapStatus"], "ok")

    def test_a_slowed_clip_is_cut_by_ffmpeg_played_from_its_start_and_reports_the_rate(self):
        from unittest import mock
        timers = []

        class FakeTimer:
            def __init__(self, interval, fn):
                self.interval = interval
                self.daemon = False
                timers.append(self)
            def start(self): pass
            def cancel(self): pass

        start, end = 649, 754
        clipMs = chunkToMs(end) - chunkToMs(start)
        with mock.patch("gui.vocab_review_api.pickBestAudioForStem", return_value="x.mp3"),                 mock.patch("gui.vocab_review_api.ensureAudioForPlayback", return_value=("x.mp3", None)),                 mock.patch("gui.vocab_review_api.makeSlowClip", return_value="slow.wav") as slow,                 mock.patch("gui.vocab_review_api.threading.Timer", FakeTimer),                 mock.patch("gui.vocab_review_api.pygame.mixer.music") as music,                 mock.patch.object(self.api, "_ensureMixer"):
            data = self.api.playOccurrenceAudio("TWICE", "Doughnut", start, end, 0, True, 0.5)["data"]
        slow.assert_called_once_with("x.mp3", chunkToMs(start) / 1000, clipMs / 1000, 0.5)
        music.load.assert_called_once_with("slow.wav")
        music.play.assert_called_once_with()                       # from the slice's own start: no MP3 seek
        self.assertEqual((data["rate"], data["playMs"]), (0.5, clipMs / 0.5))
        self.assertAlmostEqual(timers[-1].interval, clipMs / 1000 / 0.5)

    def test_atempo_filter_chains_below_half_speed(self):
        from core.util_functions import atempoFilter
        self.assertEqual(atempoFilter(0.75), "atempo=0.75")
        self.assertEqual(atempoFilter(0.5), "atempo=0.5")
        self.assertEqual(atempoFilter(0.35), "atempo=0.5,atempo=0.7")
        for bad in (0.1, 1.5):
            with self.assertRaises(ValueError):
                atempoFilter(bad)

    def test_syllable_takes_are_separate_from_word_takes_and_auto_prefers_syllables(self):
        timing = self._tapCard()
        word = self._save({0: 932, 1: 951, 3: 1010})                           # a word take (unit "word")
        self.assertEqual(timing()["data"]["tapUnit"], "word")
        syl = self.api.saveTaps("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1",
                                {0: 932, 1: 951, 4: 1010}, "syllable", 1.0)
        self.assertTrue(syl["ok"], syl)
        self.assertEqual(syl["data"]["timing"]["words"][:3], ["こ", "い", "お"])
        auto = timing()["data"]
        self.assertEqual((auto["tapUnit"], auto["tapStatus"], auto["unit"]), ("syllable", "ok", "syllable"))
        only = timing(unit="word")["data"]                                       # the word view still has its take
        self.assertEqual((only["tapUnit"], only["unit"]), ("word", "word"))
        # clearing the syllable take falls back to the word take
        self.api.saveTaps("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1", {}, "syllable")
        self.assertEqual(timing()["data"]["tapUnit"], "word")
        self.assertEqual(word["data"]["timing"]["tapStatus"], "ok")

    def test_the_default_lag_is_real_time_so_a_slowed_take_assumes_less_song_time(self):
        self._tapCard()
        res = self.api.saveTaps("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1",
                                {3: 1000}, "word", 0.5)
        self.assertEqual(res["data"]["lag"], 1.88)                               # 150 ms * 0.5 / 40 ms, rounded

    def _nudge(self, anchors, index, delta, unit="word"):
        return self.api.nudgeTake("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1",
                                  anchors, unit, index, delta)

    def test_a_saved_take_also_keeps_the_raw_taps_and_speed(self):
        timing = self._tapCard()
        self._save({0: 932, 1: 951, 3: 1010})
        applied = timing()["data"]
        self.assertEqual(applied["anchors"], {0: 922, 1: 941, 3: 1000})          # lag 10 removed
        self.assertEqual(applied["raw"], {0: 932, 1: 951, 3: 1010})
        self.assertEqual(applied["tapRate"], 1.0)

    def test_preview_returns_the_corrected_anchors_a_nudge_starts_from(self):
        self._tapCard()
        res = self.api.previewTaps("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1",
                                   {0: 932, 1: 951, 3: 1010}, "word", 1.0)
        self.assertEqual(res["data"]["anchors"], {0: 922, 1: 941, 3: 1000})

    def test_nudge_moves_one_beat_without_saving_or_removing_any_lag(self):
        self._tapCard()
        res = self._nudge({0: 922, 1: 941, 3: 1000}, 3, 2)
        self.assertEqual((res["data"]["status"], res["data"]["anchors"][3], res["data"]["beatStart"]), ("moved", 1002, 1002))
        self.assertFalse(os.path.exists(os.path.join("saved_labels", "TWICE", "taps", "Doughnut_taps.json")))
        locked = self._nudge({0: 922, 1: 941, 3: 1000}, 1, 2)                     # い starts the second row
        self.assertEqual((locked["data"]["status"], locked["data"]["anchors"][1]), ("locked", 941))
        reset = self._nudge({0: 922, 1: 941, 3: 1000}, 3, None)
        self.assertEqual((reset["data"]["status"], 3 in reset["data"]["anchors"]), ("reset", False))

    def test_saving_a_corrected_take_writes_the_chunks_as_given_and_keeps_raw_taps(self):
        timing = self._tapCard()
        self._save({0: 932, 1: 951, 3: 1010})                                    # lag 10 measured and removed once
        saved = self.api.saveCorrectedTake("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1",
                                           {0: 922, 1: 941, 3: 1004})
        self.assertTrue(saved["ok"], saved)
        applied = timing()["data"]
        self.assertEqual(applied["anchors"], {0: 922, 1: 941, 3: 1004})          # NOT 994: lag is not taken off twice
        self.assertEqual(applied["raw"], {0: 932, 1: 951, 3: 1010})              # raw taps survive the nudge
        self.assertEqual(applied["tapLag"], 10)

    def test_a_fresh_take_saved_corrected_stores_its_own_raw_taps_lag_and_rate(self):
        timing = self._tapCard()
        saved = self.api.saveCorrectedTake("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1",
                                           {0: 922, 3: 1004}, "word", 0.5, {0: 927, 3: 1009}, 5.0)
        self.assertTrue(saved["ok"], saved)
        applied = timing()["data"]
        self.assertEqual((applied["anchors"], applied["raw"], applied["tapRate"], applied["tapLag"]),
                         ({0: 922, 3: 1004}, {0: 927, 3: 1009}, 0.5, 5.0))

    def test_preview_taps_times_a_take_without_saving_it(self):
        timing = self._tapCard()
        res = self.api.previewTaps("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1",
                                   {0: 932, 1: 951, 3: 1010}, "word", 1.0)
        self.assertTrue(res["ok"], res)
        data = res["data"]
        self.assertEqual((data["lag"], data["measured"]), (10, True))
        starts = {w["text"]: w["startChunk"] for w in data["timing"]["pieces"] if w["isWord"]}
        estimate = {w["text"]: w["startChunk"] for w in data["estimate"]["pieces"] if w["isWord"]}
        self.assertEqual(starts["してから"], 1000)                      # tapped 1010 minus the 10 chunk lag
        self.assertNotEqual(estimate["してから"], 1000)                 # the untouched estimate differs
        self.assertFalse(os.path.exists(os.path.join("saved_labels", "TWICE", "taps")))   # nothing written
        self.assertEqual(timing()["data"]["tapStatus"], "none")

    def test_get_kanji_reading_is_hiragana_without_spaces(self):
        self.assertEqual(self.api.getKanjiReading("恋")["data"], "こい")
        self.assertEqual(self.api.getKanjiReading("手を振って")["data"], "ておふって")   # sung pronunciation

    def test_get_line_timing_reads_a_split_kanji_from_the_lyrics_file(self):
        from unittest import mock
        from core.lyric_text import READING_OPEN as RO, READING_CLOSE as RC, PAUSE_MARK as M
        rows = [["Sana", 922, 936, False, False], ["Sana", 941, 1045, False, False]]
        raw = "恋" + RO + "こ" + M + "い" + RC + "をしてから"
        entries = [{"lyricId": "L1", "korean": raw}]
        with mock.patch("gui.vocab_review_api.loadRawLabels", return_value=rows), \
                mock.patch("core.vocab_sync._readLyricEntries", return_value=entries):
            res = self.api.getLineTiming("TWICE", "Doughnut", 922, 1045, "恋をしてから", ["Sana"], "Japanese", "L1")
        koi = [p for p in res["data"]["pieces"] if p["isWord"]][0]
        self.assertEqual([p["startChunk"] for p in koi["parts"]], [922, 941])
        self.assertEqual("".join(p["text"] for p in res["data"]["pieces"]), "恋をしてから")

    def test_shutdown_stops_the_clip_cancels_the_timer_and_releases_the_mixer(self):
        from unittest import mock

        class FakeTimer:
            cancelled = False
            def cancel(self): FakeTimer.cancelled = True

        self.api._stopTimer = FakeTimer()
        self.api._mixerReady = True
        with mock.patch("gui.vocab_review_api.pygame.mixer") as mixer:
            res = self.api.shutdown()
        self.assertTrue(res["ok"], res)
        self.assertTrue(FakeTimer.cancelled)
        mixer.music.stop.assert_called_once()
        mixer.quit.assert_called_once()
        self.assertFalse(self.api._mixerReady)

    def test_shutdown_is_safe_when_nothing_ever_played(self):
        self.assertEqual(self.api.shutdown(), {"ok": True, "data": None})

    def test_clip_start_offset_lead_is_configurable(self):
        from core.util_functions import clipStartOffsetMs, WORD_LEAD_MS, MIN_TAIL_MS
        d = 12000
        self.assertEqual(clipStartOffsetMs(d, 0.5, 0), 6000)                       # no lead: exactly the word
        self.assertEqual(clipStartOffsetMs(d, 0.5), 6000 - WORD_LEAD_MS)           # default unchanged
        self.assertEqual(clipStartOffsetMs(d, 1.0, 0), d - MIN_TAIL_MS)            # tail guard still applies

    def test_exact_play_from_here_adds_no_further_lead(self):
        from unittest import mock
        from core.util_functions import clipStartOffsetMs, WORD_LEAD_MS

        class FakeTimer:
            def __init__(self, interval, fn): self.interval = interval; self.daemon = False
            def start(self): pass
            def cancel(self): pass

        # Doughnut/Nayeon line 2 starts at row start 649 of the span 631-903: its playFraction is already the
        # row start itself, so playback must begin exactly there (not 300 ms earlier, inside line 1).
        start, end = 631, 903
        clipMs = chunkToMs(end) - chunkToMs(start)
        fraction = (649 - start) / (end - start)
        with mock.patch("gui.vocab_review_api.pickBestAudioForStem", return_value="x.mp3"), \
                mock.patch("gui.vocab_review_api.ensureAudioForPlayback", return_value=("x.wav", None)), \
                mock.patch("gui.vocab_review_api.threading.Timer", FakeTimer), \
                mock.patch("gui.vocab_review_api.pygame.mixer.music") as music, \
                mock.patch.object(self.api, "_ensureMixer"):
            self.assertTrue(self.api.playOccurrenceAudio("TWICE", "Doughnut", start, end, fraction, True)["ok"])
            exactStart = music.play.call_args.kwargs["start"]
            self.api.playOccurrenceAudio("TWICE", "Doughnut", start, end, fraction)          # default: with lead
            leadStart = music.play.call_args.kwargs["start"]
        self.assertAlmostEqual(exactStart, chunkToMs(649) / 1000, delta=0.002)
        self.assertAlmostEqual(exactStart - leadStart, WORD_LEAD_MS / 1000, delta=0.002)

    def test_preview_pause_edit_reports_words_per_row_for_unsaved_text(self):
        from unittest import mock
        from core.lyric_text import EDITOR_PAUSE_GLYPH as G
        rows = [["Tzuyu", 2456, 2473, False, False], ["Tzuyu", 2478, 2579, False, False],
                ["Tzuyu", 2605, 2828, False, False]]
        text = "そ" + G + "ばにいなくても" + G + "\n切れずにリンクしてるね\nMemory の余韻に浸っていたい"
        with mock.patch("gui.vocab_review_api.loadRawLabels", return_value=rows):
            res = self.api.previewPauseEdit("TWICE", "Doughnut", 2456, 2828, ["Tzuyu"], "Japanese", text)
        self.assertTrue(res["ok"], res)
        data = res["data"]
        self.assertEqual((data["rowCount"], data["markers"]), (3, 2))        # every pause marked: rows - 1
        self.assertEqual([r["words"] for r in data["rows"]][:2], ["そ", "ばに いなくても"])
        self.assertEqual(data["rows"][0]["seconds"], 0.7)                      # 17 chunks * 40 ms

    def test_preview_pause_edit_without_a_span_returns_none(self):
        self.assertEqual(self.api.previewPauseEdit("G", "S", 5, None, [], "Japanese", "x"), {"ok": True, "data": None})

    def _inTempLyricsDir(self, entries):
        """Put saved_labels/BTS/Stay Gold_lyrics.json in this test's own tempdir (setUp already chdir'd into
        it, and the API reads relative paths) - never the real saved_labels."""
        import json
        os.makedirs(os.path.join("saved_labels", "BTS"))
        path = os.path.join("saved_labels", "BTS", "Stay Gold_lyrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=4)
        return path

    def test_pause_edit_round_trip_for_any_song(self):
        import json
        from core.lyric_text import PAUSE_MARK, EDITOR_PAUSE_GLYPH
        words = "時計の針さえ\n動きを止めるよ"
        path = self._inTempLyricsDir([{"lyricId": "v1", "korean": "時計の" + PAUSE_MARK + "針さえ\n動きを止めるよ"}])
        got = self.api.getLyricForPauseEdit("BTS", "Stay Gold", "v1", words)
        self.assertTrue(got["ok"], got)
        self.assertEqual(got["data"]["text"], "時計の" + EDITOR_PAUSE_GLYPH + "針さえ\n動きを止めるよ")   # visible stand-in
        res = self.api.savePauseMarkers("BTS", "Stay Gold", "v1", words,
                                        "時計の" + EDITOR_PAUSE_GLYPH + "針さえ\n動きを" + EDITOR_PAUSE_GLYPH + "止めるよ")
        self.assertTrue(res["ok"], res)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(json.load(f)[0]["korean"], "時計の" + PAUSE_MARK + "針さえ\n動きを" + PAUSE_MARK + "止めるよ")

    def test_pause_edit_refuses_wording_changes_with_a_readable_message(self):
        words = "時計の針さえ"
        path = self._inTempLyricsDir([{"lyricId": "v1", "korean": words}])
        with open(path, "rb") as f:
            before = f.read()
        res = self.api.savePauseMarkers("BTS", "Stay Gold", "v1", words, "時間の針さえ")
        self.assertFalse(res["ok"])
        self.assertIn("Only pause markers", res["error"])
        with open(path, "rb") as f:
            self.assertEqual(f.read(), before)

    def test_pause_edit_reports_a_missing_lyric_or_song(self):
        self._inTempLyricsDir([{"lyricId": "v1", "korean": "時計の針さえ"}])
        self.assertFalse(self.api.getLyricForPauseEdit("BTS", "Stay Gold", "zzz", "違う")["ok"])
        self.assertFalse(self.api.getLyricForPauseEdit("TWICE", "Doughnut", "v1", "時計の針さえ")["ok"])

    def _timingWithMarkerInLyricsFile(self, lyricId):
        from unittest import mock
        from core.lyric_text import PAUSE_MARK
        # Stay Gold card 21: the pause after 君を is inside a written line, so the author marked it.
        rows = [["J-Hope", 2727, 2832, False, False], ["J-Hope", 2838, 2880, False, False],
                ["J-Hope", 2886, 2996, False, False]]
        entries = [{"lyricId": "f7d2", "korean": "君を" + PAUSE_MARK + "優しく\nいただくのさ"}]
        with mock.patch("gui.vocab_review_api.loadRawLabels", return_value=rows), \
                mock.patch("core.vocab_sync._readLyricEntries", return_value=entries):
            # The occurrence line in the DB is clean - no marker.
            return self.api.getLineTiming("BTS", "Stay Gold", 2798, 2996, "君を優しく\nいただくのさ",
                                          ["J-Hope"], "Japanese", lyricId)

    def test_get_line_timing_recovers_the_pause_marker_by_lyric_id(self):
        res = self._timingWithMarkerInLyricsFile("f7d2")
        self.assertTrue(res["ok"], res)
        pieces = res["data"]["pieces"]
        self.assertEqual("".join(p["text"] for p in pieces), "君を優しく\nいただくのさ")   # marker never shows
        starts = {p["text"]: p["startChunk"] for p in pieces if p["isWord"]}
        self.assertEqual((starts["君を"], starts["優しく"]), (2798, 2838))

    def test_get_line_timing_recovers_the_pause_marker_by_matching_the_clean_text(self):
        res = self._timingWithMarkerInLyricsFile(None)
        starts = {p["text"]: p["startChunk"] for p in res["data"]["pieces"] if p["isWord"]}
        self.assertEqual(starts["優しく"], 2838)

    def test_get_line_timing_without_a_span_returns_none(self):
        self.assertEqual(self.api.getLineTiming("G", "S", 100, None, "x", [], "Japanese"), {"ok": True, "data": None})
        self.assertEqual(self.api.getLineTiming("G", "S", 100, 100, "x", [], "Japanese"), {"ok": True, "data": None})


if __name__ == "__main__":
    unittest.main()
