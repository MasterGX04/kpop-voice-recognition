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
