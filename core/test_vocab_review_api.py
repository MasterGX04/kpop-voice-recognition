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


if __name__ == "__main__":
    unittest.main()
