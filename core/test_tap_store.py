"""
Tests for core/tap_store.py. Uses a temp directory as the project root, never the real saved_labels.
Run with: venv/Scripts/python.exe -m unittest core.test_tap_store -v
"""

import os
import tempfile
import unittest

from core import tap_store
from core.tap_store import TapStoreError, getTake, saveTake, tapsPath, wordsHash

WORDS = ["手", "を", "振って", "背を"]


class TapStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def test_file_lives_in_a_taps_subfolder_of_the_group(self):
        path = tapsPath("TWICE", "Doughnut", self.root)
        self.assertEqual(os.path.normpath(path), os.path.normpath(
            os.path.join(self.root, "saved_labels", "TWICE", "taps", "Doughnut_taps.json")))

    def test_no_file_means_no_take(self):
        self.assertEqual(getTake("TWICE", "Doughnut", "L1", WORDS, self.root)["status"], "none")

    def test_round_trip_creates_the_folder_and_returns_integer_indexed_anchors(self):
        saveTake("TWICE", "Doughnut", "L1", WORDS, {0: 631.04, 2: 675.2}, lag=4.5, root=self.root)
        take = getTake("TWICE", "Doughnut", "L1", WORDS, self.root)
        self.assertEqual(take["status"], "ok")
        self.assertEqual(take["anchors"], {0: 631.0, 2: 675.2})
        self.assertEqual(take["lag"], 4.5)
        self.assertTrue(take["takenAt"])

    def test_changed_words_make_the_take_stale_and_apply_nothing(self):
        saveTake("TWICE", "Doughnut", "L1", WORDS, {0: 631}, root=self.root)
        take = getTake("TWICE", "Doughnut", "L1", ["手", "を", "振", "って", "背を"], self.root)
        self.assertEqual((take["status"], take["anchors"]), ("stale", {}))

    def test_other_lyrics_in_the_file_are_untouched_and_an_empty_take_clears_only_its_own(self):
        saveTake("TWICE", "Doughnut", "L1", WORDS, {0: 631}, root=self.root)
        saveTake("TWICE", "Doughnut", "L2", ["a", "b"], {1: 900}, root=self.root)
        saveTake("TWICE", "Doughnut", "L1", WORDS, {}, root=self.root)
        self.assertEqual(getTake("TWICE", "Doughnut", "L1", WORDS, self.root)["status"], "none")
        self.assertEqual(getTake("TWICE", "Doughnut", "L2", ["a", "b"], self.root)["anchors"], {1: 900.0})

    def test_an_unreadable_file_is_refused_and_never_overwritten(self):
        path = tapsPath("TWICE", "Doughnut", self.root)
        os.makedirs(os.path.dirname(path))
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not json")
        with self.assertRaises(TapStoreError):
            saveTake("TWICE", "Doughnut", "L1", WORDS, {0: 631}, root=self.root)
        with open(path, encoding="utf-8") as f:
            self.assertEqual(f.read(), "{not json")

    def test_a_lyric_without_an_id_cannot_be_saved(self):
        with self.assertRaises(TapStoreError):
            saveTake("TWICE", "Doughnut", None, WORDS, {0: 1}, root=self.root)
        self.assertEqual(getTake("TWICE", "Doughnut", None, WORDS, self.root)["status"], "none")

    def test_hash_depends_on_words_and_order(self):
        self.assertEqual(wordsHash(WORDS), wordsHash(list(WORDS)))
        self.assertNotEqual(wordsHash(WORDS), wordsHash(WORDS[::-1]))
        self.assertNotEqual(wordsHash(["ab", "c"]), wordsHash(["a", "bc"]))

    def test_no_temp_file_is_left_behind(self):
        saveTake("TWICE", "Doughnut", "L1", WORDS, {0: 631}, root=self.root)
        self.assertEqual(os.listdir(os.path.dirname(tapsPath("TWICE", "Doughnut", self.root))), ["Doughnut_taps.json"])

    def test_raw_taps_and_rate_are_stored_beside_the_applied_chunks(self):
        saveTake("TWICE", "Doughnut", "L1", WORDS, {0: 631.0, 2: 676.0}, lag=4.5, root=self.root,
                 raw={0: 635.5, 2: 680.123}, rate=0.5)
        take = getTake("TWICE", "Doughnut", "L1", WORDS, self.root)
        self.assertEqual((take["raw"], take["rate"]), ({0: 635.5, 2: 680.12}, 0.5))
        self.assertEqual(take["anchors"], {0: 631.0, 2: 676.0})

    def test_a_take_without_raw_taps_still_loads(self):
        saveTake("TWICE", "Doughnut", "L1", WORDS, {0: 631.0}, lag=4.5, root=self.root)
        take = getTake("TWICE", "Doughnut", "L1", WORDS, self.root)
        self.assertEqual((take["status"], take["raw"], take["rate"]), ("ok", None, None))


if __name__ == "__main__":
    unittest.main()
