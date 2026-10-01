"""
Tests for core/vocab_sync.py - the compile/scan function that ingests Kanji/Hanja vocab from
saved_labels/*/*_lyrics.json into the SQLite store. Stdlib unittest, tempdir-based like
core.test_kanji_reference.CrossSongIndexTests. Run with:
    python -m unittest core.test_vocab_sync -v
"""

import codecs
import json
import os
import shutil
import tempfile
import unittest

from core import vocab_sync, vocab_store_ja, vocab_store_ko


class ScanSongForVocabTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def _writeLyricsFile(self, group, song, entries):
        path = f"saved_labels/{group}/{song}_lyrics.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with codecs.open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)

    def test_missing_lyrics_file_is_a_no_op(self):
        result = vocab_sync.scanSongForVocab("TWICE", "NeverSaved")
        self.assertEqual(result, {"jaAdded": 0, "koAdded": 0, "occurrencesLinked": 0})

    def test_japanese_entry_is_scanned_into_vocab_ja(self):
        self._writeLyricsFile("TWICE", "SongA", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"],
             "lyricId": "a1", "startChunk": 10, "linkedLabel": {"member": "Nayeon", "startChunk": 15, "endChunk": 40}},
        ])

        result = vocab_sync.scanSongForVocab("TWICE", "SongA")
        self.assertGreater(result["jaAdded"], 0)

        due = vocab_store_ja.getDueCards("reading", limit=50)
        lemmas = {c["lemma"] for c in due}
        self.assertIn("時間", lemmas)

    def test_korean_entry_is_scanned_into_vocab_ko(self):
        self._writeLyricsFile("ITZY", "SongB", [
            {"language": "Korean", "korean": "학교에 가고 싶어", "memberName": ["Yeji"],
             "lyricId": "b1", "startChunk": 5, "linkedLabel": None},
        ])

        result = vocab_sync.scanSongForVocab("ITZY", "SongB")
        self.assertGreater(result["koAdded"], 0)

        due = vocab_store_ko.getDueCards("reading", limit=50)
        lemmas = {c["lemma"] for c in due}
        self.assertIn("학교", lemmas)

    def test_unlinked_lyric_gets_inferred_span_and_manual_link_still_wins(self):
        os.makedirs("saved_labels/TWICE", exist_ok=True)
        with codecs.open("saved_labels/TWICE/SongL_labels.json", "w", encoding="utf-8") as f:
            json.dump([["Nayeon", 111, 200, False, False], ["Jihyo", 511, 600, False, False]], f)
        self._writeLyricsFile("TWICE", "SongL", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"], "startChunk": 100},
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Jihyo"], "lyricId": "m1",
             "startChunk": 500, "linkedLabel": {"member": "Jihyo", "startChunk": 511, "endChunk": 600}},
        ])
        vocab_sync.scanSongForVocab("TWICE", "SongL")
        chunks = {(o["startChunk"], o["endChunk"]) for c in vocab_store_ja.getDueCards("reading", limit=50)
                  if c["lemma"] == "時間" for o in vocab_store_ja.getOccurrences(c["vocabId"])}
        self.assertEqual(chunks, {(111, 200), (511, 600)})

    def test_rescanning_unlinked_lyrics_does_not_duplicate_and_refreshes_spans(self):
        self._writeLyricsFile("TWICE", "SongU", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"], "startChunk": 100},
        ])
        vocab_sync.scanSongForVocab("TWICE", "SongU")
        vocab_sync.scanSongForVocab("TWICE", "SongU")
        card = next(c for c in vocab_store_ja.getDueCards("reading", limit=50) if c["lemma"] == "時間")
        occurrences = vocab_store_ja.getOccurrences(card["vocabId"], limit=10)
        self.assertEqual(len(occurrences), 1)
        self.assertEqual((occurrences[0]["startChunk"], occurrences[0]["endChunk"]), (111, 311))

    def test_unbounded_hand_linked_span_is_capped(self):
        os.makedirs("saved_labels/TWICE", exist_ok=True)
        with codecs.open("saved_labels/TWICE/SongC_labels.json", "w", encoding="utf-8") as f:
            json.dump([["Nayeon", 111, 130, False, False], ["Nayeon", 140, 5000, False, False]], f)
        self._writeLyricsFile("TWICE", "SongC", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"], "lyricId": "c1",
             "startChunk": 100, "linkedLabel": {"member": "Nayeon", "startChunk": 111, "endChunk": 130}},
        ])
        vocab_sync.scanSongForVocab("TWICE", "SongC")
        card = next(c for c in vocab_store_ja.getDueCards("reading", limit=50) if c["lemma"] == "時間")
        occ = vocab_store_ja.getOccurrences(card["vocabId"])[0]
        self.assertEqual(occ["endChunk"] - occ["startChunk"], 250)

    def test_rescanning_the_same_song_does_not_duplicate(self):
        self._writeLyricsFile("TWICE", "SongA", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"],
             "lyricId": "a1", "startChunk": 10, "linkedLabel": None},
        ])

        vocab_sync.scanSongForVocab("TWICE", "SongA")
        result2 = vocab_sync.scanSongForVocab("TWICE", "SongA")

        self.assertEqual(result2["jaAdded"], 0)
        self.assertEqual(result2["occurrencesLinked"], 0)

    def test_scan_all_songs_covers_every_group(self):
        self._writeLyricsFile("TWICE", "SongA", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"], "lyricId": "a1"},
        ])
        self._writeLyricsFile("ITZY", "SongB", [
            {"language": "Korean", "korean": "학교에 가고 싶어", "memberName": ["Yeji"], "lyricId": "b1"},
        ])

        summary = vocab_sync.scanAllSongsForVocab()
        self.assertIn("TWICE/SongA", summary)
        self.assertIn("ITZY/SongB", summary)

    def test_song_html_report_is_regenerated_from_db(self):
        self._writeLyricsFile("TWICE", "SongA", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"], "lyricId": "a1"},
        ])
        vocab_sync.scanSongForVocab("TWICE", "SongA")
        self.assertTrue(os.path.exists("saved_labels/TWICE/SongA_kanji_reference.html"))

    def test_cross_song_index_is_regenerated_from_db_after_scan_all(self):
        self._writeLyricsFile("TWICE", "SongA", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"], "lyricId": "a1"},
        ])
        vocab_sync.scanAllSongsForVocab()
        self.assertTrue(os.path.exists("kanji_reference/word_index.json"))
        self.assertTrue(os.path.exists("kanji_reference/by_song.html"))


if __name__ == "__main__":
    unittest.main()
