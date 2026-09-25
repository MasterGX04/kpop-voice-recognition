"""
Tests for core/vocab_store_ja.py. Stdlib unittest, tempdir-based like
core.test_kanji_reference.SongVocabPersistenceTests. Run with:
    python -m unittest core.test_vocab_store_ja -v
"""

import os
import shutil
import tempfile
import unittest

from core import vocab_store_ja
from core.vocab_db import getConnection


def _entry(lemma, category="onyomi", cognate=None):
    return {
        "surface": lemma, "reading": "じかん", "lemma": lemma, "lemmaReading": "じかん",
        "category": category, "chineseCognate": cognate,
        "japaneseMeaning": {"status": "found", "pos": ["n"], "gloss": ["time"]},
        "mandarinPinyin": {"traditional": lemma, "pinyin": "shi2 jian1"},
    }


class UpsertVocabTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_new_word_creates_vocab_and_srs_card(self):
        vocabId, isNew = vocab_store_ja.upsertVocab(_entry("時間"))
        self.assertTrue(isNew)

        due = vocab_store_ja.getDueCards("reading", limit=10)
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["lemma"], "時間")

    def test_confirmed_cognate_defaults_meaning_known(self):
        cognate = {"status": "confirmed", "traditional": "時間", "pinyin": "shi2 jian1", "gloss": ["time"]}
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間", cognate=cognate))

        conn = getConnection()
        row = conn.execute(
            "SELECT meaning_state, reading_state FROM srs_card_ja WHERE vocab_ja_id = ?", (vocabId,)
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 1)  # meaning known
        self.assertEqual(row[1], 0)  # reading still unseen

    def test_not_attested_cognate_still_gets_a_pinyin_but_meaning_defaults_unknown(self):
        cognate = {"status": "not_attested", "traditional": "殘業", "pinyinFallback": "can2 ye4"}
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("残業", category="onyomi", cognate=cognate))

        conn = getConnection()
        row = conn.execute(
            "SELECT meaning_state, cognate_pinyin FROM srs_card_ja s "
            "JOIN vocab_ja v ON v.id = s.vocab_ja_id WHERE s.vocab_ja_id = ?", (vocabId,)
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 0)
        self.assertEqual(row[1], "can2 ye4")

    def test_kunyomi_word_without_cognate_defaults_meaning_unknown(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("出会う", category="kunyomi", cognate=None))

        conn = getConnection()
        row = conn.execute(
            "SELECT meaning_state FROM srs_card_ja WHERE vocab_ja_id = ?", (vocabId,)
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 0)

    def test_upsert_by_lemma_replaces_rather_than_duplicates(self):
        idA, isNewA = vocab_store_ja.upsertVocab(_entry("時間"))
        idB, isNewB = vocab_store_ja.upsertVocab(_entry("時間"))
        self.assertTrue(isNewA)
        self.assertFalse(isNewB)
        self.assertEqual(idA, idB)

    def test_add_occurrence_is_idempotent(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        inserted1 = vocab_store_ja.addOccurrence(
            vocabId, "TWICE", "TestSong", ["Nayeon"], "時間がない", "lyric-1", 10, 20
        )
        inserted2 = vocab_store_ja.addOccurrence(
            vocabId, "TWICE", "TestSong", ["Nayeon"], "時間がない", "lyric-1", 10, 20
        )
        self.assertTrue(inserted1)
        self.assertFalse(inserted2)

        occurrences = vocab_store_ja.getOccurrences(vocabId)
        self.assertEqual(len(occurrences), 1)
        self.assertEqual(occurrences[0]["singers"], ["Nayeon"])

    def test_submit_review_moves_card_out_of_due_queue(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        self.assertEqual(len(vocab_store_ja.getDueCards("reading")), 1)

        vocab_store_ja.submitReview(vocabId, "reading", "good")

        self.assertEqual(len(vocab_store_ja.getDueCards("reading")), 0)

    def test_list_all_vocab_ignores_due_status(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        vocab_store_ja.submitReview(vocabId, "reading", "good")
        vocab_store_ja.submitReview(vocabId, "meaning", "good")

        self.assertEqual(vocab_store_ja.getDueCards("reading"), [])
        allWords = vocab_store_ja.listAllVocab()
        self.assertEqual([w["lemma"] for w in allWords], ["時間"])

    def test_update_meaning_overwrites_gloss_and_flips_status_to_found(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("残業", cognate=None))
        vocab_store_ja.updateMeaning(vocabId, ["overtime work (manually filled in)"])

        card = vocab_store_ja.listAllVocab()[0]
        self.assertEqual(card["meaning"]["status"], "found")
        self.assertEqual(card["meaning"]["gloss"], ["overtime work (manually filled in)"])

    def test_rescan_does_not_clobber_a_manually_edited_meaning(self):
        # Regression for a real user report: they hand-corrected a meaning, then re-ran Compile
        # (which just calls upsertVocab again with the same raw lookup result), and the manual
        # edit silently reverted because upsertVocab used to overwrite meaning_json unconditionally.
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("残業", cognate=None))
        vocab_store_ja.updateMeaning(vocabId, ["overtime work (manually filled in)"])

        # Simulate a rescan: the exact same entry, as analyzeSelection() would produce it again.
        vocab_store_ja.upsertVocab(_entry("残業", cognate=None))

        card = vocab_store_ja.listAllVocab()[0]
        self.assertEqual(card["meaning"]["gloss"], ["overtime work (manually filled in)"])

    def test_rescan_still_refreshes_meaning_for_an_unlocked_word(self):
        # A word nobody has manually edited yet must still pick up a freshly re-analyzed meaning
        # (e.g. if the underlying dictionary lookup logic improves) - locking is opt-in via
        # updateMeaning(), not the default for every word.
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        entryWithNewMeaning = _entry("時間")
        entryWithNewMeaning["japaneseMeaning"] = {"status": "found", "pos": ["n"], "gloss": ["updated meaning"]}
        vocab_store_ja.upsertVocab(entryWithNewMeaning)

        card = vocab_store_ja.listAllVocab()[0]
        self.assertEqual(card["meaning"]["gloss"], ["updated meaning"])

    def test_delete_vocab_removes_word_and_its_occurrences(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        vocab_store_ja.addOccurrence(vocabId, "TWICE", "TestSong", ["Nayeon"], "時間がない", "l1", 1, 2)

        vocab_store_ja.deleteVocab(vocabId)

        self.assertEqual(vocab_store_ja.listAllVocab(), [])
        self.assertEqual(vocab_store_ja.getOccurrences(vocabId), [])


if __name__ == "__main__":
    unittest.main()
