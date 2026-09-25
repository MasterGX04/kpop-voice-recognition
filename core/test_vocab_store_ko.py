"""
Tests for core/vocab_store_ko.py. Stdlib unittest, tempdir-based. Run with:
    python -m unittest core.test_vocab_store_ko -v
"""

import os
import shutil
import tempfile
import unittest

from core import vocab_store_ko
from core.vocab_db import getConnection


def _entry(lemma, hanjaCandidates=None, meaning=None):
    return {
        "surface": lemma, "lemma": lemma,
        "meaning": meaning or {"status": "found", "pos": "noun", "gloss": ["fire"]},
        "hanjaCandidates": hanjaCandidates or [],
    }


class UpsertVocabTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_ambiguous_word_stores_every_hanja_candidate(self):
        # 화 -> six unrelated real Hanja (see core/korean_hanja.py: lookupHanja()'s own docstring).
        candidates = [
            {"hanja": h, "gloss": [g], "pos": "noun", "pinyin": p}
            for h, g, p in [("火", "fire", "huo3"), ("禍", "misfortune", "huo4"),
                             ("和", "harmony", "he2"), ("化", "change", "hua4"),
                             ("畫", "picture", "hua4"), ("靴", "shoe", "xue1")]
        ]
        vocabId, isNew = vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=candidates))
        self.assertTrue(isNew)

        conn = getConnection()
        rows = conn.execute(
            "SELECT hanja_form FROM vocab_ko_hanja WHERE vocab_ko_id = ?", (vocabId,)
        ).fetchall()
        conn.close()
        self.assertEqual({r[0] for r in rows}, {"火", "禍", "和", "化", "畫", "靴"})

    def test_native_word_has_no_hanja_candidates_and_meaning_defaults_unknown(self):
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("너무", hanjaCandidates=[]))

        conn = getConnection()
        hanjaCount = conn.execute(
            "SELECT COUNT(*) FROM vocab_ko_hanja WHERE vocab_ko_id = ?", (vocabId,)
        ).fetchone()[0]
        meaningState = conn.execute(
            "SELECT meaning_state FROM srs_card_ko WHERE vocab_ko_id = ?", (vocabId,)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(hanjaCount, 0)
        self.assertEqual(meaningState, 0)

    def test_sino_korean_word_defaults_meaning_known(self):
        candidates = [{"hanja": "學校", "gloss": ["school"], "pos": "noun", "pinyin": "xue2 xiao4"}]
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("학교", hanjaCandidates=candidates))

        conn = getConnection()
        meaningState = conn.execute(
            "SELECT meaning_state FROM srs_card_ko WHERE vocab_ko_id = ?", (vocabId,)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(meaningState, 1)

    def test_re_upsert_replaces_hanja_candidates_rather_than_accumulating(self):
        vocabId1, _ = vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=[
            {"hanja": "火", "gloss": ["fire"], "pos": "noun", "pinyin": "huo3"}
        ]))
        vocabId2, isNew = vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=[
            {"hanja": "火", "gloss": ["fire"], "pos": "noun", "pinyin": "huo3"},
            {"hanja": "禍", "gloss": ["misfortune"], "pos": "noun", "pinyin": "huo4"},
        ]))
        self.assertEqual(vocabId1, vocabId2)
        self.assertFalse(isNew)

        conn = getConnection()
        count = conn.execute(
            "SELECT COUNT(*) FROM vocab_ko_hanja WHERE vocab_ko_id = ?", (vocabId1,)
        ).fetchone()[0]
        conn.close()
        self.assertEqual(count, 2)

    def test_add_occurrence_is_idempotent(self):
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("학교"))
        inserted1 = vocab_store_ko.addOccurrence(
            vocabId, "ITZY", "TestSong", ["Yeji"], "학교에 가자", "lyric-1", 5, 15
        )
        inserted2 = vocab_store_ko.addOccurrence(
            vocabId, "ITZY", "TestSong", ["Yeji"], "학교에 가자", "lyric-1", 5, 15
        )
        self.assertTrue(inserted1)
        self.assertFalse(inserted2)
        self.assertEqual(len(vocab_store_ko.getOccurrences(vocabId)), 1)

    def test_list_all_vocab_ignores_due_status(self):
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("학교"))
        vocab_store_ko.submitReview(vocabId, "reading", "good")

        self.assertEqual(vocab_store_ko.getDueCards("reading"), [])
        allWords = vocab_store_ko.listAllVocab()
        self.assertEqual([w["lemma"] for w in allWords], ["학교"])

    def test_update_meaning_overwrites_gloss_and_flips_status_to_found(self):
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("너무", meaning={"status": "not_found"}))
        vocab_store_ko.updateMeaning(vocabId, ["too much (manually filled in)"])

        card = vocab_store_ko.listAllVocab()[0]
        self.assertEqual(card["meaning"]["status"], "found")
        self.assertEqual(card["meaning"]["gloss"], ["too much (manually filled in)"])

    def test_delete_vocab_removes_word_hanja_and_occurrences(self):
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=[
            {"hanja": "火", "gloss": ["fire"], "pos": "noun", "pinyin": "huo3"}
        ]))
        vocab_store_ko.addOccurrence(vocabId, "ITZY", "TestSong", ["Yeji"], "화가 났어", "l1", 1, 2)

        vocab_store_ko.deleteVocab(vocabId)

        self.assertEqual(vocab_store_ko.listAllVocab(), [])
        self.assertEqual(vocab_store_ko.getOccurrences(vocabId), [])

    def test_keep_only_hanja_candidate_deletes_the_rest(self):
        candidates = [
            {"hanja": h, "gloss": [g], "pos": "noun", "pinyin": p}
            for h, g, p in [("火", "fire", "huo3"), ("禍", "misfortune", "huo4"), ("和", "harmony", "he2")]
        ]
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=candidates))
        card = vocab_store_ko.listAllVocab()[0]
        keepId = next(c["hanjaId"] for c in card["hanjaCandidates"] if c["hanja"] == "禍")

        vocab_store_ko.keepOnlyHanjaCandidate(vocabId, keepId)

        remaining = vocab_store_ko.listAllVocab()[0]["hanjaCandidates"]
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["hanja"], "禍")

    def test_keep_only_hanja_candidate_drops_stale_cognate_link(self):
        from core import vocab_store_ja, vocab_link

        vocab_store_ja.upsertVocab({
            "surface": "禍", "reading": "か", "lemma": "禍", "lemmaReading": "か",
            "category": "onyomi",
            "chineseCognate": {"status": "confirmed", "traditional": "禍", "pinyin": "huo4", "gloss": ["misfortune"]},
            "japaneseMeaning": {"status": "found", "pos": ["n"], "gloss": ["misfortune"]},
            "mandarinPinyin": {"traditional": "禍", "pinyin": "huo4"},
        })
        candidates = [
            {"hanja": h, "gloss": [g], "pos": "noun", "pinyin": p}
            for h, g, p in [("火", "fire", "huo3"), ("禍", "misfortune", "huo4")]
        ]
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=candidates))
        vocab_link.syncCognateLinks()
        self.assertEqual(len(vocab_link.getLinkedWords("ko", vocabId)), 1)

        card = vocab_store_ko.listAllVocab()[0]
        keepId = next(c["hanjaId"] for c in card["hanjaCandidates"] if c["hanja"] == "火")
        vocab_store_ko.keepOnlyHanjaCandidate(vocabId, keepId)

        self.assertEqual(vocab_link.getLinkedWords("ko", vocabId), [])

    def test_clear_hanja_candidates_removes_a_single_wrong_candidate(self):
        # 해 ("sun/day", native) incorrectly matched against 害 ("harm") - a real Sino-Korean
        # reading, but only for compounds like 재해/유해, not for bare native 해. Not "ambiguous"
        # (only one candidate), just wrong - keepOnlyHanjaCandidate has nothing to contrast
        # against here, so this needs its own "remove everything" escape hatch.
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("해", hanjaCandidates=[
            {"hanja": "害", "gloss": ["harm"], "pos": "noun", "pinyin": "hai4"}
        ]))

        vocab_store_ko.clearHanjaCandidates(vocabId)

        card = vocab_store_ko.listAllVocab()[0]
        self.assertEqual(card["hanjaCandidates"], [])

    def test_clear_hanja_candidates_drops_stale_cognate_link(self):
        from core import vocab_store_ja, vocab_link

        vocab_store_ja.upsertVocab({
            "surface": "害", "reading": "がい", "lemma": "害", "lemmaReading": "がい",
            "category": "onyomi",
            "chineseCognate": {"status": "confirmed", "traditional": "害", "pinyin": "hai4", "gloss": ["harm"]},
            "japaneseMeaning": {"status": "found", "pos": ["n"], "gloss": ["harm"]},
            "mandarinPinyin": {"traditional": "害", "pinyin": "hai4"},
        })
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("해", hanjaCandidates=[
            {"hanja": "害", "gloss": ["harm"], "pos": "noun", "pinyin": "hai4"}
        ]))
        vocab_link.syncCognateLinks()
        self.assertEqual(len(vocab_link.getLinkedWords("ko", vocabId)), 1)

        vocab_store_ko.clearHanjaCandidates(vocabId)

        self.assertEqual(vocab_link.getLinkedWords("ko", vocabId), [])

    def test_rescan_does_not_undo_keep_only_hanja_candidate(self):
        # Regression for a real user report: they resolved every ambiguous word via "Keep only
        # this" in the Review screen, then re-ran Compile (which just calls upsertVocab again with
        # the same raw lookup result for every word), and all of their resolutions silently
        # reverted back to the full multi-candidate list.
        candidates = [
            {"hanja": h, "gloss": [g], "pos": "noun", "pinyin": p}
            for h, g, p in [("火", "fire", "huo3"), ("禍", "misfortune", "huo4")]
        ]
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=candidates))
        card = vocab_store_ko.listAllVocab()[0]
        keepId = next(c["hanjaId"] for c in card["hanjaCandidates"] if c["hanja"] == "禍")
        vocab_store_ko.keepOnlyHanjaCandidate(vocabId, keepId)

        # Simulate a rescan: the exact same entry (both original candidates), as
        # analyzeKoreanSelection() would produce it again from the unchanged lyric text.
        vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=candidates))

        remaining = vocab_store_ko.listAllVocab()[0]["hanjaCandidates"]
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["hanja"], "禍")

    def test_rescan_does_not_undo_clear_hanja_candidates(self):
        candidates = [{"hanja": "害", "gloss": ["harm"], "pos": "noun", "pinyin": "hai4"}]
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("해", hanjaCandidates=candidates))
        vocab_store_ko.clearHanjaCandidates(vocabId)

        vocab_store_ko.upsertVocab(_entry("해", hanjaCandidates=candidates))

        card = vocab_store_ko.listAllVocab()[0]
        self.assertEqual(card["hanjaCandidates"], [])

    def test_rescan_does_not_clobber_a_manually_edited_meaning(self):
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("너무", meaning={"status": "not_found"}))
        vocab_store_ko.updateMeaning(vocabId, ["too much (manually filled in)"])

        vocab_store_ko.upsertVocab(_entry("너무", meaning={"status": "not_found"}))

        card = vocab_store_ko.listAllVocab()[0]
        self.assertEqual(card["meaning"]["gloss"], ["too much (manually filled in)"])

    def test_rescan_still_refreshes_hanja_for_an_unresolved_word(self):
        # An untouched word must still pick up newly-discovered candidates on rescan (e.g. lyrics
        # were edited, or the dictionary lookup improved) - locking is opt-in via
        # keepOnlyHanjaCandidate()/clearHanjaCandidates(), not the default for every word.
        vocabId, _ = vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=[
            {"hanja": "火", "gloss": ["fire"], "pos": "noun", "pinyin": "huo3"}
        ]))
        vocab_store_ko.upsertVocab(_entry("화", hanjaCandidates=[
            {"hanja": "火", "gloss": ["fire"], "pos": "noun", "pinyin": "huo3"},
            {"hanja": "禍", "gloss": ["misfortune"], "pos": "noun", "pinyin": "huo4"},
        ]))

        remaining = vocab_store_ko.listAllVocab()[0]["hanjaCandidates"]
        self.assertEqual(len(remaining), 2)


if __name__ == "__main__":
    unittest.main()
