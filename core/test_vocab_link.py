"""
Tests for core/vocab_link.py's cross-language cognate linking. Stdlib unittest, tempdir-based.
Run with:
    python -m unittest core.test_vocab_link -v
"""

import os
import shutil
import tempfile
import unittest

from core import vocab_store_ja, vocab_store_ko, vocab_link


def _jaEntry(lemma, cognateForm, cognatePinyin):
    return {
        "surface": lemma, "reading": "がっこう", "lemma": lemma, "lemmaReading": "がっこう",
        "category": "onyomi",
        "chineseCognate": {"status": "confirmed", "traditional": cognateForm, "pinyin": cognatePinyin, "gloss": ["school"]},
        "japaneseMeaning": {"status": "found", "pos": ["n"], "gloss": ["school"]},
        "mandarinPinyin": {"traditional": cognateForm, "pinyin": cognatePinyin},
    }


def _koEntry(lemma, hanjaForm=None, pinyin=None):
    candidates = [{"hanja": hanjaForm, "gloss": ["school"], "pos": "noun", "pinyin": pinyin}] if hanjaForm else []
    return {
        "surface": lemma, "lemma": lemma,
        "meaning": {"status": "found", "pos": "noun", "gloss": ["school"]},
        "hanjaCandidates": candidates,
    }


class CognateLinkTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_shared_cognate_form_links_ja_and_ko(self):
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("学校", "學校", "xue2 xiao4"))
        koId, _ = vocab_store_ko.upsertVocab(_koEntry("학교", "學校", "xue2 xiao4"))

        linkCount = vocab_link.syncCognateLinks()
        self.assertEqual(linkCount, 1)

        jaLinks = vocab_link.getLinkedWords("ja", jaId)
        self.assertEqual(len(jaLinks), 1)
        self.assertEqual(jaLinks[0]["vocabId"], koId)
        self.assertEqual(jaLinks[0]["lemma"], "학교")

        koLinks = vocab_link.getLinkedWords("ko", koId)
        self.assertEqual(koLinks[0]["lemma"], "学校")

    def test_unrelated_words_are_not_linked(self):
        vocab_store_ja.upsertVocab(_jaEntry("時間", "時間", "shi2 jian1"))
        vocab_store_ko.upsertVocab(_koEntry("너무"))

        self.assertEqual(vocab_link.syncCognateLinks(), 0)

    def test_sync_is_idempotent(self):
        vocab_store_ja.upsertVocab(_jaEntry("学校", "學校", "xue2 xiao4"))
        vocab_store_ko.upsertVocab(_koEntry("학교", "學校", "xue2 xiao4"))

        vocab_link.syncCognateLinks()
        secondCount = vocab_link.syncCognateLinks()
        self.assertEqual(secondCount, 1)  # not doubled


class CognateBridgeTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_linked_pair_gives_full_three_way_bridge_from_japanese_side(self):
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("学校", "學校", "xue2 xiao4"))
        koId, _ = vocab_store_ko.upsertVocab(_koEntry("학교", "學校", "xue2 xiao4"))
        vocab_link.syncCognateLinks()

        bridge = vocab_link.getCognateBridge("ja", jaId)
        self.assertEqual(bridge["cognateForm"], "學校")
        self.assertEqual(bridge["japanese"], {"vocabId": jaId, "lemma": "学校", "lemmaReading": "がっこう"})
        self.assertEqual(bridge["chinese"], {"status": "confirmed", "pinyin": "xue2 xiao4", "gloss": ["school"]})
        self.assertEqual(bridge["korean"]["vocabId"], koId)
        self.assertEqual(bridge["korean"]["candidates"], [
            {"hanjaForm": "學校", "pinyin": "xue2 xiao4", "gloss": ["school"], "linked": True}
        ])

    def test_linked_pair_gives_full_three_way_bridge_from_korean_side(self):
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("学校", "學校", "xue2 xiao4"))
        koId, _ = vocab_store_ko.upsertVocab(_koEntry("학교", "學校", "xue2 xiao4"))
        vocab_link.syncCognateLinks()

        bridge = vocab_link.getCognateBridge("ko", koId)
        self.assertEqual(bridge["cognateForm"], "學校")
        self.assertEqual(bridge["japanese"], {"vocabId": jaId, "lemma": "学校", "lemmaReading": "がっこう"})
        self.assertEqual(bridge["chinese"], {"status": "confirmed", "pinyin": "xue2 xiao4", "gloss": ["school"]})
        self.assertEqual(bridge["korean"]["vocabId"], koId)

    def test_unlinked_japanese_word_still_shows_its_own_confirmed_cognate(self):
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("時間", "時間", "shi2 jian1"))

        bridge = vocab_link.getCognateBridge("ja", jaId)
        self.assertEqual(bridge["cognateForm"], "時間")
        self.assertEqual(bridge["chinese"]["pinyin"], "shi2 jian1")
        self.assertIsNone(bridge["korean"])

    def test_unlinked_sino_japanese_word_gets_predicted_hangul(self):
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("生活", "生活", "sheng1 huo2"))
        bridge = vocab_link.getCognateBridge("ja", jaId)
        self.assertIsNone(bridge["korean"])
        self.assertEqual(bridge["predictedKorean"]["hangul"], "생활")

    def test_linked_word_shows_real_korean_not_a_prediction(self):
        jaId, _ = vocab_store_ja.upsertVocab(_jaEntry("学校", "學校", "xue2 xiao4"))
        vocab_store_ko.upsertVocab(_koEntry("학교", "學校", "xue2 xiao4"))
        vocab_link.syncCognateLinks()
        self.assertIsNone(vocab_link.getCognateBridge("ja", jaId)["predictedKorean"])

    def test_native_word_has_no_predicted_hangul(self):
        entry = _jaEntry("言葉", "言葉", "yan2 ye4")
        entry["category"] = "jukujigo"
        jaId, _ = vocab_store_ja.upsertVocab(entry)
        self.assertIsNone(vocab_link.getCognateBridge("ja", jaId)["predictedKorean"])

    def test_native_japanese_word_with_no_cognate_has_no_chinese_or_korean_leg(self):
        entry = {
            "surface": "離す", "reading": "はなす", "lemma": "離す", "lemmaReading": "はなす",
            "category": "kunyomi", "chineseCognate": None,
            "japaneseMeaning": {"status": "found", "pos": ["v"], "gloss": ["to separate"]},
            "mandarinPinyin": {"traditional": "離", "pinyin": "li2"},
        }
        jaId, _ = vocab_store_ja.upsertVocab(entry)

        bridge = vocab_link.getCognateBridge("ja", jaId)
        self.assertIsNone(bridge["cognateForm"])
        self.assertIsNone(bridge["chinese"])
        self.assertIsNone(bridge["korean"])

    def test_unlinked_korean_word_shows_its_own_candidates_with_no_fabricated_chinese_leg(self):
        koId, _ = vocab_store_ko.upsertVocab(_koEntry("학교", "學校", "xue2 xiao4"))

        bridge = vocab_link.getCognateBridge("ko", koId)
        self.assertIsNone(bridge["cognateForm"])
        self.assertIsNone(bridge["japanese"])
        self.assertIsNone(bridge["chinese"])
        self.assertEqual(bridge["korean"]["candidates"], [
            {"hanjaForm": "學校", "pinyin": "xue2 xiao4", "gloss": ["school"], "linked": False}
        ])

    def test_ambiguous_korean_word_marks_only_the_actually_linked_candidate(self):
        koEntry = {
            "surface": "화", "lemma": "화",
            "meaning": {"status": "found", "pos": "noun", "gloss": ["fire"]},
            "hanjaCandidates": [
                {"hanja": "火", "gloss": ["fire"], "pos": "noun", "pinyin": "huo3"},
                {"hanja": "和", "gloss": ["harmony"], "pos": "noun", "pinyin": "he2"},
            ],
        }
        koId, _ = vocab_store_ko.upsertVocab(koEntry)
        vocab_store_ja.upsertVocab(_jaEntry("火", "火", "huo3"))
        vocab_link.syncCognateLinks()

        bridge = vocab_link.getCognateBridge("ko", koId)
        self.assertEqual(bridge["cognateForm"], "火")
        linkedFlags = {c["hanjaForm"]: c["linked"] for c in bridge["korean"]["candidates"]}
        self.assertEqual(linkedFlags, {"火": True, "和": False})


if __name__ == "__main__":
    unittest.main()
