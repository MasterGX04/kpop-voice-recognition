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


if __name__ == "__main__":
    unittest.main()
