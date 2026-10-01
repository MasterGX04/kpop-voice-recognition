"""
Tests for core/korean_vocab.py: analyzeKoreanSelection().

Stdlib unittest only, matching this project's test style. Run with:
    python -m unittest core.test_korean_vocab -v
"""

import unittest

from core.korean_vocab import analyzeKoreanSelection


class AnalyzeKoreanSelectionTests(unittest.TestCase):
    def test_fused_ending_word_gets_a_real_meaning_not_a_joined_lemma(self):
        # Regression test for the real bug this function used to have: it re-derived each word's
        # meaning from scratch via lookupKoreanMeaning(entry["lemma"], entry["tag"]), which broke
        # the moment a merged content+ending group's lemma became a "+"-joined compound (see
        # core.korean_grammar_breakdown._mergeGroup's own comment) - 다른 (다르다 "different" +
        # its adnominal ending) used to be stored under the unlookupable lemma "다르다+ᆫ".
        results = analyzeKoreanSelection("태생부터 다른 사람")
        dareun = next(r for r in results if r["surface"] == "다른")
        self.assertEqual(dareun["lemma"], "다르다")
        self.assertEqual(dareun["meaning"]["status"], "found")
        self.assertIn("different", dareun["meaning"]["gloss"][0])

    def test_bieup_irregular_derived_verb_gets_its_base_adjectives_meaning(self):
        results = analyzeKoreanSelection("우릴 부러워하네")
        bureowo = next(r for r in results if r["surface"] == "부러워하")
        self.assertEqual(bureowo["lemma"], "부럽다")
        self.assertEqual(bureowo["meaning"]["status"], "found")

    def test_function_words_are_excluded_from_vocab(self):
        results = analyzeKoreanSelection("우릴 부러워하네")
        self.assertNotIn("을", [r["surface"] for r in results])
        self.assertNotIn("네", [r["surface"] for r in results])


if __name__ == "__main__":
    unittest.main()
