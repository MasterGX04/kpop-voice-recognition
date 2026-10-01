"""
Tests for core/japanese_hangul.py - Japanese Sino-vocabulary -> Korean hangul cognate prediction.
Uses the real bundled dictionaries (CC-CEDICT, Korean Wiktionary index, Kiwi) rather than mocks,
since the whole point is how those gates interact; cases below are the empirically-found ones.

Run with: python -m unittest core.test_japanese_hangul -v
"""

import unittest

from core.japanese_hangul import lookupHangulCognate


def _hangul(lemma, category="onyomi"):
    result = lookupHangulCognate(lemma, category)
    return result["hangul"] if result else None


class ConvertsChineseOriginWordsTests(unittest.TestCase):
    def test_common_sino_words(self):
        self.assertEqual(_hangul("生活"), "생활")
        self.assertEqual(_hangul("大學"), "대학")
        self.assertEqual(_hangul("時間"), "시간")

    def test_shinjitai_is_converted_via_traditional_first(self):
        # Raw hanja.translate("予告") gives 여고 and ("証明") 정명 - wrong; via Traditional it's right.
        self.assertEqual(_hangul("予告"), "예고")
        self.assertEqual(_hangul("証明"), "증명")
        self.assertEqual(_hangul("経済"), "경제")

    def test_japan_coined_word_chinese_later_adopted_via_variant_character(self):
        # 化粧室 is not in CC-CEDICT, Taiwan's 化妝室 is; Korean uses 화장실 (Wiktionary keys 化粧室).
        for spelling in ("化粧室", "化妝室"):
            result = lookupHangulCognate(spelling, "onyomi")
            self.assertEqual(result["hangul"], "화장실")
            self.assertEqual(result["tier"], "attested")

    def test_suru_verb_uses_its_kanji_stem(self):
        self.assertEqual(_hangul("生活する", "mixed"), "생활")

    def test_tier_is_attested_when_wiktionary_confirms(self):
        self.assertEqual(lookupHangulCognate("生活", "onyomi")["tier"], "attested")
        self.assertTrue(lookupHangulCognate("生活", "onyomi")["gloss"])

    def test_tier_is_plausible_when_only_kiwi_knows_it(self):
        result = lookupHangulCognate("大胆", "onyomi")
        self.assertEqual((result["hangul"], result["tier"], result["gloss"]), ("대담", "plausible", []))


class RejectsNonCognatesTests(unittest.TestCase):
    def test_native_japanese_categories_never_convert(self):
        self.assertIsNone(lookupHangulCognate("言葉", "jukujigo"))
        self.assertIsNone(lookupHangulCognate("時計", "jukujigo"))
        self.assertIsNone(lookupHangulCognate("名前", "kunyomi"))

    def test_japan_only_coinage_chinese_never_adopted(self):
        for word in ("本当", "写真", "不動産", "丁度"):
            self.assertIsNone(lookupHangulCognate(word, "onyomi"), word)

    def test_chinese_attested_but_not_a_korean_word(self):
        # In CEDICT, but 장합 / 편당 / 면강 are not Korean words (장합 is only a person's name).
        for word in ("場合", "便當", "勉強"):
            self.assertIsNone(lookupHangulCognate(word, "onyomi"), word)

    def test_single_kanji_is_not_a_word(self):
        self.assertIsNone(lookupHangulCognate("愛", "onyomi"))

    def test_kana_containing_or_non_kanji_lemmas(self):
        self.assertIsNone(lookupHangulCognate("気持ち", "mixed"))
        self.assertIsNone(lookupHangulCognate("ありがとう", "onyomi"))
        self.assertIsNone(lookupHangulCognate("", "onyomi"))


if __name__ == "__main__":
    unittest.main()
