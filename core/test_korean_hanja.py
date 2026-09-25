"""
Tests for Phase 1 of core/korean_hanja.py.

Stdlib unittest only, matching core/test_kanji_reference.py's style. Run with:
    python -m unittest core.test_korean_hanja -v
"""

import unittest

from core.korean_hanja import lookupHanja


class LookupHanjaTests(unittest.TestCase):
    def test_unambiguous_sino_korean_word(self):
        candidates = lookupHanja("학교")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["hanja"], "學校")
        self.assertIn("school", candidates[0]["gloss"][0])

    def test_hanja_candidates_carry_mandarin_pinyin(self):
        # Added per direct user request - the same Mandarin pinyin mnemonic already built for
        # Japanese Kanji (core/kanji_reference.py: lookupChineseCognate), reused here since
        # Korean Hanja are themselves Traditional Chinese character forms.
        candidates = lookupHanja("학교")
        self.assertEqual(candidates[0]["pinyin"], "xue2 xiao4")

    def test_variant_slash_form_still_gets_pinyin(self):
        # 畫/畵 (one of 화's six candidates) lists two orthographic variants separated by "/" -
        # must still resolve to real pinyin via the first variant, not crash or return garbage.
        candidates = lookupHanja("화")
        huaCandidate = next(c for c in candidates if c["hanja"] == "畫/畵")
        self.assertEqual(huaCandidate["pinyin"], "hua4")

    def test_six_way_homograph(self):
        # .claude/KOREAN_HANJA_PLAN.md's canonical example: 화 alone corresponds to at least six
        # unrelated real Hanja words sharing the same modern pronunciation - the whole reason this
        # feature always returns a list rather than picking one.
        candidates = lookupHanja("화")
        hanjaForms = {c["hanja"] for c in candidates}
        self.assertEqual(hanjaForms, {"火", "禍", "和", "化", "畫/畵", "靴"})

    def test_native_word_returns_empty_list(self):
        # 목소리 ("voice") is a purely native Korean word with no Hanja origin at all - must
        # return [] rather than fabricating a character-level guess.
        self.assertEqual(lookupHanja("목소리"), [])
        self.assertEqual(lookupHanja("너무"), [])

    def test_word_absent_from_index_returns_empty_list(self):
        self.assertEqual(lookupHanja("asdfqwerzxcv"), [])

    def test_mostly_native_word_with_one_rare_hanja_homograph(self):
        # 사랑 is overwhelmingly used as the native word for "love" (no Hanja), but a distinct,
        # much rarer Sino-Korean word (舍廊, "salon/hall in a traditional Korean house") happens to
        # share the exact same spelling - a real case of the same honesty principle as the 화
        # six-way split, just with only one real Hanja candidate instead of six. This must NOT be
        # silently dropped just because the common sense of the word is native.
        candidates = lookupHanja("사랑")
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["hanja"], "舍廊")

    def test_tag_filters_out_unrelated_homograph_under_a_different_pos(self):
        # Real bug found via direct user testing: 이 as the native proximal determiner ("this",
        # Kiwi tag MM) was showing six unrelated Hanja candidates - 二(two)/李(surname)/理/利/釐/伊 -
        # that are each real words, but only ever surface under a completely different part of
        # speech (a numeral, a surname, ...), never as this determiner. Passing the token's own
        # tag must narrow the result to nothing, since the "det" entry for 이 has no Hanja at all.
        self.assertEqual(lookupHanja("이", "MM"), [])
        # Without a tag (no grammatical context to disambiguate with), the original unfiltered
        # behavior is preserved - every real candidate across every homograph.
        self.assertGreaterEqual(len(lookupHanja("이")), 6)

    def test_tag_keeps_real_homographs_sharing_the_same_pos(self):
        # 화's six candidates are all real for an ordinary noun-tagged 화 (NNG) - unlike 이's case,
        # these six genuinely share the same part of speech as each other (three "noun", three
        # "suffix"), so tag-filtering must narrow to the matching subset, not wipe out real
        # candidates just because more than one exists.
        nounOnly = lookupHanja("화", "NNG")
        self.assertEqual({c["hanja"] for c in nounOnly}, {"火", "禍", "和"})
        suffixOnly = lookupHanja("화", "XSN")
        self.assertEqual({c["hanja"] for c in suffixOnly}, {"化", "畫/畵", "靴"})


if __name__ == "__main__":
    unittest.main()
