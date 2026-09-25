"""
Tests for core/japanese_utils.py's romaji/hiragana reading conversion, focused on the
macron-Hepburn romaji change (kanjiLineToReading: "romaji" now collapses a genuine chouonpu
long vowel into a single ā/ī/ū/ē/ō instead of the ambiguous "ou"-vs-"oo" digraph spelling).

Stdlib unittest only, matching core/test_kanji_reference.py's style. Run with:
    python -m unittest core.test_japanese_utils -v
"""

import unittest

from core.japanese_utils import kanjiLineToReading


class MacronRomajiTests(unittest.TestCase):
    def test_ou_style_long_vowel_gets_a_macron(self):
        # 東京 - the chouonpu comes from the more common "ou" digraph spelling.
        self.assertEqual(kanjiLineToReading("東京", "romaji"), "tōkyō")

    def test_oo_style_long_vowel_also_gets_the_same_macron(self):
        # 遠く - the chouonpu here comes from the rarer "oo" digraph spelling (とおく, not とうく) -
        # must collapse to the identical ō as the "ou" case above, since they're the same sound.
        self.assertEqual(kanjiLineToReading("遠く", "romaji"), "tōku")

    def test_short_long_vowel_word_gets_a_macron(self):
        self.assertEqual(kanjiLineToReading("そう", "romaji"), "sō")

    def test_ei_long_vowel_also_gets_a_macron(self):
        # 先生 - a long e written as えい (not a chouonpu at all in the dictionary spelling), but
        # UniDic's phonetic `pron` reading marks it as an actual long vowel too (センセー) - same
        # sound, so it gets the same macron treatment as ー-based long vowels.
        self.assertEqual(kanjiLineToReading("先生", "romaji"), "sensē")

    def test_sokuon_consonant_doubling_is_unaffected(self):
        # 学校 has both a real chouonpu (こ->こう, collapses to ō) AND an unrelated doubled
        # CONSONANT from the small っ (がっこう -> "gakkou", the "kk") - only the vowel doublet
        # should collapse; the consonant doubling must survive untouched.
        self.assertEqual(kanjiLineToReading("学校", "romaji"), "gakkō")

    def test_two_distinct_morae_that_merely_look_similar_are_not_collapsed(self):
        # 思う (omou) and 追う (ou) each end in a genuine two-mora "vowel + u" sequence, NOT a
        # chouonpu long vowel - confirmed via testing that UniDic's `pron` never marks these with
        # "ー" (unlike a real long vowel), so pykakasi romanizes them as two DIFFERENT adjacent
        # vowel letters that must NOT be merged into a macron.
        self.assertEqual(kanjiLineToReading("思う", "romaji"), "omou")
        self.assertEqual(kanjiLineToReading("追う", "romaji"), "ou")

    def test_hiragana_output_is_unaffected_by_the_macron_change(self):
        # The macron collapsing is a romaji-only concern - hiragana output must keep showing the
        # real, conventional kana spelling (とうきょう), never a macron (not valid kana at all).
        self.assertEqual(kanjiLineToReading("東京", "hiragana"), "とうきょう")
        # とうく, not the real orthographic とおく - a pre-existing simplification unrelated to
        # this change (_expandLongVowelMark defaults to the far more common "u"-row spelling for
        # an o-class chouonpu; see its own docstring).
        self.assertEqual(kanjiLineToReading("遠く", "hiragana"), "とうく")

    def test_ordinary_line_without_long_vowels_is_unaffected(self):
        self.assertEqual(kanjiLineToReading("見透かす気持ちを", "romaji"), "misukasu kimochi o")

    def test_sokuon_split_across_a_token_boundary_still_geminates(self):
        # fugashi splits あって into two tokens, あっ + て - a naively-inserted word-boundary
        # space between them breaks kakasi's gemination (it needs the consonant immediately
        # after the sokuon), mis-romanizing it as literal "atsu te" instead of "atte".
        self.assertEqual(kanjiLineToReading("あっても", "romaji"), "atte mo")

    def test_sokuon_within_a_single_token_is_unaffected(self):
        # たった is tokenized as one word, so this must keep working the same as before.
        self.assertEqual(kanjiLineToReading("たった", "romaji"), "tatta")

    def test_watashi_reading_override(self):
        # UniDic's default reading for 私 is the formal watakushi, not the near-universal
        # watashi lyrics actually use - see _READING_OVERRIDES.
        self.assertEqual(kanjiLineToReading("私", "romaji"), "watashi")
        self.assertEqual(kanjiLineToReading("私たち", "romaji"), "watashi tachi")

    def test_ashita_reading_override(self):
        # Real bug report: 明日への衝動 romanized as "asu e no shōdō" - UniDic's default reading
        # for 明日 is the more formal/literary あす, not the near-universal casual あした lyrics
        # actually use. 明日香 ("Asuka", a name) tokenizes as one fused, differently-spelled token
        # and must be unaffected.
        self.assertEqual(kanjiLineToReading("明日への衝動", "romaji"), "ashita e no shōdō")
        self.assertEqual(kanjiLineToReading("明日香", "romaji"), "asuka")

    def test_kokoro_suffix_after_a_katakana_loanword_is_not_shin(self):
        # Real bug report: ファイティング心 ("fighting spirit") romanized 心 as "shin" (on'yomi),
        # not the native "gokoro" (kun'yomi, rendaku'd) a hybrid loanword+心 coinage actually
        # takes. Confirmed via testing this is a genuine UniDic ambiguity, not a labeling error in
        # the lyrics: 心 as a bound suffix (接尾辞) not already in UniDic's fixed compound
        # dictionary reads しん regardless of whether the whole compound is really Sino-Japanese
        # (好奇心) or a native/hybrid coinage (ファイティング心) - the preceding word's script
        # (katakana = loanword, never genuinely Sino-Japanese) is the deciding signal here.
        self.assertEqual(
            kanjiLineToReading("弱さとファイティング心で 強さをゲット重ねたら", "romaji"),
            "yowa sa to faiteingu gokoro de tsuyo sa o getto kasane tara",
        )

    def test_kokoro_suffix_after_kanji_stays_onyomi_shin(self):
        # 好奇心 (kōkishin, "curiosity") is a genuine Sino-Japanese compound that happens to hit
        # the exact same 接尾辞 code path as ファイティング心 (好奇 isn't in UniDic's fixed
        # whole-word dictionary as a single 好奇心 token) - must NOT be overridden to "gokoro".
        self.assertEqual(kanjiLineToReading("好奇心", "romaji"), "kōki shin")

    def test_embedded_english_doubled_vowel_is_not_macronized(self):
        # Real bug report: an English word/interjection sitting in the same lyric line as
        # Japanese text (e.g. "Ooh yeah") got its own "oo" collapsed into a macron the same as a
        # real chouonpu, since the macron-collapse used to run on the whole joined romaji string
        # rather than only the chunks kakasi actually converted FROM Japanese script.
        self.assertEqual(kanjiLineToReading("Good morning", "romaji"), "Good morning")
        self.assertEqual(kanjiLineToReading("Cool", "romaji"), "Cool")
        self.assertEqual(kanjiLineToReading("好きだよ Ooh yeah", "romaji"), "suki da yo Ooh yeah")

    def test_japanese_macron_still_applies_next_to_untouched_english(self):
        self.assertEqual(kanjiLineToReading("東京 Good", "romaji"), "tōkyō Good")

    def test_kokoro_as_a_full_fixed_compound_is_unaffected(self):
        # 遊び心/女心/親心 are already correct straight from UniDic's dictionary as single fused
        # tokens (not the 心-as-接尾辞 path at all) - confirms the override doesn't need to (and
        # doesn't) touch these.
        self.assertEqual(kanjiLineToReading("遊び心", "romaji"), "asobigokoro")
        self.assertEqual(kanjiLineToReading("女心", "romaji"), "onnagokoro")


if __name__ == "__main__":
    unittest.main()
