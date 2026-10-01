"""
Tests for core/cloze.py: buildClozeCard(). Stdlib unittest only. Run with:
    python -m unittest core.test_cloze -v
"""

import unittest

from core.cloze import buildClozeCard, orderOccurrencesForCloze, BLANK_MARKER


def _occ(lyricLine):
    return {"lyricLine": lyricLine}


class BuildClozeCardTests(unittest.TestCase):
    def test_japanese_happy_path_blanks_the_matched_surface(self):
        card = buildClozeCard("Japanese", "食べる", "食べたくない")
        self.assertEqual(card["blankedLine"], BLANK_MARKER + "たくない")
        self.assertEqual(card["answerSurface"], "食べ")
        self.assertEqual(card["gloss"], "to eat")

    def test_korean_happy_path_blanks_the_matched_surface(self):
        # 다른 (surface) is 다르다's ("different") own inflected form in this real line - the
        # match is by lemma, not by the vocab row's stored `surface`, which may be a different
        # inflected form entirely.
        card = buildClozeCard("Korean", "다르다", "태생부터 다른 사람")
        self.assertEqual(card["blankedLine"], "태생부터 " + BLANK_MARKER + " 사람")
        self.assertEqual(card["answerSurface"], "다른")

    def test_rest_of_line_is_preserved_character_for_character(self):
        line = "태생부터 다른 사람"
        card = buildClozeCard("Korean", "다르다", line)
        blanked = card["blankedLine"]
        self.assertEqual(blanked.replace(BLANK_MARKER, card["answerSurface"]), line)

    def test_no_matching_lemma_returns_none(self):
        # A lemma that never appears in this line at all - the caller (getClozeCardDetail) is
        # expected to try another real occurrence rather than treat this as an error.
        self.assertIsNone(buildClozeCard("Japanese", "存在しない単語", "食べたくない"))

    def test_function_word_lemma_is_never_matched_as_the_answer(self):
        # たい ("want to") is a real lemma in this line, but it's a function-role entry - cloze
        # only ever blanks the tracked CONTENT word itself, never a grammar particle/ending, even
        # if its lemma happens to be passed in by mistake.
        self.assertIsNone(buildClozeCard("Japanese", "たい", "食べたくない"))


class OrderOccurrencesForClozeTests(unittest.TestCase):
    """
    Real user feedback: a uniformly random occurrence line for cloze made it feel like "which
    song is this from" rather than "do you know this word" - a word only tested via a long, dense
    rap verse needs the whole passage recalled before the blank is even attemptable. Biasing
    toward shorter lines targets exactly that, without ever discarding a word's only occurrence.
    """

    def test_shorter_lines_always_come_before_longer_ones(self):
        short1, short2 = _occ("短い一行"), _occ("もう一つ短い")
        long1, long2 = _occ("これはとても長くて複雑な一行の歌詞です"), _occ("さらにもっと長くて難解な二行目の歌詞です")
        occurrences = [long1, short1, long2, short2]

        ordered = orderOccurrencesForCloze(occurrences)

        shortSet = {id(short1), id(short2)}
        firstHalf = ordered[:2]
        self.assertTrue(all(id(o) in shortSet for o in firstHalf))

    def test_a_word_with_only_long_occurrences_still_returns_all_of_them(self):
        # Deprioritized, never discarded - a word with no short line at all must still be
        # quizzable from what it has.
        occurrences = [_occ("長い一行目です"), _occ("これも長い二行目です")]
        ordered = orderOccurrencesForCloze(occurrences)
        self.assertEqual(len(ordered), 2)
        self.assertCountEqual(ordered, occurrences)

    def test_empty_list_returns_empty_list(self):
        self.assertEqual(orderOccurrencesForCloze([]), [])

    def test_single_occurrence_is_returned_as_is(self):
        occ = _occ("一行だけ")
        self.assertEqual(orderOccurrencesForCloze([occ]), [occ])


if __name__ == "__main__":
    unittest.main()
