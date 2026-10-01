"""
Tests for core/karaoke_timing.py. Pure functions, no disk I/O. Japanese cases need fugashi/unidic
(the project venv has them): run with  venv/Scripts/python.exe -m unittest core.test_karaoke_timing -v

Fixture rows are the real BTS "Stay Gold" J-Hope rows around chunk 2635 (a 3-line card sung as three
back-to-back rows, then a 2-row card with a short pause), copied rather than read from the live data.
"""

import unittest

from core.karaoke_timing import (kanaMorae, englishSyllables, segmentLine, buildStretches,
                                 timeLine, REST_BEATS)
from core.lyric_text import PAUSE_MARK as M

STAY_GOLD_ROWS = [
    ["J-Hope", 2635, 2677, False, False], ["Jungkook", 2658, 2677, True, False],
    ["J-Hope", 2682, 2722, False, False], ["Jungkook", 2704, 2722, True, False],
    ["J-Hope", 2727, 2832, False, False], ["Jungkook", 2752, 2832, True, False],
    ["J-Hope", 2832, 2837, False, True],                      # ad-lib, not lyric text
    ["J-Hope", 2838, 2880, False, False], ["Jungkook", 2838, 2880, True, False],
    ["J-Hope", 2879, 2888, False, True],
    ["Jungkook", 2886, 2992, True, False], ["J-Hope", 2886, 2996, False, False],
]
CARD_20 = "気づかれないように\n近づいてく Slowly\n予告するよ Baby 無防備な"
CARD_21 = "君を優しく\nいただくのさ\n君の深いところ now…"


def _words(result):
    return [p for p in result["pieces"] if p["isWord"]]


def _text(result):
    return "".join(p["text"] for p in result["pieces"])


class BeatCountTests(unittest.TestCase):
    def test_kana_morae(self):
        self.assertEqual(kanaMorae("キミ"), 2)
        self.assertEqual(kanaMorae("ヤサシク"), 4)
        self.assertEqual(kanaMorae("きゃく"), 2)        # small ゃ fuses with き
        self.assertEqual(kanaMorae("ガッコー"), 4)      # ッ and ー are beats

    def test_english_syllables(self):
        self.assertEqual([englishSyllables(w) for w in ("Baby", "Slowly", "Moon", "Light", "now", "love")],
                         [2, 2, 1, 1, 1, 1])


class SegmentTests(unittest.TestCase):
    def test_pieces_reassemble_to_the_exact_text(self):
        for text in (CARD_20, CARD_21, "魅惑的な Moon Light\n今宵も眠らない"):
            self.assertEqual("".join(p["text"] for p in segmentLine(text, "Japanese")), text)

    def test_kanji_word_is_weighted_by_reading_not_characters(self):
        words = {p["text"]: p["weight"] for p in segmentLine("君を優しく", "Japanese") if p["isWord"]}
        self.assertEqual(words, {"君を": 3, "優しく": 4})   # 君を = キミ+オ, not 2 characters

    def test_particle_and_prefix_join_their_word(self):
        words = [p["text"] for p in segmentLine("予告するよ Baby 無防備な", "Japanese") if p["isWord"]]
        self.assertEqual(words, ["予告", "するよ", "Baby", "無防備な"])

    def test_korean_words_weighted_by_syllable_blocks(self):
        words = [(p["text"], p["weight"]) for p in segmentLine("안녕 하세요", "Korean") if p["isWord"]]
        self.assertEqual(words, [("안녕", 2), ("하세요", 3)])
        self.assertEqual("".join(p["text"] for p in segmentLine("안녕 하세요", "Korean")), "안녕 하세요")


class StretchTests(unittest.TestCase):
    def test_ad_lib_rows_and_other_members_are_ignored(self):
        self.assertEqual(buildStretches(STAY_GOLD_ROWS, ["J-Hope"], 2635, 2996),
                         [(2635, 2677), (2682, 2722), (2727, 2832), (2838, 2880), (2886, 2996)])

    def test_overlapping_rows_of_the_cards_members_fuse(self):
        # J-Hope + Jungkook both credited: Jungkook's backing 2886-2992 sits inside J-Hope's row.
        self.assertEqual(buildStretches(STAY_GOLD_ROWS, ["J-Hope", "Jungkook"], 2838, 2996),
                         [(2838, 2880), (2886, 2996)])

    def test_every_gap_is_a_boundary_only_real_overlap_fuses(self):
        # Rows the author drew are exact: touching and 1-chunk gaps stay separate; only an overlapping
        # (duplicate/nested) row of the same member joins the stretch it overlaps.
        rows = [["A", 100, 150, False, False], ["A", 150, 200, False, False], ["A", 201, 250, False, False],
                ["A", 240, 300, False, False]]
        self.assertEqual(buildStretches(rows, ["A"], 100, 300), [(100, 150), (150, 200), (201, 300)])

    def test_rows_are_clipped_to_the_span(self):
        rows = [["A", 50, 150, False, False], ["A", 160, 400, False, False]]
        self.assertEqual(buildStretches(rows, ["A"], 100, 200), [(100, 150), (160, 200)])

    def test_no_usable_row_gives_the_whole_span(self):
        self.assertEqual(buildStretches([["B", 100, 200, False, False]], ["A"], 100, 200), [(100, 200)])
        self.assertEqual(buildStretches([], None, 100, 200), [(100, 200)])


class TimeLineTests(unittest.TestCase):
    def test_three_lines_three_rows_each_line_starts_at_its_row(self):
        result = timeLine(CARD_20, "Japanese", 2635, 2832, STAY_GOLD_ROWS, ["J-Hope"])
        self.assertEqual(result["stretches"], 3)
        first = {w["text"]: w["startChunk"] for w in _words(result)}
        self.assertEqual(first["気づかれない"], 2635)
        self.assertEqual(first["近づいてく"], 2682)
        self.assertEqual(first["予告"], 2727)
        self.assertEqual(_text(result), CARD_20)

    def test_no_word_starts_inside_a_pause(self):
        result = timeLine(CARD_21, "Japanese", 2838, 2996, STAY_GOLD_ROWS, ["J-Hope", "Jungkook"])
        for w in _words(result):
            self.assertFalse(2880 <= w["startChunk"] < 2886, w)
            self.assertFalse(2880 < w["endChunk"] < 2886, w)
        self.assertEqual(_text(result), CARD_21)

    def test_card_that_starts_mid_row_with_a_pause_inside_a_line_needs_a_marker(self):
        # Stay Gold card 21 as confirmed by the author: 君を is sung at 2798 at the END of the
        # 2727-2832 row (the card's startChunk 2787 + 11, deliberately mid-row), pause, 優しく alone in
        # 2838-2880, pause, then いただくのさ + 君の深いところ now sung straight through 2886-2996.
        # The pause after 君を is inside a written line, so the author marks it.
        marked = "君を" + M + "優しく\nいただくのさ\n君の深いところ now…"
        result = timeLine(marked, "Japanese", 2798, 2992, STAY_GOLD_ROWS, ["J-Hope", "Jungkook"])
        self.assertEqual(result["stretches"], 3)
        starts = {w["text"]: w["startChunk"] for w in _words(result)}
        self.assertEqual(starts["君を"], 2798)
        self.assertEqual(starts["優しく"], 2838)
        self.assertEqual(starts["いただくのさ"], 2886)
        self.assertGreater(starts["君の"], 2886)          # same stretch as いただくのさ: no pause between
        self.assertEqual(_text(result), CARD_21)          # the marker never shows in the text

    def test_without_the_marker_the_line_break_prior_wins(self):
        # Why the marker exists: unmarked, 君を優しく is kept together (cut at the written line break).
        result = timeLine(CARD_21, "Japanese", 2798, 2992, STAY_GOLD_ROWS, ["J-Hope", "Jungkook"])
        starts = {w["text"]: w["startChunk"] for w in _words(result)}
        self.assertNotEqual(starts["優しく"], 2838)

    def test_one_continuous_row_spreads_words_by_beats(self):
        rows = [["RM", 100, 400, False, False]]
        result = timeLine("君を優しく", "Japanese", 100, 400, rows, ["RM"])
        kimi, yasashiku = _words(result)
        self.assertEqual((kimi["startChunk"], kimi["fraction"]), (100, 0.0))
        self.assertAlmostEqual(yasashiku["startChunk"], 100 + 300 * 3 / 7, places=1)   # 3 of 7 beats in

    def test_no_rows_still_gives_monotonic_fractions_in_range(self):
        result = timeLine(CARD_21, "Japanese", 1000, 1200, [], None)
        fractions = [w["fraction"] for w in _words(result)]
        self.assertEqual(fractions, sorted(fractions))
        self.assertTrue(all(0 <= f <= 1 for f in fractions))
        self.assertEqual(fractions[0], 0.0)

    def test_line_break_beats_a_slightly_better_proportion_mid_line(self):
        # Two rows 40/60 of the time, text 4/6 beats split by a line break: cut goes at the break.
        rows = [["A", 100, 140, False, False], ["A", 150, 210, False, False]]
        result = timeLine("ほしぞら\nなみだのうた", "Japanese", 100, 210, rows, ["A"])
        words = _words(result)
        afterBreak = next(p for i, p in enumerate(result["pieces"])
                          if p["isWord"] and "\n" in result["pieces"][i - 1]["text"])
        self.assertEqual(words[0]["startChunk"], 100)
        self.assertEqual(afterBreak["startChunk"], 150)

    def test_pieces_carry_fractions_only_on_words(self):
        result = timeLine("안녕 하세요", "Korean", 0, 100, [["A", 0, 100, False, False]], ["A"])
        for p in result["pieces"]:
            self.assertEqual("fraction" in p, p["isWord"])


V_ROWS = [["V", 1133, 1179, False, False], ["V", 1182, 1226, False, False], ["V", 1229, 1273, False, False],
          ["V", 1276, 1338, False, False], ["V", 1367, 1479, False, False]]


class StayGoldVTests(unittest.TestCase):
    """V's card (startChunk 1122): 5 rows only 3 chunks apart = 5 phrases, marked by the author with a pause
    after 時計の and after 動きを (the other two cuts are line breaks). Real rows and text."""

    def _starts(self, text):
        result = timeLine(text, "Japanese", 1133, 1479, V_ROWS, ["V"])
        return result, {w["text"]: w["startChunk"] for w in _words(result)}

    def test_each_phrase_starts_on_its_own_row(self):
        marked = "時計の" + M + "針さえ\n動きを" + M + "止めるよ\nUh let it glow"
        result, starts = self._starts(marked)
        self.assertEqual(result["stretches"], 5)
        self.assertEqual((starts["時計の"], starts["針さえ"], starts["動きを"], starts["止めるよ"], starts["Uh"]),
                         (1133, 1182, 1229, 1276, 1367))
        self.assertEqual(_text(result), "時計の針さえ\n動きを止めるよ\nUh let it glow")

    def test_the_rows_alone_already_place_the_phrases_without_markers(self):
        _, starts = self._starts("時計の針さえ\n動きを止めるよ\nUh let it glow")
        self.assertEqual(starts["Uh"], 1367)            # the last phrase starts on the last row either way


class PauseMarkerTests(unittest.TestCase):
    def test_marker_splits_words_flags_the_one_before_and_never_shows(self):
        pieces = segmentLine("君を" + M + "優しく", "Japanese")
        words = [p for p in pieces if p["isWord"]]
        self.assertEqual([(w["text"], w["pauseAfter"]) for w in words], [("君を", True), ("優しく", False)])
        self.assertEqual("".join(p["text"] for p in pieces), "君を優しく")

    def test_marker_is_a_hard_word_break_even_mid_word(self):
        words = [p["text"] for p in segmentLine("가나" + M + "다라", "Korean") if p["isWord"]]
        self.assertEqual(words, ["가나", "다라"])

    def test_leading_double_and_trailing_markers_are_harmless(self):
        pieces = segmentLine(M + "가 나" + M + M + "다" + M, "Korean")
        self.assertEqual("".join(p["text"] for p in pieces), "가 나다")
        self.assertEqual([p["pauseAfter"] for p in pieces if p["isWord"]], [False, True, True])

    ROWS = [["A", 100, 200, False, False], ["A", 210, 310, False, False]]

    def test_marker_overrides_the_text_time_proportion(self):
        # Words weigh 1 / 4 / 2 beats and the two rows are equal length, so unmarked the best cut is
        # after word 2 (5/7 of the text vs 1/2 of the time). A marker after word 1 wins anyway.
        unmarked = {w["text"]: w["startChunk"] for w in _words(timeLine("가 나다라마 바사", "Korean", 100, 310, self.ROWS, ["A"]))}
        marked = {w["text"]: w["startChunk"] for w in _words(timeLine("가" + M + " 나다라마 바사", "Korean", 100, 310, self.ROWS, ["A"]))}
        self.assertEqual(unmarked["바사"], 210)
        self.assertEqual(marked["나다라마"], 210)

    def test_marker_with_no_label_gap_becomes_a_rest_inside_the_stretch(self):
        rows = [["A", 100, 300, False, False]]
        plain = {w["text"]: w["startChunk"] for w in _words(timeLine("가나 다라", "Korean", 100, 300, rows, ["A"]))}
        rested = {w["text"]: w["startChunk"] for w in _words(timeLine("가나" + M + " 다라", "Korean", 100, 300, rows, ["A"]))}
        self.assertEqual(plain["다라"], 200.0)                                   # 2 of 4 beats
        # 가나 (2 beats) + rest, then 다라 (2 beats), all inside one 200-chunk stretch.
        self.assertAlmostEqual(rested["다라"], 100 + 200 * (2 + REST_BEATS) / (4 + REST_BEATS), places=1)
        self.assertGreater(rested["다라"], plain["다라"])      # the pause pushes the next word later

    def test_marker_at_a_real_gap_adds_no_rest(self):
        # The marker coincides with the 200-210 gap, so it is a cut, not an in-stretch rest.
        result = timeLine("가나" + M + " 다라", "Korean", 100, 310, self.ROWS, ["A"])
        starts = {w["text"]: w["startChunk"] for w in _words(result)}
        self.assertEqual((starts["가나"], starts["다라"]), (100, 210))
        self.assertEqual(result["stretches"], 2)


if __name__ == "__main__":
    unittest.main()
