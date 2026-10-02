"""
Tests for core/karaoke_timing.py. Pure functions, no disk I/O. Japanese cases need fugashi/unidic
(the project venv has them): run with  venv/Scripts/python.exe -m unittest core.test_karaoke_timing -v

Fixture rows are the real BTS "Stay Gold" J-Hope rows around chunk 2635 (a 3-line card sung as three
back-to-back rows, then a 2-row card with a short pause), copied rather than read from the live data.
"""

import unittest

from core.karaoke_timing import (kanaMorae, englishSyllables, segmentLine, buildStretches,
                                 timeLine, REST_BEATS, calibrateLag, splitMorae, suggestRate, rowCheck, nudgeAnchor, isHeldVowel)
from core.lyric_text import stripAll, PAUSE_MARK as M, READING_OPEN as RO, READING_CLOSE as RC

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


NAYEON_ROWS = [["Nayeon", 631, 642, False, False], ["Nayeon", 649, 754, False, False],
               ["Nayeon", 762, 830, False, False], ["Nayeon", 835, 903, False, False]]


class PlayFromHereTests(unittest.TestCase):
    """TWICE Doughnut, Nayeon's first card: 4 rows only ~0.3 s apart (and the first just 0.44 s long). A plain
    300 ms lead-in before a clicked word started playback inside the PREVIOUS line (clicking line 2 played the
    end of line 1). `playFraction` must never reach back before the word's own stretch."""

    SPAN = (631, 903)

    def _result(self):
        return timeLine("手を振って\n背を向けた瞬間に\nすぐにさみしさにやられた", "Japanese",
                        self.SPAN[0], self.SPAN[1], NAYEON_ROWS, ["Nayeon"])

    def _chunk(self, fraction):
        return self.SPAN[0] + fraction * (self.SPAN[1] - self.SPAN[0])

    def test_the_first_word_of_every_row_plays_from_exactly_that_rows_start(self):
        firstWords = {}
        for w in _words(self._result()):
            firstWords.setdefault(round(w["startChunk"]), w)
        for rowStart in (631, 649, 762, 835):
            w = firstWords[rowStart]
            self.assertAlmostEqual(self._chunk(w["playFraction"]), rowStart, delta=0.05, msg=w["text"])

    def test_no_word_ever_plays_from_before_its_own_stretch(self):
        stretches = [(r[1], r[2]) for r in NAYEON_ROWS]
        for w in _words(self._result()):
            stretchStart = max(s for s, _ in stretches if s <= w["startChunk"] + 0.01)
            self.assertGreaterEqual(self._chunk(w["playFraction"]) + 0.05, stretchStart, w["text"])
            self.assertLessEqual(w["playFraction"], w["fraction"])          # a lead-in, never a lag

    def test_a_word_well_inside_a_stretch_still_gets_the_lead_in(self):
        mid = [w for w in _words(self._result()) if w["text"] == "瞬間に"][0]      # estimated onset ~701, row starts 649
        self.assertAlmostEqual(self._chunk(mid["fraction"]) - self._chunk(mid["playFraction"]), 7.5, delta=0.1)


class DoughnutMarkedCardTests(unittest.TestCase):
    """Nayeon's first card with the author's marker 手▾を振って: 4 phrases, 4 rows (631-642 手, 649-754 を振って - a
    slow held phrase, 4 morae in 105 chunks - 762-830 背を向けた瞬間に, 835-903 すぐにさみしさにやられた). Before the fix the
    tempo-blind text-vs-time guess put を振って AND 背を向けた瞬間に in the 105-chunk row, so clicking 背 played から
    を振って."""

    SPAN = (631, 903)
    MARKED = "手" + M + "を振って\n背を向けた瞬間に\nすぐにさみしさにやられた"

    def _starts(self, text):
        result = timeLine(text, "Japanese", self.SPAN[0], self.SPAN[1], NAYEON_ROWS, ["Nayeon"])
        return result, {w["text"]: w for w in _words(result)}

    def test_each_phrase_is_exactly_one_row(self):
        result, words = self._starts(self.MARKED)
        self.assertEqual(result["stretches"], 4)
        self.assertEqual(words["手"]["startChunk"], 631)
        self.assertEqual(words["を"]["startChunk"], 649)
        self.assertEqual(words["背を"]["startChunk"], 762)        # row 3 start, NOT somewhere inside row 2
        self.assertEqual(words["すぐに"]["startChunk"], 835)      # row 4 start

    def test_clicking_背_plays_from_背_not_from_を振って(self):
        _, words = self._starts(self.MARKED)
        playChunk = self.SPAN[0] + words["背を"]["playFraction"] * (self.SPAN[1] - self.SPAN[0])
        self.assertAlmostEqual(playChunk, 762, delta=0.05)
        self.assertGreaterEqual(playChunk, 754)                  # not inside row 2 (を振って ends at 754)

    def test_unmarked_card_still_uses_the_tempo_aware_search(self):
        # Same text, no marker: nothing the author asserted, so the old behaviour stands (and 背 is wrong here -
        # which is exactly what the marker is for).
        _, words = self._starts(self.MARKED.replace(M, ""))
        self.assertLess(words["背を"]["startChunk"], 754)

    def test_marker_that_does_not_make_phrases_equal_rows_does_not_force_anything(self):
        # 3 rows but 4 phrases (marker + 2 line breaks): the search decides, as for Stay Gold card 21.
        rows = NAYEON_ROWS[:3]
        result = timeLine(self.MARKED, "Japanese", 631, 830, rows, ["Nayeon"])
        self.assertEqual(result["stretches"], 3)


TZUYU_ROWS = [["Tzuyu", 2456, 2473, False, False], ["Tzuyu", 2478, 2579, False, False],
              ["Tzuyu", 2605, 2828, False, False]]
SANA_ROWS = [["Sana", 922, 936, False, False], ["Sana", 941, 1045, False, False], ["Sana", 1070, 1233, False, False],
             ["Sana", 1235, 1251, False, False], ["Sana", 1253, 1289, False, False]]


class MergedLinesTests(unittest.TestCase):
    """Doughnut Tzuyu / Sana: one held syllable, a pause, the rest of line 1, then lines 2+3 sung as ONE long row.
    Nothing in the text says two lines share a row, and a tempo-blind text-vs-time guess cut line 2 mid-way
    (Tzuyu) and left a row empty (Sana). With a marker at EVERY pause (rows - 1) the markers decide."""

    def test_tzuyu_every_pause_marked(self):
        text = "そ" + M + "ばにいなくても" + M + "\n切れずにリンクしてるね\nMemory の余韻に浸っていたい (I, I, I, I)"
        result = timeLine(text, "Japanese", 2456, 2828, TZUYU_ROWS, ["Tzuyu"])
        self.assertEqual([r[2] for r in result["rows"]][:2], [["そ"], ["ばに", "いなくても"]])
        self.assertEqual(result["rows"][2][2][0], "切れずに")                       # lines 2+3 share the long row
        starts = {w["text"]: w["startChunk"] for w in _words(result)}
        self.assertEqual((starts["そ"], starts["ばに"], starts["切れずに"]), (2456, 2478, 2605))
        self.assertEqual(result["markers"], 2)

    def test_tzuyu_without_the_second_marker_is_guessed_wrong(self):
        text = "そ" + M + "ばにいなくても\n切れずにリンクしてるね\nMemory の余韻に浸っていたい (I, I, I, I)"
        result = timeLine(text, "Japanese", 2456, 2828, TZUYU_ROWS, ["Tzuyu"])
        starts = {w["text"]: w["startChunk"] for w in _words(result)}
        self.assertLess(starts["切れずに"], 2605)       # tempo-blind guess crams line 2's start into row 2

    def test_sana_every_pause_marked_leaves_no_row_empty(self):
        text = ("恋" + M + "をしてから" + M + "\n君に埋め尽くされた Mind\nしあわせと切なさが忙しいな" + M
                + " (Na na" + M + " na na)")
        result = timeLine(text, "Japanese", 922, 1289, SANA_ROWS, ["Sana"])
        self.assertEqual(result["stretches"], 5)
        rows = [r[2] for r in result["rows"]]
        self.assertEqual(rows[0], ["恋"])
        self.assertEqual(rows[1][0], "を")
        self.assertIn("Mind", rows[2])                       # 君に…Mind AND しあわせと…忙しいな share row 3
        self.assertIn("忙しいな", rows[2])
        self.assertEqual(rows[3], ["Na", "na"])
        self.assertEqual(rows[4], ["na", "na"])
        self.assertTrue(all(rows))                           # no row left empty

    def test_rows_report_their_words_and_the_marker_count(self):
        result = timeLine("가" + M + " 나", "Korean", 100, 310,
                          [["A", 100, 200, False, False], ["A", 210, 310, False, False]], ["A"])
        self.assertEqual([(a, b, w) for a, b, w in result["rows"]], [(100, 200, ["가"]), (210, 310, ["나"])])
        self.assertEqual(result["markers"], 1)


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


class SplitKanjiTests(unittest.TestCase):
    """恋 is sung こ in one row and い at the start of the next (Doughnut, Sana). A reading annotation lets a pause sit
    INSIDE the kanji word: 恋 stays one word on screen, its kana parts are timed apart."""
    TEXT = ("恋" + RO + "こ" + M + "い" + RC + "をしてから" + M + "\n君に埋め尽くされた Mind\n"
            "しあわせと切なさが忙しいな" + M + " (Na na" + M + " na na)")

    def test_the_word_is_one_piece_with_a_part_per_row(self):
        result = timeLine(self.TEXT, "Japanese", 922, 1289, SANA_ROWS, ["Sana"])
        koi = _words(result)[0]
        self.assertEqual(koi["text"], "恋")
        self.assertEqual([p["reading"] for p in koi["parts"]], ["こ", "い"])
        self.assertEqual(koi["parts"][0]["startChunk"], 922)       # こ at the first row start, exact
        self.assertEqual(koi["parts"][1]["startChunk"], 941)       # い at the second row start, exact
        self.assertEqual((koi["startChunk"], koi["endChunk"]), (922, koi["parts"][1]["endChunk"]))

    def test_pieces_still_reassemble_without_the_annotation(self):
        result = timeLine(self.TEXT, "Japanese", 922, 1289, SANA_ROWS, ["Sana"])
        self.assertEqual(_text(result), stripAll(self.TEXT))
        self.assertFalse(any(p.get("continuation") for p in result["pieces"]))

    def test_the_marker_inside_the_reading_counts_as_a_pause_and_rows_are_exact(self):
        result = timeLine(self.TEXT, "Japanese", 922, 1289, SANA_ROWS, ["Sana"])
        self.assertEqual(result["markers"], 4)                      # 5 rows, 4 pauses: こ|い, してから, 忙しいな, Na
        rows = [r[2] for r in result["rows"]]
        self.assertEqual(rows[0], ["恋(こ)"])
        self.assertEqual(rows[1][:2], ["恋(い)", "を"])
        self.assertTrue(all(rows))

    def test_the_particle_after_the_split_word_is_its_own_word(self):
        words = [w["text"] for w in _words(timeLine(self.TEXT, "Japanese", 922, 1289, SANA_ROWS, ["Sana"]))]
        self.assertEqual(words[:2], ["恋", "を"])

    def test_a_reading_without_a_pause_only_sets_the_beats(self):
        text = "恋" + RO + "こいこい" + RC + "を"
        word = _words(timeLine(text, "Japanese", 0, 100, [], None))[0]
        self.assertEqual(word["weight"], 4)
        self.assertEqual(len(word["parts"]), 1)

    def test_an_annotation_with_no_kanji_or_no_kana_is_ignored(self):
        for text in ("あ" + RO + "い" + M + "う" + RC + "え", "恋" + RO + "x" + RC + "を"):
            result = timeLine(text, "Japanese", 0, 100, [], None)
            self.assertEqual(_text(result), stripAll(text))
            self.assertTrue(all("parts" not in w for w in _words(result)))


class TapAnchorTests(unittest.TestCase):
    """Tapped onsets (core.tap_store) override the estimate; the author's row starts always win."""
    ONE_ROW = [["A", 100, 300, False, False]]
    TWO_ROWS = [["A", 100, 200, False, False], ["A", 210, 310, False, False]]

    @staticmethod
    def _starts(result):
        return {w["text"]: (w["startChunk"], w["source"]) for w in _words(result)}

    def test_no_anchors_changes_nothing_but_labels_the_source(self):
        result = timeLine("가나 다라 마바", "Korean", 100, 300, self.ONE_ROW, ["A"])
        self.assertEqual(self._starts(result), {"가나": (100, "row"), "다라": (166.7, "estimated"),
                                                "마바": (233.3, "estimated")})
        self.assertEqual(result["words"], ["가나", "다라", "마바"])
        self.assertEqual(result["rowStarts"], {0: 100})

    def test_a_tap_moves_its_word_and_respreads_only_the_gap_before_it(self):
        result = timeLine("가나 다라 마바", "Korean", 100, 300, self.ONE_ROW, ["A"], anchors={2: 250})
        self.assertEqual(self._starts(result), {"가나": (100, "row"), "다라": (175, "estimated"),
                                                "마바": (250, "tapped")})

    def test_the_first_word_of_a_row_stays_on_the_row_start(self):
        result = timeLine("가나 다라 마바 사아", "Korean", 100, 310, self.TWO_ROWS, ["A"], anchors={2: 230, 3: 280})
        starts = self._starts(result)
        self.assertEqual(starts["마바"], (210, "row"))        # the tap at 230 loses to the labelled row start
        self.assertEqual(starts["사아"], (280, "tapped"))

    def test_a_tap_outside_its_stretch_is_clamped_and_never_goes_backwards(self):
        result = timeLine("가나 다라 마바", "Korean", 100, 300, self.ONE_ROW, ["A"], anchors={1: 900, 2: 120})
        starts = self._starts(result)
        self.assertEqual(starts["다라"][0], 300)               # clamped to the stretch end
        self.assertEqual(starts["마바"][0], 300)               # an earlier tap never moves before the previous one

    def test_with_no_labelled_rows_the_first_tap_is_honoured(self):
        result = timeLine("가나 다라", "Korean", 100, 300, [], None, anchors={0: 130})
        self.assertEqual(self._starts(result)["가나"], (130, "tapped"))

    def test_each_kana_part_of_a_split_kanji_takes_its_own_tap(self):
        text = "恋" + RO + "こ" + M + "い" + RC + "を"
        rows = [["S", 0, 40, False, False], ["S", 50, 150, False, False]]
        result = timeLine(text, "Japanese", 0, 150, rows, ["S"], anchors={1: 70})
        self.assertEqual(result["words"], ["恋(こ)", "恋(い)", "を"])
        koi = _words(result)[0]
        self.assertEqual([p["startChunk"] for p in koi["parts"]], [0, 50])     # い is the row start: the tap loses

    def test_lag_is_the_median_gap_between_taps_and_exact_row_starts(self):
        lag, spread = calibrateLag({0: 112, 3: 331, 5: 999}, {0: 100, 3: 320})   # 5 is not a row start
        self.assertEqual((lag, spread), (11.5, 1))
        self.assertEqual(calibrateLag({0: 112}, {0: 100}), (None, None))         # one tap is not enough
        self.assertEqual(calibrateLag({}, {}), (None, None))


class SyllableUnitTests(unittest.TestCase):
    """unit="syllable": one tap per sung beat (every kana; ー and っ count like ひ・と・つ), per Hangul block, and one
    per whole English word. Row assignment still happens per word; syllables are split out afterwards."""
    ONE_ROW = [["A", 100, 200, False, False]]

    def test_split_morae_counts_long_marks_and_sokuon_as_beats_and_fuses_small_kana(self):
        self.assertEqual(splitMorae("シュンカン"), ["シュ", "ン", "カ", "ン"])
        self.assertEqual(splitMorae("やっつ"), ["や", "っ", "つ"])
        self.assertEqual(splitMorae("ラーメン"), ["ラ", "ー", "メ", "ン"])
        self.assertEqual(len(splitMorae("キミ")), kanaMorae("キミ"))

    def test_japanese_words_split_by_reading_and_the_word_starts_with_its_first_syllable(self):
        result = timeLine("振って", "Japanese", 100, 200, self.ONE_ROW, ["A"], unit="syllable")
        self.assertEqual(result["words"], ["ふ", "っ", "て"])
        word = _words(result)[0]
        self.assertEqual([(x["label"], x["startChunk"]) for x in word["syllables"]],
                         [("ふ", 100), ("っ", 133.3), ("て", 166.7)])
        self.assertEqual((word["startChunk"], word["endChunk"]), (100, 200))

    def test_a_held_mark_stays_a_mark_and_hold_beats_are_flagged(self):
        result = timeLine("ラーメン 振って 瞬間", "Japanese", 0, 100, [], None, unit="syllable")
        self.assertEqual(result["words"], ["ラ", "ー", "メ", "ン", "ふ", "っ", "て", "しゅ", "ん", "か", "ん"])
        flags = [(x["label"], x["hold"]) for w in _words(result) for x in w["syllables"]]
        self.assertEqual([lab for lab, hold in flags if hold], ["ー", "ン", "っ", "ん", "ん"])

    def test_the_particle_is_sung_as_pronounced(self):
        self.assertEqual(timeLine("手を", "Japanese", 0, 100, [], None, unit="syllable")["words"], ["て", "お"])

    def test_korean_blocks_and_english_words_are_one_unit_each(self):
        self.assertEqual(timeLine("사랑해요 Baby", "Korean", 0, 100, [], None, unit="syllable")["words"],
                         ["사", "랑", "해", "요", "Baby"])
        self.assertEqual(timeLine("あい Baby", "Japanese", 0, 100, [], None, unit="syllable")["words"], ["あ", "い", "Baby"])

    def test_a_syllable_tap_respreads_only_the_gap_before_the_next_tap(self):
        result = timeLine("振って", "Japanese", 100, 200, self.ONE_ROW, ["A"], anchors={1: 150}, unit="syllable")
        sy = {x["label"]: (x["startChunk"], x["source"]) for x in _words(result)[0]["syllables"]}
        self.assertEqual(sy, {"ふ": (100, "row"), "っ": (150, "tapped"), "て": (175, "estimated")})

    def test_each_kana_of_a_split_kanji_part_is_a_unit_and_the_part_break_still_cuts_rows(self):
        text = "恋" + RO + "こ" + M + "い" + RC + "を"
        rows = [["S", 0, 40, False, False], ["S", 50, 150, False, False]]
        result = timeLine(text, "Japanese", 0, 150, rows, ["S"], unit="syllable")
        self.assertEqual(result["words"], ["こ", "い", "お"])
        self.assertEqual(result["rowStarts"], {0: 0, 1: 50})
        self.assertEqual([x["label"] for x in _words(result)[0]["syllables"]], ["こ", "い"])

    def test_word_mode_is_untouched_by_the_new_option(self):
        a = timeLine("振って 背を", "Japanese", 100, 200, self.ONE_ROW, ["A"])
        self.assertEqual(a["unit"], "word")
        self.assertTrue(all("syllables" not in w for w in _words(a)))

    def test_pace_suggests_a_slower_speed_only_when_the_busiest_stretch_is_too_fast(self):
        self.assertEqual([suggestRate(x) for x in (2, 4, 5, 7, 20)], [1.0, 1.0, 0.75, 0.5, 0.35])
        slow = timeLine("あい", "Japanese", 0, 100, [["A", 0, 100, False, False]], ["A"], unit="syllable")
        self.assertEqual(slow["pace"], {"fastest": 0.5, "suggestedRate": 1.0})


class TapsDecideTheRowTests(unittest.TestCase):
    """TWICE "Funny Valentine", Momo's first card. The label rows say she sings "Moon night あなたが" | "くれた" |
    "曖昧なこの感情は", but the author's markers sit after "night" and after あなた - and 2 markers + 1 == 3 rows made the
    markers the row boundaries, so correct taps were clamped into the wrong rows and the take came out wildly off."""
    ROWS = [["Momo", 271, 356, False, False], ["Momo", 360, 379, False, False], ["Momo", 383, 451, False, False]]
    TEXT = "Moon night" + M + "\nあなた" + M + "がくれた　曖昧なこの感情は"
    # When each of the 21 units (Moon night あ な た が く れ た あ い ま い な こ の か ん じょ ー わ) is really sung.
    TRUE = {0: 271, 1: 299, 2: 331, 3: 339, 4: 347, 5: 353, 6: 361, 7: 369, 8: 376, 9: 384, 10: 390, 11: 396, 12: 402,
            13: 408, 14: 414, 15: 420, 16: 426, 17: 432, 18: 438, 19: 444, 20: 448}

    def _time(self, **kw):
        return timeLine(self.TEXT, "Japanese", 271, 451, self.ROWS, ["Momo"], unit="syllable", **kw)

    @staticmethod
    def _rowOf(result):
        return {label: k for k, (_, _, words) in enumerate(result["rows"]) for label in words}

    def test_without_taps_the_markers_are_believed(self):
        result = self._time()
        self.assertEqual(len(result["words"]), 21)
        self.assertEqual(result["reassigned"], 0)
        starts = [u["startChunk"] for w in _words(result) for u in w["syllables"]]
        self.assertEqual(starts[2], 360)                    # あ at the start of the row the markers chose

    def test_taps_move_the_beats_into_the_rows_they_were_sung_in(self):
        result = self._time(anchors=dict(self.TRUE))
        starts = [u["startChunk"] for w in _words(result) for u in w["syllables"]]
        self.assertEqual(starts[2:6], [331, 339, 347, 353])           # あ な た が stay where they were tapped
        self.assertEqual(starts[6], 360)                              # く opens the 360-379 row: the label wins
        self.assertGreater(result["reassigned"], 0)
        self.assertEqual(self._rowOf(result)["Moon"], 0)
        self.assertEqual([u["startChunk"] for w in _words(result) for u in w["syllables"]][9], 383)   # opens row 3: label wins

    def test_untapped_beats_stay_inside_the_rows_of_the_taps_around_them(self):
        taps = {i: t for i, t in self.TRUE.items() if i not in (3, 4, 7)}          # na, ta, re skipped
        starts = [u["startChunk"] for w in _words(self._time(anchors=taps)) for u in w["syllables"]]
        self.assertTrue(331 < starts[3] < starts[4] < 353)                           # between あ and が
        self.assertTrue(361 < starts[7] < 376)                                       # between く and た

    def test_lag_comes_from_row_starts_by_time_when_the_text_put_units_in_the_wrong_rows(self):
        raw = {i: t + 5 for i, t in self.TRUE.items()}                               # a steady 200 ms late
        base = self._time()
        byUnit = calibrateLag(raw, base["rowStarts"])                                 # the old way: wild gaps -> no lag
        self.assertEqual(byUnit, (None, None))
        lag, spread = calibrateLag(raw, base["rowStarts"], base["stretchStarts"])
        self.assertTrue(4 <= lag <= 7, lag)       # rows 4 chunks apart + a 5 chunk lag: the last tap of a row lands
                                                  # AFTER the next row starts, and must not be mistaken for its first
        # on a slowed clip the same person lags less in song time, so the "too soon" bar drops with the speed
        slowRaw = {i: t + 2.5 for i, t in self.TRUE.items()}
        self.assertTrue(1.5 <= calibrateLag(slowRaw, base["rowStarts"], base["stretchStarts"], minLag=1.5)[0] <= 4)

    def test_row_check_reports_how_far_each_rows_first_tap_is_from_its_label_start(self):
        base = self._time()
        good = {i: t for i, t in self.TRUE.items()}                       # lag already removed
        self.assertEqual(rowCheck(good, base["stretchStarts"]), [0.0, 1.0, 1.0])
        late = {i: t + 25 for i, t in self.TRUE.items()}                  # a take a whole second late
        self.assertEqual(rowCheck(late, base["stretchStarts"])[0], 25.0)   # row 1 is unambiguous; later rows can pick a neighbour's tap
        self.assertEqual(rowCheck({}, base["stretchStarts"]), [])

    def test_two_english_words_are_two_words_two_units_and_two_taps(self):
        """"Moon night" is two words in every mode, with or without the marker after it, and stay two even when the
        take has them tapped almost together (or clamped onto the same chunk)."""
        for unit in ("word", "syllable"):
            result = self._time() if unit == "syllable" else timeLine(self.TEXT, "Japanese", 271, 451, self.ROWS, ["Momo"])
            self.assertEqual(result["words"][:2], ["Moon", "night"])
            self.assertEqual([w["text"] for w in _words(result)][:2], ["Moon", "night"])
        for night in (281, 273, 271):
            result = self._time(anchors={0: 271, 1: night})
            moon, nightWord = _words(result)[:2]
            self.assertEqual((moon["text"], nightWord["text"]), ("Moon", "night"))
            self.assertEqual((moon["startChunk"], nightWord["startChunk"]), (271, night))
        self.assertEqual(segmentLine("Moon night", "Japanese")[0]["text"], "Moon")
        self.assertEqual(len([p for p in segmentLine("Moon night", "Japanese") if p["isWord"]]), 2)

    def test_lag_by_unit_is_still_used_when_the_text_got_the_rows_right(self):
        base = timeLine("가나 다라", "Korean", 100, 310, [["A", 100, 200, False, False], ["A", 210, 310, False, False]], ["A"])
        self.assertEqual(calibrateLag({0: 112, 1: 222}, base["rowStarts"], base["stretchStarts"]), (12, 0))


class NudgeTests(unittest.TestCase):
    """The fine-tune pass: move one beat of a take by whole chunks, inside its own row, never past a fixed neighbour."""
    TWO_ROWS = [["A", 100, 200, False, False], ["A", 210, 310, False, False]]
    TEXT = "가나 다라 마바 사아 자차 카타"      # estimate: beats 0-2 in the first row (0 on its start), 3-5 in the second

    def _timed(self, anchors=None):
        return timeLine(self.TEXT, "Korean", 100, 310, self.TWO_ROWS, ["A"], anchors=anchors or {})

    def test_slots_list_every_beat_with_its_row_and_source(self):
        slots = self._timed({1: 130})["slots"]
        self.assertEqual([s["label"] for s in slots], ["가나", "다라", "마바", "사아", "자차", "카타"])
        self.assertEqual([s["row"] for s in slots], [0, 0, 0, 1, 1, 1])
        self.assertEqual([s["source"] for s in slots], ["row", "tapped", "estimated", "row", "estimated", "estimated"])
        self.assertEqual(slots[1]["startChunk"], 130)

    def test_nudging_a_tapped_beat_moves_exactly_that_beat_and_leaves_the_other_anchors_alone(self):
        anchors = {1: 130, 4: 260}
        out, status = nudgeAnchor(self._timed(anchors), anchors, 1, 1)
        self.assertEqual((status, out), ("moved", {1: 131, 4: 260}))
        self.assertEqual(self._timed(out)["slots"][1]["startChunk"], 131)

    def test_an_estimated_beat_becomes_tapped_when_nudged(self):
        timing = self._timed({})
        estimate = timing["slots"][1]["startChunk"]
        out, status = nudgeAnchor(timing, {}, 1, 5)
        self.assertEqual((status, out), ("moved", {1: round(estimate + 5, 1)}))
        self.assertEqual(self._timed(out)["slots"][1]["source"], "tapped")

    def test_the_first_beat_of_each_label_row_is_locked(self):
        for index in (0, 3):
            out, status = nudgeAnchor(self._timed({1: 130}), {1: 130}, index, 3)
            self.assertEqual((status, out), ("locked", {1: 130}))

    def test_a_beat_never_crosses_the_tapped_beat_next_to_it(self):
        anchors = {1: 130, 2: 140}
        out, status = nudgeAnchor(self._timed(anchors), anchors, 1, 20)
        self.assertEqual((status, out[1]), ("moved", 140))      # stops ON the neighbour, not past it
        out, status = nudgeAnchor(self._timed(out), out, 1, 1)
        self.assertEqual((status, out[1]), ("edge", 140))
        out, status = nudgeAnchor(self._timed(out), out, 2, -20)
        self.assertEqual((status, out[2]), ("edge", 140))       # and the neighbour cannot step back over it either

    def test_a_beat_stays_inside_its_own_row(self):
        anchors = {2: 190}
        out, status = nudgeAnchor(self._timed(anchors), anchors, 2, 50)
        self.assertEqual((status, out[2]), ("moved", 200))      # the row ends at 200 (the next starts at 210)
        out, status = nudgeAnchor(self._timed(out), out, 2, -500)
        self.assertEqual(out[2], 100)                            # floor: the row start, where beat 0 sits

    def test_estimated_beats_in_between_do_not_block_a_nudge(self):
        anchors = {2: 180}
        out, status = nudgeAnchor(self._timed(anchors), anchors, 2, -20)
        self.assertEqual((status, out[2]), ("moved", 160))       # beat 1 is only an estimate: it re-spreads

    def test_reset_puts_the_beat_back_to_the_estimate(self):
        out, status = nudgeAnchor(self._timed({1: 130}), {1: 130}, 1, None)
        self.assertEqual((status, out), ("reset", {}))
        self.assertEqual(nudgeAnchor(self._timed({}), {}, 1, None)[1], "edge")      # nothing to reset

    def test_an_unknown_beat_is_an_error(self):
        with self.assertRaises(ValueError):
            nudgeAnchor(self._timed({}), {}, 99, 1)


class HeldVowelTests(unittest.TestCase):
    """Funny Valentine, Momo sings 感情 as "gan jyou" and 重要性 as "juu you sei": the held vowel is not a syllable."""

    @staticmethod
    def _units(text):
        result = timeLine(text, "Japanese", 0, 400, [], None, unit="syllable")
        return [(s["label"], s["hold"], s["held"]) for s in result["slots"]]

    def test_the_rule(self):
        for prev, beat in (("じょ", "う"), ("じゅ", "う"), ("く", "う"), ("せ", "い"), ("お", "お"), ("こ", "お")):
            self.assertTrue(isHeldVowel(prev, beat), (prev, beat))
        for prev, beat in (("か", "う"), ("い", "い"), ("あ", "あ"), ("こ", "い"), ("", "う")):
            self.assertFalse(isHeldVowel(prev, beat), (prev, beat))

    def test_real_card_words_come_out_as_the_singer_counts_them(self):
        kanjou = self._units("感情は")
        # the reading spells the long vowel as ー: か ん じょ ー は  (ん is a hold, ー is held)
        self.assertEqual([(l, h) for l, h, _ in kanjou][:2], [("か", False), ("ん", True)])
        self.assertTrue(any(l == "ー" and (h or he) for l, h, he in kanjou))
        taps = [l for l, h, he in kanjou if not h and not (he or l == "ー")]
        self.assertEqual(taps, ["か", "じょ", "わ"])                      # "gan jyou" + the particle
        juuyou = self._units("二人の時間の重要性")
        self.assertEqual([l for l, h, he in juuyou if not h and not (he or l == "ー")],
                         ["ふ", "た", "り", "の", "じ", "か", "の", "じゅ", "よ", "せ"])

    def test_a_long_vowel_written_in_kana_is_a_held_beat_too(self):
        units = self._units("かんじょう")
        self.assertEqual([(l, h) for l, h, he in units], [("か", False), ("ん", True), ("じょ", False), ("ー", True)])


if __name__ == "__main__":
    unittest.main()
