"""
Tests for core/lyric_text.py - the single owner of the "|" colour-split and the invisible pause marker.
Run with: python -m unittest core.test_lyric_text -v
"""

import unittest

from core.lyric_text import (PAUSE_MARK, EDITOR_PAUSE_GLYPH, hasPauseMarks, stripForDisplay, stripAll,
                             stripAllWithSelection, toEditorText, fromEditorText, findRawLyricText)

M = PAUSE_MARK


class MarkerTests(unittest.TestCase):
    def test_marker_is_the_invisible_separator_and_editor_glyph_is_a_triangle(self):
        self.assertEqual(ord(PAUSE_MARK), 0x2063)
        self.assertEqual(ord(EDITOR_PAUSE_GLYPH), 0x25BE)

    def test_strip_for_display_keeps_the_colour_split(self):
        self.assertEqual(stripForDisplay(f"君を{M}|優しく"), "君を|優しく")

    def test_strip_all_removes_both_markers(self):
        self.assertEqual(stripAll(f"君を{M}|優しく"), "君を優しく")

    def test_none_and_empty_are_safe(self):
        for fn in (stripForDisplay, stripAll, toEditorText, fromEditorText):
            self.assertEqual(fn(None), "")
            self.assertEqual(fn(""), "")
        self.assertFalse(hasPauseMarks(None))

    def test_marker_survives_python_strip_so_callers_must_use_this_module(self):
        # The reason the choke point exists: a marker-only line is NOT empty to str.strip().
        self.assertTrue(M.strip())
        self.assertEqual(stripAll(M).strip(), "")

    def test_editor_round_trip(self):
        stored = f"君を{M}優しく\n{M}いただくのさ"
        shown = toEditorText(stored)
        self.assertNotIn(M, shown)
        self.assertIn(EDITOR_PAUSE_GLYPH, shown)
        self.assertEqual(fromEditorText(shown), stored)

    def test_editor_text_without_markers_is_unchanged(self):
        self.assertEqual(toEditorText("君を優しく"), "君を優しく")
        self.assertEqual(fromEditorText("君を優しく"), "君を優しく")

    def test_has_pause_marks(self):
        self.assertTrue(hasPauseMarks(f"a{M}b"))
        self.assertFalse(hasPauseMarks("a|b"))


class SelectionTests(unittest.TestCase):
    def test_selection_offsets_follow_the_text_after_stripping(self):
        text = f"君を{M}優しく|いただく"             # 優しく sits at 3..6 before stripping
        clean, start, end = stripAllWithSelection(text, 3, 6)
        self.assertEqual(clean, "君を優しくいただく")
        self.assertEqual(clean[start:end], "優しく")

    def test_selection_to_end_of_text(self):
        text = f"君を{M}優しく"
        clean, start, end = stripAllWithSelection(text, 3, len(text))
        self.assertEqual(clean[start:end], "優しく")

    def test_selection_before_any_marker_is_unchanged(self):
        clean, start, end = stripAllWithSelection(f"君を{M}優しく", 0, 2)
        self.assertEqual((start, end, clean[start:end]), (0, 2, "君を"))

    def test_no_markers_is_identity(self):
        self.assertEqual(stripAllWithSelection("君を優しく", 1, 3), ("君を優しく", 1, 3))


class FindRawTests(unittest.TestCase):
    ENTRIES = [
        {"lyricId": "a", "korean": f"君を{M}優しく"},
        {"lyricId": "b", "korean": "いただくのさ"},
    ]

    def test_matches_by_lyric_id(self):
        self.assertEqual(findRawLyricText(self.ENTRIES, "a", "君を優しく"), f"君を{M}優しく")

    def test_falls_back_to_matching_the_clean_text(self):
        self.assertEqual(findRawLyricText(self.ENTRIES, None, "君を優しく"), f"君を{M}優しく")
        self.assertEqual(findRawLyricText(self.ENTRIES, None, "いただくのさ"), "いただくのさ")

    def test_unknown_returns_none(self):
        self.assertIsNone(findRawLyricText(self.ENTRIES, "zzz", "違う"))
        self.assertIsNone(findRawLyricText([], None, ""))


class ReadersIgnoreTheMarkerTests(unittest.TestCase):
    """Every analysis path must give the same answer with and without the marker (and the "|" split)."""

    LINE = "君を優しく いただくのさ"
    MARKED = "君を" + M + "優しく " + M + "いただく|のさ"

    def test_reading_conversion(self):
        from core.japanese_utils import kanjiLineToReading
        for fmt in ("romaji", "hiragana"):
            self.assertEqual(kanjiLineToReading(self.MARKED, fmt), kanjiLineToReading(self.LINE, fmt))

    def test_kanji_analysis_of_a_highlighted_word_after_stripping(self):
        from core.kanji_reference import analyzeSelection
        raw = "君を" + M + "優しく"                       # highlight 優しく (offsets 3..6 in the raw text)
        clean, start, end = stripAllWithSelection(raw, 3, 6)
        self.assertEqual([r["surface"] for r in analyzeSelection(clean, start, end)],
                         [r["surface"] for r in analyzeSelection("君を優しく", 2, 5)])


if __name__ == "__main__":
    unittest.main()
