"""
Tests for core/lyric_file.py (editing a lyric's pause markers in a song's lyrics file from outside the Tk app).
All against a tempdir; never the real saved_labels.
Run with: python -m unittest core.test_lyric_file -v
"""

import json
import os
import shutil
import tempfile
import unittest

from core.lyric_file import LyricEditError, getRawLyric, lyricsPath, setPauseMarkers
from core.lyric_text import PAUSE_MARK as M

V = "時計の針さえ\n動きを止めるよ\nUh let it glow"
OTHER = "君を優しく\nいただくのさ"


def _entries():
    return [
        {"lyricId": "v1", "linkedLabel": {"member": "V", "startChunk": 1133, "endChunk": 1179},
         "language": "Japanese", "memberName": ["V"], "korean": V, "romanization": "tokei no hari sae",
         "english": "Even the hand of a clock", "startChunk": 1122, "isAdLib": False, "adLibDuration": 0,
         "anchorMode": "startChunk"},
        {"lyricId": "j1", "linkedLabel": None, "language": "Japanese", "memberName": ["J-Hope"],
         "korean": OTHER, "romanization": "", "english": "", "startChunk": 2787, "isAdLib": False,
         "adLibDuration": 0, "anchorMode": "startChunk"},
    ]


class LyricFileTests(unittest.TestCase):
    def setUp(self):
        self.root = tempfile.mkdtemp()
        self.path = lyricsPath("BTS", "Stay Gold", self.root)
        os.makedirs(os.path.dirname(self.path))

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _write(self, entries, newline="\n"):
        text = json.dumps(entries, ensure_ascii=False, indent=4).replace("\n", newline)
        with open(self.path, "wb") as f:
            f.write(text.encode("utf-8"))

    def _read(self):
        with open(self.path, "rb") as f:
            raw = f.read()
        return raw, json.loads(raw.decode("utf-8"))

    def test_adds_markers_and_changes_nothing_else(self):
        self._write(_entries())
        marked = "時計の" + M + "針さえ\n動きを" + M + "止めるよ\nUh let it glow"
        written = setPauseMarkers("BTS", "Stay Gold", "v1", "", marked, self.root)
        self.assertEqual(written, marked)
        _, data = self._read()
        expected = _entries()
        expected[0]["korean"] = marked
        self.assertEqual(data, expected)                     # only that one field of that one lyric moved

    def test_removing_a_marker_works_too(self):
        marked = "時計の" + M + "針さえ\n動きを止めるよ\nUh let it glow"
        entries = _entries(); entries[0]["korean"] = marked
        self._write(entries)
        setPauseMarkers("BTS", "Stay Gold", "v1", "", V, self.root)
        self.assertEqual(self._read()[1][0]["korean"], V)

    def test_found_by_clean_line_when_the_occurrence_has_no_lyric_id(self):
        self._write(_entries())
        setPauseMarkers("BTS", "Stay Gold", None, OTHER, "君を" + M + "優しく\nいただくのさ", self.root)
        self.assertEqual(self._read()[1][1]["korean"], "君を" + M + "優しく\nいただくのさ")

    def test_changing_the_wording_is_refused_and_the_file_is_untouched(self):
        self._write(_entries())
        before = self._read()[0]
        with self.assertRaises(LyricEditError) as ctx:
            setPauseMarkers("BTS", "Stay Gold", "v1", "", V.replace("時計", "時間"), self.root)
        self.assertIn("Only pause markers", str(ctx.exception))
        self.assertEqual(self._read()[0], before)

    def test_changing_the_colour_split_is_refused(self):
        self._write(_entries())
        with self.assertRaises(LyricEditError):
            setPauseMarkers("BTS", "Stay Gold", "v1", "", V.replace("時計の", "時計|の"), self.root)

    def test_unknown_lyric_is_refused(self):
        self._write(_entries())
        with self.assertRaises(LyricEditError):
            setPauseMarkers("BTS", "Stay Gold", "nope", "違う歌詞", V, self.root)
        with self.assertRaises(LyricEditError):
            getRawLyric("BTS", "Stay Gold", "nope", "違う歌詞", self.root)

    def test_missing_or_unreadable_file_is_refused_and_never_overwritten(self):
        with self.assertRaises(LyricEditError):
            setPauseMarkers("BTS", "Stay Gold", "v1", "", V, self.root)          # no file yet
        with open(self.path, "wb") as f:
            f.write(b"{ this is not json")
        with self.assertRaises(LyricEditError):
            setPauseMarkers("BTS", "Stay Gold", "v1", "", V, self.root)
        with open(self.path, "rb") as f:
            self.assertEqual(f.read(), b"{ this is not json")

    def test_line_endings_are_preserved(self):
        for newline in ("\n", "\r\n"):
            self._write(_entries(), newline)
            setPauseMarkers("BTS", "Stay Gold", "v1", "", "時計の" + M + "針さえ\n動きを止めるよ\nUh let it glow", self.root)
            raw, _ = self._read()
            self.assertEqual(b"\r\n" in raw, newline == "\r\n")
            if newline == "\r\n":
                self.assertEqual(raw.count(b"\n"), raw.count(b"\r\n"))   # no bare LF crept in

    def test_no_change_does_not_rewrite_the_file(self):
        self._write(_entries())
        before = os.path.getmtime(self.path)
        os.utime(self.path, (before - 100, before - 100))
        setPauseMarkers("BTS", "Stay Gold", "v1", "", V, self.root)
        self.assertEqual(os.path.getmtime(self.path), before - 100)

    def test_no_temp_file_is_left_behind(self):
        self._write(_entries())
        setPauseMarkers("BTS", "Stay Gold", "v1", "", "時計の" + M + "針さえ\n動きを止めるよ\nUh let it glow", self.root)
        self.assertEqual(os.listdir(os.path.dirname(self.path)), ["Stay Gold_lyrics.json"])

    def test_get_raw_lyric_returns_the_marker_bearing_text(self):
        entries = _entries(); entries[0]["korean"] = "時計の" + M + "針さえ"
        self._write(entries)
        self.assertEqual(getRawLyric("BTS", "Stay Gold", "v1", "", self.root), ("v1", "時計の" + M + "針さえ"))


if __name__ == "__main__":
    unittest.main()
