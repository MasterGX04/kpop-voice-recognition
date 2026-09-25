"""
Tests for core/vocab_db.py's schema creation. Stdlib unittest only. Run with:
    python -m unittest core.test_vocab_db -v
"""

import os
import shutil
import tempfile
import unittest

from core.vocab_db import getConnection


class SchemaCreationTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_all_tables_are_created(self):
        conn = getConnection()
        try:
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
        finally:
            conn.close()
        expected = {
            "vocab_ja", "srs_card_ja", "vocab_occurrence_ja",
            "vocab_ko", "vocab_ko_hanja", "srs_card_ko", "vocab_occurrence_ko",
            "cognate_link",
        }
        self.assertTrue(expected.issubset(tables))

    def test_foreign_keys_are_enforced(self):
        conn = getConnection()
        try:
            with self.assertRaises(Exception):
                conn.execute(
                    "INSERT INTO srs_card_ja (vocab_ja_id, meaning_due_ts, reading_due_ts) VALUES (999, 0, 0)"
                )
        finally:
            conn.close()

    def test_calling_get_connection_twice_does_not_fail(self):
        getConnection().close()
        getConnection().close()


class SchemaMigrationTests(unittest.TestCase):
    """
    Regression coverage for a real scenario: a database created before meaning_locked/hanja_locked
    existed (any already-populated data/vocab_srs.db from before this fix) must get those columns
    added in place - "CREATE TABLE IF NOT EXISTS" alone is a no-op once the table already exists,
    so without an explicit ALTER TABLE migration, an existing user's database would keep silently
    missing them forever, and the "don't clobber my manual edits on rescan" fix would never apply.
    """

    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_pre_existing_database_gets_new_columns_added_without_losing_data(self):
        import sqlite3

        os.makedirs("data", exist_ok=True)
        oldConn = sqlite3.connect("data/vocab_srs.db")
        oldConn.executescript(
            """
            CREATE TABLE vocab_ko (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL UNIQUE,
                surface TEXT,
                meaning_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            """
        )
        oldConn.execute(
            "INSERT INTO vocab_ko (lemma, surface, meaning_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("화", "화", '{"status": "found", "pos": "noun", "gloss": ["fire"]}', "2026-01-01", "2026-01-01"),
        )
        oldConn.commit()
        oldConn.close()

        conn = getConnection()
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(vocab_ko)").fetchall()}
            self.assertIn("meaning_locked", columns)
            self.assertIn("hanja_locked", columns)

            row = conn.execute(
                "SELECT lemma, meaning_locked, hanja_locked FROM vocab_ko WHERE lemma = ?", ("화",)
            ).fetchone()
            self.assertEqual(row[0], "화")
            self.assertEqual(row[1], 0)  # new column defaults to unlocked; existing row preserved
            self.assertEqual(row[2], 0)
        finally:
            conn.close()

    def test_vocab_ja_also_gets_meaning_locked_column(self):
        import sqlite3

        os.makedirs("data", exist_ok=True)
        oldConn = sqlite3.connect("data/vocab_srs.db")
        oldConn.executescript(
            """
            CREATE TABLE vocab_ja (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                lemma TEXT NOT NULL UNIQUE,
                lemma_reading TEXT, surface TEXT, category TEXT, meaning_json TEXT,
                cognate_form TEXT, cognate_status TEXT, cognate_pinyin TEXT,
                cognate_gloss_json TEXT, mnemonic_pinyin_json TEXT,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            """
        )
        oldConn.close()

        conn = getConnection()
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(vocab_ja)").fetchall()}
            self.assertIn("meaning_locked", columns)
        finally:
            conn.close()


if __name__ == "__main__":
    unittest.main()
