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
            "cognate_link", "review_log",
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

    def test_pre_existing_srs_cards_get_cloze_columns_backfilled_to_due_now(self):
        # Milestone 5: a database created before the cloze_* columns existed must get them added
        # in place, with cloze_due_ts backfilled to a real "due now" timestamp (not left at the
        # literal 0 ALTER TABLE ADD COLUMN's static default requires) - see _migrateColumns()'s own
        # comment for why a static default can't express "now" directly.
        import sqlite3
        import time

        os.makedirs("data", exist_ok=True)
        oldConn = sqlite3.connect("data/vocab_srs.db")
        oldConn.executescript(
            """
            CREATE TABLE vocab_ja (
                id INTEGER PRIMARY KEY AUTOINCREMENT, lemma TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE srs_card_ja (
                vocab_ja_id INTEGER PRIMARY KEY,
                meaning_state INTEGER NOT NULL DEFAULT 0, reading_state INTEGER NOT NULL DEFAULT 0,
                meaning_box INTEGER NOT NULL DEFAULT 1, meaning_ease REAL NOT NULL DEFAULT 2.5,
                meaning_due_ts INTEGER NOT NULL, meaning_last_reviewed_ts INTEGER,
                reading_box INTEGER NOT NULL DEFAULT 1, reading_ease REAL NOT NULL DEFAULT 2.5,
                reading_due_ts INTEGER NOT NULL, reading_last_reviewed_ts INTEGER
            );
            """
        )
        oldConn.execute(
            "INSERT INTO vocab_ja (lemma, created_at, updated_at) VALUES ('食べる', '2026-01-01', '2026-01-01')"
        )
        oldConn.execute(
            "INSERT INTO srs_card_ja (vocab_ja_id, meaning_due_ts, reading_due_ts) VALUES (1, 0, 0)"
        )
        oldConn.commit()
        oldConn.close()

        before = int(time.time())
        conn = getConnection()
        try:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(srs_card_ja)").fetchall()}
            for col in ("cloze_box", "cloze_ease", "cloze_due_ts", "cloze_last_reviewed_ts"):
                self.assertIn(col, columns)

            row = conn.execute(
                "SELECT cloze_box, cloze_ease, cloze_due_ts, cloze_last_reviewed_ts FROM srs_card_ja WHERE vocab_ja_id = 1"
            ).fetchone()
            self.assertEqual(row[0], 1)
            self.assertEqual(row[1], 2.5)
            self.assertGreaterEqual(row[2], before)  # backfilled to "now", not left at 0
            self.assertIsNone(row[3])

            indexNames = {
                r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'").fetchall()
            }
            self.assertIn("idx_srs_ja_cloze_due", indexNames)
        finally:
            conn.close()

    def test_migration_does_not_reset_cloze_due_ts_on_a_later_reconnect(self):
        # Real risk found while designing the backfill: since ALTER TABLE ADD COLUMN needs a
        # static default, cloze_due_ts is added as 0 and then backfilled to "now" in the SAME call
        # that adds it - but that backfill must never re-fire on a later getConnection() call
        # against an already-migrated database, or real review progress would keep getting reset
        # to "due now" every time the app restarts.
        import sqlite3

        os.makedirs("data", exist_ok=True)
        oldConn = sqlite3.connect("data/vocab_srs.db")
        oldConn.executescript(
            """
            CREATE TABLE vocab_ja (
                id INTEGER PRIMARY KEY AUTOINCREMENT, lemma TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE srs_card_ja (
                vocab_ja_id INTEGER PRIMARY KEY,
                meaning_box INTEGER NOT NULL DEFAULT 1, meaning_ease REAL NOT NULL DEFAULT 2.5,
                meaning_due_ts INTEGER NOT NULL, meaning_last_reviewed_ts INTEGER,
                reading_box INTEGER NOT NULL DEFAULT 1, reading_ease REAL NOT NULL DEFAULT 2.5,
                reading_due_ts INTEGER NOT NULL, reading_last_reviewed_ts INTEGER
            );
            """
        )
        oldConn.execute(
            "INSERT INTO vocab_ja (lemma, created_at, updated_at) VALUES ('食べる', '2026-01-01', '2026-01-01')"
        )
        oldConn.execute(
            "INSERT INTO srs_card_ja (vocab_ja_id, meaning_due_ts, reading_due_ts) VALUES (1, 0, 0)"
        )
        oldConn.commit()
        oldConn.close()

        conn1 = getConnection()
        # Simulate a real cloze review having advanced the due date far into the future.
        conn1.execute("UPDATE srs_card_ja SET cloze_due_ts = 4102444800 WHERE vocab_ja_id = 1")
        conn1.commit()
        conn1.close()

        # Simulate a fresh process re-opening the same (now-migrated) database file.
        from core import vocab_db
        vocab_db._initializedDbPaths.clear()

        conn2 = getConnection()
        try:
            row = conn2.execute("SELECT cloze_due_ts FROM srs_card_ja WHERE vocab_ja_id = 1").fetchone()
            self.assertEqual(row[0], 4102444800)  # untouched by the second migration pass
        finally:
            conn2.close()

    def _makeLeitnerDb(self):
        import sqlite3
        os.makedirs("data", exist_ok=True)
        c = sqlite3.connect("data/vocab_srs.db")
        c.executescript(
            """
            CREATE TABLE vocab_ja (
                id INTEGER PRIMARY KEY AUTOINCREMENT, lemma TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            );
            CREATE TABLE srs_card_ja (
                vocab_ja_id INTEGER PRIMARY KEY,
                meaning_state INTEGER NOT NULL DEFAULT 0, reading_state INTEGER NOT NULL DEFAULT 0,
                meaning_box INTEGER NOT NULL DEFAULT 1, meaning_ease REAL NOT NULL DEFAULT 2.5,
                meaning_due_ts INTEGER NOT NULL, meaning_last_reviewed_ts INTEGER,
                reading_box INTEGER NOT NULL DEFAULT 1, reading_ease REAL NOT NULL DEFAULT 2.5,
                reading_due_ts INTEGER NOT NULL, reading_last_reviewed_ts INTEGER,
                cloze_box INTEGER NOT NULL DEFAULT 1, cloze_ease REAL NOT NULL DEFAULT 2.5,
                cloze_due_ts INTEGER NOT NULL, cloze_last_reviewed_ts INTEGER
            );
            INSERT INTO vocab_ja (lemma, created_at, updated_at) VALUES ('a', 'x', 'x'), ('b', 'x', 'x');
            -- card 1: never reviewed, but meaning_state=1 is the OLD "known" flag
            INSERT INTO srs_card_ja (vocab_ja_id, meaning_state, meaning_due_ts, reading_due_ts, cloze_due_ts)
                VALUES (1, 1, 100, 100, 100);
            -- card 2: reading reviewed at box 3 (7 days), ease 1.3 -> hard word
            INSERT INTO srs_card_ja (vocab_ja_id, reading_box, reading_ease, reading_due_ts,
                reading_last_reviewed_ts, meaning_due_ts, cloze_due_ts)
                VALUES (2, 3, 1.3, 5000, 4000, 100, 100);
            """
        )
        c.commit()
        c.close()

    def test_fsrs_migration_maps_new_and_reviewed_cards(self):
        self._makeLeitnerDb()
        conn = getConnection()
        try:
            q = ("SELECT {t}_state, {t}_stability, {t}_difficulty, {t}_reps, {t}_due_ts "
                 "FROM srs_card_ja WHERE vocab_ja_id = ?")
            # never reviewed -> New everywhere, old 'known' flag wiped
            for track in ("meaning", "reading", "cloze"):
                self.assertEqual(conn.execute(q.format(t=track), (1,)).fetchone()[:4], (0, None, None, 0))
            # reviewed -> Review, stability from box interval scaled by ease, difficulty from ease
            state, stability, difficulty, reps, due = conn.execute(q.format(t="reading"), (2,)).fetchone()
            self.assertEqual(state, 2)
            self.assertAlmostEqual(stability, 7 * 1.3 / 2.5)
            self.assertAlmostEqual(difficulty, 9.0)
            self.assertEqual(due, 5000)  # due_ts untouched
            self.assertEqual(conn.execute(q.format(t="meaning"), (2,)).fetchone()[0], 0)
        finally:
            conn.close()

    def test_fsrs_migration_is_a_no_op_on_reconnect(self):
        self._makeLeitnerDb()
        conn = getConnection()
        conn.execute("UPDATE srs_card_ja SET reading_stability = 99.0, reading_state = 3 WHERE vocab_ja_id = 2")
        conn.commit()
        conn.close()

        from core import vocab_db
        vocab_db._initializedDbPaths.clear()
        conn = getConnection()
        try:
            row = conn.execute("SELECT reading_stability, reading_state FROM srs_card_ja WHERE vocab_ja_id = 2").fetchone()
            self.assertEqual(row, (99.0, 3))
        finally:
            conn.close()

    def test_review_log_table_has_expected_columns(self):
        conn = getConnection()
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(review_log)").fetchall()}
        finally:
            conn.close()
        self.assertTrue({"language", "vocab_id", "track", "rating", "state_before", "reviewed_ts",
                         "elapsed_days", "duration_ms"} <= cols)


if __name__ == "__main__":
    unittest.main()
