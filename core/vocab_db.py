"""
SQLite schema and connection helper for the Japanese Kanji / Korean Hanja vocab + SRS store (see
.claude/ plan doc). Japanese and Korean get separate tables rather than one shared "vocab" table:
Japanese has a single onyomi/kunyomi/mixed/jukujigo category and at most one Chinese cognate, while
a Korean Hangul spelling can map to several unrelated real Hanja at once (e.g. 화 -> 火/禍/和/化/
畫/靴 - core/korean_hanja.py: lookupHanja()), so its cognate data needs a child table, not a scalar
column. cognate_link bridges the two languages when they share the same Chinese-cognate root.

Path is a plain relative "data/vocab_srs.db" (like saved_labels/ and kanji_reference/ elsewhere in
this project), not a __file__-relative absolute path, so tests that chdir into a throwaway tempdir
(see core/test_kanji_reference.py's SongVocabPersistenceTests) get a fully isolated DB.
"""

import os
import sqlite3

_DB_PATH = "data/vocab_srs.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS vocab_ja (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma TEXT NOT NULL UNIQUE,
    lemma_reading TEXT,
    surface TEXT,
    category TEXT,
    meaning_json TEXT,
    meaning_locked INTEGER NOT NULL DEFAULT 0,
    cognate_form TEXT,
    cognate_status TEXT,
    cognate_pinyin TEXT,
    cognate_gloss_json TEXT,
    mnemonic_pinyin_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS srs_card_ja (
    vocab_ja_id INTEGER PRIMARY KEY REFERENCES vocab_ja(id) ON DELETE CASCADE,
    meaning_state INTEGER NOT NULL DEFAULT 0,
    reading_state INTEGER NOT NULL DEFAULT 0,
    meaning_box INTEGER NOT NULL DEFAULT 1,
    meaning_ease REAL NOT NULL DEFAULT 2.5,
    meaning_due_ts INTEGER NOT NULL,
    meaning_last_reviewed_ts INTEGER,
    reading_box INTEGER NOT NULL DEFAULT 1,
    reading_ease REAL NOT NULL DEFAULT 2.5,
    reading_due_ts INTEGER NOT NULL,
    reading_last_reviewed_ts INTEGER
);

CREATE TABLE IF NOT EXISTS vocab_occurrence_ja (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vocab_ja_id INTEGER NOT NULL REFERENCES vocab_ja(id) ON DELETE CASCADE,
    group_name TEXT NOT NULL,
    song_title TEXT NOT NULL,
    singer_names_json TEXT,
    lyric_line TEXT,
    lyric_id TEXT,
    start_chunk INTEGER,
    end_chunk INTEGER,
    added_at TEXT NOT NULL,
    UNIQUE(vocab_ja_id, lyric_id)
);

CREATE TABLE IF NOT EXISTS vocab_ko (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    lemma TEXT NOT NULL UNIQUE,
    surface TEXT,
    meaning_json TEXT,
    meaning_locked INTEGER NOT NULL DEFAULT 0,
    hanja_locked INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS vocab_ko_hanja (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vocab_ko_id INTEGER NOT NULL REFERENCES vocab_ko(id) ON DELETE CASCADE,
    hanja_form TEXT NOT NULL,
    pinyin TEXT,
    pos TEXT,
    gloss_json TEXT
);

CREATE TABLE IF NOT EXISTS srs_card_ko (
    vocab_ko_id INTEGER PRIMARY KEY REFERENCES vocab_ko(id) ON DELETE CASCADE,
    meaning_state INTEGER NOT NULL DEFAULT 0,
    reading_state INTEGER NOT NULL DEFAULT 0,
    meaning_box INTEGER NOT NULL DEFAULT 1,
    meaning_ease REAL NOT NULL DEFAULT 2.5,
    meaning_due_ts INTEGER NOT NULL,
    meaning_last_reviewed_ts INTEGER,
    reading_box INTEGER NOT NULL DEFAULT 1,
    reading_ease REAL NOT NULL DEFAULT 2.5,
    reading_due_ts INTEGER NOT NULL,
    reading_last_reviewed_ts INTEGER
);

CREATE TABLE IF NOT EXISTS vocab_occurrence_ko (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vocab_ko_id INTEGER NOT NULL REFERENCES vocab_ko(id) ON DELETE CASCADE,
    group_name TEXT NOT NULL,
    song_title TEXT NOT NULL,
    singer_names_json TEXT,
    lyric_line TEXT,
    lyric_id TEXT,
    start_chunk INTEGER,
    end_chunk INTEGER,
    added_at TEXT NOT NULL,
    UNIQUE(vocab_ko_id, lyric_id)
);

CREATE TABLE IF NOT EXISTS cognate_link (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    vocab_ja_id INTEGER NOT NULL REFERENCES vocab_ja(id) ON DELETE CASCADE,
    vocab_ko_id INTEGER NOT NULL REFERENCES vocab_ko(id) ON DELETE CASCADE,
    cognate_form TEXT NOT NULL,
    UNIQUE(vocab_ja_id, vocab_ko_id)
);

CREATE INDEX IF NOT EXISTS idx_srs_ja_meaning_due ON srs_card_ja(meaning_due_ts);
CREATE INDEX IF NOT EXISTS idx_srs_ja_reading_due ON srs_card_ja(reading_due_ts);
CREATE INDEX IF NOT EXISTS idx_srs_ko_meaning_due ON srs_card_ko(meaning_due_ts);
CREATE INDEX IF NOT EXISTS idx_srs_ko_reading_due ON srs_card_ko(reading_due_ts);
CREATE INDEX IF NOT EXISTS idx_occurrence_ja_vocab ON vocab_occurrence_ja(vocab_ja_id);
CREATE INDEX IF NOT EXISTS idx_occurrence_ko_vocab ON vocab_occurrence_ko(vocab_ko_id);
CREATE INDEX IF NOT EXISTS idx_cognate_link_form ON cognate_link(cognate_form);
"""


_initializedDbPaths = set()

# (table, column, ddlType) added after the original schema shipped - existing on-disk databases
# need these bolted on with ALTER TABLE, since "CREATE TABLE IF NOT EXISTS" is a no-op once the
# table already exists (confirmed real: a user's actual data/vocab_srs.db, already populated from
# an earlier session, was missing these entirely - _SCHEMA alone would never have added them).
_ADDED_COLUMNS = [
    ("vocab_ja", "meaning_locked", "INTEGER NOT NULL DEFAULT 0"),
    ("vocab_ko", "meaning_locked", "INTEGER NOT NULL DEFAULT 0"),
    ("vocab_ko", "hanja_locked", "INTEGER NOT NULL DEFAULT 0"),
]


def _migrateColumns(conn: sqlite3.Connection):
    for table, column, ddlType in _ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddlType}")


def getConnection() -> sqlite3.Connection:
    """
    Open a connection to data/vocab_srs.db, creating the schema on first use. Foreign keys are
    off by default in sqlite3 - turned on explicitly so ON DELETE CASCADE actually cascades.

    The schema script (and the column migration below) only actually run once per distinct
    database file per process - re-running 8 CREATE TABLE + 7 CREATE INDEX statements (even as
    harmless no-ops) on every single getConnection() call added up to real, measured overhead once
    a caller opens thousands of short-lived connections in a loop (core.vocab_sync's compile-all-
    songs scan: ~4,262 calls for a full library). Keyed by absolute path (not a single global flag)
    so tests that chdir into a fresh tempdir per test still get their own database properly
    initialized.

    timeout=30 (vs. sqlite3's 5s default) gives a bit more headroom for a long-running bulk write
    (core.vocab_sync holds one connection open for an entire scan) to not collide with a quick
    interactive call (e.g. the Vocab Review screen) elsewhere in the app.
    """
    absPath = os.path.abspath(_DB_PATH)
    os.makedirs(os.path.dirname(absPath), exist_ok=True)
    conn = sqlite3.connect(absPath, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    if absPath not in _initializedDbPaths:
        conn.executescript(_SCHEMA)
        _migrateColumns(conn)
        conn.commit()
        _initializedDbPaths.add(absPath)
    return conn
