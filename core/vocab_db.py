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
import time

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
    reading_last_reviewed_ts INTEGER,
    cloze_box INTEGER NOT NULL DEFAULT 1,
    cloze_ease REAL NOT NULL DEFAULT 2.5,
    cloze_due_ts INTEGER NOT NULL,
    cloze_last_reviewed_ts INTEGER
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
    reading_last_reviewed_ts INTEGER,
    cloze_box INTEGER NOT NULL DEFAULT 1,
    cloze_ease REAL NOT NULL DEFAULT 2.5,
    cloze_due_ts INTEGER NOT NULL,
    cloze_last_reviewed_ts INTEGER
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

CREATE TABLE IF NOT EXISTS review_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    language TEXT NOT NULL,
    vocab_id INTEGER NOT NULL,
    track TEXT NOT NULL,
    rating TEXT NOT NULL,
    state_before INTEGER NOT NULL,
    stability_before REAL,
    difficulty_before REAL,
    due_before_ts INTEGER,
    reviewed_ts INTEGER NOT NULL,
    elapsed_days REAL,
    duration_ms INTEGER
);

CREATE INDEX IF NOT EXISTS idx_review_log_reviewed ON review_log(reviewed_ts);
CREATE INDEX IF NOT EXISTS idx_review_log_card ON review_log(language, vocab_id, track);
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
    # A hand-picked display spelling (e.g. どんな -> 何様 as a Chinese-reader mnemonic); rescans keep it.
    ("vocab_ja", "surface_locked", "INTEGER NOT NULL DEFAULT 0"),
    ("vocab_ko", "meaning_locked", "INTEGER NOT NULL DEFAULT 0"),
    ("vocab_ko", "hanja_locked", "INTEGER NOT NULL DEFAULT 0"),
    # Milestone 5 (.claude/FLASHCARD_WEB_UPGRADE_PLAN.md): an independent cloze-drill SRS track,
    # mirroring the meaning_*/reading_* column groups exactly. cloze_due_ts is NOT NULL with no
    # static literal default that means "due now" - ALTER TABLE ADD COLUMN only allows a constant
    # default, so it's added as 0 here and immediately backfilled to a real "due now" timestamp
    # below, once, only for the table(s) that just got the column added this call (see
    # _migrateColumns) - never unconditionally, or every process start would reset real cloze
    # review progress back to "due now" once already migrated.
    ("srs_card_ja", "cloze_box", "INTEGER NOT NULL DEFAULT 1"),
    ("srs_card_ja", "cloze_ease", "REAL NOT NULL DEFAULT 2.5"),
    ("srs_card_ja", "cloze_due_ts", "INTEGER NOT NULL DEFAULT 0"),
    ("srs_card_ja", "cloze_last_reviewed_ts", "INTEGER"),
    ("srs_card_ko", "cloze_box", "INTEGER NOT NULL DEFAULT 1"),
    ("srs_card_ko", "cloze_ease", "REAL NOT NULL DEFAULT 2.5"),
    ("srs_card_ko", "cloze_due_ts", "INTEGER NOT NULL DEFAULT 0"),
    ("srs_card_ko", "cloze_last_reviewed_ts", "INTEGER"),
]

# FSRS (.claude/FLASHCARD_FSRS_PLAN.md, M1): per track (meaning/reading/cloze) x language. `*_state`
# is 0 New / 1 Learning / 2 Review / 3 Relearning (see core.srs_fsrs). meaning_state/reading_state
# already existed as an unused "known" flag (0/1) - the ALTER is skipped for them, and the one-time
# backfill in _migrateColumns() rewrites their values into the FSRS meaning. The old *_box/*_ease
# columns stay in place, unused, so a rollback is possible.
for _table in ("srs_card_ja", "srs_card_ko"):
    for _track in ("meaning", "reading", "cloze"):
        _ADDED_COLUMNS += [
            (_table, f"{_track}_state", "INTEGER NOT NULL DEFAULT 0"),
            (_table, f"{_track}_step", "INTEGER NOT NULL DEFAULT 0"),
            (_table, f"{_track}_stability", "REAL"),
            (_table, f"{_track}_difficulty", "REAL"),
            (_table, f"{_track}_reps", "INTEGER NOT NULL DEFAULT 0"),
            (_table, f"{_track}_lapses", "INTEGER NOT NULL DEFAULT 0"),
            (_table, f"{_track}_suspended", "INTEGER NOT NULL DEFAULT 0"),
        ]

# The old Leitner ladder, only used to translate an old box into an initial FSRS stability.
_LEGACY_BOX_TO_DAYS = [1, 3, 7, 14, 30, 90]


def _backfillFsrs(conn: sqlite3.Connection, table: str):
    """Old rows -> FSRS. Never reviewed (last_reviewed_ts NULL) -> New. Reviewed -> Review with
    stability = the old box interval scaled by ease, difficulty mapped from ease (2.5 -> 5,
    1.3 -> 9, higher ease easier, clamped 1-10). Approximate on purpose: FSRS self-corrects within
    a couple of reviews. Runs only in the call that just added the *_stability columns."""
    boxDays = "CASE MIN(MAX({t}_box, 1), 6) " + " ".join(
        f"WHEN {i + 1} THEN {d}" for i, d in enumerate(_LEGACY_BOX_TO_DAYS)
    ) + " END"
    for track in ("meaning", "reading", "cloze"):
        days = boxDays.format(t=track)
        conn.execute(
            f"""UPDATE {table} SET
                    {track}_state = CASE WHEN {track}_last_reviewed_ts IS NULL THEN 0 ELSE 2 END,
                    {track}_stability = CASE WHEN {track}_last_reviewed_ts IS NULL THEN NULL
                        ELSE MAX(0.1, ({days}) * {track}_ease / 2.5) END,
                    {track}_difficulty = CASE WHEN {track}_last_reviewed_ts IS NULL THEN NULL
                        ELSE MIN(10.0, MAX(1.0, 5.0 - ({track}_ease - 2.5) * (4.0 / 1.2))) END,
                    {track}_reps = CASE WHEN {track}_last_reviewed_ts IS NULL THEN 0 ELSE 1 END"""
        )


def _migrateColumns(conn: sqlite3.Connection):
    justAddedClozeDue = set()
    justAddedFsrs = set()
    for table, column, ddlType in _ADDED_COLUMNS:
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddlType}")
            if column == "cloze_due_ts":
                justAddedClozeDue.add(table)
            if column == "meaning_stability":
                justAddedFsrs.add(table)

    # idx_srs_*_cloze_due can't live in the static _SCHEMA block (that executescript runs BEFORE
    # this function, so it would fail with "no such column" on a pre-existing DB the ALTER TABLE
    # loop above hasn't reached yet) - safe here since cloze_due_ts always exists by this point on
    # both a brand-new DB (created via _SCHEMA) and a migrated one (just ALTER'd above).
    nowTs = int(time.time())
    for table in justAddedFsrs:
        _backfillFsrs(conn, table)
    for table in ("srs_card_ja", "srs_card_ko"):
        if table in justAddedClozeDue:
            conn.execute(f"UPDATE {table} SET cloze_due_ts = ? WHERE cloze_due_ts = 0", (nowTs,))
        suffix = table.split("_")[-1]
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_srs_{suffix}_cloze_due ON {table}(cloze_due_ts)")


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
