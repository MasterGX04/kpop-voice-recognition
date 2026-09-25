"""
Persistence for Japanese Kanji vocab + SRS state (vocab_ja/srs_card_ja/vocab_occurrence_ja - see
core/vocab_db.py). `entry` throughout is core.kanji_reference.analyzeSelection()'s per-word shape.
"""

import json

from core.vocab_db import getConnection
from core.vocab_store import now, computeNextReview


def _cognateColumns(entry: dict):
    cognate = entry.get("chineseCognate")
    if not cognate:
        return None, None, None, None
    status = cognate["status"]
    form = cognate.get("traditional")
    pinyin = cognate["pinyin"] if status == "confirmed" else cognate.get("pinyinFallback")
    glossJson = json.dumps(cognate["gloss"]) if status == "confirmed" else None
    return form, status, pinyin, glossJson


def upsertVocab(entry: dict, conn=None):
    """
    Upsert one word into vocab_ja by lemma, replacing its cognate/meaning columns with the
    freshly-analyzed values. Creates srs_card_ja only on first insert - meaning_state starts
    "known" (1) when the word has a confirmed Chinese cognate (Sino-Japanese vocabulary is
    typically already recognizable from the Chinese cognate alone), reading_state always starts
    at 0 (unseen) regardless, since reading is an active-recall skill a cognate doesn't shortcut.

    Pass an existing `conn` (and commit/close it yourself) when calling this in a loop - e.g.
    core.vocab_sync scanning hundreds of words - so the loop isn't paying for a fresh connection
    + full schema check + fsync-backed commit on every single word (measured: this was the actual
    cause of "Compile All Songs' Vocab" taking over a minute and freezing the UI the whole time).

    Returns (vocabId, isNew).
    """
    ownConn = conn is None
    if ownConn:
        conn = getConnection()
    try:
        cognateForm, cognateStatus, cognatePinyin, cognateGlossJson = _cognateColumns(entry)
        ts = now()
        row = conn.execute(
            "SELECT id, meaning_locked FROM vocab_ja WHERE lemma = ?", (entry["lemma"],)
        ).fetchone()

        if row:
            vocabId, meaningLocked = row
            # A rescan (core.vocab_sync re-running after a manual "Edit Meaning" in the Review
            # screen) must NOT clobber that manual edit back to the raw JMdict lookup - real bug
            # found via a user report: they hand-picked Hanja/meanings, re-ran Compile, and their
            # edits silently vanished because this UPDATE always overwrote every column
            # unconditionally. meaning_locked (set by updateMeaning()) is the fix.
            fields = {
                "lemma_reading": entry.get("lemmaReading"), "surface": entry.get("surface"),
                "category": entry.get("category"), "cognate_form": cognateForm,
                "cognate_status": cognateStatus, "cognate_pinyin": cognatePinyin,
                "cognate_gloss_json": cognateGlossJson,
                "mnemonic_pinyin_json": json.dumps(entry.get("mandarinPinyin")),
                "updated_at": ts,
            }
            if not meaningLocked:
                fields["meaning_json"] = json.dumps(entry.get("japaneseMeaning"))
            setClause = ", ".join(f"{col}=?" for col in fields)
            conn.execute(f"UPDATE vocab_ja SET {setClause} WHERE id=?", (*fields.values(), vocabId))
            isNew = False
        else:
            cursor = conn.execute(
                """INSERT INTO vocab_ja (
                       lemma, lemma_reading, surface, category, meaning_json, cognate_form,
                       cognate_status, cognate_pinyin, cognate_gloss_json, mnemonic_pinyin_json,
                       created_at, updated_at
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry["lemma"], entry.get("lemmaReading"), entry.get("surface"),
                    entry.get("category"), json.dumps(entry.get("japaneseMeaning")), cognateForm,
                    cognateStatus, cognatePinyin, cognateGlossJson,
                    json.dumps(entry.get("mandarinPinyin")), ts, ts,
                ),
            )
            vocabId = cursor.lastrowid
            isNew = True

            meaningState = 1 if cognateStatus == "confirmed" else 0
            conn.execute(
                """INSERT INTO srs_card_ja (
                       vocab_ja_id, meaning_state, reading_state, meaning_due_ts, reading_due_ts
                   ) VALUES (?, ?, 0, ?, ?)""",
                (vocabId, meaningState, ts, ts),
            )

        if ownConn:
            conn.commit()
        return vocabId, isNew
    finally:
        if ownConn:
            conn.close()


def addOccurrence(vocab_id: int, group: str, song: str, singer_names, lyric_line: str,
                   lyric_id, start_chunk, end_chunk, conn=None) -> bool:
    """Returns True if a new occurrence row was inserted, False if it already existed.

    Pass an existing `conn` when calling this in a loop - see upsertVocab()'s docstring."""
    ownConn = conn is None
    if ownConn:
        conn = getConnection()
    try:
        cursor = conn.execute(
            """INSERT OR IGNORE INTO vocab_occurrence_ja (
                   vocab_ja_id, group_name, song_title, singer_names_json, lyric_line, lyric_id,
                   start_chunk, end_chunk, added_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (vocab_id, group, song, json.dumps(singer_names), lyric_line, lyric_id,
             start_chunk, end_chunk, now()),
        )
        if ownConn:
            conn.commit()
        return cursor.rowcount > 0
    finally:
        if ownConn:
            conn.close()


def submitReview(vocab_id: int, track: str, rating: str):
    """track is "meaning" | "reading"."""
    conn = getConnection()
    try:
        boxCol, easeCol = f"{track}_box", f"{track}_ease"
        dueCol, lastCol = f"{track}_due_ts", f"{track}_last_reviewed_ts"
        row = conn.execute(
            f"SELECT {boxCol}, {easeCol} FROM srs_card_ja WHERE vocab_ja_id = ?", (vocab_id,)
        ).fetchone()
        if row is None:
            return
        newBox, newEase, dueTs = computeNextReview(row[0], row[1], rating)
        conn.execute(
            f"UPDATE srs_card_ja SET {boxCol}=?, {easeCol}=?, {dueCol}=?, {lastCol}=? WHERE vocab_ja_id=?",
            (newBox, newEase, dueTs, now(), vocab_id),
        )
        conn.commit()
    finally:
        conn.close()


_CARD_COLUMNS = (
    "v.id, v.lemma, v.lemma_reading, v.surface, v.category, v.meaning_json, "
    "v.cognate_form, v.cognate_pinyin, v.cognate_gloss_json, v.mnemonic_pinyin_json"
)


def _rowToCard(r) -> dict:
    return {
        "vocabId": r[0], "lemma": r[1], "lemmaReading": r[2], "surface": r[3],
        "category": r[4], "meaning": json.loads(r[5]) if r[5] else None,
        "cognateForm": r[6], "cognatePinyin": r[7],
        "cognateGloss": json.loads(r[8]) if r[8] else None,
        "mnemonicPinyin": json.loads(r[9]) if r[9] else None,
    }


def getDueCards(track: str, limit: int = 20) -> list:
    conn = getConnection()
    try:
        dueCol = f"{track}_due_ts"
        rows = conn.execute(
            f"""SELECT {_CARD_COLUMNS}
                FROM srs_card_ja s JOIN vocab_ja v ON v.id = s.vocab_ja_id
                WHERE s.{dueCol} <= ? ORDER BY s.{dueCol} ASC LIMIT ?""",
            (now(), limit),
        ).fetchall()
        return [_rowToCard(r) for r in rows]
    finally:
        conn.close()


def listAllVocab(limit: int = 500) -> list:
    """Every vocab_ja word regardless of SRS due status - for browsing/editing, not reviewing."""
    conn = getConnection()
    try:
        rows = conn.execute(
            f"SELECT {_CARD_COLUMNS} FROM vocab_ja v ORDER BY v.lemma LIMIT ?", (limit,)
        ).fetchall()
        return [_rowToCard(r) for r in rows]
    finally:
        conn.close()


def updateMeaning(vocab_id: int, gloss: list, pos=None):
    """
    Manually fill in/correct a word's meaning (e.g. a real JMdict miss). Sets meaning_locked so a
    later rescan (core.vocab_sync re-running upsertVocab) leaves this alone instead of silently
    overwriting it back to the raw JMdict lookup result.
    """
    conn = getConnection()
    try:
        row = conn.execute("SELECT meaning_json FROM vocab_ja WHERE id = ?", (vocab_id,)).fetchone()
        existing = json.loads(row[0]) if row and row[0] else {}
        meaning = {"status": "found", "pos": pos if pos is not None else existing.get("pos", []), "gloss": gloss}
        conn.execute(
            "UPDATE vocab_ja SET meaning_json = ?, meaning_locked = 1, updated_at = ? WHERE id = ?",
            (json.dumps(meaning), now(), vocab_id),
        )
        conn.commit()
    finally:
        conn.close()


def deleteVocab(vocab_id: int):
    """Delete a vocab_ja word entirely (cascades to its srs_card_ja/occurrences/cognate_link rows) -
    for glitched/unwanted entries."""
    conn = getConnection()
    try:
        conn.execute("DELETE FROM vocab_ja WHERE id = ?", (vocab_id,))
        conn.commit()
    finally:
        conn.close()


def getOccurrences(vocab_id: int, limit: int = 5) -> list:
    conn = getConnection()
    try:
        rows = conn.execute(
            """SELECT group_name, song_title, singer_names_json, lyric_line, start_chunk, end_chunk
               FROM vocab_occurrence_ja WHERE vocab_ja_id = ? LIMIT ?""",
            (vocab_id, limit),
        ).fetchall()
        return [
            {"group": r[0], "song": r[1], "singers": json.loads(r[2]) if r[2] else [],
             "lyricLine": r[3], "startChunk": r[4], "endChunk": r[5]}
            for r in rows
        ]
    finally:
        conn.close()
