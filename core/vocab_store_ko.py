"""
Persistence for Korean Hanja vocab + SRS state (vocab_ko/vocab_ko_hanja/srs_card_ko/
vocab_occurrence_ko - see core/vocab_db.py). `entry` throughout is
core.korean_vocab.analyzeKoreanSelection()'s per-word shape.
"""

import json

from core.vocab_db import getConnection
from core.vocab_store import now, computeNextReview


def upsertVocab(entry: dict, conn=None):
    """
    Upsert one word into vocab_ko by lemma. For a brand-new word, its vocab_ko_hanja candidate
    rows are inserted fresh from `entry`. For an already-known word, a rescan must NOT clobber
    manual curation done in the Review screen - real bug found via a user report: they hand-picked
    the correct Hanja for several words (keepOnlyHanjaCandidate/clearHanjaCandidates) and manually
    filled in meanings (updateMeaning), then re-ran Compile, and every one of those edits silently
    reverted because this function always deleted+reinserted the full candidate list and
    overwrote meaning_json unconditionally, with no idea a human had already decided. hanja_locked
    and meaning_locked (set by those exact functions) are the fix - a rescan just leaves a locked
    word's candidates/meaning alone, still refreshing `surface` (harmless, cosmetic) either way.

    Creates srs_card_ko only on first insert - meaning_state starts "known" (1) when at least one
    real Hanja candidate was found (Sino-Korean vocabulary), reading_state always starts at 0
    (Hangul already IS the reading, but the word's active recall still starts unseen).

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
        ts = now()
        row = conn.execute(
            "SELECT id, meaning_locked, hanja_locked FROM vocab_ko WHERE lemma = ?", (entry["lemma"],)
        ).fetchone()

        if row:
            vocabId, meaningLocked, hanjaLocked = row
            fields = {"surface": entry.get("surface"), "updated_at": ts}
            if not meaningLocked:
                fields["meaning_json"] = json.dumps(entry.get("meaning"))
            setClause = ", ".join(f"{col}=?" for col in fields)
            conn.execute(f"UPDATE vocab_ko SET {setClause} WHERE id=?", (*fields.values(), vocabId))
            isNew = False
        else:
            cursor = conn.execute(
                "INSERT INTO vocab_ko (lemma, surface, meaning_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (entry["lemma"], entry.get("surface"), json.dumps(entry.get("meaning")), ts, ts),
            )
            vocabId = cursor.lastrowid
            isNew = True
            hanjaLocked = False

        candidates = entry.get("hanjaCandidates") or []
        if not hanjaLocked:
            conn.execute("DELETE FROM vocab_ko_hanja WHERE vocab_ko_id = ?", (vocabId,))
            for c in candidates:
                conn.execute(
                    "INSERT INTO vocab_ko_hanja (vocab_ko_id, hanja_form, pinyin, pos, gloss_json) VALUES (?, ?, ?, ?, ?)",
                    (vocabId, c["hanja"], c.get("pinyin"), c.get("pos"), json.dumps(c.get("gloss"))),
                )

        if isNew:
            meaningState = 1 if candidates else 0
            conn.execute(
                """INSERT INTO srs_card_ko (
                       vocab_ko_id, meaning_state, reading_state, meaning_due_ts, reading_due_ts
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
            """INSERT OR IGNORE INTO vocab_occurrence_ko (
                   vocab_ko_id, group_name, song_title, singer_names_json, lyric_line, lyric_id,
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
    conn = getConnection()
    try:
        boxCol, easeCol = f"{track}_box", f"{track}_ease"
        dueCol, lastCol = f"{track}_due_ts", f"{track}_last_reviewed_ts"
        row = conn.execute(
            f"SELECT {boxCol}, {easeCol} FROM srs_card_ko WHERE vocab_ko_id = ?", (vocab_id,)
        ).fetchone()
        if row is None:
            return
        newBox, newEase, dueTs = computeNextReview(row[0], row[1], rating)
        conn.execute(
            f"UPDATE srs_card_ko SET {boxCol}=?, {easeCol}=?, {dueCol}=?, {lastCol}=? WHERE vocab_ko_id=?",
            (newBox, newEase, dueTs, now(), vocab_id),
        )
        conn.commit()
    finally:
        conn.close()


def _hanjaCandidatesFor(conn, vocab_id: int) -> list:
    hanjaRows = conn.execute(
        "SELECT id, hanja_form, pinyin, pos, gloss_json FROM vocab_ko_hanja WHERE vocab_ko_id = ?", (vocab_id,)
    ).fetchall()
    return [
        {"hanjaId": h[0], "hanja": h[1], "pinyin": h[2], "pos": h[3], "gloss": json.loads(h[4]) if h[4] else []}
        for h in hanjaRows
    ]


def _rowToCard(conn, r) -> dict:
    return {
        "vocabId": r[0], "lemma": r[1], "surface": r[2],
        "meaning": json.loads(r[3]) if r[3] else None,
        "hanjaCandidates": _hanjaCandidatesFor(conn, r[0]),
    }


def getDueCards(track: str, limit: int = 20) -> list:
    conn = getConnection()
    try:
        dueCol = f"{track}_due_ts"
        rows = conn.execute(
            f"""SELECT v.id, v.lemma, v.surface, v.meaning_json
                FROM srs_card_ko s JOIN vocab_ko v ON v.id = s.vocab_ko_id
                WHERE s.{dueCol} <= ? ORDER BY s.{dueCol} ASC LIMIT ?""",
            (now(), limit),
        ).fetchall()
        return [_rowToCard(conn, r) for r in rows]
    finally:
        conn.close()


def listAllVocab(limit: int = 500) -> list:
    """Every vocab_ko word regardless of SRS due status - for browsing/editing, not reviewing."""
    conn = getConnection()
    try:
        rows = conn.execute(
            "SELECT id, lemma, surface, meaning_json FROM vocab_ko ORDER BY lemma LIMIT ?", (limit,)
        ).fetchall()
        return [_rowToCard(conn, r) for r in rows]
    finally:
        conn.close()


def updateMeaning(vocab_id: int, gloss: list, pos=None):
    """
    Manually fill in/correct a word's meaning (e.g. a real Wiktionary miss). Sets meaning_locked so
    a later rescan (core.vocab_sync re-running upsertVocab) leaves this alone instead of silently
    overwriting it back to the raw lookup result.
    """
    conn = getConnection()
    try:
        row = conn.execute("SELECT meaning_json FROM vocab_ko WHERE id = ?", (vocab_id,)).fetchone()
        existing = json.loads(row[0]) if row and row[0] else {}
        meaning = {"status": "found", "pos": pos if pos is not None else existing.get("pos", ""), "gloss": gloss}
        conn.execute(
            "UPDATE vocab_ko SET meaning_json = ?, meaning_locked = 1, updated_at = ? WHERE id = ?",
            (json.dumps(meaning), now(), vocab_id),
        )
        conn.commit()
    finally:
        conn.close()


def deleteVocab(vocab_id: int):
    """Delete a vocab_ko word entirely (cascades to its hanja/srs_card_ko/occurrences/cognate_link
    rows) - for glitched entries, e.g. a vowel-contraction fusion artifact like "세계+ᆯ"."""
    conn = getConnection()
    try:
        conn.execute("DELETE FROM vocab_ko WHERE id = ?", (vocab_id,))
        conn.commit()
    finally:
        conn.close()


def keepOnlyHanjaCandidate(vocab_id: int, hanja_id: int):
    """
    Resolve an ambiguous word (e.g. 화 -> 火/禍/和/化/畫/靴) down to the one real candidate the
    user picked, deleting the rest so they stop cluttering the review screen. Also drops any
    cognate_link row pointing at a candidate that's no longer kept, and re-syncs links so the kept
    form gets linked if it wasn't already (see core.vocab_link).

    Sets hanja_locked so a later rescan (core.vocab_sync re-running upsertVocab) leaves this word's
    candidates alone instead of silently re-adding back everything just deleted here - real bug
    found via a user report: they resolved several ambiguous words, re-ran Compile, and every
    resolution reverted to the full raw candidate list because upsertVocab always re-derived it
    fresh from the dictionary lookup with no idea a human had already decided.
    """
    conn = getConnection()
    try:
        row = conn.execute(
            "SELECT hanja_form FROM vocab_ko_hanja WHERE id = ? AND vocab_ko_id = ?", (hanja_id, vocab_id)
        ).fetchone()
        if row is None:
            return
        keepForm = row[0]
        conn.execute("DELETE FROM vocab_ko_hanja WHERE vocab_ko_id = ? AND id != ?", (vocab_id, hanja_id))
        conn.execute(
            "DELETE FROM cognate_link WHERE vocab_ko_id = ? AND cognate_form != ?", (vocab_id, keepForm)
        )
        conn.execute("UPDATE vocab_ko SET hanja_locked = 1 WHERE id = ?", (vocab_id,))
        conn.commit()
    finally:
        conn.close()

    from core import vocab_link
    vocab_link.syncCognateLinks()


def clearHanjaCandidates(vocab_id: int):
    """
    Remove ALL Hanja candidates for a word, marking it as fully native. Handles the case
    keepOnlyHanjaCandidate() can't: a word with only a single (wrong) candidate, not an ambiguous
    2+ - e.g. 해 ("sun/day", a genuinely native word) getting matched against 害 ("harm"), a real
    but unrelated Sino-Korean homophone that only actually applies to compounds like 재해/유해, not
    to bare 해 meaning "sun". Also drops any now-stale cognate_link rows for this word.

    Sets hanja_locked, same reasoning as keepOnlyHanjaCandidate() - without it, the next rescan
    would silently re-add the exact candidates just removed here.
    """
    conn = getConnection()
    try:
        conn.execute("DELETE FROM vocab_ko_hanja WHERE vocab_ko_id = ?", (vocab_id,))
        conn.execute("DELETE FROM cognate_link WHERE vocab_ko_id = ?", (vocab_id,))
        conn.execute("UPDATE vocab_ko SET hanja_locked = 1 WHERE id = ?", (vocab_id,))
        conn.commit()
    finally:
        conn.close()


def getOccurrences(vocab_id: int, limit: int = 5) -> list:
    conn = getConnection()
    try:
        rows = conn.execute(
            """SELECT group_name, song_title, singer_names_json, lyric_line, start_chunk, end_chunk
               FROM vocab_occurrence_ko WHERE vocab_ko_id = ? LIMIT ?""",
            (vocab_id, limit),
        ).fetchall()
        return [
            {"group": r[0], "song": r[1], "singers": json.loads(r[2]) if r[2] else [],
             "lyricLine": r[3], "startChunk": r[4], "endChunk": r[5]}
            for r in rows
        ]
    finally:
        conn.close()
