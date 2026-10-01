"""
Persistence for Korean Hanja vocab + SRS state (vocab_ko/vocab_ko_hanja/srs_card_ko/
vocab_occurrence_ko - see core/vocab_db.py). `entry` throughout is
core.korean_vocab.analyzeKoreanSelection()'s per-word shape.
"""

import json
from collections import defaultdict

from core.vocab_db import getConnection
from core import vocab_store
from core.vocab_store import now


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

    Creates srs_card_ko only on first insert, with all three tracks New (FSRS state 0) and due
    now - see core/srs_fsrs.py.

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
            conn.execute(
                """INSERT INTO srs_card_ko (
                       vocab_ko_id, meaning_due_ts, reading_due_ts, cloze_due_ts
                   ) VALUES (?, ?, ?, ?)""",
                (vocabId, ts, ts, ts),
            )

        if ownConn:
            conn.commit()
        return vocabId, isNew
    finally:
        if ownConn:
            conn.close()


def addOccurrence(vocab_id: int, group: str, song: str, singer_names, lyric_line: str,
                   lyric_id, start_chunk, end_chunk, conn=None) -> bool:
    """Returns True if a new occurrence row was inserted, False if it already existed (in which
    case its mutable fields - notably start_chunk/end_chunk - are refreshed in place, so re-running
    a scan after the label-run resolver (core.label_runs) or the lyric text itself changes actually
    updates the cached span instead of leaving a stale one behind forever).

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
        isNew = cursor.rowcount > 0
        if not isNew:
            conn.execute(
                """UPDATE vocab_occurrence_ko
                   SET group_name = ?, song_title = ?, singer_names_json = ?, lyric_line = ?,
                       start_chunk = ?, end_chunk = ?
                   WHERE vocab_ko_id = ? AND lyric_id = ?""",
                (group, song, json.dumps(singer_names), lyric_line, start_chunk, end_chunk,
                 vocab_id, lyric_id),
            )
        if ownConn:
            conn.commit()
        return isNew
    finally:
        if ownConn:
            conn.close()


def submitReview(vocab_id: int, track: str, rating: str, nowTs=None, durationMs=None):
    """track is "meaning" | "reading" | "cloze"; rating is "again" | "hard" | "good" | "easy"."""
    conn = getConnection()
    try:
        vocab_store.submitReview(conn, "srs_card_ko", "vocab_ko_id", "ko", vocab_id, track,
                                 rating, nowTs, durationMs)
        conn.commit()
    finally:
        conn.close()


def previewIntervals(vocab_id: int, track: str, nowTs=None):
    """{again, hard, good, easy} -> seconds until due, or None if the word has no card."""
    conn = getConnection()
    try:
        return vocab_store.previewIntervals(conn, "srs_card_ko", "vocab_ko_id", vocab_id, track, nowTs)
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
        "suspended": bool(conn.execute(
            "SELECT meaning_suspended AND reading_suspended AND cloze_suspended "
            "FROM srs_card_ko WHERE vocab_ko_id = ?", (r[0],)).fetchone()[0]),
    }


def _cardById(conn, vocabId):
    row = conn.execute(
        "SELECT id, lemma, surface, meaning_json FROM vocab_ko WHERE id = ?", (vocabId,)).fetchone()
    return _rowToCard(conn, row) if row else None


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


def getNextCard(track: str, nowTs=None, newLimit=vocab_store.NEW_PER_DAY,
                learnAheadSecs: int = vocab_store.LEARN_AHEAD_SECS, song=None, shuffleNew=False):
    """Next card for the study queue (see vocab_store.pickNextCard), with a "queueType" key
    ("learning" | "review" | "new"), or None when nothing is left. `song` = (group, song_title)
    limits the queue to that song's words; `shuffleNew` randomizes new-card order."""
    conn = getConnection()
    try:
        onlyIds = vocab_store.songVocabIds(conn, "vocab_occurrence_ko", *song) if song else None
        picked = vocab_store.pickNextCard(
            conn, "ko", "srs_card_ko", "vocab_ko_id", "vocab_occurrence_ko", "vocab_ko_id",
            track, nowTs, newLimit, learnAheadSecs, onlyIds=onlyIds, shuffleNew=shuffleNew,
        )
        if picked is None:
            return None
        vocabId, queueType = picked
        row = conn.execute(
            "SELECT id, lemma, surface, meaning_json FROM vocab_ko WHERE id = ?", (vocabId,)
        ).fetchone()
        card = _rowToCard(conn, row)
        card["queueType"] = queueType
        return card
    finally:
        conn.close()


def _mutate(fn, vocab_ids, **kw):
    conn = getConnection()
    try:
        fn(conn, "srs_card_ko", "vocab_ko_id", [vocab_ids] if isinstance(vocab_ids, int) else vocab_ids, **kw)
        conn.commit()
    finally:
        conn.close()


def setSuspended(vocab_ids, suspended: bool):
    """Suspend/unsuspend every track of the given word(s) (an id or a list of ids)."""
    _mutate(vocab_store.setSuspended, vocab_ids, suspended=suspended)


def markKnown(vocab_ids, nowTs=None):
    """"I already know this" for the given word(s) - see vocab_store.markKnown."""
    _mutate(vocab_store.markKnown, vocab_ids, nowTs=nowTs)


def listTriageCandidates(limit: int = 100) -> list:
    conn = getConnection()
    try:
        return vocab_store.triageCandidates(
            conn, "srs_card_ko", "vocab_ko_id", "vocab_ko", "vocab_occurrence_ko",
            "vocab_ko_id", "NULL", limit)
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


def listSongs() -> list:
    """[{"group", "song", "wordCount"}] for the Group -> Song picker."""
    conn = getConnection()
    try:
        return vocab_store.listSongs(conn, "vocab_occurrence_ko")
    finally:
        conn.close()


def listSongVocab(group: str, song: str) -> list:
    """Every word in one song as cards, in lyric order - no row cap (unlike listAllVocab)."""
    conn = getConnection()
    try:
        ids = vocab_store.songVocabIds(conn, "vocab_occurrence_ko", group, song)
        return [c for c in (_cardById(conn, i) for i in ids) if c]
    finally:
        conn.close()


def getOccurrences(vocab_id: int, limit: int = 5, group=None, song=None) -> list:
    """`group`/`song` (both) narrow to one song's occurrences, in lyric order."""
    conn = getConnection()
    try:
        where, args, order = "vocab_ko_id = ?", [vocab_id], ""
        if group is not None and song is not None:
            where += " AND group_name = ? AND song_title = ?"
            args += [group, song]
            order = "ORDER BY start_chunk, id"
        rows = conn.execute(
            f"""SELECT group_name, song_title, singer_names_json, lyric_line, start_chunk, end_chunk, lyric_id
               FROM vocab_occurrence_ko WHERE {where} {order} LIMIT ?""",
            (*args, limit),
        ).fetchall()
        return [
            {"group": r[0], "song": r[1], "singers": json.loads(r[2]) if r[2] else [],
             "lyricLine": r[3], "startChunk": r[4], "endChunk": r[5], "lyricId": r[6]}
            for r in rows
        ]
    finally:
        conn.close()


def getCoOccurrenceGraph(min_shared_songs: int = 1) -> dict:
    """Korean counterpart of core.vocab_store_ja.getCoOccurrenceGraph() - see its docstring for the
    full design (Milestone 4 of .claude/FLASHCARD_WEB_UPGRADE_PLAN.md). Same shape, same
    store-layer-only scope, just over vocab_occurrence_ko/vocab_ko instead."""
    conn = getConnection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT vocab_ko_id, group_name, song_title FROM vocab_occurrence_ko"
        ).fetchall()

        wordsBySong = defaultdict(set)
        for vocabId, group, song in rows:
            wordsBySong[(group, song)].add(vocabId)

        pairSongs = defaultdict(set)
        for song, words in wordsBySong.items():
            wordList = sorted(words)
            for i in range(len(wordList)):
                for j in range(i + 1, len(wordList)):
                    pairSongs[(wordList[i], wordList[j])].add(song)

        edges = [
            {
                "a": a, "b": b, "sharedSongs": len(songs),
                "songs": [{"group": g, "song": s} for g, s in sorted(songs)],
            }
            for (a, b), songs in pairSongs.items()
            if len(songs) >= min_shared_songs
        ]

        nodeIds = {vocabId for edge in edges for vocabId in (edge["a"], edge["b"])}
        if not nodeIds:
            return {"nodes": [], "edges": []}

        placeholders = ",".join("?" for _ in nodeIds)
        lemmaRows = conn.execute(
            f"SELECT id, lemma FROM vocab_ko WHERE id IN ({placeholders})", tuple(nodeIds)
        ).fetchall()
        nodes = [{"vocabId": r[0], "lemma": r[1]} for r in lemmaRows]

        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()
