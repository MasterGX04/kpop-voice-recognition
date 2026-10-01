"""
Persistence for Japanese Kanji vocab + SRS state (vocab_ja/srs_card_ja/vocab_occurrence_ja - see
core/vocab_db.py). `entry` throughout is core.kanji_reference.analyzeSelection()'s per-word shape.
"""

import json
from collections import defaultdict

from core.vocab_db import getConnection
from core import vocab_store
from core.kanji_reference import _containsKanji
from core.vocab_store import now


def _cognateColumns(entry: dict):
    cognate = entry.get("chineseCognate")
    if not cognate:
        return None, None, None, None
    status = cognate["status"]
    form = cognate.get("traditional")
    pinyin = cognate["pinyin"] if status == "confirmed" else cognate.get("pinyinFallback")
    glossJson = json.dumps(cognate["gloss"]) if status == "confirmed" else None
    return form, status, pinyin, glossJson


def pickSurface(lemma: str, existing, new):
    """
    Which spelling a word is displayed with (the flashcard prompt). vocab_ja is keyed by lemma, but
    every occurrence has its own surface, and upsertVocab() used to overwrite `surface` with
    whichever occurrence was scanned last - so 今 flipped to "いま" whenever a kana-spelled line
    (愛をいま) happened to come last. Rules, so the result no longer depends on scan order:
      - a kanji surface of a kanji lemma displays as the lemma (dictionary form: 離せる -> 離す);
      - otherwise never let a kana spelling replace an existing kanji one;
      - otherwise keep what's stored (a kana-only word keeps its first spelling).
    """
    if _containsKanji(new) and _containsKanji(lemma):
        return lemma
    if existing and (_containsKanji(existing) or not _containsKanji(new)):
        return existing
    return new


def upsertVocab(entry: dict, conn=None):
    """
    Upsert one word into vocab_ja by lemma, replacing its cognate/meaning columns with the
    freshly-analyzed values. Creates srs_card_ja only on first insert, with all three tracks New (FSRS state 0) and due
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
        cognateForm, cognateStatus, cognatePinyin, cognateGlossJson = _cognateColumns(entry)
        ts = now()
        row = conn.execute(
            "SELECT id, meaning_locked, surface, surface_locked FROM vocab_ja WHERE lemma = ?", (entry["lemma"],)
        ).fetchone()

        if row:
            vocabId, meaningLocked, existingSurface, surfaceLocked = row
            # A rescan (core.vocab_sync re-running after a manual "Edit Meaning" in the Review
            # screen) must NOT clobber that manual edit back to the raw JMdict lookup - real bug
            # found via a user report: they hand-picked Hanja/meanings, re-ran Compile, and their
            # edits silently vanished because this UPDATE always overwrote every column
            # unconditionally. meaning_locked (set by updateMeaning()) is the fix.
            fields = {
                "lemma_reading": entry.get("lemmaReading"),
                "category": entry.get("category"), "cognate_form": cognateForm,
                "cognate_status": cognateStatus, "cognate_pinyin": cognatePinyin,
                "cognate_gloss_json": cognateGlossJson,
                "mnemonic_pinyin_json": json.dumps(entry.get("mandarinPinyin")),
                "updated_at": ts,
            }
            if not surfaceLocked:
                fields["surface"] = pickSurface(entry["lemma"], existingSurface, entry.get("surface"))
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
                    entry["lemma"], entry.get("lemmaReading"),
                    pickSurface(entry["lemma"], None, entry.get("surface")), entry.get("category"), json.dumps(entry.get("japaneseMeaning")), cognateForm,
                    cognateStatus, cognatePinyin, cognateGlossJson,
                    json.dumps(entry.get("mandarinPinyin")), ts, ts,
                ),
            )
            vocabId = cursor.lastrowid
            isNew = True

            conn.execute(
                """INSERT INTO srs_card_ja (
                       vocab_ja_id, meaning_due_ts, reading_due_ts, cloze_due_ts
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
            """INSERT OR IGNORE INTO vocab_occurrence_ja (
                   vocab_ja_id, group_name, song_title, singer_names_json, lyric_line, lyric_id,
                   start_chunk, end_chunk, added_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (vocab_id, group, song, json.dumps(singer_names), lyric_line, lyric_id,
             start_chunk, end_chunk, now()),
        )
        isNew = cursor.rowcount > 0
        if not isNew:
            conn.execute(
                """UPDATE vocab_occurrence_ja
                   SET group_name = ?, song_title = ?, singer_names_json = ?, lyric_line = ?,
                       start_chunk = ?, end_chunk = ?
                   WHERE vocab_ja_id = ? AND lyric_id = ?""",
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
        vocab_store.submitReview(conn, "srs_card_ja", "vocab_ja_id", "ja", vocab_id, track,
                                 rating, nowTs, durationMs)
        conn.commit()
    finally:
        conn.close()


def previewIntervals(vocab_id: int, track: str, nowTs=None):
    """{again, hard, good, easy} -> seconds until due, or None if the word has no card."""
    conn = getConnection()
    try:
        return vocab_store.previewIntervals(conn, "srs_card_ja", "vocab_ja_id", vocab_id, track, nowTs)
    finally:
        conn.close()


_CARD_COLUMNS = (
    "v.id, v.lemma, v.lemma_reading, v.surface, v.category, v.meaning_json, "
    "v.cognate_form, v.cognate_pinyin, v.cognate_gloss_json, v.mnemonic_pinyin_json, "
    "(SELECT s.meaning_suspended AND s.reading_suspended AND s.cloze_suspended "
    "FROM srs_card_ja s WHERE s.vocab_ja_id = v.id)"
)


def _rowToCard(r) -> dict:
    return {
        "vocabId": r[0], "lemma": r[1], "lemmaReading": r[2], "surface": r[3],
        "category": r[4], "meaning": json.loads(r[5]) if r[5] else None,
        "cognateForm": r[6], "cognatePinyin": r[7],
        "cognateGloss": json.loads(r[8]) if r[8] else None,
        "mnemonicPinyin": json.loads(r[9]) if r[9] else None,
        "suspended": bool(r[10]),
    }


def _cardById(conn, vocabId):
    row = conn.execute(f"SELECT {_CARD_COLUMNS} FROM vocab_ja v WHERE v.id = ?", (vocabId,)).fetchone()
    return _rowToCard(row) if row else None


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


def getNextCard(track: str, nowTs=None, newLimit=vocab_store.NEW_PER_DAY,
                learnAheadSecs: int = vocab_store.LEARN_AHEAD_SECS, song=None, shuffleNew=False):
    """Next card for the study queue (see vocab_store.pickNextCard), with a "queueType" key
    ("learning" | "review" | "new"), or None when nothing is left. `song` = (group, song_title)
    limits the queue to that song's words; `shuffleNew` randomizes new-card order."""
    conn = getConnection()
    try:
        onlyIds = vocab_store.songVocabIds(conn, "vocab_occurrence_ja", *song) if song else None
        picked = vocab_store.pickNextCard(
            conn, "ja", "srs_card_ja", "vocab_ja_id", "vocab_occurrence_ja", "vocab_ja_id",
            track, nowTs, newLimit, learnAheadSecs, onlyIds=onlyIds, shuffleNew=shuffleNew,
        )
        if picked is None:
            return None
        vocabId, queueType = picked
        row = conn.execute(f"SELECT {_CARD_COLUMNS} FROM vocab_ja v WHERE v.id = ?", (vocabId,)).fetchone()
        card = _rowToCard(row)
        card["queueType"] = queueType
        return card
    finally:
        conn.close()


def _mutate(fn, vocab_ids, **kw):
    conn = getConnection()
    try:
        fn(conn, "srs_card_ja", "vocab_ja_id", [vocab_ids] if isinstance(vocab_ids, int) else vocab_ids, **kw)
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
            conn, "srs_card_ja", "vocab_ja_id", "vocab_ja", "vocab_occurrence_ja",
            "vocab_ja_id", "v.lemma_reading", limit)
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


def updateSurface(vocab_id: int, surface: str):
    """
    Manually set the spelling a word is displayed with (the flashcard prompt) - e.g. a kanji
    spelling for a kana-written word (どんな -> 何様) as a Chinese-reader mnemonic. Sets
    surface_locked so a rescan (pickSurface) leaves it alone. Display only: lemma, reading and the
    cloze drill (which blanks the real lyric text) are unaffected. Blank resets to automatic.
    """
    surface = (surface or "").strip()
    conn = getConnection()
    try:
        if surface:
            conn.execute(
                "UPDATE vocab_ja SET surface = ?, surface_locked = 1, updated_at = ? WHERE id = ?",
                (surface, now(), vocab_id),
            )
        else:
            conn.execute(
                "UPDATE vocab_ja SET surface = lemma, surface_locked = 0, updated_at = ? WHERE id = ?",
                (now(), vocab_id),
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


def listSongs() -> list:
    """[{"group", "song", "wordCount"}] for the Group -> Song picker."""
    conn = getConnection()
    try:
        return vocab_store.listSongs(conn, "vocab_occurrence_ja")
    finally:
        conn.close()


def listSongVocab(group: str, song: str) -> list:
    """Every word in one song as cards, in lyric order - no row cap (unlike listAllVocab)."""
    conn = getConnection()
    try:
        ids = vocab_store.songVocabIds(conn, "vocab_occurrence_ja", group, song)
        return [c for c in (_cardById(conn, i) for i in ids) if c]
    finally:
        conn.close()


def getOccurrences(vocab_id: int, limit: int = 5, group=None, song=None) -> list:
    """`group`/`song` (both) narrow to one song's occurrences, in lyric order."""
    conn = getConnection()
    try:
        where, args, order = "vocab_ja_id = ?", [vocab_id], ""
        if group is not None and song is not None:
            where += " AND group_name = ? AND song_title = ?"
            args += [group, song]
            order = "ORDER BY start_chunk, id"
        rows = conn.execute(
            f"""SELECT group_name, song_title, singer_names_json, lyric_line, start_chunk, end_chunk, lyric_id
               FROM vocab_occurrence_ja WHERE {where} {order} LIMIT ?""",
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
    """
    Cross-word co-occurrence graph (Milestone 4 of .claude/FLASHCARD_WEB_UPGRADE_PLAN.md): which
    words cluster together because they turn up in the same song(s) - a genuinely different
    question from getOccurrences() (every song *one* word appears in). Nodes are every vocab_ja
    word with at least one edge; edges connect two words that share `min_shared_songs`+ songs in
    common, weighted by how many songs they share (and which ones).

    Deliberately store-layer only, independent of any UI - see the plan's Milestone 4 scope note
    ("design and unit-test this query layer as its own reviewable step before building any graph
    UI on top of it").

    Returns {"nodes": [{"vocabId", "lemma"}, ...], "edges": [{"a", "b", "sharedSongs",
    "songs": [{"group", "song"}, ...]}, ...]}. `a` < `b` (vocabId order) so each pair appears
    exactly once, never twice.
    """
    conn = getConnection()
    try:
        rows = conn.execute(
            "SELECT DISTINCT vocab_ja_id, group_name, song_title FROM vocab_occurrence_ja"
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
            f"SELECT id, lemma FROM vocab_ja WHERE id IN ({placeholders})", tuple(nodeIds)
        ).fetchall()
        nodes = [{"vocabId": r[0], "lemma": r[1]} for r in lemmaRows]

        return {"nodes": nodes, "edges": edges}
    finally:
        conn.close()
