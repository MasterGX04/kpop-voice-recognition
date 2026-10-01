"""
Language-agnostic SRS plumbing shared by core/vocab_store_ja.py and core/vocab_store_ko.py. The
scheduling math itself lives in core/srs_fsrs.py (FSRS); this module only maps a srs_card_ja/
srs_card_ko row <-> a card-state dict and writes the review_log row, so both languages share one
implementation.
"""

import json
import random
import time
from datetime import datetime
from typing import Optional

from core import srs_fsrs

TRACKS = ("meaning", "reading", "cloze")


def now() -> int:
    return int(time.time())


def _checkTrack(track: str):
    # Track names are interpolated into column names below - never let anything else through.
    if track not in TRACKS:
        raise ValueError(f"unknown track {track!r}")


def _loadCardState(conn, table: str, idCol: str, vocabId: int, track: str) -> Optional[dict]:
    row = conn.execute(
        f"""SELECT {track}_state, {track}_step, {track}_stability, {track}_difficulty,
                   {track}_due_ts, {track}_last_reviewed_ts, {track}_reps, {track}_lapses
            FROM {table} WHERE {idCol} = ?""",
        (vocabId,),
    ).fetchone()
    if row is None:
        return None
    return {"state": row[0], "step": row[1], "stability": row[2], "difficulty": row[3],
            "dueTs": row[4], "lastReviewTs": row[5], "reps": row[6], "lapses": row[7]}


def submitReview(conn, table: str, idCol: str, language: str, vocabId: int, track: str,
                 rating: str, nowTs: Optional[int] = None, durationMs: Optional[int] = None) -> bool:
    """Apply one rating ("again|hard|good|easy") to a card and append the review_log row, in the
    caller's transaction (commit is the caller's). Returns False if the word has no card."""
    _checkTrack(track)
    if rating not in srs_fsrs.RATINGS:
        raise ValueError(f"unknown rating {rating!r}")
    nowTs = now() if nowTs is None else nowTs
    before = _loadCardState(conn, table, idCol, vocabId, track)
    if before is None:
        return False

    after = srs_fsrs.review(before, rating, nowTs)
    lapsed = before["state"] == srs_fsrs.STATE_REVIEW and rating == "again"
    elapsedDays = (
        (nowTs - before["lastReviewTs"]) / 86400 if before["lastReviewTs"] else None
    )

    conn.execute(
        f"""UPDATE {table} SET {track}_state=?, {track}_step=?, {track}_stability=?,
                {track}_difficulty=?, {track}_due_ts=?, {track}_last_reviewed_ts=?,
                {track}_reps={track}_reps + 1, {track}_lapses={track}_lapses + ?
            WHERE {idCol}=?""",
        (after["state"], after["step"], after["stability"], after["difficulty"],
         after["dueTs"], nowTs, 1 if lapsed else 0, vocabId),
    )
    conn.execute(
        """INSERT INTO review_log (language, vocab_id, track, rating, state_before,
               stability_before, difficulty_before, due_before_ts, reviewed_ts, elapsed_days,
               duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (language, vocabId, track, rating, before["state"], before["stability"],
         before["difficulty"], before["dueTs"], nowTs, elapsedDays, durationMs),
    )
    return True


def previewIntervals(conn, table: str, idCol: str, vocabId: int, track: str,
                     nowTs: Optional[int] = None) -> Optional[dict]:
    """{again, hard, good, easy} -> seconds until due, for the four rating buttons."""
    _checkTrack(track)
    nowTs = now() if nowTs is None else nowTs
    state = _loadCardState(conn, table, idCol, vocabId, track)
    if state is None:
        return None
    return srs_fsrs.previewIntervals(state, nowTs)


# ---------------------------------------------------------------------------------------------
# Anki-style study queue (.claude/FLASHCARD_FSRS_PLAN.md, M3). Stateless: everything is derived
# from the srs_card_* rows and review_log, so the Python side holds no session state.
# ---------------------------------------------------------------------------------------------

# None = no daily cap on new cards (user preference: fast memorizer). M5 exposes this as a setting.
NEW_PER_DAY = None
# If nothing else is left, a learning card due within this window is shown early rather than
# ending the session with a "done" screen while a 1-10 minute step is still pending (Anki's
# "learn ahead" behaviour).
LEARN_AHEAD_SECS = 20 * 60


def _startOfDay(nowTs: int) -> int:
    """Local midnight - the daily new-card cap and sibling burying both reset here."""
    d = datetime.fromtimestamp(nowTs)
    return int(d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())


def newCardsIntroducedToday(conn, language: str, nowTs: int) -> int:
    """First-ever reviews today for this language (state_before = 0), across all tracks - the
    daily counter is derived from review_log, no separate stored counter to drift."""
    return conn.execute(
        "SELECT COUNT(*) FROM review_log WHERE language = ? AND state_before = 0 AND reviewed_ts >= ?",
        (language, _startOfDay(nowTs)),
    ).fetchone()[0]


def _buriedIds(conn, language: str, track: str, nowTs: int) -> set:
    """Words with a DIFFERENT track reviewed today: their sibling cards wait until tomorrow so a
    cloze card can't leak the meaning answer (and vice versa). Same-track relearning is allowed."""
    rows = conn.execute(
        "SELECT DISTINCT vocab_id FROM review_log WHERE language = ? AND track != ? AND reviewed_ts >= ?",
        (language, track, _startOfDay(nowTs)),
    ).fetchall()
    return {r[0] for r in rows}


def listSongs(conn, occTable: str) -> list:
    """Every (group, song) that has vocab occurrences, with its distinct word count - the data
    behind the flashcard screen's Group -> Song picker. Sorted group then song, case-insensitive."""
    rows = conn.execute(
        f"SELECT group_name, song_title, COUNT(DISTINCT {_occIdCol(occTable)}) "
        f"FROM {occTable} GROUP BY group_name, song_title"
    ).fetchall()
    songs = [{"group": g, "song": t, "wordCount": n} for g, t, n in rows]
    songs.sort(key=lambda x: (x["group"].casefold(), x["song"].casefold()))
    return songs


def _occIdCol(occTable: str) -> str:
    return "vocab_ja_id" if occTable.endswith("_ja") else "vocab_ko_id"


def songVocabIds(conn, occTable: str, group: str, song: str) -> list:
    """Vocab ids appearing in one song, in the order they first occur in it (so the list reads
    like the lyric sheet, which is what makes finding a specific line quick)."""
    idCol = _occIdCol(occTable)
    rows = conn.execute(
        f"SELECT {idCol}, MIN(COALESCE(start_chunk, 0)), MIN(id) FROM {occTable} "
        f"WHERE group_name = ? AND song_title = ? GROUP BY {idCol} ORDER BY 2, 3",
        (group, song),
    ).fetchall()
    return [r[0] for r in rows]


def _complementIds(conn, table: str, idCol: str, keepIds) -> set:
    keep = set(keepIds)
    return {r[0] for r in conn.execute(f"SELECT {idCol} FROM {table}")} - keep


def pickNextCard(conn, language: str, table: str, idCol: str, occTable: str, occIdCol: str,
                 track: str, nowTs: Optional[int] = None, newLimit: Optional[int] = NEW_PER_DAY,
                 learnAheadSecs: int = LEARN_AHEAD_SECS, onlyIds=None, shuffleNew: bool = False):
    """
    Returns (vocabId, queueType) for the next card to study, or None when the session is done.
    Order: 1) learning/relearning cards due now (earliest first), 2) review cards due, lowest
    retrievability first with a random tiebreak (so the order changes every session), 3) new cards
    while today's cap allows, most-frequent-in-your-lyrics first, 4) learn-ahead. Suspended cards
    and buried siblings are never returned.

    `onlyIds` restricts the queue to those words (e.g. one song's vocabulary); `shuffleNew` picks
    new cards at random instead of most-frequent-first.
    """
    _checkTrack(track)
    nowTs = now() if nowTs is None else nowTs
    buried = _buriedIds(conn, language, track, nowTs)
    if onlyIds is not None:
        buried = buried | _complementIds(conn, table, idCol, onlyIds)
    base = f"FROM {table} WHERE {track}_suspended = 0"

    def learning(cutoff):
        rows = conn.execute(
            f"SELECT {idCol} {base} AND {track}_state IN (1, 3) AND {track}_due_ts <= ? "
            f"ORDER BY {track}_due_ts ASC",
            (cutoff,),
        ).fetchall()
        return next((r[0] for r in rows if r[0] not in buried), None)

    found = learning(nowTs)
    if found is not None:
        return found, "learning"

    rows = conn.execute(
        f"SELECT {idCol}, {track}_step, {track}_stability, {track}_difficulty, {track}_due_ts, "
        f"{track}_last_reviewed_ts {base} AND {track}_state = 2 AND {track}_due_ts <= ?",
        (nowTs,),
    ).fetchall()
    scored = []
    for vocabId, step, stability, difficulty, dueTs, lastTs in rows:
        if vocabId in buried:
            continue
        cs = {"state": srs_fsrs.STATE_REVIEW, "step": step, "stability": stability,
              "difficulty": difficulty, "dueTs": dueTs, "lastReviewTs": lastTs}
        scored.append((srs_fsrs.retrievability(cs, nowTs), random.random(), vocabId))
    if scored:
        return min(scored)[2], "review"

    if newLimit is None or newCardsIntroducedToday(conn, language, nowTs) < newLimit:
        rows = conn.execute(
            f"""SELECT s.{idCol} FROM {table} s
                LEFT JOIN (SELECT {occIdCol} AS vid, COUNT(*) AS n FROM {occTable} GROUP BY {occIdCol}) o
                    ON o.vid = s.{idCol}
                WHERE s.{track}_suspended = 0 AND s.{track}_state = 0
                ORDER BY COALESCE(o.n, 0) DESC, s.{idCol} ASC"""
        ).fetchall()
        candidates = [r[0] for r in rows if r[0] not in buried]
        if candidates:
            return (random.choice(candidates) if shuffleNew else candidates[0]), "new"

    found = learning(nowTs + learnAheadSecs)
    if found is not None:
        return found, "learning"
    return None


# ---------------------------------------------------------------------------------------------
# Known-word handling (M4): suspend, "I already know this", bulk triage.
# ---------------------------------------------------------------------------------------------

def setSuspended(conn, table: str, idCol: str, vocabIds, suspended: bool, tracks=TRACKS):
    """Suspended cards are never returned by the study queue (see pickNextCard). Caller commits."""
    for track in tracks:
        _checkTrack(track)
    sets = ", ".join(f"{t}_suspended = ?" for t in tracks)
    flag = 1 if suspended else 0
    for vocabId in vocabIds:
        conn.execute(f"UPDATE {table} SET {sets} WHERE {idCol} = ?", (*([flag] * len(tracks)), vocabId))


def markKnown(conn, table: str, idCol: str, vocabIds, nowTs: Optional[int] = None,
              stabilityDays: float = srs_fsrs.KNOWN_STABILITY_DAYS, tracks=TRACKS):
    """
    "I already know this": seed each track straight into Review with a long stability (see
    srs_fsrs.seedKnownCard). A track that already has MORE stability than the seed is left alone -
    this never downgrades real progress. Not written to review_log: it isn't a review, and fake
    ratings would pollute the daily new-card count and the future weight optimizer (M6).
    Caller commits.
    """
    for track in tracks:
        _checkTrack(track)
    nowTs = now() if nowTs is None else nowTs
    seed = srs_fsrs.seedKnownCard(nowTs, stabilityDays)
    for vocabId in vocabIds:
        for track in tracks:
            conn.execute(
                f"""UPDATE {table} SET {track}_state=?, {track}_step=?, {track}_stability=?,
                        {track}_difficulty=?, {track}_due_ts=?, {track}_last_reviewed_ts=?,
                        {track}_suspended=0
                    WHERE {idCol}=? AND ({track}_stability IS NULL OR {track}_stability < ?)""",
                (seed["state"], seed["step"], seed["stability"], seed["difficulty"], seed["dueTs"],
                 seed["lastReviewTs"], vocabId, stabilityDays),
            )


def triageCandidates(conn, cardTable: str, idCol: str, vocabTable: str, occTable: str,
                     occIdCol: str, readingExpr: str, limit: int = 100) -> list:
    """
    Untouched words (New on every track, not suspended) ranked by how often they occur in your
    saved lyrics - the words most likely to be 私/僕/本当-style things you already know, and the
    ones the New queue serves first. Returns dicts with the fields the triage list shows.
    """
    rows = conn.execute(
        f"""SELECT v.id, v.lemma, v.surface, {readingExpr}, v.meaning_json, COALESCE(o.n, 0)
            FROM {cardTable} s
            JOIN {vocabTable} v ON v.id = s.{idCol}
            LEFT JOIN (SELECT {occIdCol} AS vid, COUNT(*) AS n FROM {occTable} GROUP BY {occIdCol}) o
                ON o.vid = v.id
            WHERE s.meaning_state = 0 AND s.reading_state = 0 AND s.cloze_state = 0
              AND s.meaning_suspended = 0 AND s.reading_suspended = 0 AND s.cloze_suspended = 0
            ORDER BY COALESCE(o.n, 0) DESC, v.id ASC LIMIT ?""",
        (limit,),
    ).fetchall()
    out = []
    for vocabId, lemma, surface, reading, meaningJson, count in rows:
        meaning = json.loads(meaningJson) if meaningJson else {}
        out.append({
            "vocabId": vocabId, "lemma": lemma, "surface": surface, "reading": reading,
            "gloss": "; ".join((meaning.get("gloss") or [])[:2]), "occurrences": count,
        })
    return out
