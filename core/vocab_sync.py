"""
Compile/scan vocab from a song's (or every song's) lyrics into the SQLite vocab store - the
primary way both Japanese Kanji and Korean Hanja vocab gets populated, replacing a manual per-word
"dedicated button" (see .claude/ plan doc): you choose which song(s) to scan, rather than being
nagged on every edit. Idempotent - safe to re-run on a song any number of times, so this same
function is both the one-time backfill of existing saved_labels/*/*_kanji_vocab.json-era Japanese
vocab and the ongoing way new vocab gets added as lyrics keep getting added/edited.

Also regenerates the Japanese HTML reports (saved_labels/<group>/<song>_kanji_reference.html and
kanji_reference/by_song.html) from the DB, superseding core.kanji_reference's JSON-backed
addWordToSongReference/scanLyricsForKanjiVocab as the live data source - those functions and their
existing tests are left untouched (still correct for what they claim to do), simply unused by any
new code path, per the plan's "DB becomes source of truth going forward" decision.
"""

import codecs
import glob
import json
import os

from core.kanji_reference import (
    analyzeSelection,
    _isJapaneseLyricEntry,
    renderSongReferenceHtml,
    renderWordIndexBySongHtml,
)
from core.korean_vocab import analyzeKoreanSelection
from core import vocab_store_ja, vocab_store_ko, vocab_link
from core.vocab_db import getConnection
from core.label_runs import resolveAllSpans
from core.lyric_text import stripAll
from core.song_stats import loadRawLabels

_SAVED_LABELS_GLOB = "saved_labels/*/*_lyrics.json"

_HANGUL_RANGE = (0xAC00, 0xD7A3)

_WORD_INDEX_DIR = "kanji_reference"
_WORD_INDEX_PATH = os.path.join(_WORD_INDEX_DIR, "word_index.json")


def _containsHangul(text: str) -> bool:
    return any(_HANGUL_RANGE[0] <= ord(ch) <= _HANGUL_RANGE[1] for ch in text)


def _isKoreanLyricEntry(lyricEntry: dict) -> bool:
    text = lyricEntry.get("korean") or ""
    return lyricEntry.get("language") == "Korean" or _containsHangul(text)


def _resolveChunks(lyricEntry: dict, span: tuple = None):
    """
    `span` is this lyric's entry from core.label_runs.resolveAllSpans (run once per song over its
    labels+lyrics): the merged whole-line span for a resolvable hand link, the lead-in-corrected
    inferred span for a broken or missing link. Falls back to the lyric's own bare startChunk (no
    independent end) when nothing could be resolved.

    No blanket length cap here: core.label_runs caps only spans with no next-lyric boundary (see
    MAX_UNBOUNDED_CLIP_CHUNKS). A blanket cap truncated long bounded lines (slow ballads, raps).
    """
    if span:
        return span
    return lyricEntry.get("startChunk"), None


def _lyricsPath(group: str, song: str) -> str:
    return f"saved_labels/{group}/{song}_lyrics.json"


def _cognateDictFromColumns(status, form, pinyin, glossJson):
    if status == "confirmed":
        return {"status": "confirmed", "traditional": form, "pinyin": pinyin,
                "gloss": json.loads(glossJson) if glossJson else []}
    if status == "not_attested":
        return {"status": "not_attested", "traditional": form, "pinyinFallback": pinyin}
    return None


def _renderSongHtmlFromDb(group: str, song: str):
    conn = getConnection()
    try:
        rows = conn.execute(
            """SELECT v.surface, v.lemma, v.lemma_reading, v.category, v.meaning_json,
                      v.cognate_form, v.cognate_status, v.cognate_pinyin, v.cognate_gloss_json,
                      v.mnemonic_pinyin_json, v.updated_at, o.lyric_line
               FROM vocab_occurrence_ja o JOIN vocab_ja v ON v.id = o.vocab_ja_id
               WHERE o.group_name = ? AND o.song_title = ?
               GROUP BY v.id""",
            (group, song),
        ).fetchall()
    finally:
        conn.close()

    entries = []
    for (surface, lemma, lemmaReading, category, meaningJson, cognateForm, cognateStatus,
         cognatePinyin, cognateGlossJson, mnemonicJson, updatedAt, lyricLine) in rows:
        entries.append({
            "surface": surface or lemma, "reading": lemmaReading, "lemma": lemma,
            "lemmaReading": lemmaReading, "category": category,
            "chineseCognate": _cognateDictFromColumns(cognateStatus, cognateForm, cognatePinyin, cognateGlossJson),
            "japaneseMeaning": json.loads(meaningJson) if meaningJson else None,
            "mandarinPinyin": json.loads(mnemonicJson) if mnemonicJson else None,
            "sourceLine": lyricLine or "", "dateAdded": (updatedAt or "")[:10],
        })

    renderSongReferenceHtml(group, song, entries)


def _renderCrossSongHtmlFromDb():
    conn = getConnection()
    try:
        vocabRows = conn.execute(
            """SELECT id, lemma, lemma_reading, category, cognate_form, cognate_status,
                      cognate_pinyin, cognate_gloss_json, meaning_json, mnemonic_pinyin_json
               FROM vocab_ja"""
        ).fetchall()
        occRows = conn.execute(
            "SELECT vocab_ja_id, group_name, song_title, singer_names_json, lyric_id FROM vocab_occurrence_ja"
        ).fetchall()
    finally:
        conn.close()

    occurrencesByVocab = {}
    for vocabId, group, song, singerJson, lyricId in occRows:
        occurrencesByVocab.setdefault(vocabId, []).append({
            "group": group, "song": song,
            "memberName": json.loads(singerJson) if singerJson else [],
            "lyricId": lyricId,
        })

    index = {}
    for (vocabId, lemma, lemmaReading, category, cognateForm, cognateStatus, cognatePinyin,
         cognateGlossJson, meaningJson, mnemonicJson) in vocabRows:
        index[lemma] = {
            "reading": lemmaReading, "category": category,
            "chineseCognate": _cognateDictFromColumns(cognateStatus, cognateForm, cognatePinyin, cognateGlossJson),
            "japaneseMeaning": json.loads(meaningJson) if meaningJson else None,
            "mandarinPinyin": json.loads(mnemonicJson) if mnemonicJson else None,
            "occurrences": occurrencesByVocab.get(vocabId, []),
        }

    os.makedirs(_WORD_INDEX_DIR, exist_ok=True)
    with codecs.open(_WORD_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    renderWordIndexBySongHtml(index)


def _readLyricEntries(path: str):
    if not os.path.exists(path):
        return None
    with codecs.open(path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, list) else None


def _scanOneSong(conn, group: str, song: str) -> dict:
    """
    The per-song scan body, taking an already-open connection so a full-library scan
    (scanAllSongsForVocab) can share one connection/one commit across every song instead of
    opening a fresh one per word - see upsertVocab()'s docstring for why that matters.
    """
    result = {"jaAdded": 0, "koAdded": 0, "occurrencesLinked": 0}
    lyricEntries = _readLyricEntries(_lyricsPath(group, song))
    if lyricEntries is None:
        return result

    rawLabels = loadRawLabels(group, song)
    spans = resolveAllSpans(rawLabels, lyricEntries)

    # Unlinked lyrics have no lyricId, and SQLite treats NULLs as distinct in the tables'
    # UNIQUE(vocab_id, lyric_id), so addOccurrence's INSERT OR IGNORE never matches them - a rescan
    # would pile up a second copy of every unlinked occurrence. They're fully derived from the
    # lyrics file, so drop and regenerate them.
    for table in ("vocab_occurrence_ja", "vocab_occurrence_ko"):
        conn.execute(f"DELETE FROM {table} WHERE group_name = ? AND song_title = ? AND lyric_id IS NULL",
                     (group, song))

    for lyricIndex, lyricEntry in enumerate(lyricEntries):
        # Analysis and the stored lyric_line see clean text: no "|" colour split, no pause marker.
        text = stripAll(lyricEntry.get("korean") or "")
        if not text.strip():
            continue

        singerNames = lyricEntry.get("memberName")
        if not isinstance(singerNames, list):
            singerNames = [singerNames] if singerNames else []
        lyricId = lyricEntry.get("lyricId")
        startChunk, endChunk = _resolveChunks(lyricEntry, spans.get(lyricIndex))

        if _isJapaneseLyricEntry(lyricEntry):
            for wordResult in analyzeSelection(text, 0, len(text)):
                vocabId, isNew = vocab_store_ja.upsertVocab(wordResult, conn=conn)
                if isNew:
                    result["jaAdded"] += 1
                if vocab_store_ja.addOccurrence(
                        vocabId, group, song, singerNames, text, lyricId, startChunk, endChunk, conn=conn):
                    result["occurrencesLinked"] += 1
        elif _isKoreanLyricEntry(lyricEntry):
            for wordResult in analyzeKoreanSelection(text):
                vocabId, isNew = vocab_store_ko.upsertVocab(wordResult, conn=conn)
                if isNew:
                    result["koAdded"] += 1
                if vocab_store_ko.addOccurrence(
                        vocabId, group, song, singerNames, text, lyricId, startChunk, endChunk, conn=conn):
                    result["occurrencesLinked"] += 1

    return result


def scanSongForVocab(group: str, song: str) -> dict:
    """
    Read saved_labels/<group>/<song>_lyrics.json and upsert every Kanji/Hanja word found into the
    vocab DB, linking each to this occurrence (song/singer/lyric line/audio chunk span), then
    regenerate this song's Japanese HTML reference report from the DB. One connection, one commit
    for the whole song (see upsertVocab()'s docstring).
    """
    conn = getConnection()
    try:
        result = _scanOneSong(conn, group, song)
        conn.commit()
    finally:
        conn.close()

    vocab_link.syncCognateLinks()
    _renderSongHtmlFromDb(group, song)
    return result


def scanAllSongsForVocab(labelsGlob: str = _SAVED_LABELS_GLOB) -> dict:
    """
    Scan every saved_labels/<group>/<song>_lyrics.json. Used both for the one-time backfill of
    existing Japanese vocab and for periodic re-syncs as new lyrics get added.

    One connection, one commit for the ENTIRE run (not per-song, not per-word) - previously this
    opened a fresh connection (re-running the full schema script) and committed individually for
    every single word/occurrence: ~4,262 times for a 35-song library, measured at 64-85 seconds
    and freezing the whole app the entire time since it ran synchronously on the Tk main thread.
    Cognate-link syncing and cross-song index rendering also now happen once at the end instead of
    once per song.
    """
    summary = {}
    conn = getConnection()
    try:
        for path in sorted(glob.glob(labelsGlob)):
            group = os.path.basename(os.path.dirname(path))
            filename = os.path.basename(path)
            song = filename[: -len("_lyrics.json")] if filename.endswith("_lyrics.json") else filename
            summary[f"{group}/{song}"] = _scanOneSong(conn, group, song)
        conn.commit()
    finally:
        conn.close()

    vocab_link.syncCognateLinks()
    for key in summary:
        group, song = key.split("/", 1)
        _renderSongHtmlFromDb(group, song)
    _renderCrossSongHtmlFromDb()
    return summary
