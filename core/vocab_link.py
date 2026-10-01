"""
Cross-language bridge: link a Japanese vocab_ja row and a Korean vocab_ko row when they share the
same Chinese-cognate root. Both sides already resolve against the same CC-CEDICT traditional-form
space (core.kanji_reference.lookupChineseCognate for Japanese; core.korean_hanja reuses that same
function for Hanja pinyin) - so this is a pure DB join on already-computed data, no new lookup.

Linking by shared English gloss instead/in addition is a deferred future idea (see the plan doc),
not built here.
"""

import json

from core.vocab_db import getConnection
from core.japanese_hangul import lookupHangulCognate


def _koHanjaCandidates(conn, koId: int, linkedHanjaForm=None) -> list:
    rows = conn.execute(
        "SELECT hanja_form, pinyin, gloss_json FROM vocab_ko_hanja WHERE vocab_ko_id = ?", (koId,)
    ).fetchall()
    return [
        {
            "hanjaForm": r[0], "pinyin": r[1], "gloss": json.loads(r[2]) if r[2] else [],
            "linked": r[0] == linkedHanjaForm,
        }
        for r in rows
    ]


def syncCognateLinks() -> int:
    """
    Recompute cognate_link from the current vocab_ja/vocab_ko_hanja rows. Cheap enough to call
    after every scan (core.vocab_sync). Returns the number of link rows now present.
    """
    conn = getConnection()
    try:
        jaRows = conn.execute(
            "SELECT id, cognate_form FROM vocab_ja WHERE cognate_status = 'confirmed' AND cognate_form IS NOT NULL"
        ).fetchall()
        koRows = conn.execute(
            "SELECT DISTINCT vocab_ko_id, hanja_form FROM vocab_ko_hanja"
        ).fetchall()

        koByForm = {}
        for koId, form in koRows:
            koByForm.setdefault(form, set()).add(koId)

        count = 0
        for jaId, form in jaRows:
            for koId in koByForm.get(form, ()):
                conn.execute(
                    "INSERT OR IGNORE INTO cognate_link (vocab_ja_id, vocab_ko_id, cognate_form) VALUES (?, ?, ?)",
                    (jaId, koId, form),
                )

        conn.commit()
        count = conn.execute("SELECT COUNT(*) FROM cognate_link").fetchone()[0]
        return count
    finally:
        conn.close()


def getLinkedWords(language: str, vocab_id: int) -> list:
    """
    `language` is "ja" or "ko" - the language of `vocab_id`. Returns the cognate cousins on the
    *other* side, as [{"vocabId", "lemma", "cognateForm"}, ...].
    """
    conn = getConnection()
    try:
        if language == "ja":
            rows = conn.execute(
                """SELECT k.id, k.lemma, l.cognate_form FROM cognate_link l
                   JOIN vocab_ko k ON k.id = l.vocab_ko_id WHERE l.vocab_ja_id = ?""",
                (vocab_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT j.id, j.lemma, l.cognate_form FROM cognate_link l
                   JOIN vocab_ja j ON j.id = l.vocab_ja_id WHERE l.vocab_ko_id = ?""",
                (vocab_id,),
            ).fetchall()
        return [{"vocabId": r[0], "lemma": r[1], "cognateForm": r[2]} for r in rows]
    finally:
        conn.close()


def getCognateBridge(language: str, vocab_id: int) -> dict:
    """
    Three-way Japanese Kanji / Korean Hanja / Chinese-source comparison for a Sino-vocabulary word
    (Milestone 3 of .claude/FLASHCARD_WEB_UPGRADE_PLAN.md - the "Tri-Lingual Ideographic Bridge"
    idea from Study_tool_ideas.txt). `language` is "ja" or "ko" - the language of `vocab_id`, same
    convention as getLinkedWords().

    Pure reuse of already-persisted columns, no new dictionary lookups: vocab_ja.cognate_status/
    cognate_pinyin/cognate_gloss_json are the only place a *confirmed* CC-CEDICT status+gloss are
    persisted (core.kanji_reference.lookupChineseCognate(), run at scan time) - vocab_ko_hanja only
    ever persists pinyin (core.korean_hanja._hanjaPinyin() calls the same lookup but only keeps the
    pinyin half). So the "chinese" leg below is always sourced from a vocab_ja row: either the
    queried word itself (language="ja"), or its linked cousin reached via cognate_link
    (language="ko" - see getLinkedWords()).

    Returns {
        "cognateForm": the shared Traditional-Chinese head form, or None if this word has no
                        cognate at all (native kunyomi/jukujigo Japanese, or a Korean word with no
                        Hanja reading),
        "japanese": {"vocabId", "lemma", "lemmaReading"} or None - populated whenever a vocab_ja
                    row is reachable (itself, or via a link),
        "korean": {"vocabId", "lemma", "candidates": [{"hanjaForm", "pinyin", "gloss", "linked"},
                   ...]} or None - `candidates` lists every real Hanja candidate this Korean word
                   has (plural when still ambiguous, see core.korean_hanja.lookupHanja), with
                   "linked" marking the one (if any) matching `cognateForm`,
        "chinese": {"status", "pinyin", "gloss"} or None - only ever populated when a vocab_ja row
                   was reached (see above); never fabricated for an unlinked Korean word just
                   because it has its own (pinyin-only) Hanja data.
    }
    An unlinked word degrades gracefully: shows what it has (its own side, fully populated), with
    the other side and "chinese" simply None - never a fabricated third leg.

    For language="ja" only, the result also carries "predictedKorean": None, or
    core.japanese_hangul.lookupHangulCognate()'s {"hangul", "hanja", "tier", "gloss"} when no real
    Korean word is linked (a predicted hangul is never shown alongside a real linked one).
    """
    conn = getConnection()
    try:
        if language == "ja":
            row = conn.execute(
                """SELECT lemma, lemma_reading, cognate_form, cognate_status, cognate_pinyin,
                          cognate_gloss_json, category FROM vocab_ja WHERE id = ?""",
                (vocab_id,),
            ).fetchone()
            if row is None:
                return {"cognateForm": None, "japanese": None, "korean": None, "chinese": None}
            lemma, lemmaReading, cognateForm, cognateStatus, cognatePinyin, cognateGlossJson, category = row

            japanese = {"vocabId": vocab_id, "lemma": lemma, "lemmaReading": lemmaReading}
            chinese = None
            if cognateForm:
                chinese = {
                    "status": cognateStatus, "pinyin": cognatePinyin,
                    "gloss": json.loads(cognateGlossJson) if cognateGlossJson else [],
                }

            korean = None
            linked = getLinkedWords("ja", vocab_id)
            if linked:
                koId, koLemma = linked[0]["vocabId"], linked[0]["lemma"]
                korean = {
                    "vocabId": koId, "lemma": koLemma,
                    "candidates": _koHanjaCandidates(conn, koId, cognateForm),
                }

            # No real Korean word from the user's own vocab is linked: predict the hangul instead
            # (core.japanese_hangul - only Chinese-attested on'yomi compounds ever get one).
            predictedKorean = None if korean else lookupHangulCognate(lemma, category)

            return {"cognateForm": cognateForm, "japanese": japanese, "korean": korean,
                    "chinese": chinese, "predictedKorean": predictedKorean}

        elif language == "ko":
            row = conn.execute("SELECT lemma FROM vocab_ko WHERE id = ?", (vocab_id,)).fetchone()
            if row is None:
                return {"cognateForm": None, "japanese": None, "korean": None, "chinese": None}
            koLemma = row[0]

            linked = getLinkedWords("ko", vocab_id)
            if not linked:
                return {
                    "cognateForm": None, "japanese": None, "chinese": None,
                    "korean": {
                        "vocabId": vocab_id, "lemma": koLemma,
                        "candidates": _koHanjaCandidates(conn, vocab_id),
                    },
                }

            jaId, cognateForm = linked[0]["vocabId"], linked[0]["cognateForm"]
            jaRow = conn.execute(
                """SELECT lemma, lemma_reading, cognate_status, cognate_pinyin, cognate_gloss_json
                   FROM vocab_ja WHERE id = ?""",
                (jaId,),
            ).fetchone()
            japanese = None
            chinese = None
            if jaRow:
                jaLemma, jaLemmaReading, cognateStatus, cognatePinyin, cognateGlossJson = jaRow
                japanese = {"vocabId": jaId, "lemma": jaLemma, "lemmaReading": jaLemmaReading}
                chinese = {
                    "status": cognateStatus, "pinyin": cognatePinyin,
                    "gloss": json.loads(cognateGlossJson) if cognateGlossJson else [],
                }

            korean = {
                "vocabId": vocab_id, "lemma": koLemma,
                "candidates": _koHanjaCandidates(conn, vocab_id, cognateForm),
            }
            return {"cognateForm": cognateForm, "japanese": japanese, "korean": korean, "chinese": chinese}

        else:
            raise ValueError(f"Unknown language: {language!r}")
    finally:
        conn.close()
