"""
Cross-language bridge: link a Japanese vocab_ja row and a Korean vocab_ko row when they share the
same Chinese-cognate root. Both sides already resolve against the same CC-CEDICT traditional-form
space (core.kanji_reference.lookupChineseCognate for Japanese; core.korean_hanja reuses that same
function for Hanja pinyin) - so this is a pure DB join on already-computed data, no new lookup.

Linking by shared English gloss instead/in addition is a deferred future idea (see the plan doc),
not built here.
"""

from core.vocab_db import getConnection


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
