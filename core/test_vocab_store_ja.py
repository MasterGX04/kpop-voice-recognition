"""
Tests for core/vocab_store_ja.py. Stdlib unittest, tempdir-based like
core.test_kanji_reference.SongVocabPersistenceTests. Run with:
    python -m unittest core.test_vocab_store_ja -v
"""

import os
import shutil
import tempfile
import unittest

from core import vocab_store_ja
from core.vocab_db import getConnection


def _entry(lemma, category="onyomi", cognate=None):
    return {
        "surface": lemma, "reading": "じかん", "lemma": lemma, "lemmaReading": "じかん",
        "category": category, "chineseCognate": cognate,
        "japaneseMeaning": {"status": "found", "pos": ["n"], "gloss": ["time"]},
        "mandarinPinyin": {"traditional": lemma, "pinyin": "shi2 jian1"},
    }


class UpsertVocabTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_new_word_creates_vocab_and_srs_card(self):
        vocabId, isNew = vocab_store_ja.upsertVocab(_entry("時間"))
        self.assertTrue(isNew)

        due = vocab_store_ja.getDueCards("reading", limit=10)
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["lemma"], "時間")

    def test_new_word_starts_new_on_every_track(self):
        cognate = {"status": "confirmed", "traditional": "時間", "pinyin": "shi2 jian1", "gloss": ["time"]}
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間", cognate=cognate))

        conn = getConnection()
        row = conn.execute(
            "SELECT meaning_state, reading_state FROM srs_card_ja WHERE vocab_ja_id = ?", (vocabId,)
        ).fetchone()
        conn.close()
        self.assertEqual(row, (0, 0))  # FSRS state 0 = New; no more cognate 'known' seeding

    def test_not_attested_cognate_still_gets_a_pinyin_but_meaning_defaults_unknown(self):
        cognate = {"status": "not_attested", "traditional": "殘業", "pinyinFallback": "can2 ye4"}
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("残業", category="onyomi", cognate=cognate))

        conn = getConnection()
        row = conn.execute(
            "SELECT meaning_state, cognate_pinyin FROM srs_card_ja s "
            "JOIN vocab_ja v ON v.id = s.vocab_ja_id WHERE s.vocab_ja_id = ?", (vocabId,)
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 0)
        self.assertEqual(row[1], "can2 ye4")

    def test_kunyomi_word_without_cognate_defaults_meaning_unknown(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("出会う", category="kunyomi", cognate=None))

        conn = getConnection()
        row = conn.execute(
            "SELECT meaning_state FROM srs_card_ja WHERE vocab_ja_id = ?", (vocabId,)
        ).fetchone()
        conn.close()
        self.assertEqual(row[0], 0)

    def test_kana_occurrence_does_not_overwrite_kanji_surface(self):
        # Real bug: 今 was displayed as "いま" because a kana-spelled line (愛をいま) was scanned last.
        def entry(surface):
            e = _entry("今", category="kunyomi")
            e["surface"] = surface
            return e
        vocabId, _ = vocab_store_ja.upsertVocab(entry("今"))
        vocab_store_ja.upsertVocab(entry("いま"))
        vocab_store_ja.upsertVocab(entry("今"))
        vocab_store_ja.upsertVocab(entry("いま"))
        self.assertEqual(vocab_store_ja.listAllVocab()[0]["surface"], "今")

    def test_pick_surface_rules(self):
        p = vocab_store_ja.pickSurface
        self.assertEqual(p("今", "いま", "今"), "今")          # legacy kana surface is upgraded
        self.assertEqual(p("今", "今", "いま"), "今")          # kana never downgrades kanji
        self.assertEqual(p("離す", None, "離せる"), "離す")     # kanji conjugation -> dictionary form
        self.assertEqual(p("此処", None, "ここ"), "ここ")       # kana-only spelling stays as written
        self.assertEqual(p("此処", "ここ", "ココ"), "ここ")     # kana keeps its first spelling
        self.assertEqual(p("ちょっと", None, "ちょっと"), "ちょっと")

    def test_upsert_by_lemma_replaces_rather_than_duplicates(self):
        idA, isNewA = vocab_store_ja.upsertVocab(_entry("時間"))
        idB, isNewB = vocab_store_ja.upsertVocab(_entry("時間"))
        self.assertTrue(isNewA)
        self.assertFalse(isNewB)
        self.assertEqual(idA, idB)

    def test_add_occurrence_is_idempotent(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        inserted1 = vocab_store_ja.addOccurrence(
            vocabId, "TWICE", "TestSong", ["Nayeon"], "時間がない", "lyric-1", 10, 20
        )
        inserted2 = vocab_store_ja.addOccurrence(
            vocabId, "TWICE", "TestSong", ["Nayeon"], "時間がない", "lyric-1", 10, 20
        )
        self.assertTrue(inserted1)
        self.assertFalse(inserted2)

        occurrences = vocab_store_ja.getOccurrences(vocabId)
        self.assertEqual(len(occurrences), 1)
        self.assertEqual(occurrences[0]["singers"], ["Nayeon"])

    def test_submit_review_moves_card_out_of_due_queue(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        self.assertEqual(len(vocab_store_ja.getDueCards("reading")), 1)

        vocab_store_ja.submitReview(vocabId, "reading", "good")

        self.assertEqual(len(vocab_store_ja.getDueCards("reading")), 0)

    def test_all_four_ratings_update_card_and_write_review_log(self):
        # Expected values are the verified table in .claude/FLASHCARD_FSRS_PLAN.md (fuzzing off).
        from fsrs import Scheduler
        from core import srs_fsrs
        srs_fsrs.setScheduler(Scheduler(enable_fuzzing=False))
        self.addCleanup(srs_fsrs.setScheduler, Scheduler())

        t0 = 1_800_000_000
        for rating, state, seconds in (("again", 1, 60), ("hard", 1, 330), ("good", 1, 600), ("easy", 2, 8 * 86400)):
            vocabId, _ = vocab_store_ja.upsertVocab(_entry(f"w-{rating}"))
            vocab_store_ja.submitReview(vocabId, "meaning", rating, nowTs=t0, durationMs=1234)

            conn = getConnection()
            card = conn.execute(
                "SELECT meaning_state, meaning_due_ts, meaning_stability, meaning_reps, meaning_last_reviewed_ts "
                "FROM srs_card_ja WHERE vocab_ja_id = ?", (vocabId,)).fetchone()
            log = conn.execute(
                "SELECT rating, state_before, stability_before, reviewed_ts, duration_ms, language "
                "FROM review_log WHERE vocab_id = ?", (vocabId,)).fetchall()
            conn.close()

            self.assertEqual(card[0], state, rating)
            self.assertAlmostEqual(card[1] - t0, seconds, delta=1)
            self.assertIsNotNone(card[2])
            self.assertEqual((card[3], card[4]), (1, t0))
            self.assertEqual(log, [(rating, 0, None, t0, 1234, "ja")])

    def test_lapse_is_counted_and_log_records_prior_state(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        t0 = 1_800_000_000
        vocab_store_ja.submitReview(vocabId, "reading", "easy", nowTs=t0)
        conn = getConnection()
        dueTs = conn.execute("SELECT reading_due_ts FROM srs_card_ja").fetchone()[0]
        conn.close()
        vocab_store_ja.submitReview(vocabId, "reading", "again", nowTs=dueTs)
        conn = getConnection()
        lapses, state = conn.execute("SELECT reading_lapses, reading_state FROM srs_card_ja").fetchone()
        last = conn.execute("SELECT state_before, elapsed_days FROM review_log ORDER BY id DESC").fetchone()
        conn.close()
        self.assertEqual((lapses, state), (1, 3))
        self.assertEqual(last[0], 2)
        self.assertGreater(last[1], 7)

    def test_preview_intervals_has_four_ordered_buttons(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        preview = vocab_store_ja.previewIntervals(vocabId, "meaning", nowTs=1_800_000_000)
        self.assertEqual(set(preview), {"again", "hard", "good", "easy"})
        self.assertLess(preview["again"], preview["easy"])

    def test_invalid_track_or_rating_is_rejected(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        with self.assertRaises(ValueError):
            vocab_store_ja.submitReview(vocabId, "meaning; DROP TABLE x", "good")
        with self.assertRaises(ValueError):
            vocab_store_ja.submitReview(vocabId, "meaning", "perfect")

    def test_new_word_is_immediately_due_for_cloze_too(self):
        # Milestone 5: cloze is a fully independent SRS track, seeded the same way meaning/reading
        # already are - a brand-new word should be immediately due for cloze review too, not left
        # NULL/never-due just because submitReview("cloze", ...) has never been generically
        # exercised against a real word before.
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        due = vocab_store_ja.getDueCards("cloze", limit=10)
        self.assertEqual([c["lemma"] for c in due], ["時間"])

    def test_submit_cloze_review_moves_card_out_of_due_queue(self):
        # submitReview()/getDueCards() are fully generic on the track name (see their own
        # f-string column interpolation) - reusing them directly for "cloze" needs no new store
        # function, just the schema columns to exist.
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        vocab_store_ja.submitReview(vocabId, "cloze", "good")
        self.assertEqual(vocab_store_ja.getDueCards("cloze"), [])
        # Meaning/reading tracks are untouched by a cloze review - independent due-dates.
        self.assertEqual(len(vocab_store_ja.getDueCards("reading")), 1)
        self.assertEqual(len(vocab_store_ja.getDueCards("meaning")), 1)

    def test_list_all_vocab_ignores_due_status(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        vocab_store_ja.submitReview(vocabId, "reading", "good")
        vocab_store_ja.submitReview(vocabId, "meaning", "good")

        self.assertEqual(vocab_store_ja.getDueCards("reading"), [])
        allWords = vocab_store_ja.listAllVocab()
        self.assertEqual([w["lemma"] for w in allWords], ["時間"])

    def test_update_meaning_overwrites_gloss_and_flips_status_to_found(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("残業", cognate=None))
        vocab_store_ja.updateMeaning(vocabId, ["overtime work (manually filled in)"])

        card = vocab_store_ja.listAllVocab()[0]
        self.assertEqual(card["meaning"]["status"], "found")
        self.assertEqual(card["meaning"]["gloss"], ["overtime work (manually filled in)"])

    def test_rescan_does_not_clobber_a_manually_edited_meaning(self):
        # Regression for a real user report: they hand-corrected a meaning, then re-ran Compile
        # (which just calls upsertVocab again with the same raw lookup result), and the manual
        # edit silently reverted because upsertVocab used to overwrite meaning_json unconditionally.
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("残業", cognate=None))
        vocab_store_ja.updateMeaning(vocabId, ["overtime work (manually filled in)"])

        # Simulate a rescan: the exact same entry, as analyzeSelection() would produce it again.
        vocab_store_ja.upsertVocab(_entry("残業", cognate=None))

        card = vocab_store_ja.listAllVocab()[0]
        self.assertEqual(card["meaning"]["gloss"], ["overtime work (manually filled in)"])

    def test_rescan_still_refreshes_meaning_for_an_unlocked_word(self):
        # A word nobody has manually edited yet must still pick up a freshly re-analyzed meaning
        # (e.g. if the underlying dictionary lookup logic improves) - locking is opt-in via
        # updateMeaning(), not the default for every word.
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        entryWithNewMeaning = _entry("時間")
        entryWithNewMeaning["japaneseMeaning"] = {"status": "found", "pos": ["n"], "gloss": ["updated meaning"]}
        vocab_store_ja.upsertVocab(entryWithNewMeaning)

        card = vocab_store_ja.listAllVocab()[0]
        self.assertEqual(card["meaning"]["gloss"], ["updated meaning"])

    def test_manual_surface_survives_a_rescan_and_blank_resets_it(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("どんな", cognate=None))
        vocab_store_ja.updateSurface(vocabId, "何様")
        vocab_store_ja.upsertVocab(_entry("どんな", cognate=None))
        self.assertEqual(vocab_store_ja.listAllVocab()[0]["surface"], "何様")

        vocab_store_ja.updateSurface(vocabId, "")
        self.assertEqual(vocab_store_ja.listAllVocab()[0]["surface"], "どんな")

    def test_delete_vocab_removes_word_and_its_occurrences(self):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        vocab_store_ja.addOccurrence(vocabId, "TWICE", "TestSong", ["Nayeon"], "時間がない", "l1", 1, 2)

        vocab_store_ja.deleteVocab(vocabId)

        self.assertEqual(vocab_store_ja.listAllVocab(), [])
        self.assertEqual(vocab_store_ja.getOccurrences(vocabId), [])


class CoOccurrenceGraphTests(unittest.TestCase):
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def test_two_words_sharing_one_song_get_one_edge(self):
        jikanId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        yumeId, _ = vocab_store_ja.upsertVocab(_entry("夢"))
        vocab_store_ja.addOccurrence(jikanId, "TWICE", "TestSong", [], "時間だ", "l1", 0, 10)
        vocab_store_ja.addOccurrence(yumeId, "TWICE", "TestSong", [], "夢を見る", "l2", 10, 20)

        graph = vocab_store_ja.getCoOccurrenceGraph()
        self.assertEqual({n["vocabId"] for n in graph["nodes"]}, {jikanId, yumeId})
        self.assertEqual(len(graph["edges"]), 1)
        edge = graph["edges"][0]
        self.assertEqual({edge["a"], edge["b"]}, {jikanId, yumeId})
        self.assertEqual(edge["sharedSongs"], 1)
        self.assertEqual(edge["songs"], [{"group": "TWICE", "song": "TestSong"}])

    def test_words_that_never_share_a_song_have_no_edge(self):
        jikanId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        yumeId, _ = vocab_store_ja.upsertVocab(_entry("夢"))
        vocab_store_ja.addOccurrence(jikanId, "TWICE", "SongA", [], "時間だ", "l1", 0, 10)
        vocab_store_ja.addOccurrence(yumeId, "TWICE", "SongB", [], "夢を見る", "l2", 0, 10)

        graph = vocab_store_ja.getCoOccurrenceGraph()
        self.assertEqual(graph["nodes"], [])
        self.assertEqual(graph["edges"], [])

    def test_shared_song_count_accumulates_across_multiple_songs(self):
        jikanId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        yumeId, _ = vocab_store_ja.upsertVocab(_entry("夢"))
        vocab_store_ja.addOccurrence(jikanId, "TWICE", "SongA", [], "時間だ", "l1", 0, 10)
        vocab_store_ja.addOccurrence(yumeId, "TWICE", "SongA", [], "夢を見る", "l2", 10, 20)
        vocab_store_ja.addOccurrence(jikanId, "TWICE", "SongB", [], "時間だ", "l3", 0, 10)
        vocab_store_ja.addOccurrence(yumeId, "TWICE", "SongB", [], "夢を見る", "l4", 10, 20)

        graph = vocab_store_ja.getCoOccurrenceGraph()
        self.assertEqual(len(graph["edges"]), 1)
        self.assertEqual(graph["edges"][0]["sharedSongs"], 2)

    def test_min_shared_songs_filters_out_weak_edges(self):
        jikanId, _ = vocab_store_ja.upsertVocab(_entry("時間"))
        yumeId, _ = vocab_store_ja.upsertVocab(_entry("夢"))
        vocab_store_ja.addOccurrence(jikanId, "TWICE", "SongA", [], "時間だ", "l1", 0, 10)
        vocab_store_ja.addOccurrence(yumeId, "TWICE", "SongA", [], "夢を見る", "l2", 10, 20)

        graph = vocab_store_ja.getCoOccurrenceGraph(min_shared_songs=2)
        self.assertEqual(graph["nodes"], [])
        self.assertEqual(graph["edges"], [])

    def test_three_words_in_one_song_form_a_full_triangle(self):
        ids = [vocab_store_ja.upsertVocab(_entry(w))[0] for w in ("時間", "夢", "気")]
        for i, vocabId in enumerate(ids):
            vocab_store_ja.addOccurrence(vocabId, "TWICE", "TestSong", [], "line", f"l{i}", i, i + 1)

        graph = vocab_store_ja.getCoOccurrenceGraph()
        self.assertEqual(len(graph["nodes"]), 3)
        self.assertEqual(len(graph["edges"]), 3)  # every pair among 3 words


if __name__ == "__main__":
    unittest.main()
