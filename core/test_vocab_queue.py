"""
Tests for the Anki-style study queue (core.vocab_store.pickNextCard via vocab_store_ja/_ko
getNextCard) - M3 of .claude/FLASHCARD_FSRS_PLAN.md. `nowTs` is injected everywhere so no test
depends on the wall clock. Run with: python -m unittest core.test_vocab_queue -v
"""

import os
import shutil
import tempfile
import unittest
from datetime import datetime

from fsrs import Scheduler

from core import srs_fsrs, vocab_store_ja, vocab_store_ko
from core.vocab_db import getConnection

# Local noon, so "today" is unambiguous whatever the machine's timezone is.
NOW = int(datetime(2026, 10, 15, 12, 0).timestamp())
DAY = 86400


def _entry(lemma):
    return {
        "surface": lemma, "reading": "r", "lemma": lemma, "lemmaReading": "r", "category": "kunyomi",
        "chineseCognate": None, "japaneseMeaning": {"status": "found", "pos": ["n"], "gloss": ["x"]},
        "mandarinPinyin": None,
    }


class _Helpers:
    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)
        srs_fsrs.setScheduler(Scheduler(enable_fuzzing=False))

    def tearDown(self):
        srs_fsrs.setScheduler(Scheduler())
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def _word(self, lemma, occurrences=0):
        vocabId, _ = vocab_store_ja.upsertVocab(_entry(lemma))
        for i in range(occurrences):
            vocab_store_ja.addOccurrence(vocabId, "G", f"S{i}", [], "line", f"{lemma}-{i}", 0, 1)
        return vocabId

    def _set(self, vocabId, track="reading", **cols):
        conn = getConnection()
        sets = ", ".join(f"{track}_{k} = ?" for k in cols)
        conn.execute(f"UPDATE srs_card_ja SET {sets} WHERE vocab_ja_id = ?", (*cols.values(), vocabId))
        conn.commit()
        conn.close()

    def _makeReview(self, vocabId, stability=10.0, dueOffset=-DAY, track="reading"):
        self._set(vocabId, track, state=2, stability=stability, difficulty=5.0,
                  due_ts=NOW + dueOffset, last_reviewed_ts=NOW + dueOffset - int(stability * DAY))

    def _next(self, track="reading", **kw):
        return vocab_store_ja.getNextCard(track, nowTs=NOW, **kw)


class QueueTests(_Helpers, unittest.TestCase):
    def test_empty_db_returns_none(self):
        self.assertIsNone(self._next())

    def test_priority_learning_then_review_then_new(self):
        new = self._word("new")
        review = self._word("review")
        learning = self._word("learning")
        self._makeReview(review)
        self._set(learning, state=1, step=1, stability=2.0, difficulty=5.0,
                  due_ts=NOW - 60, last_reviewed_ts=NOW - 700)

        order = []
        for _ in range(3):
            card = self._next()
            order.append((card["lemma"], card["queueType"]))
            # take it off the queue by pushing it far into the future
            self._set(card["vocabId"], state=2, stability=10.0, difficulty=5.0,
                      due_ts=NOW + 30 * DAY, last_reviewed_ts=NOW)
        self.assertEqual(order, [("learning", "learning"), ("review", "review"), ("new", "new")])
        self.assertIsNone(self._next(newLimit=0))

    def test_reviews_ordered_by_lowest_retrievability(self):
        fresh = self._word("fresh")      # only just due
        forgotten = self._word("forgotten")  # long overdue
        self._makeReview(fresh, stability=10.0, dueOffset=-60)
        self._makeReview(forgotten, stability=10.0, dueOffset=-60 * DAY)
        self.assertEqual(self._next()["lemma"], "forgotten")

    def test_review_not_yet_due_is_not_returned(self):
        w = self._word("later")
        self._makeReview(w, dueOffset=+2 * DAY)
        self.assertIsNone(self._next(newLimit=0))

    def test_daily_new_cap_and_seeding_from_30_new_10_due_2_learning(self):
        for i in range(30):
            self._word(f"n{i}")
        for i in range(10):
            self._makeReview(self._word(f"r{i}"), dueOffset=-DAY)
        for i in range(2):
            self._set(self._word(f"l{i}"), state=3, step=0, stability=2.0, difficulty=5.0,
                      due_ts=NOW - 10, last_reviewed_ts=NOW - 700)

        seen = {"learning": 0, "review": 0, "new": 0}
        for _ in range(60):
            card = self._next(newLimit=15)
            if card is None:
                break
            seen[card["queueType"]] += 1
            # a real Good on the same track logs it and moves it out of today's queue
            vocab_store_ja.submitReview(card["vocabId"], "reading", "easy", nowTs=NOW)
        self.assertEqual(seen["learning"], 2)
        self.assertEqual(seen["review"], 10)
        self.assertEqual(seen["new"], 15)  # cap honoured, 15 new left for tomorrow

    def test_default_has_no_daily_new_cap(self):
        for i in range(40):
            self._word(f"n{i}")
        seen = 0
        while self._next() is not None and seen < 100:
            card = self._next()
            vocab_store_ja.submitReview(card["vocabId"], "reading", "easy", nowTs=NOW)
            seen += 1
        self.assertEqual(seen, 40)

    def test_new_cards_ordered_by_occurrence_count(self):
        self._word("rare", occurrences=1)
        self._word("common", occurrences=5)
        self._word("never", occurrences=0)
        self.assertEqual(self._next()["lemma"], "common")

    def test_again_card_returns_once_its_step_elapses(self):
        w = self._word("only")
        vocab_store_ja.submitReview(w, "reading", "again", nowTs=NOW)  # due in 60s
        # no learn-ahead: not due yet -> nothing to show
        self.assertIsNone(self._next(newLimit=0, learnAheadSecs=0))
        # with learn-ahead, the pending step is offered early rather than ending the session
        self.assertEqual(self._next(newLimit=0)["lemma"], "only")
        # and once the step elapses it is simply due
        card = vocab_store_ja.getNextCard("reading", nowTs=NOW + 61, newLimit=0, learnAheadSecs=0)
        self.assertEqual((card["lemma"], card["queueType"]), ("only", "learning"))

    def test_siblings_are_buried_until_tomorrow(self):
        w = self._word("word")
        vocab_store_ja.submitReview(w, "meaning", "good", nowTs=NOW)
        # reading + cloze cards of the same word wait; same track (relearning) is still allowed
        self.assertIsNone(self._next("reading"))
        self.assertIsNone(self._next("cloze"))
        self.assertIsNotNone(vocab_store_ja.getNextCard("meaning", nowTs=NOW, newLimit=0))
        # next local day the sibling is available again
        self.assertEqual(vocab_store_ja.getNextCard("reading", nowTs=NOW + DAY)["lemma"], "word")

    def test_suspended_cards_are_skipped(self):
        w = self._word("sleepy")
        self._set(w, suspended=1)
        self.assertIsNone(self._next())

    def test_new_cap_counts_first_reviews_across_tracks_per_language(self):
        a, b = self._word("a"), self._word("b")
        vocab_store_ja.submitReview(a, "reading", "good", nowTs=NOW)
        # cap of 1 already used today, so no further new cards even on another track
        self.assertIsNone(vocab_store_ja.getNextCard("meaning", nowTs=NOW, newLimit=1))
        self.assertEqual(vocab_store_ja.getNextCard("meaning", nowTs=NOW, newLimit=2)["lemma"], "b")

    def test_korean_queue_returns_card_with_hanja_candidates(self):
        vocab_store_ko.upsertVocab({
            "surface": "학교", "lemma": "학교",
            "meaning": {"status": "found", "gloss": ["school"]},
            "hanjaCandidates": [{"hanja": "學校", "gloss": ["school"], "pos": "noun", "pinyin": "xue2 xiao4"}],
        })
        card = vocab_store_ko.getNextCard("reading", nowTs=NOW)
        self.assertEqual((card["lemma"], card["queueType"]), ("학교", "new"))
        self.assertEqual(len(card["hanjaCandidates"]), 1)


class KnownWordTests(_Helpers, unittest.TestCase):
    """M4: suspend, "I already know this", triage."""

    def test_suspend_hides_word_from_queue_and_unsuspend_restores_it(self):
        w = self._word("私")
        vocab_store_ja.setSuspended(w, True)
        self.assertIsNone(self._next())
        self.assertIsNone(self._next("cloze"))
        self.assertTrue(vocab_store_ja.listAllVocab()[0]["suspended"])
        vocab_store_ja.setSuspended(w, False)
        self.assertEqual(self._next()["lemma"], "私")
        self.assertFalse(vocab_store_ja.listAllVocab()[0]["suspended"])

    def test_known_word_leaves_new_queue_and_is_due_in_about_60_days(self):
        w = self._word("私")
        other = self._word("other")
        vocab_store_ja.markKnown(w, nowTs=NOW)
        self.assertEqual(self._next()["lemma"], "other")  # 私 no longer New
        conn = getConnection()
        for track in ("meaning", "reading", "cloze"):
            state, stability, due = conn.execute(
                f"SELECT {track}_state, {track}_stability, {track}_due_ts FROM srs_card_ja WHERE vocab_ja_id = ?",
                (w,)).fetchone()
            self.assertEqual((state, stability), (2, 60.0))
            self.assertEqual(due - NOW, 60 * DAY)  # 90% retention: interval == stability
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM review_log").fetchone()[0], 0)
        conn.close()
        # not due yet -> only "other" is offered, then nothing
        vocab_store_ja.markKnown(other, nowTs=NOW)
        self.assertIsNone(self._next())
        # ...but once the 60 days pass it is tested once, as a review
        later = vocab_store_ja.getNextCard("reading", nowTs=NOW + 61 * DAY)
        self.assertEqual(later["queueType"], "review")

    def test_mark_known_never_downgrades_real_progress(self):
        w = self._word("strong")
        self._makeReview(w, stability=200.0, dueOffset=+100 * DAY)
        vocab_store_ja.markKnown(w, nowTs=NOW)
        conn = getConnection()
        stability, due = conn.execute(
            "SELECT reading_stability, reading_due_ts FROM srs_card_ja").fetchone()
        conn.close()
        self.assertEqual((stability, due), (200.0, NOW + 100 * DAY))

    def test_mark_known_also_unsuspends(self):
        w = self._word("w")
        vocab_store_ja.setSuspended(w, True)
        vocab_store_ja.markKnown(w, nowTs=NOW)
        self.assertFalse(vocab_store_ja.listAllVocab()[0]["suspended"])

    def test_triage_lists_untouched_words_by_frequency_and_drops_handled_ones(self):
        self._word("rare", occurrences=1)
        common = self._word("common", occurrences=4)
        touched = self._word("touched", occurrences=9)
        suspended = self._word("suspended", occurrences=8)
        vocab_store_ja.submitReview(touched, "reading", "good", nowTs=NOW)
        vocab_store_ja.setSuspended(suspended, True)

        rows = vocab_store_ja.listTriageCandidates()
        self.assertEqual([r["lemma"] for r in rows], ["common", "rare"])
        self.assertEqual(rows[0]["occurrences"], 4)
        self.assertEqual(rows[0]["gloss"], "x")

        vocab_store_ja.markKnown([common, rows[1]["vocabId"]], nowTs=NOW)
        self.assertEqual(vocab_store_ja.listTriageCandidates(), [])
        self.assertEqual(len(vocab_store_ja.listTriageCandidates(limit=0)), 0)

    def test_korean_suspend_known_and_triage(self):
        vocab_store_ko.upsertVocab({"surface": "너무", "lemma": "너무",
                                    "meaning": {"status": "found", "gloss": ["too"]}, "hanjaCandidates": []})
        rows = vocab_store_ko.listTriageCandidates()
        self.assertEqual([r["lemma"] for r in rows], ["너무"])
        self.assertIsNone(rows[0]["reading"])
        vocab_id = rows[0]["vocabId"]
        vocab_store_ko.setSuspended([vocab_id], True)
        self.assertIsNone(vocab_store_ko.getNextCard("reading", nowTs=NOW))
        self.assertTrue(vocab_store_ko.listAllVocab()[0]["suspended"])
        vocab_store_ko.markKnown(vocab_id, nowTs=NOW)
        self.assertEqual(vocab_store_ko.listTriageCandidates(), [])


if __name__ == "__main__":
    unittest.main()
