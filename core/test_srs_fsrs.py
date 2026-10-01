import unittest

from fsrs import Scheduler

from core import srs_fsrs
from core.srs_fsrs import (
    STATE_LEARNING, STATE_NEW, STATE_RELEARNING, STATE_REVIEW,
    formatInterval, newCardState, previewIntervals, retrievability, review,
)

T0 = 1_800_000_000
DAY = 86400


class SrsFsrsTests(unittest.TestCase):
    def setUp(self):
        srs_fsrs.setScheduler(Scheduler(enable_fuzzing=False))

    def tearDown(self):
        srs_fsrs.setScheduler(Scheduler())

    def test_new_card_good_walks_learning_steps_then_graduates(self):
        c = review(newCardState(T0), "good", T0)
        self.assertEqual(c["state"], STATE_LEARNING)
        self.assertEqual(c["dueTs"] - T0, 600)  # 10 min
        c2 = review(c, "good", c["dueTs"])
        self.assertEqual(c2["state"], STATE_REVIEW)
        self.assertEqual(round((c2["dueTs"] - c["dueTs"]) / DAY), 2)

    def test_easy_on_new_card_graduates_immediately(self):
        c = review(newCardState(T0), "easy", T0)
        self.assertEqual(c["state"], STATE_REVIEW)
        self.assertEqual(round((c["dueTs"] - T0) / DAY), 8)

    def test_again_on_review_card_relearns_and_raises_difficulty(self):
        c = review(newCardState(T0), "easy", T0)
        lapsed = review(c, "again", c["dueTs"])
        self.assertEqual(lapsed["state"], STATE_RELEARNING)
        self.assertGreater(lapsed["difficulty"], c["difficulty"])
        self.assertEqual(lapsed["dueTs"] - c["dueTs"], 600)

    def test_good_streak_grows_without_a_cap_below_ten_years(self):
        c, now = newCardState(T0), T0
        for _ in range(14):
            c = review(c, "good", now)
            now = c["dueTs"]
        self.assertGreater((c["dueTs"] - c["lastReviewTs"]) / DAY, 3650)

    def test_preview_matches_actual_review(self):
        c = review(newCardState(T0), "good", T0)
        preview = previewIntervals(c, c["dueTs"])
        for name, seconds in preview.items():
            self.assertEqual(review(c, name, c["dueTs"])["dueTs"] - c["dueTs"], seconds)
        self.assertLess(preview["again"], preview["good"])
        self.assertLess(preview["good"], preview["easy"])

    def test_state_round_trips_through_plain_values(self):
        c = review(newCardState(T0), "good", T0)
        self.assertEqual(srs_fsrs._fromCard(srs_fsrs._toCard(c)), c)

    def test_retrievability_about_target_at_due_time(self):
        c = review(newCardState(T0), "easy", T0)
        self.assertAlmostEqual(retrievability(c, c["dueTs"]), 0.9, delta=0.02)
        self.assertLess(retrievability(c, c["dueTs"] + 10 * DAY), 0.87)
        self.assertEqual(retrievability(newCardState(T0), T0), 0.0)

    def test_new_card_state_is_new(self):
        self.assertEqual(newCardState(T0)["state"], STATE_NEW)

    def test_format_interval(self):
        self.assertEqual(formatInterval(30), "<1m")
        self.assertEqual(formatInterval(600), "10m")
        self.assertEqual(formatInterval(3 * 3600), "3h")
        self.assertEqual(formatInterval(8 * DAY), "8d")
        self.assertEqual(formatInterval(730 * DAY), "2.0y")


if __name__ == "__main__":
    unittest.main()
