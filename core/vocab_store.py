"""
Language-agnostic SRS math shared by core/vocab_store_ja.py and core/vocab_store_ko.py - no table
access here, just the Leitner/SM-2-lite update rule both apply to their own srs_card_ja/
srs_card_ko rows.
"""

import time

_MIN_EASE = 1.3
_BOX_TO_DAYS = [1, 3, 7, 14, 30, 90]


def now() -> int:
    return int(time.time())


def computeNextReview(box: int, ease: float, rating: str):
    """
    rating is "again" | "good" | "easy". Returns (newBox, newEase, dueTimestamp).

    "again" drops back to box 1 and penalizes ease (floored at 1.3, SM-2's own floor) - the word
    wasn't actually recalled, so it needs to be seen again soon. "good" advances one box; "easy"
    advances two boxes and nudges ease up, the same asymmetry SM-2 uses to reward an easy recall
    faster than a merely-correct one. Box is clamped into _BOX_TO_DAYS's range so it always maps
    to a real interval.
    """
    if rating == "again":
        newBox = 1
        newEase = max(_MIN_EASE, ease - 0.2)
    elif rating == "easy":
        newBox = box + 2
        newEase = ease + 0.15
    else:  # "good"
        newBox = box + 1
        newEase = ease

    newBox = max(1, min(newBox, len(_BOX_TO_DAYS)))
    intervalDays = _BOX_TO_DAYS[newBox - 1] * (newEase / 2.5)
    dueTimestamp = now() + int(intervalDays * 86400)
    return newBox, newEase, dueTimestamp
