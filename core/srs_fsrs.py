"""
FSRS scheduling wrapper (.claude/FLASHCARD_FSRS_PLAN.md, M0). The ONLY module that imports `fsrs`:
everything outside speaks plain ints/dicts (unix-second timestamps, 0-3 state codes), so library
types never reach the SQLite layer and a future FSRS-7 swap is a one-file change.

A "card state" dict is {state, step, stability, difficulty, dueTs, lastReviewTs}. state 0 = New is
OUR concept - the library has no New state (a fresh Card is already Learning with stability None),
so New == stability is None; 1/2/3 are the library's Learning/Review/Relearning.
"""

from datetime import datetime, timezone

from fsrs import Card, Rating, Scheduler, State

STATE_NEW, STATE_LEARNING, STATE_REVIEW, STATE_RELEARNING = 0, 1, 2, 3

RATINGS = {"again": Rating.Again, "hard": Rating.Hard, "good": Rating.Good, "easy": Rating.Easy}

_scheduler = Scheduler()


def setScheduler(scheduler: Scheduler):
    """Swap the module scheduler (tests pass Scheduler(enable_fuzzing=False); M5 will pass the
    user's retention/steps)."""
    global _scheduler
    _scheduler = scheduler


def newCardState(nowTs: int) -> dict:
    return {"state": STATE_NEW, "step": 0, "stability": None, "difficulty": None,
            "dueTs": nowTs, "lastReviewTs": None}


def _dt(ts: int) -> datetime:
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _ts(dt: datetime) -> int:
    return int(dt.timestamp())


def _toCard(cs: dict) -> Card:
    isNew = cs["stability"] is None
    return Card(
        card_id=0,
        state=State.Learning if isNew else State(cs["state"]),
        step=0 if isNew else cs["step"],
        stability=cs["stability"],
        difficulty=cs["difficulty"],
        due=_dt(cs["dueTs"]),
        last_review=_dt(cs["lastReviewTs"]) if cs.get("lastReviewTs") else None,
    )


def _fromCard(card: Card) -> dict:
    return {
        "state": int(card.state.value),
        "step": card.step if card.step is not None else 0,
        "stability": card.stability,
        "difficulty": card.difficulty,
        "dueTs": _ts(card.due),
        "lastReviewTs": _ts(card.last_review) if card.last_review else None,
    }


def review(cardState: dict, rating: str, nowTs: int) -> dict:
    """Returns the new card state after `rating` ("again|hard|good|easy") at `nowTs`."""
    newCard, _log = _scheduler.review_card(_toCard(cardState), RATINGS[rating], _dt(nowTs))
    return _fromCard(newCard)


def previewIntervals(cardState: dict, nowTs: int) -> dict:
    """{again, hard, good, easy} -> seconds until each rating would make the card due again.
    review_card() doesn't mutate its input, so this is just four calls."""
    return {name: review(cardState, name, nowTs)["dueTs"] - nowTs for name in RATINGS}


def retrievability(cardState: dict, nowTs: int) -> float:
    """Recall probability right now; a never-reviewed card is 0."""
    if cardState["stability"] is None or not cardState.get("lastReviewTs"):
        return 0.0
    return _scheduler.get_card_retrievability(_toCard(cardState), _dt(nowTs))


KNOWN_STABILITY_DAYS = 60.0
KNOWN_DIFFICULTY = 5.0


def seedKnownCard(nowTs: int, stabilityDays: float = KNOWN_STABILITY_DAYS) -> dict:
    """
    Card state for "I already know this": straight into Review with a long stability, so it is
    still tested once later at the target retention instead of never (rating Easy on a New card
    only gives ~8 days). Due date = the interval at which recall hits the scheduler's desired
    retention for that stability (60 days at the default 90%). Uses the scheduler's private
    _next_interval - verified against fsrs 6.3.2 - and falls back to stability-in-days (exact at
    90%) if a future version drops it.
    """
    try:
        days = _scheduler._next_interval(stability=stabilityDays)
    except Exception:
        days = round(stabilityDays)
    return {"state": STATE_REVIEW, "step": 0, "stability": stabilityDays,
            "difficulty": KNOWN_DIFFICULTY, "dueTs": nowTs + int(days) * 86400,
            "lastReviewTs": nowTs}


def formatInterval(seconds: int) -> str:
    """Button label text: '<1m', '10m', '3h', '8d', '2.1y'."""
    if seconds < 60:
        return "<1m"
    if seconds < 3600:
        return f"{round(seconds / 60)}m"
    if seconds < 86400:
        return f"{round(seconds / 3600)}h"
    days = seconds / 86400
    if days < 365:
        return f"{round(days)}d"
    return f"{days / 365:.1f}y"
