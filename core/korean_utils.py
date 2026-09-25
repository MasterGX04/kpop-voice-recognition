"""Shared Kiwi (kiwipiepy) tokenizer accessor for the Korean lyric mode."""

_kiwi = None


def _getKiwi():
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi()
    return _kiwi


def getKiwi():
    """Public accessor for the shared Kiwi singleton (used by core/korean_grammar_breakdown.py)."""
    return _getKiwi()
