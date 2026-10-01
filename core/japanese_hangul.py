"""
Japanese Sino-vocabulary -> Korean hangul cognate prediction (e.g. 生活 -> 생활, 化粧室 -> 화장실).

Scope is deliberately narrow (per the user): only convert a Japanese word when it is a real
Chinese-origin word - an on'yomi compound attested in Chinese (CC-CEDICT), which also covers
Japan-coined words (和製漢語) that Chinese later adopted, e.g. 化妝室. Native Japanese words
(kun'yomi, jukujigo) and Japan-only coinages Chinese never took (本当, 手続き, 写真) get NO hangul.

Pipeline, in order (each gate can reject):
  1. Shape: kanji-only stem of 2+ characters, optionally followed by する. A single kanji only
     gives the character's Sino-Korean sound (雪 -> 설), which is not a word.
  2. Chinese attestation: the Traditional form (core.kanji_reference.toTraditional) - or a
     character-variant spelling of it, see _CHAR_VARIANTS - is a CC-CEDICT word. 化粧室 is not
     in CEDICT but Taiwan's 化妝室 is; without variants your own best example would be rejected.
  3. Conversion: the `hanja` package (character-by-character Sino-Korean reading) applied to the
     TRADITIONAL form. Applied to raw shinjitai it is wrong (予告 -> 여고, 証明 -> 정명).
  4. Korean existence: the hangul must be a real Korean word, else 場合 -> 장합 / 便當 -> 편당 /
     勉強 -> 면강 (all CEDICT-attested, all nonsense in Korean) would slip through.
       tier "attested": Korean Wiktionary lists this Hanja spelling (or the hangul word carries it)
       tier "plausible": not in Wiktionary, but Kiwi's dictionary knows it as a noun.
     Neither -> rejected.

Known limit (documented, not solved): a real Korean word can mean something different from the
Japanese one (大丈夫 -> 대장부 "great man"; also true of CEDICT's own 大丈夫 gloss). Existence
gates cannot see that, so the result carries the Korean gloss for a human to compare.
"""

import re

from core.kanji_reference import lookupChineseCognate, toTraditional
from core.korean_dictionary import allEntries

_KANJI_STEM = re.compile(r"^[一-鿿]{2,}$")
_HANGUL_ONLY = re.compile(r"^[가-힣]+$")
_CONVERTIBLE_CATEGORIES = ("onyomi", "mixed")

# Japanese/Korean spelling -> spelling Chinese dictionaries (esp. Taiwan) use, for characters
# where toTraditional() leaves the Japanese form unchanged. Small on purpose: add pairs as real
# misses turn up. 粧/妝: 化粧室 vs 化妝室.
_CHAR_VARIANTS = {
    "粧": "妝",
}
_REVERSE_VARIANTS = {v: k for k, v in _CHAR_VARIANTS.items()}


def _hanjaModule():
    """Lazy + optional: the `hanja` package is a soft dependency (requirements.txt)."""
    try:
        import hanja
        return hanja
    except ImportError:
        return None


def _splitStem(lemma: str):
    """Returns the kanji stem, or None when the lemma isn't 'kanji-only[+する]'."""
    stem = lemma[:-2] if lemma.endswith("する") else lemma
    return stem if _KANJI_STEM.match(stem) else None


def _chineseForms(stem: str) -> list:
    """Traditional form first, then the same with each variant substitution applied (both
    directions: Japanese/Korean 粧 -> Chinese 妝, and a Chinese-spelled input back to 粧)."""
    trad = toTraditional(stem)
    forms = [trad]
    for table in (_CHAR_VARIANTS, _REVERSE_VARIANTS):
        variant = "".join(table.get(ch, ch) for ch in trad)
        if variant not in forms:
            forms.append(variant)
    return forms


def _wiktionaryAttests(hanjaForms, hangul: str):
    """Returns a Korean gloss list if Wiktionary ties one of these Hanja spellings to `hangul`."""
    for form in hanjaForms:
        for entry in allEntries(form):
            for gloss in entry.get("gloss", []):
                if ("hanja form of " + hangul) in gloss.lower():
                    return entry["gloss"]
    for entry in allEntries(hangul):
        if entry.get("hanja") and any(f in entry["hanja"].split("/") for f in hanjaForms):
            return entry["gloss"]
    return None


def _kiwiKnowsNoun(hangul: str) -> bool:
    # NNP (proper noun) is excluded on purpose: 場合 -> 장합 is a Three Kingdoms general's name.
    from core.korean_utils import getKiwi
    tokens = getKiwi().tokenize(hangul)
    return len(tokens) == 1 and tokens[0].form == hangul and tokens[0].tag in ("NNG", "XR", "NR")


def lookupHangulCognate(lemma: str, category: str):
    """
    Returns None (no Korean cognate to claim), or
        {"hangul", "hanja", "tier": "attested"|"plausible", "gloss": [...],
         "chineseStatus": "confirmed"}
    `hanja` is the Chinese-attested spelling that passed gate 2. `gloss` is Korean Wiktionary's
    gloss for the tier-"attested" case, [] for "plausible". Never fabricates: any failed gate
    returns None. `category` is the classifyReading() result the vocab row already stores.
    """
    if category not in _CONVERTIBLE_CATEGORIES:
        return None
    stem = _splitStem(lemma)
    if stem is None:
        return None
    hanjaLib = _hanjaModule()
    if hanjaLib is None:
        return None

    forms = _chineseForms(stem)
    attestedForm = next((f for f in forms if lookupChineseCognate(f)["status"] == "confirmed"), None)
    if attestedForm is None:
        return None  # Japan-only coinage (or native word): not Chinese-attested

    hangul = hanjaLib.translate(attestedForm, "substitution")
    if not _HANGUL_ONLY.match(hangul):
        return None

    # Also try the input spelling itself: Korean Wiktionary keys 化粧室 (Japanese/Korean 粧), not 化妝室.
    gloss = _wiktionaryAttests(list(dict.fromkeys(forms + [stem])), hangul)
    if gloss is not None:
        return {"hangul": hangul, "hanja": attestedForm, "tier": "attested",
                "gloss": gloss, "chineseStatus": "confirmed"}
    if _kiwiKnowsNoun(hangul):
        return {"hangul": hangul, "hanja": attestedForm, "tier": "plausible",
                "gloss": [], "chineseStatus": "confirmed"}
    return None
