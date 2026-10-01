"""Kanji reference collector - Milestones 1-6: Shinjitai->Traditional conversion, on'yomi/
kun'yomi classification, Chinese-cognate lookup via CC-CEDICT, per-song vocab persistence, a
cross-song shared-vocabulary index scanned from saved_labels/, and Japanese-meaning lookup via
JMdict (for every word, not just onyomi/mixed).

See .claude/KANJI_REFERENCE_PLAN.md for the full design.
"""

import codecs
import glob
import html as htmlEscape
import json
import os
import re
from collections import defaultdict

from core.japanese_utils import getTagger, normalizeVariantKanji, overrideReading, tokenReading
from core.lyric_text import stripAll

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "jp2t")

_charDict = None
_phraseDict = None
_maxPhraseLen = 1


def _loadDict(filename):
    result = {}
    path = os.path.join(_DATA_DIR, filename)
    with codecs.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            key = parts[0]
            value = parts[1].split(" ")[0]  # first candidate when multiple are listed
            result[key] = value
    return result


def _ensureLoaded():
    global _charDict, _phraseDict, _maxPhraseLen
    if _charDict is None:
        _charDict = _loadDict("JPShinjitaiCharacters.txt")
    if _phraseDict is None:
        _phraseDict = _loadDict("JPShinjitaiPhrases.txt")
        _maxPhraseLen = max((len(k) for k in _phraseDict), default=1)


def toTraditional(text: str) -> str:
    """
    Convert Japanese Shinjitai Kanji in `text` to their Traditional Chinese equivalent forms
    (e.g. 楽->樂, 会->會), using OpenCC's JPShinjitai dictionaries (see data/jp2t/NOTICE.md).

    Longest-match first against the phrase dictionary (compound words where a per-character
    swap alone would be wrong), falling back to a per-character lookup, then to the character
    unchanged if it isn't in either table (true for the large majority of Kanji, which don't
    differ between Japanese and Traditional Chinese forms at all).
    """
    _ensureLoaded()

    result = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        maxLen = min(_maxPhraseLen, n - i)
        for length in range(maxLen, 1, -1):
            candidate = text[i:i + length]
            if candidate in _phraseDict:
                result.append(_phraseDict[candidate])
                i += length
                matched = True
                break
        if matched:
            continue

        ch = text[i]
        result.append(_charDict.get(ch, ch))
        i += 1

    return "".join(result)


# --- Milestone 2: on'yomi/kun'yomi classification ---

_KATAKANA_TO_HIRAGANA = str.maketrans({
    chr(code): chr(code - 0x60) for code in range(0x30A1, 0x30F7)
})

_READINGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "kanjidic2", "readings.json")
_kanjiReadings = None


def _katakanaToHiragana(text: str) -> str:
    return text.translate(_KATAKANA_TO_HIRAGANA)


def _loadKanjiReadings():
    global _kanjiReadings
    if _kanjiReadings is None:
        with codecs.open(_READINGS_PATH, "r", encoding="utf-8") as f:
            _kanjiReadings = json.load(f)
    return _kanjiReadings


def _isKanji(ch: str) -> bool:
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF


def _matchSegmentation(surface: str, reading: str, charIndex: int, readIndex: int, allowOn: bool, allowKun: bool) -> bool:
    """
    Recursive backtracking match: can `reading` (hiragana) be fully accounted for by walking
    `surface` character by character, using only the allowed reading type(s) for each Kanji?

    Any non-Kanji character (okurigana, particles) must be consumed as part of a specific Kanji's
    own declared kun'yomi okurigana (see the "okuri" handling below) - it is never independently
    "free" just because it happens to match the reading text literally. A real bug found via a
    real word, 切ない (setsunai, "heartrending") illustrates exactly why: 切's on'yomi せつ
    matches the start of the reading せつない, and an earlier version of this function then let
    the leftover "ない" pass through via a blanket literal-identity check with no requirement
    that it be legitimate okurigana for anything - which wrongly classified 切ない as a pure
    on'yomi (Sino-Japanese) word. Real on'yomi compounds are always 100% Kanji with zero embedded
    kana (時間, 瞬間, 残業 all have none) - 切ない's "ない" is a native adjective-forming suffix,
    unrelated to any of 切's actual dictionary readings (its kun'yomi is all about "cutting" -
    き.る/き.れる - not "nai"). Letting the classifier treat it as free led directly to a second,
    downstream bug: lookupChineseCognate()/getMandarinPinyin() then ran on the full lemma "切ない"
    including the kana, silently pasting the literal kana into a "Chinese" pinyin/traditional
    string (e.g. "qie1 な い") - nonsense for a word that was never a real Sino-Japanese compound
    to begin with. Removing this blanket pass-through doesn't lose any real word: genuine
    kun'yomi-with-okurigana words (出会う, 危ない, 少ない) already fully consume their own
    trailing kana through the explicit "okuri" branch below in the very same recursive step as
    their kanji, so they never reach this function pointed at a bare kana character at all.
    """
    if charIndex == len(surface) and readIndex == len(reading):
        return True
    if charIndex == len(surface) or readIndex > len(reading):
        return False

    ch = surface[charIndex]

    if not _isKanji(ch):
        return False

    entry = _loadKanjiReadings().get(ch)
    if not entry:
        return False

    if allowOn:
        for onReading in entry["on"]:
            onHira = _katakanaToHiragana(onReading)
            length = len(onHira)
            if reading[readIndex:readIndex + length] == onHira:
                if _matchSegmentation(surface, reading, charIndex + 1, readIndex + length, allowOn, allowKun):
                    return True

    if allowKun:
        for kunReading in entry["kun"]:
            core, _, okuri = kunReading.partition(".")
            length = len(core)
            if reading[readIndex:readIndex + length] != core:
                continue
            if not okuri:
                if _matchSegmentation(surface, reading, charIndex + 1, readIndex + length, allowOn, allowKun):
                    return True
                continue
            # Okurigana must literally appear next in the surface text (it's already kana).
            okuriLen = len(okuri)
            if surface[charIndex + 1:charIndex + 1 + okuriLen] == okuri:
                if _matchSegmentation(
                    surface, reading, charIndex + 1 + okuriLen, readIndex + length + okuriLen, allowOn, allowKun
                ):
                    return True

    return False


def classifyReading(surface: str, reading: str) -> str:
    """
    Classify a word's reading as "onyomi" (Sino-Japanese compound), "kunyomi" (native Japanese
    word), "mixed" (jūbako/yutō - one part on'yomi, one part kun'yomi), or "jukujigo" (idiomatic
    whole-word reading, e.g. 今日=kyou, that can't be decomposed per character at all).

    `reading` should already be in hiragana (see resolveContextReading).
    """
    if _matchSegmentation(surface, reading, 0, 0, allowOn=True, allowKun=False):
        return "onyomi"
    if _matchSegmentation(surface, reading, 0, 0, allowOn=False, allowKun=True):
        return "kunyomi"
    if _matchSegmentation(surface, reading, 0, 0, allowOn=True, allowKun=True):
        return "mixed"
    return "jukujigo"


def resolveContextReading(fullText: str, startOffset: int, endOffset: int):
    """
    Tokenize `fullText` and return the tokens overlapping the character range
    [startOffset, endOffset). This resolves the *actual* in-context reading rather than
    tokenizing the highlighted substring in isolation - critical because a single Kanji's
    reading is context-dependent (e.g. bare "時" tokenizes as "ジ" with no context, but as
    "とき" in a real sentence like "時が止まる").
    """
    tagger = getTagger()
    pos = 0
    matchedTokens = []

    for word in tagger(normalizeVariantKanji(fullText)):
        # fugashi/MeCab never emits whitespace (spaces, newlines) as a token of its own -
        # it's dropped from `surface` entirely and only reported via the *next* token's
        # `white_space` attribute. Skipping this means `pos` silently drifts out of sync
        # with the real string offset after every space/newline in the text (e.g. a
        # multi-line lyric field, or Japanese text with an embedded English word), which
        # then misattributes every token after the first skipped character - confirmed to
        # make a genuinely-highlighted Kanji word fail to match at all.
        pos += len(word.white_space)
        tokenLen = len(word.surface)
        tokenStart, tokenEnd = pos, pos + tokenLen
        if tokenEnd > startOffset and tokenStart < endOffset:
            matchedTokens.append(word)
        pos = tokenEnd

    return matchedTokens


def _containsKanji(text: str) -> bool:
    return any(_isKanji(ch) for ch in text)


_KATAKANA_RANGE = (0x30A0, 0x30FF)


def _isKatakanaOnly(text: str) -> bool:
    return bool(text) and all(_KATAKANA_RANGE[0] <= ord(ch) <= _KATAKANA_RANGE[1] for ch in text)


def _hasJapaneseScript(text: str) -> bool:
    """True if `text` contains any Kanji, hiragana or katakana. Japanese lyrics are full of English
    ("How I am gonna find it", "me", "oh") and fugashi tokenizes those as ordinary nouns, so without
    this check they became junk Japanese flashcards. Digits/Latin/punctuation-only tokens fail."""
    return any(
        _isKanji(ch) or 0x3040 <= ord(ch) <= 0x30FF  # hiragana + katakana blocks (incl. ー)
        for ch in text
    )


# Content/function-word classification for fugashi/UniDic tokens - lives here (not in
# core/grammar_breakdown.py, which needs the exact same distinction) because that module already
# imports _containsKanji/lookupJapaneseMeaning FROM this one; putting these here instead of there
# keeps that a one-way, non-circular dependency.
_FUNCTION_POS1 = {"助詞", "助動詞"}
# 空白 (whitespace, e.g. the full-width "　" some lyrics use as a mid-line separator) carries no
# grammar of its own, same reasoning as punctuation - never a real vocabulary word.
_SKIP_POS1 = {"補助記号", "記号", "空白"}

# UniDic marks a token as grammaticalized (used as a light verb / auxiliary stem rather than its
# literal dictionary sense) via pos2, not pos1 - see core/grammar_breakdown.py's own docstring for
# the real words (しまう/いる/なる/よう) this was found against.
_GRAMMATICALIZED_POS2 = {"助動詞語幹", "非自立可能"}


def _pos2Of(word):
    return getattr(word.feature, "pos2", None)


def isContentWord(word) -> bool:
    """
    True for an ordinary content word (noun/verb/adjective/adverb/...) as opposed to punctuation/
    whitespace, a particle/auxiliary-verb, or a grammaticalized light-verb/auxiliary-stem use of
    an otherwise-ordinary lemma. Kanji-presence plays no part in this - a kana-only content word
    (ちょっと, とても) is exactly as much a "content word" as a Kanji-bearing one; see
    analyzeSelection()'s own comment for why that distinction matters for vocab intake.
    """
    pos1 = word.feature.pos1
    if pos1 in _SKIP_POS1 or pos1 in _FUNCTION_POS1:
        return False
    return _pos2Of(word) not in _GRAMMATICALIZED_POS2


def _analyzeToken(word) -> dict:
    """
    Build one word-level result from a single fugashi token.

    Classification is run against the token's *lemma* (dictionary/citation form) rather than
    its raw conjugated surface - e.g. 出会った ("deatta", te/ta-form of 出会う) has okurigana that
    doesn't match KANJIDIC2's dictionary-form entry for 会 ("あ.う") at all, since conjugation
    changes the okurigana (会う -> 会っ-て/-た). Classifying the lemma 出会う instead sidesteps
    that entirely, since a lemma is always in citation form. Nouns/particles are unaffected
    (their lemma equals their surface).
    """
    surface = word.surface
    reading = _katakanaToHiragana(tokenReading(word))
    lemma = word.feature.lemma or surface  # unidic gives None for some unknown tokens
    lemmaReading = _katakanaToHiragana(getattr(word.feature, "lForm", None) or tokenReading(word))

    # unidic-lite bakes a "-<POS category>" disambiguator suffix directly into the lemma for
    # some common homograph-prone words - confirmed via real saved_labels data: 私 (BTS/Let Go)
    # and 君 (TWICE/Funny Valentine) both come back as "私-代名詞"/"君-代名詞" ("-pronoun"), not
    # bare "私"/"君". A literal ASCII hyphen never appears in a genuine Japanese lemma, so
    # stripping everything from the first hyphen onward recovers the real dictionary form - this
    # was silently misclassifying both as "jukujigo" instead of kunyomi, and made JMdict lookups
    # (Milestone 6) fail outright, since "私-代名詞" isn't a real kanji spelling in JMdict.
    if "-" in lemma:
        lemma = lemma.split("-", 1)[0]

    # unidic-lite sometimes stores an inconsistent, katakana-only lemma for certain proper nouns
    # (e.g. 東京's lemma comes back as "トウキョウ", not "東京" - a dictionary data gap, not
    # something fixable in code, and confirmed not universal: 日本's lemma is correctly "日本").
    # If the lemma lost the kanji the surface actually has, it can't be a real citation form -
    # fall back to classifying the surface directly instead of trusting the broken lemma.
    if _containsKanji(surface) and not _containsKanji(lemma):
        lemma, lemmaReading = surface, reading

    # lForm bypasses tokenReading()'s casual-register overrides (明日 -> アス, 私 -> ワタクシ), so
    # re-apply them here or flashcards show あす/わたくし while the lyric romaji says ashita.
    casual = overrideReading(lemma)
    if casual:
        lemmaReading = _katakanaToHiragana(casual)

    category = classifyReading(lemma, lemmaReading)

    # Chinese-cognate lookup only makes sense for Sino-Japanese readings (onyomi/mixed) - a
    # kunyomi/jukujigo word is native Japanese, so never fabricate a Chinese equivalent for it.
    chineseCognate = lookupChineseCognate(lemma) if category in ("onyomi", "mixed") else None

    # Unlike chineseCognate, the word's own Japanese meaning (Milestone 6, JMdict) is looked up
    # for EVERY word regardless of category - a kunyomi word has no Chinese shortcut at all, so
    # this is the only source of meaning it ever gets.
    japaneseMeaning = lookupJapaneseMeaning(lemma, lemmaReading)

    # Also unconditional (Milestone 7) - a purely phonetic memorization aid (Mandarin pinyin for
    # the Kyuujitai/Traditional form), never an attestation claim - see getMandarinPinyin().
    mandarinPinyin = getMandarinPinyin(lemma, chineseCognate)

    return {
        "surface": surface,
        "reading": reading,
        "lemma": lemma,
        "lemmaReading": lemmaReading,
        "category": category,
        "chineseCognate": chineseCognate,
        "japaneseMeaning": japaneseMeaning,
        "mandarinPinyin": mandarinPinyin,
    }


def analyzeSelection(fullText: str, startOffset: int, endOffset: int) -> list:
    """
    Full Milestone 2+3+6 result for a highlighted span: one entry per real word overlapping the
    highlighted range - every Kanji-containing word, PLUS every kana-only content word (real gap
    found via a live-vocab audit: this used to keep Kanji-containing words only, which silently
    dropped every kana-only content word - ちょっと/とても/これ/etc - from vocab intake entirely,
    not just particles/punctuation as the exclusion was originally intended to cover). A
    katakana-only surface is still excluded here (loanwords are out of scope for this filter, not
    "not a word"), and a real function word (particle/auxiliary-verb/grammaticalized light-verb
    use) is still excluded via isContentWord() - see its own docstring. Highlighting a single word
    gives a one-item list; highlighting a whole line gives a per-word breakdown of every eligible
    word in it. Tokens with no Japanese script at all (English words embedded in a Japanese lyric)
    are excluded - see _hasJapaneseScript().
    """
    tokens = resolveContextReading(fullText, startOffset, endOffset)
    return [
        _analyzeToken(word) for word in tokens
        if _hasJapaneseScript(word.surface)
        and (_containsKanji(word.surface) or (isContentWord(word) and not _isKatakanaOnly(word.surface)))
    ]


# --- Milestone 3: Chinese-cognate lookup via CC-CEDICT ---

_CEDICT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "cedict", "index.json")
_cedictIndex = None


def _loadCedictIndex():
    global _cedictIndex
    if _cedictIndex is None:
        with codecs.open(_CEDICT_PATH, "r", encoding="utf-8") as f:
            _cedictIndex = json.load(f)
    return _cedictIndex


def _normalizePinyin(pinyin: str) -> str:
    """
    Both KANJIDIC2 and CC-CEDICT encode the umlaut vowel "u" (as in lu:3 = lu with an umlaut,
    3rd tone) using a plain ASCII "u:" rather than the actual character - a historical plain-text
    convention neither source ever expands. Left as-is, real words come out wrong (履 -> "lu:3"
    instead of a real pinyin spelling) - found while adding per-character pinyin display for 履
    (lu:3), which made this pre-existing bug visible for the first time (no earlier example in
    this project happened to contain a "u:" character). Fixed by replacing "u:" with the actual
    "ü" character, so 履 renders as "lü3" rather than the raw source encoding.
    """
    return pinyin.replace("u:", "ü")


def _fallbackPinyin(word: str) -> str:
    """
    Build a constructed, best-guess pinyin for a word that isn't attested in CC-CEDICT, by
    concatenating each character's own best-available pinyin reading. This is NOT a real Chinese
    reading for the word as a whole (the word doesn't exist in Chinese) - it exists only to give a
    rough sense of how the characters "would" sound in Mandarin. Never labelled as anything but
    constructed.
    """
    return " ".join(_charPinyin(ch) for ch in word)


def _charPinyin(ch: str) -> str:
    """
    Best-available Mandarin pinyin for a single character, preferring CC-CEDICT's own listing
    over KANJIDIC2's `pinyin` field when the character is a CC-CEDICT headword in its own right.

    Real bug found via the user's own 言葉 example: KANJIDIC2's `pinyin` field is borrowed from
    Unihan for Japanese-context cross-reference, and its ordering is NOT reliably "most common
    reading first" - confirmed directly against kanjidic2.xml: 葉's own `<reading r_type="pinyin">`
    elements are listed `xie2`, `ye4`, `she4`, in that literal document order (xie2 is a real but
    archaic/classical reading - it's the reading of 叶, the character later reused as 葉's
    Simplified form, in its own original, unrelated sense "to harmonize/rhyme" - not a parsing
    bug, just an unhelpful default for this purpose). Naively taking index 0 silently produced
    "yan2 xie2" for 言葉 instead of the expected "yan2 ye4". CC-CEDICT, being a dictionary of
    actual Chinese usage rather than a Japanese-context cross-reference, doesn't have this
    problem for the common case - checked against every character already exercised by this
    project (殘, 業, 時, 間, 気, 會, 出, 夢, 履, 甘, 手, 私) before relying on it, and all of them
    already agreed with KANJIDIC2's index-0 pick, meaning 葉 is a real, narrow miss, not a sign
    that the whole approach was unreliable.

    CC-CEDICT's own single-character entries can *also* have their own ordering quirk: many
    characters that double as common surnames list the surname reading first, capitalized (e.g.
    葉 -> "Ye4"/surname before "ye4"/leaf; 業 -> "Ye4"/surname before "ye4"/occupation) - a real,
    intentional MDBG convention (capitalization marks a proper-noun reading), not a defect. Skip
    straight past those by preferring the first entry whose pinyin does NOT start with a capital
    letter, falling back to the first entry outright only if every single one is a surname
    reading (i.e. the character isn't ordinarily used as a common word in Chinese at all).

    Falls back to KANJIDIC2's own pinyin[0] only when the character isn't a CC-CEDICT headword at
    all (confirmed this happens for at least 気); falls back to the bare character itself if
    neither source has anything (e.g. actual kana mixed into a kunyomi verb's lemma, such as the
    く in 履く - see getMandarinPinyin()'s Kanji-only filtering, which normally keeps this branch
    from even being reached for kana, but it's kept here as a safe default regardless).
    """
    cedictEntries = _loadCedictIndex().get(ch)
    if cedictEntries:
        nonSurname = [e for e in cedictEntries if not e["pinyin"][:1].isupper()]
        candidates = nonSurname or cedictEntries[:1]

        # CEDICT's own per-character ordering is file order, not frequency - confirmed real miss:
        # 行 lists hang2 ("row/profession/bank") first and xing2 ("to walk/go/OK") third, even
        # though xing2 is by far the more common reading for a kun'yomi verb like 行く (iku, "to
        # go"). There's no real frequency data to rank against, but the number of distinct
        # senses/idioms CEDICT attaches to a reading is a decent proxy for how central it is (行:
        # xing2 has 9, hang2 has 6, heng2 - only used in one fixed compound - has 1). Rather than
        # commit to a single, possibly-wrong pick, show the top 2 distinct readings (ties keep
        # their original relative order) so the rarer-but-first-listed reading never gets shown
        # alone as if it were the only or obviously-correct one.
        ranked = sorted(candidates, key=lambda e: -len(e["gloss"]))
        pinyins = list(dict.fromkeys(_normalizePinyin(e["pinyin"]) for e in ranked))
        return "/".join(pinyins[:2])

    entry = _loadKanjiReadings().get(ch)
    if entry and entry.get("pinyin"):
        return _normalizePinyin(entry["pinyin"][0])

    return ch


def lookupChineseCognate(word: str) -> dict:
    """
    Look up the Chinese equivalent of an on'yomi/mixed Japanese word. Must only be called for
    "onyomi"/"mixed" classifications (see classifyReading) - a kun'yomi or jukujigo word is
    native Japanese and has no Chinese cognate to find, real or otherwise.

    Converts `word` to Traditional Chinese forms first (Milestone 1's toTraditional()), then
    looks it up in CC-CEDICT:
      - Found -> {"status": "confirmed", "traditional": ..., "pinyin": ..., "gloss": [...]}
        using CC-CEDICT's first-listed reading/sense (a word can have several; the first is
        MDBG's primary entry). Note this only confirms the character string is a real Chinese
        word - it does not guarantee the meaning matches the Japanese sense (e.g. 大丈夫/大丈夫
        means "a manly man" in Chinese, not "it's okay" as in Japanese - a false friend, not a
        bug in the lookup).
      - Not found -> {"status": "not_attested", "traditional": ..., "pinyinFallback": ...},
        a Japan-coined term (和製漢語) with a constructed (not real) fallback pinyin.
    """
    traditional = toTraditional(word)
    index = _loadCedictIndex()
    entries = index.get(traditional)

    if entries:
        # Real bug found while adding Hanja pinyin: a single character that also doubles as a
        # common Chinese surname (e.g. 火) lists that surname reading FIRST, capitalized
        # ("Huo3"/"surname Huo"), ahead of its ordinary-word entry ("huo3"/"fire") - same
        # CC-CEDICT convention _charPinyin() already skips past (see its own docstring). Taking
        # entries[0] here fabricated a "surname Huo" gloss for what should read as "fire".
        nonSurname = [e for e in entries if not e["pinyin"][:1].isupper()]
        candidates = nonSurname or entries[:1]

        # Same file-order-isn't-frequency-order issue _charPinyin() already found for 行 also
        # applies here: 化 alone lists "hua1" (an obscure variant of 花) before "hua4" (the far
        # more relevant "-ization" reading) - ranking by attached sense count as a rough proxy for
        # how central a reading is (same heuristic, same justification as _charPinyin) picks the
        # more useful entry instead of trusting CC-CEDICT's raw file order. Ties keep their
        # original relative order (stable sort).
        best = max(candidates, key=lambda e: len(e["gloss"]))
        return {
            "status": "confirmed",
            "traditional": traditional,
            "pinyin": _normalizePinyin(best["pinyin"]),
            "gloss": best["gloss"],
        }

    return {
        "status": "not_attested",
        "traditional": traditional,
        "pinyinFallback": _fallbackPinyin(traditional),
    }


def getMandarinPinyin(lemma: str, chineseCognate: dict) -> dict:
    """
    Return the best-available Mandarin pinyin reading for `lemma`'s Kyuujitai/Traditional form,
    for EVERY word regardless of on'yomi/kun'yomi category - a purely phonetic memorization aid,
    never a claim about whether `lemma` is a real Chinese word (that claim belongs to
    chineseCognate alone, and stays gated to onyomi/mixed).

    This is a deliberate, narrower addition on top of the project's founding rule ("never
    fabricate a Chinese equivalent for a native Japanese word" - see the 出会う worked example at
    the top of this doc): showing how the characters *sound* in isolation is always true, whether
    or not the compound is attested as a Chinese word, so it doesn't fabricate anything the way
    presenting it as a "confirmed cognate" would. Added per direct user request as a mnemonic
    technique - visualizing a kun'yomi word's Kyuujitai form through its own Chinese-character
    knowledge (e.g. 夢 read as kunyomi "yume" in Japanese, but recalling Mandarin "meng4" via the
    shared character is a real, useful memory hook even though 夢 the *word* isn't a Chinese-Japanese
    cognate in the linguistic sense chineseCognate cares about).

    Reuses chineseCognate's own traditional-form conversion and pinyin when one was already
    computed (onyomi/mixed) - both its "confirmed" (real CC-CEDICT pinyin) and "not_attested"
    (constructed fallback) branches already carry exactly this data. Only for kunyomi/jukujigo
    words (chineseCognate is always None there) does this compute fresh via toTraditional() +
    _fallbackPinyin() - the same construction as the "not_attested" fallback, just run
    unconditionally instead of only for onyomi/mixed misses.

    Kunyomi verbs commonly carry real okurigana kana in their lemma (履く, not just 履) - unlike
    onyomi/mixed lemmas, which classifyReading() only ever accepts as pure-Kanji strings.
    _fallbackPinyin() has no pinyin for a kana character, so it would otherwise paste the raw
    kana literally into the result (e.g. "lü3 く"); Kanji characters are extracted first here so
    only the parts that actually have a Mandarin reading are shown.

    Returns {"traditional": ..., "pinyin": ...} - always populated, never None.
    """
    if chineseCognate is not None:
        pinyin = chineseCognate["pinyin"] if chineseCognate["status"] == "confirmed" else chineseCognate["pinyinFallback"]
        return {"traditional": chineseCognate["traditional"], "pinyin": pinyin}

    kanjiOnly = "".join(ch for ch in lemma if _isKanji(ch))
    traditional = toTraditional(kanjiOnly)
    return {"traditional": traditional, "pinyin": _fallbackPinyin(traditional)}


# --- Milestone 6: Japanese-meaning lookup via JMdict (every word, not just onyomi/mixed) ---

_JMDICT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "jmdict", "index.json")
_jmdictIndex = None


def _loadJmdictIndex():
    global _jmdictIndex
    if _jmdictIndex is None:
        with codecs.open(_JMDICT_PATH, "r", encoding="utf-8") as f:
            _jmdictIndex = json.load(f)
    return _jmdictIndex


def lookupJapaneseMeaning(kanji: str, reading: str) -> dict:
    """
    Look up a word's own English meaning via JMdict - run for **every** Kanji word regardless of
    on'yomi/kun'yomi classification (unlike lookupChineseCognate(), which only ever runs for
    onyomi/mixed). This is the only source of meaning a kunyomi word gets, since it has no
    Chinese-cognate shortcut at all (e.g. 離す, "to separate" - a native Japanese verb the
    Chinese-lookup tier never even attempts).

    A kanji spelling can map to more than one JMdict entry with unrelated meanings - real cases
    found via testing:
      - Same reading, different kanji (homophones): 離す and 話す are both read はなす, but mean
        "to separate" and "to speak" respectively - JMdict keys them as separate entries, so
        looking up by (kanji, reading) together (not reading alone) resolves the right one.
      - Same kanji, different reading, unrelated meaning: 湯 read ゆ means "hot water" (the
        everyday Japanese sense), but the same kanji read タン means "soup" - a reading borrowed
        specifically for the Chinese sense of the character. This is a second, independent
        illustration of the same fact `lookupChineseCognate()`'s false-friend note already
        documents: a shared character does not imply a shared meaning across languages, and here
        it doesn't even imply a shared meaning across *readings of the same character*.

    Returns {"status": "found", "pos": [...], "gloss": [...]} using the entry whose readings list
    contains `reading` (falls back to the first entry for that kanji spelling if no reading
    matches exactly, e.g. a small reading mismatch from fugashi), taking the first-listed sense
    (JMdict orders senses by frequency/primacy) - or {"status": "not_found"} if the kanji
    spelling isn't in JMdict at all (rare; JMdict's ~229k kanji headwords cover the large
    majority of real vocabulary).
    """
    index = _loadJmdictIndex()
    candidates = index.get(kanji)
    if not candidates:
        return {"status": "not_found"}

    match = next((c for c in candidates if reading in c["readings"]), candidates[0])
    firstSense = match["senses"][0]
    return {
        "status": "found",
        "pos": firstSense["pos"],
        "gloss": firstSense["gloss"],
    }


# --- Milestone 4: personal per-song vocab persistence ---


def _vocabJsonPath(group: str, song: str) -> str:
    return f"saved_labels/{group}/{song}_kanji_vocab.json"


def _referenceHtmlPath(group: str, song: str) -> str:
    return f"saved_labels/{group}/{song}_kanji_reference.html"


def loadSongVocab(group: str, song: str) -> list:
    """Load the saved Kanji vocab entries for one song (see addWordToSongReference), or []
    if nothing has been saved for it yet."""
    path = _vocabJsonPath(group, song)
    if not os.path.exists(path):
        return []
    with codecs.open(path, "r", encoding="utf-8", errors="ignore") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
    return data if isinstance(data, list) else []


def addWordToSongReference(group: str, song: str, entry: dict) -> list:
    """
    Upsert one vocab entry into `<song>_kanji_vocab.json`, keyed by **lemma** - the same dedupe
    key philosophy as classifyReading() itself, so re-highlighting a different conjugation of an
    already-saved word (e.g. 出会う after 出会った was saved first) replaces the existing entry
    instead of creating a duplicate. `entry` is expected to carry analyzeSelection()'s per-word
    fields (surface/lemma/lemmaReading/category/chineseCognate) plus save metadata
    (dateAdded/sourceLine) - see .claude/KANJI_REFERENCE_PLAN.md's Milestone 4 schema.

    The JSON file is the source of truth; `<song>_kanji_reference.html` is a derived view that
    gets fully regenerated from the updated list on every call, never hand-edited or patched.
    Returns the full updated vocab list.
    """
    entries = loadSongVocab(group, song)
    lemma = entry["lemma"]

    for i, existing in enumerate(entries):
        if existing.get("lemma") == lemma:
            entries[i] = entry
            break
    else:
        entries.append(entry)

    path = _vocabJsonPath(group, song)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with codecs.open(path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=4)

    renderSongReferenceHtml(group, song, entries)
    return entries


def _renderCognateCellHtml(cognate) -> str:
    if not cognate:
        return ""
    if cognate["status"] == "confirmed":
        gloss = "; ".join(cognate["gloss"][:2])
        return (
            f"{htmlEscape.escape(cognate['traditional'])} ({htmlEscape.escape(cognate['pinyin'])})"
            f" &mdash; {htmlEscape.escape(gloss)}"
        )
    return (
        "not attested (Japan-coined term) &mdash; constructed pinyin: "
        f"{htmlEscape.escape(cognate['pinyinFallback'])}"
    )


def _renderJapaneseMeaningCellHtml(meaning) -> str:
    if not meaning or meaning.get("status") != "found":
        return ""
    gloss = "; ".join(meaning["gloss"][:3])
    return htmlEscape.escape(gloss)


def _renderMandarinPinyinCellHtml(pinyin) -> str:
    if not pinyin:
        return ""
    return (
        f"{htmlEscape.escape(pinyin['traditional'])} "
        f"<span class='pinyin'>({htmlEscape.escape(pinyin['pinyin'])})</span>"
    )


_HTML_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: "Yu Gothic", "Hiragino Sans", "Meiryo", sans-serif; margin: 2em; }}
  h1 {{ font-size: 1.3em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  .surface {{ font-size: 1.3em; }}
  .lemma {{ font-size: 0.85em; color: #555; }}
  .pinyin {{ color: #555; }}
  .source {{ white-space: pre-line; font-size: 0.9em; color: #444; }}
</style>
</head>
<body>
<h1>{heading}</h1>
<table>
<tr>{headerRow}</tr>
{rows}
</table>
</body>
</html>
"""


def renderSongReferenceHtml(group: str, song: str, vocabEntries: list) -> str:
    """
    Regenerate `<song>_kanji_reference.html` in full from `vocabEntries` and write it to disk.
    Always a full rebuild from the JSON, never an incremental patch. Returns the path written.
    """
    rows = []
    for entry in vocabEntries:
        lemmaLine = ""
        if entry.get("lemma") != entry.get("surface"):
            lemmaLine = (
                f"<div class='lemma'>Dictionary form: {htmlEscape.escape(entry['lemma'])} "
                f"({htmlEscape.escape(entry['lemmaReading'])})</div>"
            )
        rows.append(
            "<tr>"
            f"<td class='surface'>{htmlEscape.escape(entry['surface'])}</td>"
            f"<td>{htmlEscape.escape(entry['reading'])}{lemmaLine}</td>"
            f"<td>{htmlEscape.escape(entry['category'])}</td>"
            f"<td>{_renderMandarinPinyinCellHtml(entry.get('mandarinPinyin'))}</td>"
            f"<td>{_renderJapaneseMeaningCellHtml(entry.get('japaneseMeaning'))}</td>"
            f"<td>{_renderCognateCellHtml(entry.get('chineseCognate'))}</td>"
            f"<td class='source'>{htmlEscape.escape(entry.get('sourceLine', ''))}</td>"
            f"<td>{htmlEscape.escape(entry.get('dateAdded', ''))}</td>"
            "</tr>"
        )

    htmlDoc = _HTML_PAGE_TEMPLATE.format(
        title=htmlEscape.escape(f"{song} - Kanji Reference"),
        heading=htmlEscape.escape(f"{group} — {song}: Kanji Reference ({len(vocabEntries)} words)"),
        headerRow="<th>Word</th><th>Reading</th><th>Category</th><th>Mandarin Pinyin</th>"
                  "<th>Japanese Meaning</th><th>Chinese Cognate</th><th>Source Line</th>"
                  "<th>Date Added</th>",
        rows="\n".join(rows),
    )

    path = _referenceHtmlPath(group, song)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with codecs.open(path, "w", encoding="utf-8") as f:
        f.write(htmlDoc)
    return path


# --- Milestone 5: automatic cross-song shared-vocabulary index ---

_SAVED_LABELS_GLOB = "saved_labels/*/*_lyrics.json"

# Hiragana + Katakana blocks (deliberately excludes the Kanji ranges in _isKanji). Used as a
# content-based "is this line actually Japanese" signal - see the module docstring in
# .claude/KANJI_REFERENCE_PLAN.md's Milestone 5 section for why the stored "language" tag alone
# isn't reliable enough (a real saved_labels entry was found tagged "Korean" despite being a
# genuine Japanese verse).
_KANA_RANGES = ((0x3040, 0x309F), (0x30A0, 0x30FF))


def _containsKana(text: str) -> bool:
    return any(any(lo <= ord(ch) <= hi for lo, hi in _KANA_RANGES) for ch in text)


def _isJapaneseLyricEntry(lyricEntry: dict) -> bool:
    """
    Decide whether one saved_labels lyric entry's `korean` field is actually Japanese text.

    Kana-detection is the primary signal (content-based, can't be produced by Korean Hangul or
    English text) - it catches entries mislabeled `"language": "Korean"` that the strict tag
    filter would silently miss. But kana-detection ALONE is not sufficient either: verified
    against real data that some genuinely-Japanese saved_labels entries are pure Kanji with
    *zero* kana at all (e.g. `saved_labels/TWICE/Do Not Touch_lyrics.json`'s "準備" and "時"
    fragment entries) - a kana-only filter would silently drop those. So this is a union of both
    signals: the stored tag being "Japanese", OR the text containing kana.
    """
    text = lyricEntry.get("korean") or ""
    return lyricEntry.get("language") == "Japanese" or _containsKana(text)


def scanLyricsForKanjiVocab(labelsGlob: str = _SAVED_LABELS_GLOB) -> dict:
    """
    Walk every `saved_labels/<group>/<song>_lyrics.json`, run analyzeSelection() over each
    Japanese-detected entry's full `korean` text (see _isJapaneseLyricEntry), and aggregate the
    results into a cross-song index keyed by **lemma**. No new tokenization/offset logic is
    needed here - analyzeSelection(text, 0, len(text)) already returns a full per-word breakdown
    of an entire line/field, exactly what the "Add Kanji to Document" button does when you
    highlight a whole line.

    Returns {lemma: {"reading", "category", "chineseCognate", "occurrences": [...]}}, where each
    occurrence is {"group", "song", "memberName", "lyricId"}. Repeats of the same lemma within
    the *same* song are deduped to one occurrence, but every distinct song using the word is kept
    (so a word that appears 5 times in one song, or once each in 5 songs, is recorded distinctly).

    Any lyric entry whose text contains kana but isn't tagged `"language": "Japanese"` is
    printed as a possible mislabeled entry - a safety net for new data, not a hard failure - the
    scan is never blocked on it.
    """
    index = {}

    for path in sorted(glob.glob(labelsGlob)):
        group = os.path.basename(os.path.dirname(path))
        filename = os.path.basename(path)
        song = filename[: -len("_lyrics.json")] if filename.endswith("_lyrics.json") else filename

        with codecs.open(path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                lyricEntries = json.load(f)
            except json.JSONDecodeError:
                continue
        if not isinstance(lyricEntries, list):
            continue

        seenLemmasThisSong = set()
        for lyricEntry in lyricEntries:
            text = stripAll(lyricEntry.get("korean") or "")
            if not text.strip():
                continue

            if _containsKana(text) and lyricEntry.get("language") != "Japanese":
                print(
                    f"[kanji_reference] possible mislabeled entry: {path} lyricId="
                    f"{lyricEntry.get('lyricId')} contains kana but language="
                    f"{lyricEntry.get('language')!r}"
                )

            if not _isJapaneseLyricEntry(lyricEntry):
                continue

            for result in analyzeSelection(text, 0, len(text)):
                lemma = result["lemma"]
                if lemma in seenLemmasThisSong:
                    continue
                seenLemmasThisSong.add(lemma)

                bucket = index.setdefault(lemma, {
                    "reading": result["lemmaReading"],
                    "category": result["category"],
                    "chineseCognate": result["chineseCognate"],
                    "japaneseMeaning": result["japaneseMeaning"],
                    "mandarinPinyin": result["mandarinPinyin"],
                    "occurrences": [],
                })
                bucket["occurrences"].append({
                    "group": group,
                    "song": song,
                    "memberName": lyricEntry.get("memberName"),
                    "lyricId": lyricEntry.get("lyricId"),
                })

    return index


def _distinctSongCount(occurrences: list) -> int:
    return len({(occ["group"], occ["song"]) for occ in occurrences})


_WORD_INDEX_DIR = "kanji_reference"
_WORD_INDEX_PATH = os.path.join(_WORD_INDEX_DIR, "word_index.json")
_BY_SONG_HTML_PATH = os.path.join(_WORD_INDEX_DIR, "by_song.html")


def _slugify(text: str) -> str:
    """Turn arbitrary group/song text into a safe HTML id/anchor fragment (keeps unicode
    letters/digits so Japanese titles stay legible in the URL, replaces everything else)."""
    slug = re.sub(r"[^\w]+", "-", text, flags=re.UNICODE).strip("-")
    return slug or "x"


def _songKey(group: str, song: str) -> str:
    return f"song-{_slugify(group)}--{_slugify(song)}"


_BY_SONG_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  body {{ font-family: "Yu Gothic", "Hiragino Sans", "Meiryo", sans-serif; margin: 0; }}
  .layout {{ display: flex; align-items: flex-start; }}
  .sidebar {{
    width: 240px; flex-shrink: 0; height: 100vh; overflow-y: auto; position: sticky; top: 0;
    border-right: 1px solid #ccc; padding: 1em 0; box-sizing: border-box; background: #fafafa;
  }}
  .sidebar .nav-group {{
    font-weight: bold; padding: 0.6em 1em 0.2em; color: #555; font-size: 0.82em;
    text-transform: uppercase; letter-spacing: 0.03em;
  }}
  .sidebar ul {{ list-style: none; margin: 0 0 0.4em; padding: 0; }}
  .sidebar li a {{ display: block; padding: 0.3em 1em; text-decoration: none; color: #222; }}
  .sidebar li a:hover {{ background: #eee; }}
  .sidebar li a.active {{ background: #d6e4ff; font-weight: bold; }}
  .content {{ flex: 1; padding: 1.5em 2em; max-width: 900px; }}
  h1 {{ font-size: 1.2em; color: #555; margin-top: 0; }}
  h2 {{ font-size: 1.3em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f0f0f0; }}
  .surface {{ font-size: 1.3em; }}
  .pinyin {{ color: #555; }}
  .found-in {{ margin: 0; padding-left: 1.1em; }}
  .found-in a {{ color: #1a4fa0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""

# Plain inline JS (no dependency, no CDN) - this file is opened directly from disk, not served,
# so a single delegated click handler is enough to swap the visible song panel and keep the
# sidebar/found-in links, the URL hash, and the active-song highlight all in sync.
_BY_SONG_SCRIPT = """
<script>
(function () {
  function showSong(key) {
    document.querySelectorAll('.song-panel').forEach(function (el) { el.hidden = (el.id !== key); });
    document.querySelectorAll('.song-link').forEach(function (el) {
      el.classList.toggle('active', el.dataset.song === key);
    });
    if (location.hash !== '#' + key) { history.replaceState(null, '', '#' + key); }
  }
  document.addEventListener('click', function (e) {
    var link = e.target.closest('.song-link');
    if (!link) return;
    e.preventDefault();
    showSong(link.dataset.song);
  });
  window.addEventListener('hashchange', function () {
    var key = location.hash.slice(1);
    if (document.getElementById(key)) showSong(key);
  });
  var initial = location.hash ? location.hash.slice(1) : null;
  if (!initial || !document.getElementById(initial)) {
    var first = document.querySelector('.song-panel');
    initial = first ? first.id : null;
  }
  if (initial) showSong(initial);
})();
</script>
"""


def renderWordIndexBySongHtml(index: dict) -> str:
    """
    Render every word in `index` (from scanLyricsForKanjiVocab) as a single browsable HTML page,
    split by song rather than one flat table (per user request, 2026-09-08 redesign of the
    original flat "shared_words.html"). A sidebar lists every song (grouped by artist group,
    both sorted alphabetically); clicking one swaps the visible table via the inline JS above -
    no page reload, no external JS dependency, since this file is opened straight from disk.

    Every word the song contains is listed (not just words shared with another song) - a word
    unique to one song simply has an empty "Found In" cell. When a word does recur elsewhere,
    "Found In" lists every *other* song as a clickable link that jumps straight to that song's
    panel (via the same #song-<slug> anchors the sidebar uses).

    Writes and returns the `kanji_reference/by_song.html` path.
    """
    bySong = defaultdict(list)
    for lemma, data in index.items():
        songsForWord = sorted({(occ["group"], occ["song"]) for occ in data["occurrences"]})
        for gs in songsForWord:
            bySong[gs].append((lemma, data, songsForWord))

    songKeys = sorted(bySong.keys())  # tuples sort group-first, then song - matches the request

    navItems = []
    currentGroup = None
    for group, song in songKeys:
        if group != currentGroup:
            if currentGroup is not None:
                navItems.append("</ul>")
            navItems.append(f"<div class='nav-group'>{htmlEscape.escape(group)}</div><ul>")
            currentGroup = group
        key = _songKey(group, song)
        navItems.append(
            f"<li><a href='#{key}' class='song-link' data-song='{key}'>{htmlEscape.escape(song)}</a></li>"
        )
    if songKeys:
        navItems.append("</ul>")

    panels = []
    for group, song in songKeys:
        key = _songKey(group, song)
        words = sorted(bySong[(group, song)], key=lambda item: (item[1]["reading"], item[0]))

        rows = []
        for lemma, data, songsForWord in words:
            others = [gs for gs in songsForWord if gs != (group, song)]
            if others:
                foundInHtml = "<ul class='found-in'>" + "".join(
                    f"<li><a href='#{_songKey(g, s)}' class='song-link' data-song='{_songKey(g, s)}'>"
                    f"{htmlEscape.escape(g)} / {htmlEscape.escape(s)}</a></li>"
                    for g, s in others
                ) + "</ul>"
            else:
                foundInHtml = ""
            rows.append(
                "<tr>"
                f"<td class='surface'>{htmlEscape.escape(lemma)}</td>"
                f"<td>{htmlEscape.escape(data['reading'])}</td>"
                f"<td>{htmlEscape.escape(data['category'])}</td>"
                f"<td>{_renderMandarinPinyinCellHtml(data.get('mandarinPinyin'))}</td>"
                f"<td>{_renderJapaneseMeaningCellHtml(data.get('japaneseMeaning'))}</td>"
                f"<td>{_renderCognateCellHtml(data.get('chineseCognate'))}</td>"
                f"<td>{foundInHtml}</td>"
                "</tr>"
            )

        panels.append(
            f"<section class='song-panel' id='{key}' hidden>"
            f"<h2>{htmlEscape.escape(group)} &mdash; {htmlEscape.escape(song)} ({len(words)} words)</h2>"
            "<table><tr><th>Word</th><th>Reading</th><th>Category</th><th>Mandarin Pinyin</th>"
            f"<th>Japanese Meaning</th><th>Chinese Cognate</th><th>Found In</th></tr>{''.join(rows)}</table>"
            "</section>"
        )

    bodyHtml = (
        "<div class='layout'>"
        f"<nav class='sidebar'>{''.join(navItems)}</nav>"
        f"<main class='content'><h1>Kanji Reference by Song</h1>{''.join(panels)}</main>"
        "</div>"
        f"{_BY_SONG_SCRIPT}"
    )

    htmlDoc = _BY_SONG_PAGE_TEMPLATE.format(title="Kanji Reference by Song", body=bodyHtml)

    os.makedirs(_WORD_INDEX_DIR, exist_ok=True)
    with codecs.open(_BY_SONG_HTML_PATH, "w", encoding="utf-8") as f:
        f.write(htmlDoc)
    return _BY_SONG_HTML_PATH


def buildKanjiReferenceIndex(labelsGlob: str = _SAVED_LABELS_GLOB) -> dict:
    """
    Scan saved_labels/ for Kanji vocab (scanLyricsForKanjiVocab) and write both Milestone 5
    outputs: the full `kanji_reference/word_index.json` (every word regardless of how many songs
    it appears in) and `kanji_reference/by_song.html` (the same data browsable per-song, with
    cross-song "Found In" links - see renderWordIndexBySongHtml). Returns the full index dict.
    """
    index = scanLyricsForKanjiVocab(labelsGlob)

    os.makedirs(_WORD_INDEX_DIR, exist_ok=True)
    with codecs.open(_WORD_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)

    renderWordIndexBySongHtml(index)
    return index


if __name__ == "__main__":
    _index = buildKanjiReferenceIndex()
    _sharedCount = sum(1 for _data in _index.values() if _distinctSongCount(_data["occurrences"]) >= 2)
    print(
        f"Indexed {len(_index)} words across {_SAVED_LABELS_GLOB} "
        f"({_sharedCount} appear in 2+ distinct songs)."
    )
    print(f"Wrote {_WORD_INDEX_PATH} and {_BY_SONG_HTML_PATH}")
