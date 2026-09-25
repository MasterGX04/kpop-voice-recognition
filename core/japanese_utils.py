"""Kanji -> Hiragana/Romaji reading conversion for the Japanese lyric mode."""

import re

_tagger = None
_kakasi = None

# Katakana (U+30A1-U+30F6) -> Hiragana (U+3041-U+3096)
_KATAKANA_TO_HIRAGANA = str.maketrans({
    chr(code): chr(code - 0x60) for code in range(0x30A1, 0x30F7)
})

_NO_LEADING_SPACE_POS = {"補助記号", "記号"}

# Katakana -> vowel class ("a"/"i"/"u"/"e"/"o"), used to expand the chouonpu (long-vowel
# mark "ー") into the conventionally-spelled kana it stands in for (see _expandLongVowelMark).
_KATAKANA_VOWEL_ROWS = {
    "a": "アカガサザタダナハバパマヤラワァャ",
    "i": "イキギシジチヂニヒビピミリィ",
    "u": "ウクグスズツヅヌフブプムユルゥュヴ",
    "e": "エケゲセゼテデネヘベペメレェ",
    "o": "オコゴソゾトドノホボポモヨロヲォョ",
}
_KATAKANA_VOWEL = {ch: vowel for vowel, row in _KATAKANA_VOWEL_ROWS.items() for ch in row}

# The kana conventionally written for a long vowel of each class (e/o default to the far more
# common i/u spelling - e.g. 学校=gakkou not gakkoo, 先生=sensei not sensee - occasionally wrong
# for the rarer literal-repeat words like おおきい/ねえさん, but right for the vast majority,
# especially Sino-Japanese on'yomi compounds like 重要, which is what prompted this fix).
_LONG_VOWEL_KATAKANA = {"a": "ア", "i": "イ", "u": "ウ", "e": "イ", "o": "ウ"}


def _expandLongVowelMark(katakana: str) -> str:
    result = []
    prevVowel = None
    for ch in katakana:
        if ch == "ー":
            if prevVowel:
                replacement = _LONG_VOWEL_KATAKANA[prevVowel]
                result.append(replacement)
                prevVowel = _KATAKANA_VOWEL[replacement]
            else:
                result.append(ch)
            continue
        result.append(ch)
        prevVowel = _KATAKANA_VOWEL.get(ch)
    return "".join(result)


def _getTagger():
    global _tagger
    if _tagger is None:
        import fugashi
        _tagger = fugashi.Tagger()
    return _tagger


def getTagger():
    """Public accessor for the shared fugashi Tagger singleton (used by core/kanji_reference.py)."""
    return _getTagger()


def tokenReading(word, prevWord=None) -> str:
    """Public accessor for a fugashi token's katakana reading (pron, falling back to kana/surface).
    Pass the preceding token as `prevWord` when available, so the 心-suffix on'yomi/kun'yomi
    override (see _wordReading) can apply; existing callers that don't pass it are unaffected."""
    return _tokenKatakanaReading(word, prevWord)


def _getKakasi():
    global _kakasi
    if _kakasi is None:
        import pykakasi
        _kakasi = pykakasi.kakasi()
    return _kakasi


# unidic's default reading for a few very common words doesn't match how they're actually
# read in the casual/spoken register lyrics are almost always in - e.g. 私 defaults to the
# formal ワタクシ (watakushi) rather than the near-universal ワタシ (watashi). These are
# always tokenized as their own standalone surface (confirmed: even in compounds like 私たち
# fugashi splits it back out as its own token), so a surface-keyed override is safe and won't
# misfire on unrelated tokens.
_READING_OVERRIDES = {
    "私": "ワタシ",
    # unidic's default reading for 明日 is the more formal/literary あす (asu); the near-universal
    # casual reading あした (ashita) lyrics actually use is a distinct, lower-priority entry in
    # its dictionary. Confirmed always its own standalone token (明日香 "Asuka" tokenizes as one
    # fused token with a completely different surface, so it's unaffected by this override).
    "明日": "アシタ",
}


def _isAllKatakana(text: str) -> bool:
    return bool(text) and all(0x30A0 <= ord(ch) <= 0x30FF for ch in text)


def _wordReading(word, prevWord=None):
    # 心 as a bound suffix (接尾辞, "-spirit/-mindset") is genuinely ambiguous between on'yomi
    # シン (a real Sino-Japanese compound, e.g. 好奇心 kōkishin) and kun'yomi ゴコロ (rendaku'd
    # kokoro, e.g. 遊び心 asobigokoro) - UniDic's fixed dictionary already gets the common
    # ごころ compounds right as their own single fused noun token (遊び心/女心/親心 all tokenize
    # as ONE token with pron already correct), so this only ever fires for a compound NOT already
    # in that fixed list - confirmed via testing that both 好奇心 (genuine on'yomi) and
    # ファイティング心 (a modern loanword+心 coinage) hit this same 接尾辞 path with the same シン
    # reading, meaning UniDic can't tell them apart on its own. The one reliable signal available
    # is the PRECEDING word's own script: a katakana loanword immediately before 心 is never part
    # of a genuine Sino-Japanese compound (those are always all-Kanji), so it's safe to prefer the
    # native ゴコロ reading specifically in that case.
    if (
        word.surface == "心"
        and getattr(word.feature, "pos1", None) == "接尾辞"
        and prevWord is not None
        and _isAllKatakana(prevWord.surface)
    ):
        return "ゴコロ"

    override = _READING_OVERRIDES.get(word.surface)
    if override:
        return override
    feature = word.feature
    return getattr(feature, "pron", None) or getattr(feature, "kana", None)


def _tokenKatakanaReading(word, prevWord=None) -> str:
    reading = _wordReading(word, prevWord)
    if not reading:
        return word.surface
    return _expandLongVowelMark(reading)


def _katakanaToHiragana(text: str) -> str:
    return text.translate(_KATAKANA_TO_HIRAGANA)


def _rawTokenReading(word, prevWord=None) -> str:
    """
    A token's reading with any real chouonpu ("ー") left intact, unlike _tokenKatakanaReading()
    which always expands it into a specific kana letter for conventional kana-spelling display.
    Only used by the macron-romaji path in kanjiLineToReading() - see _collapseChouonpuToMacron().
    """
    reading = _wordReading(word, prevWord)
    return reading if reading else word.surface


_MACRON_BY_DOUBLED_VOWEL = {"aa": "ā", "ii": "ī", "uu": "ū", "ee": "ē", "oo": "ō"}
_DOUBLED_VOWEL_PATTERN = re.compile("|".join(_MACRON_BY_DOUBLED_VOWEL))


def _containsJapaneseScript(text: str) -> bool:
    return any(
        0x3040 <= ord(ch) <= 0x30FF  # hiragana + katakana
        or 0x4E00 <= ord(ch) <= 0x9FFF or 0x3400 <= ord(ch) <= 0x4DBF  # kanji
        for ch in text
    )


def _collapseChouonpuToMacron(romaji: str) -> str:
    """
    Collapse a same-letter doubled vowel in Hepburn romaji (aa/ii/uu/ee/oo) into a single macron
    vowel (modified-Hepburn style, e.g. tōkyō) - both "ou" and "oo" spellings of the same real
    long-o sound become one unambiguous symbol instead of forcing a reader to remember they mean
    the same thing.

    Safe because a same-letter doublet here can ONLY come from a real chouonpu ("ー") in the
    token's phonetic `pron` reading - confirmed via testing: pykakasi doubles the preceding vowel
    letter whenever it sees an actual "ー" (トー -> "too", センセー -> "sensee"), but a token that
    merely LOOKS similar without one (思う "omou", 追う "ou" - two distinct morae, no chouonpu in
    `pron` at all) always romanizes as two DIFFERENT adjacent vowel letters, never a doublet - so
    this never fires on those. See core/test_japanese_utils.py.

    Caller must only pass this a chunk of romaji that actually came FROM Japanese script (see
    kanjiLineToReading, which checks kakasi's own per-chunk `orig` field) - kakasi passes an
    embedded English phrase straight through untouched, and this function has no way to tell an
    English "oo"/"aa" (e.g. "Good", "Cool") apart from a real chouonpu-derived one if it's ever
    run across passthrough English text instead.
    """
    return _DOUBLED_VOWEL_PATTERN.sub(lambda m: _MACRON_BY_DOUBLED_VOWEL[m.group(0)], romaji)


def kanjiLineToReading(line: str, outputFormat: str = "romaji") -> str:
    """
    Convert one line of Japanese text (Kanji/kana) into a reading line.

    outputFormat: "hiragana" or "romaji".

    Lyric lines use '|' as a UI-only marker for per-member color splits
    (see LyricBox._createColorCodedText) - it carries no linguistic meaning,
    so it's stripped before tokenizing and never appears in the output.
    """
    stripped = line.replace("|", "")
    if not stripped.strip():
        return ""

    tagger = _getTagger()
    parts = []
    prevReading = ""
    prevWord = None
    for word in tagger(stripped):
        posClass = getattr(word.feature, "pos1", "")
        # The hiragana path wants the conventional kana spelling (chouonpu expanded into a real
        # kana letter, e.g. とうきょう); the romaji path wants the real chouonpu left intact so
        # _collapseChouonpuToMacron() can tell a genuine long vowel apart from a token that just
        # looks similar (see that function's docstring) - kakasi handles a raw "ー" fine either way.
        raw = _rawTokenReading(word, prevWord)
        reading = raw if outputFormat == "romaji" else _expandLongVowelMark(raw)
        # A word boundary can fall right between a sokuon ("っ", gemination marker) and the
        # consonant it doubles (e.g. fugashi splits あって into あっ + て) - a space there breaks
        # kakasi's gemination since it's no longer adjacent to the next consonant, mis-romanizing
        # it as a literal standalone "tsu" (あって -> "atsu te" instead of "atte"). Confirmed via
        # testing: kakasi only geminates when the sokuon is immediately followed by the consonant
        # kana, with nothing in between. So never insert a space right after a trailing sokuon.
        if parts and posClass not in _NO_LEADING_SPACE_POS and not prevReading.endswith("ッ"):
            parts.append(" ")
        parts.append(reading)
        prevReading = reading
        prevWord = word

    joined = _katakanaToHiragana("".join(parts))

    if outputFormat == "hiragana":
        return joined

    kks = _getKakasi()
    parts = []
    for item in kks.convert(joined):
        hepburn = item["hepburn"]
        # kakasi passes an embedded English phrase straight through as its own chunk (`orig`
        # stays the literal English text, unconverted) - only collapse a doubled vowel when this
        # chunk actually came from Japanese script, never on a passthrough chunk (see
        # _collapseChouonpuToMacron's own docstring: real bug found where "Good"/"Cool" mixed
        # into a lyric line came out "Gōd"/"Cōl").
        if _containsJapaneseScript(item["orig"]):
            hepburn = _collapseChouonpuToMacron(hepburn)
        parts.append(hepburn)
    return "".join(parts)


def kanjiTextToReading(text: str, outputFormat: str = "romaji") -> str:
    """Convert multi-line Japanese lyric text into a reading line-by-line."""
    return "\n".join(
        kanjiLineToReading(line, outputFormat) for line in (text or "").split("\n")
    )
