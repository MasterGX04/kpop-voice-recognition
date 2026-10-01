"""
Grammar breakdown: tokenize a highlighted Korean lyric line into a stem + particle + ending
chain, each piece independently glossed (e.g. 전해져 -> 전하[to convey] + 어[connective] +
지[passive/inchoative auxiliary] + 어[connective]). Korean sibling of core/grammar_breakdown.py,
using kiwipiepy instead of fugashi. Design reference: .claude/KOREAN_GRAMMAR_BREAKDOWN_PLAN.md.
"""

from core.korean_utils import getKiwi
from core.korean_dictionary import lookupKoreanMeaning
from core.korean_hanja import lookupHanja

# Sejong-tagset punctuation (SF/SP/SS/SE/SO/SW) carries no grammar of its own and is skipped
# entirely - same treatment as 補助記号/記号 in the Japanese module. SL (foreign-script text -
# in real lyrics, almost always embedded English fragments, e.g. "Yeah"/"I"/"like"/"that") is
# skipped too, per direct user feedback: a Korean grammar breakdown has nothing to say about an
# English word, and showing it with gloss=None was pure clutter. SH/SN (Hanja characters or
# numerals actually written in the lyric, as opposed to Hangul) are NOT skipped - real Korean
# content, just spelled with a different script, so they still go through the ordinary "content"
# path below.
#
# Real, currently-live bug found via a live-DB audit: this kiwipiepy build never actually emits
# the plain "SS" tag the Sejong tagset docs describe for quotation marks/brackets - it splits it
# into "SSO" (opening '"([{) and "SSC" (closing '")]}), confirmed directly via tokenize(). Neither
# was in this set, so every quote mark and bracket in a lyric fell through to the ordinary
# "content" role and got stored as if it were a real vocabulary word (e.g. a bare "(" or "'" row
# in vocab_ko) - not stale data, the current pipeline was still producing these.
_SKIP_TAGS = {"SF", "SP", "SS", "SSO", "SSC", "SE", "SO", "SW", "SL"}

# A function-word lemma's real gloss depends on its tag, not just its lemma: the exact same
# lemma string can mean two unrelated things under two different tags - confirmed via the real
# frequency scan in .claude/KOREAN_GRAMMAR_BREAKDOWN_PLAN.md (어 as EC "connective" vs. EF
# "informal sentence-final ending"; ᆫ/은/는 as ETM "adnominal ending" vs. JX "topic marker").
# Keying per-tag (rather than one flat dict shared across every particle/ending, as the plan's
# prose literally describes) avoids that collision - same "verify against real data before
# trusting the obvious shape" discipline as every other tier in this project.
_FUNCTION_GLOSSES = {
    # JK* - case particles
    "JKS": {"이": "subject marker (after a consonant)", "가": "subject marker (after a vowel)"},
    "JKO": {
        "을": "object marker (after a consonant)",
        "를": "object marker (after a vowel)",
        "ᆯ": "object marker (contracted 를)",
    },
    "JKG": {"의": "possessive / genitive marker"},
    "JKB": {
        "에": "at / in / to (location or time)",
        "에서": "at / from (location)",
        "로": "with / by means of / toward (direction or means)",
        "으로": "with / by means of / toward (direction or means, after a consonant)",
        "에게": "to / at (a person)",
        "처럼": "like / similar to",
    },
    "JKC": {"로": "as / into (complement marker)"},
    "JKV": {"아": "vocative (addressing someone directly)", "야": "vocative (addressing someone directly)"},
    "JKQ": {"고": "quotative marker"},
    # JX/JC - topic/auxiliary/connecting particles
    "JX": {
        "은": "topic marker (after a consonant) / contrast",
        "는": "topic marker (after a vowel) / contrast",
        "ᆫ": "topic marker (contracted 은/는)",
        "도": "also / too",
        "만": "only / just",
        "까지": "even / up to",
        "부터": "from / starting at",
        "요": "polite / softening particle",
    },
    "JC": {"와": "and / with", "과": "and / with"},
    # E* - endings
    "EC": {
        "어": "connective (\"and/so/then\", links to what follows)",
        "게": "adverbializer (\"so as to\")",
        "고": "connective (\"and\", sequential/listing)",
        "지": "connective, often before negation (지 않다 = \"not\")",
        "어도": "even if / although",
        "어야": "must / have to (\"only if\")",
        "며": "while / and (simultaneous action)",
        "면": "if / when (conditional)",
        "든": "whether... or / regardless (contracted 든지)",
        "지만": "but / although",
        "면서": "while (simultaneous action)",
        "니": "since / because",
        "어서": "so / because (sequential cause)",
        "는데": "but / and (background or contrast)",
        "네": "realization connective (\"oh, ...\")",
        "듯": "as if / as though (evidentially, \"seems like\")",
    },
    "EF": {
        "어": "informal sentence-final ending",
        "다": "plain sentence-final ending",
        "지": "informal confirming ending (\"right?\")",
        "야": "informal declarative/vocative ending",
        "ᆫ다": "plain present-tense sentence-final ending (contracted ㄴ다)",
        "네": "exclamatory ending (\"oh, I see!\")",
        "ᆯ까": "deliberative question ending (\"shall I/we...?\")",
        "죠": "polite confirming ending (contracted 지요)",
        "잖아": "assertion reminder (\"you know, right?\")",
        "ᆯ게": "informal promissory/intentional ending (\"I will...\", contracted 을게)",
    },
    "ETN": {"ᆷ": "nominalizer (turns the clause into a noun, contracted 음)", "기": "nominalizer (turns the clause into a noun)"},
    "ETM": {
        "ᆫ": "adnominal ending, past/completed (contracted 은/-ㄴ)",
        "ᆯ": "adnominal ending, future/prospective (contracted 을/-ㄹ)",
        "는": "adnominal ending, present/ongoing",
        "은": "adnominal ending, past/completed",
        "을": "adnominal ending, future/prospective",
        "던": "adnominal ending, past habitual/retrospective (\"used to ~\")",
    },
    "EP": {"었": "past tense", "겠": "future / intention / conjecture", "시": "honorific"},
    # Auxiliary verb / copula tags
    "VX": {
        "지다": "passive/inchoative auxiliary (\"comes to be\")",
        "하다": "auxiliary \"to do\" (light verb after a descriptive stem)",
        "있다": "auxiliary marking an ongoing/resultant state",
        "보다": "auxiliary \"try doing\" (after a connective 어/아)",
        "주다": "auxiliary \"to do for someone\" (after 어/아)",
        "않다": "negative auxiliary \"not\" (after 지)",
        "말다": "prohibitive auxiliary \"don't\" (after 지)",
        "싶다": "auxiliary \"want to\" (after 고)",
        "가다": "auxiliary \"-ing onward / starting to\" (after 어/아)",
        "놓다": "resultative auxiliary \"do and leave it\" (after 어/아)",
        "버리다": "completive auxiliary \"end up doing / did it all\" (after 어/아)",
    },
    "VCP": {"이다": "to be (copula)"},
    "VCN": {"아니다": "to not be (negative copula)"},
    # XSA/XSV - derivational suffixes that turn a (usually Sino-Korean) noun root into an
    # adjective/verb (평온+하다 -> 평온하다 "to be tranquil", 공부+하다 -> 공부하다 "to study").
    # Grammatical glue, not an independent word - found as a real, high-frequency gap via testing:
    # without this, a bare XSA/XSV token like the 하 in 평온했던 fell into the ordinary content
    # path and got looked up (and Hanja-checked) as if it were its own word, producing an unrelated
    # gloss ("under a situation") plus noise Hanja candidates for a single grammatical suffix.
    "XSA": {"하": "-하다: adjective-forming suffix (e.g. 평온+하다, \"to be tranquil\")"},
    "XSV": {"하": "-하다: verb-forming suffix (e.g. 공부+하다, \"to study\")"},
    # XSM - adverbializer suffix (히/이), the -하다 suffix's sibling for forming adverbs instead of
    # adjectives/verbs (따뜻+이/히 -> 따뜻이/따뜻히 "warmly"). Same real gap as XSA/XSV: without
    # this, a bare XSM token fell into the ordinary content path and got looked up as if it were
    # its own word - found via testing 따스히: the lone "히" was matching an unrelated onomatopoeia
    # noun entry ("Conveys a nervous laughter or smile") instead of being recognized as grammatical
    # glue.
    "XSM": {"히": "-히/-이: adverbializer suffix (forms an adverb from a noun/root)",
            "이": "-히/-이: adverbializer suffix (forms an adverb from a noun/root)"},
}

_FUNCTION_TAG_PREFIXES = ("J", "E")
_FUNCTION_TAGS = {"VX", "VCP", "VCN", "XSA", "XSV", "XSM"}

# XR (bound root) + one of these suffix tags, when the suffix's own lemma is exactly "하", means
# the citation form actually being used is the composed adjective/verb (나른+하다 -> "languid"),
# not the bare root - see _resolveGloss(). Keyed by the Wiktionary "pos" the composed word should
# be looked up under.
_HADA_COMPOSABLE_SUFFIX_TAGS = {"XSA": "VA", "XSV": "VV"}

# A MAG-tagged adverb ending in one of these is very often a Sino-Korean noun root plus that
# adverbializer, already fused into a single Kiwi token (조심히, 정확히, ...) rather than split
# into separate XR+XSM tokens the way rarer roots like 따스/조용 are - see _contentEntry()'s
# MAG root-stripping fallback.
_ADVERB_SUFFIX_CHARS = ("히", "이")

# Hangul syllable-block composition constants (Unicode's algorithmic Hangul Syllables block,
# U+AC00-U+D7A3 = (initial * 21 + medial) * 28 + final, 19 initials x 21 medials x 28 finals -
# see _addBieupBatchim()'s docstring for why this is needed at all.
_HANGUL_BASE = 0xAC00
_HANGUL_LAST = 0xD7A3
_V_COUNT, _T_COUNT = 21, 28
_BIEUP_TAIL_INDEX = 17  # ㅂ's index in the 28-entry final-consonant table (0 = no final)

# Dependent/bound nouns (의존명사) - a small, well-known closed grammatical class in Korean that
# can never stand alone (수 없다/있다 "can't/can", 것/거 "thing" as a nominalizer, 줄 알다/모르다
# "to know/not know how to", etc.). Kiwi tags these NNB, a "content" noun tag, but they function
# as grammatical scaffolding, not real independent vocabulary - and critically, Wiktionary's own
# "pos" field has no separate category for "dependent noun" vs. an ordinary noun, so the
# tag-matching in core/korean_hanja.py: lookupHanja() (which fixed the 이/"this" case) CANNOT tell
# these apart from the unrelated Hanja homographs that happen to share the same spelling under the
# exact same "noun" pos. Found via direct user testing: 수 ("way, means" in "가눌 수 없는") was
# showing 11 completely unrelated real Hanja nouns (手/數/水/繡/首/受/髓/隋/守) that only ever
# surface as ordinary standalone nouns (NNG), never as this dependent-noun construction. Hard-coded
# because this is a real, small, closed class (the standard list of Korean 의존명사), the same
# "small closed class -> hand-curate" reasoning as the particle/ending dicts above - NOT a general
# "NNB never gets Hanja" rule, which would also wrongly suppress real Sino-Korean counter nouns
# (대, 장, 명, ...) that are also tagged NNB but genuinely are Hanja-origin words.
_NATIVE_DEPENDENT_NOUNS = {
    "수", "것", "거", "바", "줄", "데", "지", "만큼", "뿐", "대로",
    "채", "체", "양", "척", "나름", "김", "겸", "통", "터", "참", "나위", "리",
}


def _roleForTag(tag: str) -> str:
    if tag in _SKIP_TAGS:
        return "skip"
    if tag.startswith(_FUNCTION_TAG_PREFIXES) or tag in _FUNCTION_TAGS:
        return "function"
    return "content"


def _stripAdverbSuffix(lemma):
    if len(lemma) > 1 and lemma[-1] in _ADVERB_SUFFIX_CHARS:
        return lemma[:-1]
    return None


def _addBieupBatchim(syllable):
    """
    Returns `syllable` (a single plain Hangul syllable block with no existing final consonant)
    with a ㅂ batchim added - e.g. 러 -> 럽, 미 -> 밉, 까 -> 깝 - or None if `syllable` isn't a
    single batchim-less Hangul syllable. Used only by the ㅂ-irregular reversal in _resolveGloss()
    (see that call site), where the target syllable is always batchim-less by construction: the
    ㅂ moved out of it to help form the following "워" syllable in the conjugated form, so putting
    it back is always adding a batchim, never overwriting one.
    """
    if len(syllable) != 1:
        return None
    code = ord(syllable)
    if not (_HANGUL_BASE <= code <= _HANGUL_LAST):
        return None
    if (code - _HANGUL_BASE) % _T_COUNT != 0:
        return None
    return chr(code + _BIEUP_TAIL_INDEX)


def _bieupIrregularAdjective(lemma):
    """
    Reverses the ㅂ-irregular "-어하다" derivation (부럽다 "enviable" + 어하다 -> 부러워하다 "to
    envy", 밉다 + 어하다 -> 미워하다, 안타깝다 + 어하다 -> 안타까워하다) back to its base adjective's
    citation form, or None if `lemma` doesn't have the right shape to try. This is a categorical,
    exceptionless Korean sound-change (ㅂ-irregular stem + 어 always surfaces as ...워), not a
    per-word guess: the syllable immediately before "워" always corresponds to the base adjective's
    last syllable with a ㅂ batchim added back and the "워" syllable dropped entirely (부러+워 ->
    부럽, not a character-by-character substitution). Only ever used as a fallback when a direct
    lookup of the -어하다 verb itself already failed (see _resolveGloss) - the caller still verifies
    the reconstructed candidate against the real dictionary before accepting it, so a shape that
    happens to match but isn't a real word (or isn't this derivation at all) is never fabricated
    into a gloss, just silently rejected downstream.
    """
    if not lemma.endswith("워하다") or len(lemma) < 4:
        return None
    targetSyllable = lemma[-4]
    newSyllable = _addBieupBatchim(targetSyllable)
    if newSyllable is None:
        return None
    return lemma[:-4] + newSyllable + "다"


def _resolveGloss(token, nextToken):
    """
    Returns (resolvedLemma, result) - `result` is the usual {"status", "pos", "gloss"} dict, and
    `resolvedLemma` is the citation form that dict actually came from: token.lemma itself for a
    plain direct hit, or the composed/reconstructed form (e.g. "나른하다", "부럽다") whenever one of
    the fallbacks below is what actually found the entry. Callers store `resolvedLemma`, never
    token.lemma blindly - see _contentEntry()'s own comment for why that distinction matters.
    """
    # XR (bound root) - Kiwi's own tag for a root that can't stand alone (confirmed via testing:
    # 조용/나른 are tagged XR, while an ordinary noun that also happens to combine with 하다, like
    # 평온, is tagged NNG instead - Kiwi already draws this line for us). When one is immediately
    # followed by the -하다 suffix (XSA/XSV lemma 하), the citation form actually being used is the
    # composed adjective/verb, not the bare root - real gap found via testing: 나른 alone isn't in
    # the Wiktionary index at all (only 나른하다, "languid, listless", is), and 조용 alone IS
    # present but only as a self-referential stub ("Root of 조용하다 ... Rarely used alone.") that
    # isn't a usable gloss either - the composed form is the only place a real gloss exists for
    # either word.
    if token.tag == "XR" and nextToken is not None and nextToken.lemma == "하" \
            and nextToken.tag in _HADA_COMPOSABLE_SUFFIX_TAGS:
        composedLemma = token.lemma + "하다"
        composed = lookupKoreanMeaning(composedLemma, _HADA_COMPOSABLE_SUFFIX_TAGS[nextToken.tag])
        if composed["status"] == "found":
            return composedLemma, composed

    # A root followed by the adverbializer XSM (히/이, e.g. 따스+히) instead of -하다 directly -
    # best-effort: still try the root's own -하다 citation form, since Wiktionary documents that
    # far more often than a standalone -히/-이 adverb entry (confirmed: 따뜻하다 exists, 따뜻이
    # doesn't). Falls through to the plain root lookup below when neither -하다 form exists either
    # (a genuine coverage gap, e.g. 따스 has no recoverable entry anywhere in the index).
    if token.tag == "XR" and nextToken is not None and nextToken.tag == "XSM":
        for composedTag in ("VA", "VV"):
            composedLemma = token.lemma + "하다"
            composed = lookupKoreanMeaning(composedLemma, composedTag)
            if composed["status"] == "found":
                return composedLemma, composed

    result = lookupKoreanMeaning(token.lemma, token.tag)

    # A MAG-tagged adverb ending in -히/-이 is very often a Sino-Korean noun root plus that
    # adverbializer, already fused into ONE Kiwi token (조심히, 정확히, ...) rather than split into
    # separate XR+XSM tokens the way rarer roots are - only used as a fallback when the adverb's
    # own direct entry came up empty (정확히/간단히/가만히/조용히/천천히 already have perfectly
    # good direct "adv" glosses of their own; only reach for the noun root's gloss - "caution,
    # care" for 조심 - when there's nothing else, e.g. 조심히 isn't in the index at all itself).
    if result["status"] != "found" and token.tag == "MAG":
        root = _stripAdverbSuffix(token.lemma)
        if root:
            rootResult = lookupKoreanMeaning(root, "NNG")
            if rootResult["status"] == "found":
                return root, rootResult

    # -어하다 psych-adjective-to-verb derivation (좋다 -> 좋아하다 "to like", 부럽다 -> 부러워하다
    # "to envy") - a real, productive coverage gap found via a live-DB audit: Kiwi already gives
    # these their own correct citation-form lemma (e.g. "부러워하다"), but the derived verb form
    # itself is often absent from the Wiktionary index, which only documents the base adjective.
    # Only the ㅂ-irregular spelling (...워하다, see _bieupIrregularAdjective) is reconstructed here
    # - the regular case (아/어하다 stripped straight to 다) is deliberately NOT guessed, since
    # "다" appended to an arbitrary stripped stem is far more likely to collide with an unrelated
    # real word than the categorical ㅂ-irregular sound change is.
    if result["status"] != "found" and token.tag == "VV":
        baseAdjective = _bieupIrregularAdjective(token.lemma)
        if baseAdjective:
            baseResult = lookupKoreanMeaning(baseAdjective, "VA")
            if baseResult["status"] == "found":
                return baseAdjective, baseResult

    return token.lemma, result


def _resolveHanja(token):
    # A dependent noun in _NATIVE_DEPENDENT_NOUNS skips Hanja lookup entirely regardless of tag
    # matching - see that set's own comment for why POS-matching alone can't fix its case.
    if token.tag == "NNB" and token.lemma in _NATIVE_DEPENDENT_NOUNS:
        return []

    hanja = lookupHanja(token.lemma, token.tag)

    # Real bug found via direct user testing: 조심 ("caution, care") has a real Hanja entry (操心)
    # under its own noun headword, but Wiktionary essentially never records Hanja directly on an
    # "adv"-pos entry (confirmed across every -히 word checked: only 특별히 carries it directly,
    # out of 조심히/정확히/간단히/가만히/천천히) - so a fused MAG adverb like 조심히 was finding
    # nothing even though its own noun root plainly has a Hanja origin. Only used as a fallback
    # when the adverb's own direct lookup (already tried above) came up empty.
    if not hanja and token.tag == "MAG":
        root = _stripAdverbSuffix(token.lemma)
        if root:
            hanja = lookupHanja(root, "NNG")

    return hanja


def _contentEntry(token, nextToken=None) -> dict:
    resolvedLemma, result = _resolveGloss(token, nextToken)
    gloss = "; ".join(result["gloss"][:2]) if result["status"] == "found" else None

    # Merged in per direct user request: a whole-line grammar breakdown should double as a Hanja
    # lookup for every content word, not require a second highlight-one-word-at-a-time button -
    # see core/korean_hanja.py: lookupHanja(). Unconditional for every content word (mirrors
    # lookupKoreanMeaning()'s own "try every word, cheap no-op on a miss" posture); naturally []
    # for a native word with no Sino-Korean origin, so the caller can skip rendering anything for
    # it rather than printing a "no Hanja" line - that's the whole point of the merge. Passing
    # token.tag lets lookupHanja() narrow to the homograph(s) that actually match this token's
    # part of speech (see that function's own docstring for the 이/"this" false-positive this
    # fixed) instead of unioning Hanja across every unrelated word sharing the same spelling.
    hanja = _resolveHanja(token)

    # `lemma` is the RESOLVED citation form (token.lemma itself, or a composed/reconstructed form
    # whenever _resolveGloss's fallbacks are what actually found the entry - e.g. "나른하다" for
    # bound-root 나른, "부럽다" for the ㅂ-irregular verb 부러워하다) - this is what
    # core.korean_vocab.analyzeKoreanSelection stores/re-looks-up as the word's real dictionary
    # key, so it must be the form that actually matched, never the raw uninflected token.lemma a
    # fallback had to move past. `meaning` carries the already-resolved structured dict forward so
    # that caller doesn't re-run (and potentially fail to reproduce) the same fallback logic - see
    # analyzeKoreanSelection's own comment on why it stopped re-deriving from scratch.
    return {"surface": token.form, "lemma": resolvedLemma, "tag": token.tag, "role": "content",
            "gloss": gloss, "hanja": hanja, "meaning": result}


def _functionEntry(token) -> dict:
    gloss = _FUNCTION_GLOSSES.get(token.tag, {}).get(token.lemma)
    return {"surface": token.form, "lemma": token.lemma, "tag": token.tag, "role": "function",
            "gloss": gloss, "hanja": None, "meaning": None}


def _mergeGroup(text, group):
    """
    Collapse a list of (token, entry) pairs that share overlapping character spans into one
    displayed entry - see breakdownLine()'s own comment for why this grouping exists at all.
    `surface` is always reconstructed from the original text via the group's real character
    range, never trusted from any token's own `.form` - a second real bug found via testing:
    kiwipiepy's `.form` for a connective ending sometimes reports its citation-form spelling
    ("어") even when the literal written character at that exact position is a vowel-harmonized
    variant ("아", e.g. in 안아 - "hug"), a mismatch that only shows up by cross-checking against
    `start`/`len` offsets into the real string, the same category of surface-vs-citation-form
    gap `core/kanji_reference.py` already had to work around for a fugashi lemma once.
    """
    tokens = [t for t, _ in group]
    entries = [e for _, e in group]
    start = tokens[0].start
    end = max(t.start + t.len for t in tokens)
    surface = text[start:end]

    if len(entries) == 1:
        merged = dict(entries[0])
        merged["surface"] = surface
        return merged

    isContent = any(e["role"] == "content" for e in entries)
    glossParts = [e["gloss"] for e in entries if e["gloss"]]
    hanjaSource = next((e for e in entries if e["role"] == "content"), None)

    # Real bug found via a live-DB audit (223/863 vocab_ko rows, 65% of every "no meaning found"
    # row): `lemma`/`tag` here are the dictionary lookup key core.korean_vocab.analyzeKoreanSelection
    # stores a word under. Joining EVERY piece's lemma/tag unconditionally - including the attached
    # particle/ending's - produced keys like "다르다+ᆫ" or "우리+ᆯ" that can never match a real
    # headword, even though each underlying token already had its own correct citation-form lemma
    # (다르다/우리) before merging. A mixed content+function group (the overwhelmingly common case -
    # a word plus its fused ending) must use only the content token's own lemma/tag as the lookup
    # key; the joined form stays what `gloss` is built from, so a real breakdown/lyrics-editor
    # display is unaffected (it never reads `lemma`/`tag` directly - see gui/lyrics_editor.py's
    # breakDownKoreanGrammar).
    lemmaSource = [e for e in entries if e["role"] == "content"] if isContent else entries

    return {
        "surface": surface,
        "lemma": "+".join(e["lemma"] for e in lemmaSource),
        "tag": "+".join(e["tag"] for e in lemmaSource),
        "role": "content" if isContent else "function",
        "gloss": " + ".join(glossParts) if glossParts else None,
        "hanja": hanjaSource["hanja"] if hanjaSource else None,
        "meaning": hanjaSource["meaning"] if hanjaSource else None,
    }


def breakdownLine(text: str) -> list:
    """
    Break a highlighted Korean line into one entry per *written syllable group* (skipping
    punctuation and embedded foreign-script/English words), each
    `{"surface", "lemma", "tag", "role", "gloss", "hanja"}` - `role` is "content" or "function",
    `gloss` is None when nothing was found (never fabricated, same discipline as
    core/grammar_breakdown.py), and `hanja` is a list of `{"hanja", "gloss", "pos"}` candidates
    for a content word (see core/korean_hanja.py: lookupHanja() - empty when the word has no
    Sino-Korean origin) or None for a function word.

    Not literally one entry per kiwipiepy token: Korean's vowel-contracting conjugations (하다's
    하+아/어 -> 해; 되다's 되+었 -> 됐; 지다's 지+어 -> 져; ...) make Kiwi emit multiple morphemes
    that occupy the exact same written syllable, not sequential ones - confirmed directly via
    `start`/`len`: 말했듯's "하"(start=1,len=1) and "었"(start=1,len=1) both claim character
    position 1 (the single syllable "했"), and 전해져's stem token "전하"(start=0,len=2) already
    spans the position its own following connective "어"(start=1,len=1) claims too. Showing these
    as separate lines misrepresents what's actually written on the page - direct user feedback
    ("make sure not to separate 하/어 because it should just be 해"). A token whose `start` falls
    strictly before the running end of the currently-open group is fused into it (see
    _mergeGroup()); a token starting at or after that point begins a new group - this needs the
    full token list up front (not a one-token-at-a-time stream) for the same reason
    _resolveGloss()'s XR + -하다 lookahead does.
    """
    tokens = list(getKiwi().tokenize(text))

    perToken = []
    for i, token in enumerate(tokens):
        role = _roleForTag(token.tag)
        if role == "skip":
            perToken.append(None)
            continue
        if role == "content":
            nextToken = tokens[i + 1] if i + 1 < len(tokens) else None
            perToken.append(_contentEntry(token, nextToken))
        else:
            perToken.append(_functionEntry(token))

    groups = []
    groupEnd = None
    for token, entry in zip(tokens, perToken):
        if entry is None:
            continue
        tokenEnd = token.start + token.len
        if groups and token.start < groupEnd:
            groups[-1].append((token, entry))
        else:
            groups.append([(token, entry)])
        groupEnd = tokenEnd if groupEnd is None else max(groupEnd, tokenEnd)

    return [_mergeGroup(text, group) for group in groups]
