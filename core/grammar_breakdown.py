"""
Grammar breakdown: tokenize a highlighted Japanese lyric line into a stem + particle +
conjugation chain, each piece independently glossed (e.g. 食べたくない -> 食べ[to eat] +
たく[want to] + ない[not]). Design reference: .claude/GRAMMAR_BREAKDOWN_PLAN.md.
"""

from core.japanese_utils import getTagger, tokenReading
from core.kanji_reference import lookupJapaneseMeaning, _containsKanji

_KATAKANA_TO_HIRAGANA = str.maketrans({
    chr(code): chr(code - 0x60) for code in range(0x30A1, 0x30F7)
})


def _katakanaToHiragana(text: str) -> str:
    return text.translate(_KATAKANA_TO_HIRAGANA)


_FUNCTION_POS1 = {"助詞", "助動詞"}
# 空白 (whitespace, e.g. the full-width "　" some lyrics use as a mid-line separator) carries no
# grammar of its own, same reasoning as punctuation - skipped entirely, never shown as a token.
_SKIP_POS1 = {"補助記号", "記号", "空白"}

# UniDic marks a token as grammaticalized (used as a light verb / auxiliary stem rather than its
# literal dictionary sense) via pos2, not pos1 - confirmed via testing: しまう/いる/なる/etc.
# used as light verbs after a て-form (食べてしまう, 食べている) come back pos1=動詞 the same as
# their literal use, but with pos2=非自立可能 ("capable of non-independent use"); よう before だ
# (食べられるように) comes back pos1=形状詞 pos2=助動詞語幹. Routing on pos1 alone (the original
# plan) would misclassify all of these as ordinary content words and gloss their literal meaning
# (仕舞う "to put away", 様 "manner") instead of their grammatical function.
_GRAMMATICALIZED_POS2 = {"助動詞語幹", "非自立可能"}

# Seeded from the real corpus frequency scan (every 助詞/助動詞 lemma across every Japanese-
# detected line in saved_labels/*/*_lyrics.json) recorded in .claude/GRAMMAR_BREAKDOWN_PLAN.md.
# Missing lemma -> no gloss (caller shows the bare lemma), never fabricated.
_PARTICLE_GLOSSES = {
    "の": "of / possessive, or nominalizer",
    "て": "connective (\"and\"/\"then\", or links to a following auxiliary)",
    "に": "at / in / to / for",
    "は": "topic marker",
    "で": "at / by / with",
    "も": "also / too",
    "が": "subject marker",
    "を": "object marker",
    "よ": "emphatic (\"you know\")",
    "から": "from / because",
    "と": "with / and / quotative",
    "だけ": "only / just",
    "か": "question marker / or",
    "へ": "toward / to",
    "ね": "seeking agreement (\"right?\")",
    "まで": "until / as far as",
    "ば": "if (conditional)",
    "ほど": "to the extent that",
    "しか": "only (with negative)",
    "より": "than",
    "じゃん": "isn't it (colloquial)",
    "な": "prohibitive / exclamatory",
    "さ": "casual emphasis",
}

_AUXILIARY_GLOSSES = {
    "だ": "is / copula",
    "ない": "not / negative",
    "た": "past tense",
    "てる": "-ing / ongoing state (contracted ている)",
    "れる": "passive / potential / honorific",
    "たい": "want to",
    "られる": "passive / potential (of a ru-verb)",
    "ず": "without doing (negative, literary)",
    "てく": "going to ~ (contracted ていく)",
    "ます": "polite ending",
    "ちゃう": "completely / accidentally did (contracted てしまう)",
}

# Ordinary verb lemmas used as a light verb (grammaticalized, see _GRAMMATICALIZED_POS2 above)
# immediately after a て-form - the corpus scan's most frequent bigrams (でい, して, いて,
# いたい, なって, ...) all pair a て-form with one of these. Keyed on the auxiliary's own lemma,
# not the whole phrase, since it combines productively with any verb's て-form.
_TE_AUXILIARY_GLOSSES = {
    "居る": "-ing / in the state of having done (ている/てる)",
    "仕舞う": "completely / accidentally did (てしまう)",
    "成る": "come to be / end up (てなる)",
    "行く": "-ing onward / starting to (ていく)",
    "来る": "-ing up until now / starting to (てくる)",
    "置く": "in advance / for later (ておく)",
    "見る": "try doing (てみる)",
    "有る": "left in that state (てある)",
    "上げる": "do for someone (てあげる)",
    "貰う": "have someone do for you (てもらう)",
    "呉れる": "someone does for you (てくれる)",
}

# Grammaticalized 形状詞 stems (pos2=助動詞語幹, see _GRAMMATICALIZED_POS2) that attach directly
# after a verb/adjective's own conjunctive form - no preceding て involved, unlike
# _TE_AUXILIARY_GLOSSES, so checked unconditionally whenever grammaticalized is true. Confirmed
# via testing real corpus lines: そう ("~sou", evidential "looks like/seems like", distinct from
# the unrelated plain adverb そう "so" - only the auxiliary use carries this pos2 flag) and みたい
# ("~mitai", "looks like/resembles") both showed up as real gaps because their bare lemma has no
# Kanji spelling for JMdict to find - same reasoning that already applies to 様 alone (ように's
# first half, kept here too as a fallback for when it isn't consumed by that 2-token pattern).
_STEM_AUXILIARY_GLOSSES = {
    "そう": "looks like / seems like (~sou)",
    "みたい": "looks like / resembles (~mitai)",
    "様": "manner / way (~you)",
}

# 接尾辞 (suffix) tokens - a closed, productive set of grammatical suffixes UniDic splits off as
# their own token, distinct from same-spelled particles (e.g. さ here is pos1=接尾辞, not the
# sentence-final particle さ in _PARTICLE_GLOSSES above) - confirmed via testing real corpus
# lines: さみしさ (寂しい's stem + さ, "-ness") and 一つ (一 + つ, counter suffix).
_SUFFIX_GLOSSES = {
    "さ": "-ness (turns an adjective stem into an abstract noun)",
    "み": "-ness (turns an adjective stem into an abstract noun, alt.)",
    "つ": "counter suffix (things, native Japanese counting)",
}

# A small, explicitly incomplete stopgap for genuinely kana-only words that JMdict's Kanji-keyed
# index (see data/jmdict/build_index.py) can never find no matter what lemma is tried - either a
# closed demonstrative class (こそあど series: こんな/そんな/あんな/どんな, plus あの when
# UniDic tags it 感動詞 instead of 連体詞) or a katakana loanword with no Kanji spelling at all.
# Unlike the particle/auxiliary dicts, katakana loanwords are NOT a closed class - this list only
# covers what real testing against saved_labels/*/*_lyrics.json actually surfaced as a gap, and
# is expected to need occasional additions as new songs are added (same "expand opportunistically"
# posture the particle dict already uses, see .claude/GRAMMAR_BREAKDOWN_PLAN.md). A proper fix
# would need data/jmdict/build_index.py rebuilt to also index kana-only JMdict entries by reading,
# which needs the raw JMdict_e source file (not currently present in data/jmdict/) - out of scope
# here.
_CLOSED_CLASS_FALLBACK_GLOSSES = {
    "こんな": "this kind of / like this",
    "そんな": "that kind of / like that",
    "あんな": "that kind of (over there) / like that",
    "どんな": "what kind of / any kind of",
    "あの": "that (over there)",
    "ダイヤ": "diamond",
    "ゴール": "goal",
    "リンク": "link",
    "ハート": "heart",
    "リボン": "ribbon",
    "アイス": "ice / ice cream",
    "クリーム": "cream",
    "そう": "so / that way / like that",
    "もう": "already / now / anymore",
    "どう": "how / in what way",
    "せめて": "at least",
    "さようなら": "goodbye",
    "ふわふわ": "fluffy / floaty",
    "くるくる": "round and round / spinning",
    "きらきら": "sparkling / glittering",
    "ふざける": "to joke around / to fool around",
}

# Adjacent function-word lemma pairs whose combined meaning isn't recoverable from gluing their
# individual single-token glosses together (verified by tokenizing each worked example with
# fugashi and checking it against the real corpus scan - see .claude/GRAMMAR_BREAKDOWN_PLAN.md).
# Checked as a 2-token window over adjacent function-ish tokens before per-token classification.
_COMPOUND_GLOSSES = {
    ("で", "も"): "but / however (でも)",
    ("に", "は"): "as for (topic-marked location/time, には)",
    ("て", "も"): "even if / even though (ても)",
    ("と", "か"): "or something like (とか)",
    ("だ", "ば"): "if (conditional, ならば)",
    ("だ", "無い"): "is not (contracted では/じゃない)",
    ("様", "だ"): "so that / in order to (ように)",
    ("の", "だ"): "you see... / explanatory (んだ/んです)",
    ("の", "に"): "even though / despite (のに)",
}


def _lemmaOf(word) -> str:
    lemma = word.feature.lemma
    # unidic-lite bakes a "-<POS category>" disambiguator into some lemmas (see
    # core/kanji_reference.py: _analyzeToken) - strip it so lookups match the real dictionary form.
    if "-" in lemma:
        lemma = lemma.split("-", 1)[0]
    return lemma


def _pos2Of(word):
    return getattr(word.feature, "pos2", None)


def _isFunctionish(word) -> bool:
    return word.feature.pos1 in _FUNCTION_POS1 or _pos2Of(word) in _GRAMMATICALIZED_POS2


def _isTeParticle(word) -> bool:
    # UniDic's 非自立可能 pos2 flag (see _GRAMMATICALIZED_POS2) marks a verb LEMMA as *capable*
    # of light-verb use (e.g. 行く/成る/居る/仕舞う) - it is NOT context-dependent, so it stays set
    # even when the verb is used completely literally (confirmed via testing: standalone 行く in
    # "行くしかない", "to go", still comes back pos2=非自立可能). The real signal for "this
    # occurrence is a grammaticalized ~te auxiliary, not the literal verb" is adjacency: it
    # directly follows a て/で connective-particle token (食べ-て-しまう, 泳い-で-いる).
    f = word.feature
    return f.pos1 == "助詞" and _pos2Of(word) == "接続助詞" and _lemmaOf(word) in {"て", "で"}


def _contentEntry(word, lemma) -> dict:
    surface = word.surface
    lemmaReading = _katakanaToHiragana(getattr(word.feature, "lForm", None) or tokenReading(word))

    # unidic-lite sometimes stores a broken katakana-only lemma for certain proper nouns (e.g.
    # 恵's lemma comes back as メグミ, not 恵) - same finding already documented in
    # core/kanji_reference.py: _analyzeToken. A lemma that lost the Kanji the surface actually
    # has can't be a real citation form, so fall back to classifying the surface directly.
    if _containsKanji(surface) and not _containsKanji(lemma):
        lemma, lemmaReading = surface, _katakanaToHiragana(tokenReading(word))

    result = lookupJapaneseMeaning(lemma, lemmaReading)
    if result["status"] == "found":
        gloss = "; ".join(result["gloss"][:2])
    else:
        gloss = _CLOSED_CLASS_FALLBACK_GLOSSES.get(lemma) or _CLOSED_CLASS_FALLBACK_GLOSSES.get(surface)
    return {"surface": surface, "lemma": lemma, "pos1": word.feature.pos1, "role": "content", "gloss": gloss}


def _classifyToken(word, precededByTe: bool) -> dict:
    pos1 = word.feature.pos1
    lemma = _lemmaOf(word)
    grammaticalized = _pos2Of(word) in _GRAMMATICALIZED_POS2

    if grammaticalized and precededByTe and lemma in _TE_AUXILIARY_GLOSSES:
        return {"surface": word.surface, "lemma": lemma, "pos1": pos1, "role": "function",
                "gloss": _TE_AUXILIARY_GLOSSES[lemma]}

    if grammaticalized and lemma in _STEM_AUXILIARY_GLOSSES:
        return {"surface": word.surface, "lemma": lemma, "pos1": pos1, "role": "function",
                "gloss": _STEM_AUXILIARY_GLOSSES[lemma]}

    if pos1 == "接尾辞" and lemma in _SUFFIX_GLOSSES:
        return {"surface": word.surface, "lemma": lemma, "pos1": pos1, "role": "function",
                "gloss": _SUFFIX_GLOSSES[lemma]}

    if pos1 in _FUNCTION_POS1:
        gloss = _AUXILIARY_GLOSSES.get(lemma) or _PARTICLE_GLOSSES.get(lemma)
        return {"surface": word.surface, "lemma": lemma, "pos1": pos1, "role": "function", "gloss": gloss}

    # Everything else - verbs/adjectives/nouns/adjectival-nouns, but also adverbs, conjunctions,
    # prefixes, etc. (副詞/接続詞/接頭辞/...). Always attempt the JMdict lookup via the token's
    # LEMMA, not just when the raw surface has Kanji: confirmed via testing that many common
    # kana-written adverbs (ちょっと, とても) still carry a real Kanji citation-form lemma in
    # UniDic (一寸, 迚も) that JMdict *does* index by - the earlier Phase 1 assumption that
    # kana-surface content words are unglossable was wrong for these, not just a hypothetical
    # edge case (see .claude/GRAMMAR_BREAKDOWN_PLAN.md). lookupJapaneseMeaning() already returns
    # "not_found" gracefully for a lemma with no real Kanji entry, so trying costs nothing and a
    # genuine gap still falls back to a bare lemma, never a fabricated gloss.
    return _contentEntry(word, lemma)


def breakdownLine(text: str) -> list:
    """
    Break a highlighted Japanese line into one entry per token (skipping punctuation), each
    `{"surface", "lemma", "pos1", "role", "gloss"}` - `role` is "content", "function", or
    "pattern" (a matched multi-token construction, e.g. ように/んだ), and `gloss` is None when
    nothing was found (never fabricated).
    """
    tagger = getTagger()
    # UniDic tags an unrecognized Latin-alphabet run (e.g. an embedded English phrase like
    # "You will find out") as an ordinary common noun (pos1=名詞) but with lemma=None - confirmed
    # via testing. Every other function here assumes a token has a real lemma string (e.g.
    # _lemmaOf does "-" in lemma), so this crashed with "argument of type 'NoneType' is not
    # iterable" the moment an English word was part of the highlighted text. English words have
    # no Japanese grammar to break down, so they're dropped from the output entirely.
    words = [
        w for w in tagger(text)
        if w.feature.pos1 not in _SKIP_POS1 and w.feature.lemma is not None
    ]

    entries = []
    i = 0
    while i < len(words):
        if i + 1 < len(words):
            w1, w2 = words[i], words[i + 1]
            key = (_lemmaOf(w1), _lemmaOf(w2))
            gloss = _COMPOUND_GLOSSES.get(key)
            if gloss is not None and _isFunctionish(w1) and _isFunctionish(w2):
                entries.append({
                    "surface": w1.surface + w2.surface,
                    "lemma": f"{key[0]}+{key[1]}",
                    "pos1": f"{w1.feature.pos1}+{w2.feature.pos1}",
                    "role": "pattern",
                    "gloss": gloss,
                })
                i += 2
                continue

        precededByTe = i > 0 and _isTeParticle(words[i - 1])
        entries.append(_classifyToken(words[i], precededByTe))
        i += 1

    return entries


# Maps a chunk's attached particle/pattern lemma to the human-readable syntactic role it plays
# in the sentence (Object/Topic/Subject/...) - purely a display label derived from what the
# particle already grammatically does, not a guess: e.g. を always marks a direct object, so a
# chunk it attaches to is always labeled "Object". Deliberately NOT exhaustive - only particles
# with one clear, well-known role are labeled; sentence-final/emphasis particles (ね/よ/な/...)
# add no useful role label, so a chunk ending in only those is left unlabeled rather than forcing
# an awkward name onto it.
_PARTICLE_ROLE_LABELS = {
    "を": "Object",
    "は": "Topic",
    "が": "Subject",
    "に": "Target / Direction",
    "へ": "Direction",
    "で": "Location / Means",
    "と": "Quotative / With",
    "から": "Source / Reason",
    "まで": "Extent / Until",
    "より": "Comparative Baseline",
    "だ": "Predicate",
}

# Compound "pattern" entries (see _COMPOUND_GLOSSES) get their own role label, keyed the same way
# their lemma is built (f"{key[0]}+{key[1]}") - a role a naive particle-by-particle lookup can't
# assign, since the role belongs to the whole 2-token construction, not either half alone.
_PATTERN_ROLE_LABELS = {
    "で+も": "Contrast",
    "に+は": "Topic",
    "て+も": "Concessive",
    "と+か": "Alternative",
    "だ+ば": "Conditional",
    "だ+無い": "Negative Predicate",
    "様+だ": "Purpose",
    "の+だ": "Explanatory",
    "の+に": "Concessive",
}

# A chunk headed by a verb/adjective/na-adjective with no case-marking particle in its tail
# (だから every predicate-ending chunk looks like this - e.g. 知らないな's tail is just ない+な,
# neither a case particle) is the sentence's predicate almost by definition, so that's the
# fallback label when nothing in the tail matches the maps above.
_PREDICATE_HEAD_POS1 = {"動詞", "形容詞", "形状詞"}


def _labelForChunk(chunk) -> str:
    for entry in reversed(chunk["tail"]):
        label = (
            _PATTERN_ROLE_LABELS.get(entry["lemma"]) if entry["role"] == "pattern"
            else _PARTICLE_ROLE_LABELS.get(entry["lemma"])
        )
        if label:
            return label
    if chunk["head"]["pos1"] in _PREDICATE_HEAD_POS1:
        return "Predicate"
    return None


def groupIntoChunks(entries: list) -> list:
    """
    Group breakdownLine()'s flat per-token entries into bunsetsu-like chunks: a leading content
    word plus whatever trailing function/pattern words attach to it (particles, auxiliaries,
    ように/んだ-style patterns) - e.g. 待つ+ほど become one "待つほど" chunk instead of two
    unrelated-looking list rows. This is a display grouping, not a real dependency parse - a run
    of function words always attaches to the nearest PRECEDING content word, which is how real
    bunsetsu segmentation works for every pattern this module curates.

    Returns a list of {"surface", "head", "tail", "label"} where `head` is the leading entry
    (content, or a line-initial function word if the line starts with one - e.g. a bare しかし),
    `tail` is the list of entries attached after it, and `label` is a human-readable syntactic
    role (Object/Topic/Subject/Predicate/...) derived from the attached particle(s), or None
    when nothing in the tail maps to a clear role (see _labelForChunk).
    """
    chunks = []
    current = None

    for entry in entries:
        if entry["role"] == "content" or current is None:
            if current is not None:
                chunks.append(current)
            current = {"head": entry, "tail": []}
        else:
            current["tail"].append(entry)

    if current is not None:
        chunks.append(current)

    for chunk in chunks:
        chunk["surface"] = chunk["head"]["surface"] + "".join(e["surface"] for e in chunk["tail"])
        chunk["label"] = _labelForChunk(chunk)

    return chunks
