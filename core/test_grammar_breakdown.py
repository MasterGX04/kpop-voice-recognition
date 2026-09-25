"""
Tests for Phase 1 of core/grammar_breakdown.py.

Stdlib unittest only, matching core/test_kanji_reference.py's style. Run with:
    python -m unittest core.test_grammar_breakdown -v
"""

import unittest

from core.grammar_breakdown import breakdownLine, groupIntoChunks


class BreakdownLineTests(unittest.TestCase):
    def test_worked_example_tabetakunai(self):
        # The plan's canonical worked example: 食べ[to eat] + たく[want to] + ない[negative].
        entries = breakdownLine("食べたくない")
        self.assertEqual([e["surface"] for e in entries], ["食べ", "たく", "ない"])
        self.assertEqual(entries[0]["role"], "content")
        self.assertEqual(entries[0]["gloss"], "to eat")
        self.assertEqual(entries[1]["role"], "function")
        self.assertEqual(entries[1]["gloss"], "want to")
        # ない here tokenizes as pos1=形容詞 (i-adjective), not 助動詞 - a real fugashi finding,
        # not the plan doc's assumption - so it's classified "content" and glossed via JMdict
        # rather than the curated auxiliary dict, but the gloss still correctly reads as negation.
        self.assertIn("not", entries[2]["gloss"])

    def test_youni_pattern_glossed_as_one_unit(self):
        # ように doesn't recompose from よう("manner") + に(copula) glossed separately -
        # confirmed via testing (see .claude/GRAMMAR_BREAKDOWN_PLAN.md) - must be caught by the
        # compound-pattern layer as a single "pattern" entry, not two disconnected tokens.
        entries = breakdownLine("食べられるように")
        pattern = [e for e in entries if e["role"] == "pattern"]
        self.assertEqual(len(pattern), 1)
        self.assertEqual(pattern[0]["surface"], "ように")
        self.assertIn("so that", pattern[0]["gloss"])

    def test_nda_pattern_glossed_as_one_unit(self):
        entries = breakdownLine("食べたんだ")
        pattern = [e for e in entries if e["role"] == "pattern"]
        self.assertEqual(len(pattern), 1)
        self.assertEqual(pattern[0]["surface"], "んだ")
        self.assertIn("explanatory", pattern[0]["gloss"])

    def test_noni_pattern_glossed_as_contrastive(self):
        entries = breakdownLine("食べたのに")
        pattern = [e for e in entries if e["role"] == "pattern"]
        self.assertEqual(len(pattern), 1)
        self.assertEqual(pattern[0]["surface"], "のに")
        self.assertIn("even though", pattern[0]["gloss"])

    def test_te_shimau_light_verb_glossed_not_literal(self):
        # しまう here is grammaticalized (completive/regret), not its literal "to put away" -
        # must only fire when directly preceded by a て/で connective particle.
        entries = breakdownLine("食べてしまった")
        shimau = next(e for e in entries if e["surface"] == "しまっ")
        self.assertEqual(shimau["role"], "function")
        self.assertIn("accidentally", shimau["gloss"])

    def test_light_verb_lemma_used_literally_is_not_misglossed(self):
        # 行く used as the literal main verb ("to go"), NOT preceded by て/で, must fall through
        # to its ordinary content-word (JMdict) gloss instead of the てい／ていく auxiliary gloss -
        # regression check for the pos2=非自立可能-is-a-lexical-not-contextual-flag finding.
        entries = breakdownLine("行くしかない")
        iku = next(e for e in entries if e["surface"] == "行く")
        self.assertEqual(iku["role"], "content")
        self.assertIn("to go", iku["gloss"])

    def test_frequent_bare_particle_has_curated_gloss(self):
        entries = breakdownLine("これは本です")
        wa = next(e for e in entries if e["surface"] == "は")
        self.assertEqual(wa["role"], "function")
        self.assertEqual(wa["gloss"], "topic marker")

    def test_kana_written_adverb_still_glossed_via_its_kanji_lemma(self):
        # ちょっと writes in pure kana, but fugashi's lemma for it is the real (if archaic)
        # Kanji citation form 一寸, which JMdict does index - confirmed via testing that the
        # original Phase 1 assumption ("kana-surface content word == unglossable") was wrong for
        # common adverbs like this one, not just a hypothetical edge case. Regression test for
        # that bug (real-world report: ちょっと was showing "(no gloss)").
        entries = breakdownLine("ちょっとだけ")
        chotto = next(e for e in entries if e["surface"] == "ちょっと")
        self.assertIn("little", chotto["gloss"])

    def test_gap_falls_back_to_bare_lemma_without_fabricating(self):
        # マジ ("maji", slang "seriously/for real") has a genuinely kana-only lemma (まじ) with no
        # Kanji citation form anywhere in JMdict, unlike ちょっと/とても above - confirmed via
        # testing this is a real remaining gap, not a hypothetical one. Must show up with no
        # gloss rather than crashing, fabricating one, or silently disappearing from the output.
        entries = breakdownLine("マジで")
        maji = next(e for e in entries if e["surface"] == "マジ")
        self.assertIsNone(maji["gloss"])

    def test_demonstrative_kosoado_series_is_glossed(self):
        # どんな/こんな/そんな/あんな have a genuinely kana-only lemma (no Kanji form exists at
        # all) - real bug report: どんな was showing "(no gloss)" in "どんな宝石よりも".
        entries = breakdownLine("どんな宝石よりも")
        donna = next(e for e in entries if e["surface"] == "どんな")
        self.assertIn("what kind", donna["gloss"])

    def test_katakana_loanword_is_glossed(self):
        # ダイヤ ("diamond") is a katakana loanword with no Kanji spelling at all, so JMdict's
        # Kanji-keyed index can never find it - real bug report: showed "(no gloss)" in "その瞳は
        # ダイヤ". Also regression-checks the "-diagram" disambiguator-suffix fugashi bakes into
        # this specific lemma is stripped before the fallback lookup.
        entries = breakdownLine("その瞳はダイヤ")
        diamond = next(e for e in entries if e["surface"] == "ダイヤ")
        self.assertEqual(diamond["gloss"], "diamond")

    def test_sou_looks_like_auxiliary_is_glossed_and_distinguished_from_plain_sou(self):
        # そう has two unrelated senses fugashi tags differently: the evidential auxiliary
        # "~sou" (looks like/seems like, attaches directly to a verb/adjective stem, tagged
        # pos2=助動詞語幹) vs. the plain standalone adverb "so/that way" (tagged pos2=*, no Kanji
        # form, a separate real gap). Must gloss the auxiliary sense correctly without the plain
        # adverb sense leaking into it, and vice versa.
        auxEntries = breakdownLine("降りそう")
        sou = next(e for e in auxEntries if e["surface"] == "そう")
        self.assertIn("looks like", sou["gloss"])

        adverbEntries = breakdownLine("分かれ道はそう")
        plainSou = next(e for e in adverbEntries if e["surface"] == "そう")
        self.assertIn("so", plainSou["gloss"])

    def test_mitai_auxiliary_is_glossed(self):
        entries = breakdownLine("穴が空いたみたい")
        mitai = next(e for e in entries if e["surface"] == "みたい")
        self.assertIn("looks like", mitai["gloss"])

    def test_nominalizing_suffix_sa_is_glossed_and_distinguished_from_the_particle(self):
        # さ here is pos1=接尾辞 ("-ness", turns 寂しい's stem into an abstract noun), a different
        # word than the sentence-final particle さ ("casual emphasis") already in
        # _PARTICLE_GLOSSES - both share the same surface/lemma, so this only works if the two are
        # kept genuinely distinguished by pos1, not accidentally merged.
        entries = breakdownLine("さみしさにやられた")
        sa = next(e for e in entries if e["surface"] == "さ")
        self.assertEqual(sa["pos1"], "接尾辞")
        self.assertIn("ness", sa["gloss"])

    def test_counter_suffix_tsu_is_glossed(self):
        entries = breakdownLine("世界でひとつ")
        tsu = next(e for e in entries if e["surface"] == "つ")
        self.assertEqual(tsu["pos1"], "接尾辞")
        self.assertIn("counter", tsu["gloss"])

    def test_punctuation_is_skipped(self):
        entries = breakdownLine("好き。")
        self.assertTrue(all(e["surface"] != "。" for e in entries))

    def test_embedded_english_does_not_crash_and_is_dropped(self):
        # Real bug report: fugashi tags an unrecognized Latin-alphabet run (an embedded English
        # phrase) as pos1=名詞 but lemma=None, which crashed every lemma-handling function here
        # ("argument of type 'NoneType' is not iterable") the moment English text was included
        # in the highlighted selection. English words carry no Japanese grammar to break down,
        # so they must be silently dropped, not crash and not appear as a bogus entry.
        entries = breakdownLine("希少なほどに貴重なものだと You will find out")
        surfaces = [e["surface"] for e in entries]
        self.assertNotIn("You", surfaces)
        self.assertNotIn("find", surfaces)
        self.assertTrue(len(entries) > 0)

    def test_fullwidth_space_is_skipped(self):
        # Some lyrics use a full-width "　" as a mid-line separator (pos1=空白) - real bug
        # report: it was showing up as its own "(content) — (no gloss)" entry.
        entries = breakdownLine("待つほど　甘み増す")
        self.assertNotIn("　", [e["surface"] for e in entries])

    def test_proper_noun_with_broken_katakana_lemma_still_glossed(self):
        # 恵 ("megumi", blessing/grace) is a real bug report: unidic-lite stores its lemma as
        # the katakana メグミ instead of 恵, the same finding already documented in
        # core/kanji_reference.py: _analyzeToken for other proper nouns (e.g. 東京/トウキョウ).
        # JMdict is keyed on the Kanji spelling, so looking it up under the broken katakana
        # lemma silently failed - must fall back to the surface itself.
        entries = breakdownLine("恵の雨")
        megumi = next(e for e in entries if e["surface"] == "恵")
        self.assertIsNotNone(megumi["gloss"])


class GroupIntoChunksTests(unittest.TestCase):
    def test_trailing_particles_attach_to_the_preceding_content_word(self):
        # Real user feedback: a flat token-by-token list "looks like a discombobulated blob of
        # vocab" once a line has more than a few words - grouping into content-word + attached
        # particles/auxiliaries chunks (bunsetsu-like) is meant to fix that readability problem.
        entries = breakdownLine("希少なほどに貴重なものだと")
        chunks = groupIntoChunks(entries)
        surfaces = [c["surface"] for c in chunks]
        self.assertEqual(surfaces, ["希少なほどに", "貴重な", "ものだと"])
        self.assertEqual(chunks[0]["head"]["surface"], "希少")
        self.assertEqual([t["surface"] for t in chunks[0]["tail"]], ["な", "ほど", "に"])

    def test_back_to_back_content_words_are_separate_chunks(self):
        entries = breakdownLine("雨降る")
        chunks = groupIntoChunks(entries)
        self.assertEqual([c["surface"] for c in chunks], ["雨", "降る"])

    def test_line_initial_particle_becomes_its_own_head(self):
        entries = [{"surface": "も", "lemma": "も", "pos1": "助詞", "role": "function", "gloss": "also / too"}]
        chunks = groupIntoChunks(entries)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["head"]["surface"], "も")
        self.assertEqual(chunks[0]["tail"], [])

    def test_object_topic_and_comparative_labels(self):
        # Real user report: this exact line ("穢れを知らないな / その瞳はダイヤ / どんな宝石より
        # も") was the worked example for wanting role labels instead of a flat token list.
        entries = breakdownLine("穢れを知らないな\nその瞳はダイヤ\nどんな宝石よりも")
        chunks = {c["surface"]: c["label"] for c in groupIntoChunks(entries)}
        self.assertEqual(chunks["穢れを"], "Object")
        self.assertEqual(chunks["瞳は"], "Topic")
        self.assertEqual(chunks["宝石よりも"], "Comparative Baseline")

    def test_verb_chain_with_no_case_particle_is_labeled_predicate(self):
        entries = breakdownLine("穢れを知らないな")
        chunks = {c["surface"]: c["label"] for c in groupIntoChunks(entries)}
        self.assertEqual(chunks["知らないな"], "Predicate")

    def test_pattern_chunk_gets_its_own_role_label(self):
        # んだ (の+だ, explanatory) is a compound "pattern" entry, not a plain particle - its
        # label has to come from the pattern-specific map, not the single-particle one.
        entries = breakdownLine("食べたんだ")
        chunks = groupIntoChunks(entries)
        ndaChunk = next(c for c in chunks if "んだ" in c["surface"])
        self.assertEqual(ndaChunk["label"], "Explanatory")

    def test_bare_content_word_with_no_role_particle_is_unlabeled(self):
        entries = breakdownLine("その")
        chunks = groupIntoChunks(entries)
        self.assertIsNone(chunks[0]["label"])


if __name__ == "__main__":
    unittest.main()
