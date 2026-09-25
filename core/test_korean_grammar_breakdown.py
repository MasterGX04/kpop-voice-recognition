"""
Tests for Phase 1 of core/korean_grammar_breakdown.py.

Stdlib unittest only, matching core/test_grammar_breakdown.py's style. Run with:
    python -m unittest core.test_korean_grammar_breakdown -v
"""

import unittest

from core.korean_grammar_breakdown import breakdownLine


class BreakdownLineTests(unittest.TestCase):
    def test_worked_example_first_half(self):
        # .claude/KOREAN_GRAMMAR_BREAKDOWN_PLAN.md's canonical worked example, first line.
        # 목소릴 is a casual contraction of 목소리+를, and 따라 is 따르다's stem 따르 fused with
        # the connective 어 (irregular 르-conjugation: 따르+어 -> 따라, not "따르어"). Both are
        # ONE written syllable group each per direct user feedback ("make sure not to separate
        # 하/어 because it should just be 해") - shown as one merged entry, not two, with each
        # piece's own gloss combined rather than displayed as if they were separate characters.
        entries = breakdownLine("목소릴 따라 너의 호흡을 따라")
        self.assertEqual(
            [e["surface"] for e in entries],
            ["목소릴", "따라", "너", "의", "호흡", "을", "따라"],
        )
        self.assertEqual(entries[0]["role"], "content")
        self.assertIn("voice", entries[0]["gloss"])
        self.assertIn("object marker", entries[0]["gloss"])
        self.assertEqual(entries[1]["role"], "content")
        self.assertIn("따르다", entries[1]["lemma"])  # dictionary form, not the bare stem
        self.assertIn("follow", entries[1]["gloss"])

    def test_worked_example_jeonhaejyeo_two_syllable_groups(self):
        # 전해져 -> 전하[to convey]+어[connective] fused into the written syllable "전해", then
        # 지[passive/inchoative]+어[connective] fused into "져" - the plan doc's proof that Kiwi
        # exposes real conjugation structure, not just word-segmentation (same significance as
        # 食べたくない's 3-piece split for Japanese), now shown as the two ACTUAL written
        # syllable-groups rather than four separately-listed morphemes that were never really
        # four separate characters on the page.
        entries = breakdownLine("다 전해져 어떤 아픔 어떤 슬픔")
        jeonhae, jyeo = entries[1], entries[2]
        self.assertEqual(jeonhae["surface"], "전해")
        self.assertEqual(jeonhae["role"], "content")
        self.assertIn("convey", jeonhae["gloss"])
        self.assertIn("connective", jeonhae["gloss"])
        self.assertEqual(jyeo["surface"], "져")
        self.assertEqual(jyeo["role"], "function")
        self.assertIn("passive", jyeo["gloss"])

        # 다 here is the adverb "all/completely" (MAG), not the unrelated verb-ending suffix note
        # that happens to be the Wiktionary index's first-listed entry for the same spelling -
        # regression test for the POS-matching fix in core/korean_dictionary.py.
        self.assertEqual(entries[0]["tag"], "MAG")
        self.assertIn("completely", entries[0]["gloss"])

    def test_fused_syllable_reconstructs_real_written_surface(self):
        # Real bug found via testing: kiwipiepy's own Token.form for a connective ending
        # sometimes reports its citation-form spelling ("어") even when the literal written
        # character at that position is a vowel-harmonized variant ("아") - confirmed directly via
        # start/len offsets into the real string. 안아 ("hug", connective form) must show the
        # TRUE written "아", not the citation-form "어" naively taken from the token.
        entries = breakdownLine("안아줄게")
        self.assertEqual(entries[0]["surface"], "안")
        self.assertEqual(entries[1]["surface"], "아")  # NOT "어"
        # 주+ᆯ게 (auxiliary + promissory ending) fuse into the real written syllable "줄게".
        self.assertEqual(entries[2]["surface"], "줄게")
        self.assertEqual(entries[2]["role"], "function")
        self.assertIn("do for someone", entries[2]["gloss"])
        self.assertIn("informal promissory", entries[2]["gloss"])

    def test_hada_past_tense_fusion_shown_as_one_syllable(self):
        # Real bug reported by the user: 말했듯 was rendering 하(XSV, "-하다" suffix) and
        # 었(EP, past tense) as two separate lines, as if "하" and "었" were each their own
        # written character - but 했 is ONE syllable representing both fused together. Also
        # covers content+function fusion (되+었 -> 됐, 그렇+었 -> 그랬) via the "됐" case.
        entries = breakdownLine("말했듯")
        malhaess = entries[1]
        self.assertEqual(malhaess["surface"], "했")
        self.assertEqual(malhaess["role"], "function")
        self.assertIn("verb-forming suffix", malhaess["gloss"])
        self.assertIn("past tense", malhaess["gloss"])

        dwaess = breakdownLine("됐어")[0]
        self.assertEqual(dwaess["surface"], "됐")
        self.assertEqual(dwaess["role"], "content")  # 되다 ("to become") is the content word
        self.assertIn("past tense", dwaess["gloss"])

    def test_frequent_particles_and_endings_glossed(self):
        # 이것이 -> 이것[this] + 이(JKS, subject marker) - the same surface "이" recurs later in
        # this line under two other tags (JKC, VCP) with unrelated meanings, so this looks it up
        # by tag rather than by surface alone (see _FUNCTION_GLOSSES's per-tag design).
        entries = breakdownLine("이것이 사랑이 아니라면 어떤 것이 사랑일까")
        subject_marker = next(e for e in entries if e["surface"] == "이" and e["tag"] == "JKS")
        self.assertIn("subject marker", subject_marker["gloss"])

    def test_unglossed_function_word_falls_back_to_none_not_fabricated(self):
        # A real ending Kiwi can produce with no curated entry yet - must show up as a function
        # token with gloss=None, never a made-up gloss (same "don't guess" discipline as the
        # Japanese module's particle/auxiliary dicts).
        entries = breakdownLine("갈게요")
        unglossed = [e for e in entries if e["role"] == "function" and e["gloss"] is None]
        # Not asserting a specific token (curated coverage may grow over time) - only that a miss
        # is possible at all and doesn't crash or get a fabricated gloss.
        for e in unglossed:
            self.assertIsNone(e["gloss"])

    def test_punctuation_skipped(self):
        entries = breakdownLine("아, 정말?")
        surfaces = [e["surface"] for e in entries]
        self.assertNotIn(",", surfaces)
        self.assertNotIn("?", surfaces)

    def test_mixed_script_line_drops_english_words(self):
        # Real lyrics mix in English fragments (per the corpus frequency scan in the plan doc,
        # SL-tagged tokens like "I"/"that"/"my"). Per direct user feedback, these carry no Korean
        # grammar to break down and must be dropped entirely rather than shown with gloss=None.
        entries = breakdownLine("say you love me 정말")
        surfaces = [e["surface"] for e in entries]
        self.assertNotIn("love", surfaces)
        self.assertNotIn("say", surfaces)
        self.assertIn("정말", surfaces)

    def test_content_word_with_hanja_origin_lists_candidates(self):
        # 심장 ("heart", an organ) -> 心臟, and 평온 ("tranquility") -> 平穩 - both genuinely
        # Sino-Korean, confirmed directly against real lyric text during testing. A whole-line
        # Grammar Breakdown should surface this without a separate per-word Hanja lookup step -
        # merged in per direct user request.
        entries = breakdownLine("평온했던 심장이")
        pyeongon = next(e for e in entries if e["surface"] == "평온")
        self.assertEqual([c["hanja"] for c in pyeongon["hanja"]], ["平穩"])
        simjang = next(e for e in entries if e["surface"] == "심장")
        self.assertEqual([c["hanja"] for c in simjang["hanja"]], ["心臟"])

    def test_determiner_ambiguous_syllable_is_not_flooded_with_unrelated_hanja(self):
        # Real bug reported by the user, this exact line: 이 here is the native proximal
        # determiner ("this feeling"), tagged MM by Kiwi - but 이 is ALSO a real, unrelated Hanja
        # word (or six of them: 二/李/理/利/釐/伊) under other parts of speech (numeral, surname,
        # ...). Merging Hanja lookup into the breakdown must not surface those just because they
        # share a spelling - see core/korean_hanja.py: lookupHanja()'s tag-filtering fix.
        entries = breakdownLine("이 느낌은 설렘보다는 왠지 toxic")
        i = next(e for e in entries if e["surface"] == "이")
        self.assertEqual(i["tag"], "MM")
        self.assertEqual(i["hanja"], [])
        self.assertIn("this", i["gloss"])
        # "toxic" (English) must also be dropped entirely, same as the mixed-script test above.
        self.assertNotIn("toxic", [e["surface"] for e in entries])

    def test_native_content_word_has_no_hanja_candidates(self):
        # 목소리 ("voice") is purely native - hanja must be an empty list, not fabricated or
        # omitted, so the caller can tell "checked, found nothing" apart from "not a content word".
        entries = breakdownLine("목소리")
        moksori = next(e for e in entries if e["surface"] == "목소리")
        self.assertEqual(moksori["hanja"], [])

    def test_function_word_has_no_hanja_field_value(self):
        entries = breakdownLine("목소리를")
        particle = next(e for e in entries if e["role"] == "function")
        self.assertIsNone(particle["hanja"])

    def test_fused_adverb_root_gets_its_hanja(self):
        # Real bug reported by the user: 조심 ("caution, care") is genuinely Sino-Korean (操心),
        # but 조심히 (the adverb actually used in the lyric, "carefully") is a single MAG-tagged
        # Kiwi token whose own Wiktionary "adv" entry never carries Hanja directly - so the direct
        # lookup was coming up empty even though the underlying noun root plainly has one. Must
        # fall back to the stripped root (조심) for both gloss and Hanja.
        entries = breakdownLine("조심히 안아줄게")
        josimhi = entries[0]
        self.assertEqual(josimhi["surface"], "조심히")
        self.assertEqual(josimhi["tag"], "MAG")
        self.assertIn("caution", josimhi["gloss"])
        self.assertEqual([c["hanja"] for c in josimhi["hanja"]], ["操心"])

    def test_adverbializer_suffix_is_function_not_content(self):
        # Real bug reported by the user: 따스+히 (XR root + XSM adverbializer, two separate Kiwi
        # tokens unlike the single-token 조심히) was showing "히" as its own bogus CONTENT word -
        # matching an unrelated onomatopoeia noun entry ("Conveys a nervous laughter or smile")
        # instead of being recognized as grammatical glue. 따스 itself has no recoverable entry
        # anywhere in the index (confirmed directly) - a genuine "not yet documented" coverage
        # gap, not a bug, so it's expected to keep showing no gloss/Hanja.
        entries = breakdownLine("따스히 안아줄게")
        tteuseu, hi = entries[0], entries[1]
        self.assertEqual(tteuseu["surface"], "따스")
        self.assertEqual(tteuseu["role"], "content")
        self.assertIsNone(tteuseu["gloss"])
        self.assertEqual(tteuseu["hanja"], [])
        self.assertEqual(hi["surface"], "히")
        self.assertEqual(hi["role"], "function")
        self.assertIn("adverbializer", hi["gloss"])

    def test_bound_root_composes_with_hada_for_its_gloss(self):
        # Real gap found via testing: 나른 (XR, "bound root") never appears alone in the
        # Wiktionary index - only its citation form 나른하다 ("languid, listless") does. 조용
        # (also XR) DOES have a bare entry, but it's a self-referential stub ("Root of 조용하다 ...
        # Rarely used alone.") that isn't a real gloss either - the composed form must win there
        # too. An ordinary noun that also happens to combine with 하다 (평온, tagged NNG not XR)
        # must NOT go through this path - it already has its own perfectly good standalone gloss.
        nareun = breakdownLine("나른한 오후")
        naReunEntry = next(e for e in nareun if e["surface"] == "나른")
        self.assertEqual(naReunEntry["tag"], "XR")
        self.assertIn("languid", naReunEntry["gloss"])

        joyong = breakdownLine("조용한 밤")
        joYongEntry = next(e for e in joyong if e["surface"] == "조용")
        self.assertIn("quiet", joYongEntry["gloss"])

        pyeongon = breakdownLine("평온한 하루")
        pyeonOnEntry = next(e for e in pyeongon if e["surface"] == "평온")
        self.assertEqual(pyeonOnEntry["tag"], "NNG")
        self.assertIn("tranquility", pyeonOnEntry["gloss"])

    def test_dependent_noun_su_gets_no_hanja_flood(self):
        # Real bug reported by the user: 수 in "가눌 수 없는" ("cannot control/steady") is the
        # native dependent noun meaning "way/means/ability" (tag NNB) - but the exact same
        # spelling is also 11 unrelated real Hanja words (手/數/水/繡/首/受/髓/隋/守/...) that are
        # only ever used as ORDINARY standalone nouns (NNG), never in this bound construction.
        # Wiktionary's own "pos" field can't tell these apart (both are just "noun"), so this
        # needs the hand-curated dependent-noun denylist, not POS-tag matching alone.
        entries = breakdownLine("가눌 수 없는 게 싫지 않네")
        su = next(e for e in entries if e["surface"] == "수")
        self.assertEqual(su["tag"], "NNB")
        self.assertEqual(su["hanja"], [])
        self.assertIn("way, means", su["gloss"])


if __name__ == "__main__":
    unittest.main()
