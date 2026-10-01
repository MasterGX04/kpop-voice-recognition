"""
Tests for Milestones 3-6 (Chinese-cognate lookup via CC-CEDICT, per-song vocab persistence, the
cross-song shared-vocabulary index, and Japanese-meaning lookup via JMdict) in
core/kanji_reference.py.

Stdlib unittest only - no test framework is set up in this project yet. Run with:
    python -m unittest core.test_kanji_reference -v
"""

import codecs
import json
import os
import shutil
import tempfile
import unittest

from core.kanji_reference import (
    analyzeSelection,
    resolveContextReading,
    _fallbackPinyin,
    _normalizePinyin,
    lookupChineseCognate,
    lookupJapaneseMeaning,
    getMandarinPinyin,
    addWordToSongReference,
    loadSongVocab,
    renderSongReferenceHtml,
    scanLyricsForKanjiVocab,
    buildKanjiReferenceIndex,
    renderWordIndexBySongHtml,
    _containsKana,
    _isJapaneseLyricEntry,
    _songKey,
)


class LookupChineseCognateTests(unittest.TestCase):
    def test_confirmed_sino_japanese_compound(self):
        # 時間 (jikan) - the plan's canonical "real cognate" worked example.
        result = lookupChineseCognate("時間")
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["pinyin"], "shi2 jian1")
        self.assertTrue(any("time" in g for g in result["gloss"]))

    def test_shinjitai_is_converted_before_lookup(self):
        # 天気 uses Shinjitai 気 and is NOT itself in CC-CEDICT; only the Traditional
        # form 天氣 (気->氣) is. Confirms toTraditional() actually runs before the lookup,
        # not just an identity passthrough.
        result = lookupChineseCognate("天気")
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["traditional"], "天氣")
        self.assertEqual(result["pinyin"], "tian1 qi4")

    def test_wasei_kango_not_attested(self):
        # 残業 (zangyou, "overtime work") is a genuine Japan-coined Sino-style compound -
        # confirmed absent from CC-CEDICT under both its Shinjitai and Traditional (殘業) forms.
        result = lookupChineseCognate("残業")
        self.assertEqual(result["status"], "not_attested")
        self.assertEqual(result["traditional"], "殘業")
        self.assertEqual(result["pinyinFallback"], "can2 ye4")

    def test_surname_first_entry_is_skipped_in_favor_of_the_common_word_reading(self):
        # Real bug found while adding Hanja pinyin: 火 also doubles as a common Chinese surname,
        # and CC-CEDICT lists that reading first, capitalized ("Huo3"/"surname Huo") - taking
        # entries[0] fabricated a "surname Huo" gloss/pinyin for what should read "huo3"/"fire".
        result = lookupChineseCognate("火")
        self.assertEqual(result["pinyin"], "huo3")
        self.assertIn("fire", result["gloss"])

    def test_false_friend_is_still_reported_as_confirmed(self):
        # 大丈夫 exists in CC-CEDICT ("a manly man") but means something entirely different
        # in Japanese ("it's okay"). The lookup only confirms the character string is a real
        # Chinese word, not that the meaning matches - this is a known, documented limitation,
        # not a bug, so it must NOT be silently treated as "not attested".
        result = lookupChineseCognate("大丈夫")
        self.assertEqual(result["status"], "confirmed")
        self.assertEqual(result["pinyin"], "da4 zhang4 fu5")


class FallbackPinyinTests(unittest.TestCase):
    def test_uses_first_kanjidic_candidate_per_character(self):
        # 残業 isn't in CC-CEDICT, so the fallback pinyin is built by concatenating each
        # character's own best-available reading, not a real Chinese-dictionary reading.
        self.assertEqual(_fallbackPinyin("殘業"), "can2 ye4")

    def test_prefers_cedict_over_a_misleadingly_ordered_kanjidic_entry(self):
        # Real bug found via the user's own 言葉 example: KANJIDIC2's raw pinyin list for 葉 is
        # ["xie2", "ye4", "she4"] - "xie2" (a rare classical reading) is listed FIRST, ahead of
        # the everyday "leaf" reading "ye4". Naive index-0 selection produced "yan2 xie2" for
        # 言葉 instead of the expected "yan2 ye4". CC-CEDICT's own entry for 葉 is preferred
        # instead, since it's curated for actual Chinese usage.
        self.assertEqual(_fallbackPinyin("言葉"), "yan2 ye4")

    def test_skips_a_capitalized_surname_entry_in_cedict(self):
        # 葉 (and 業) are also common surnames - CC-CEDICT lists the surname reading first,
        # capitalized ("Ye4"), ahead of the common-word reading ("ye4"). The capitalized entry
        # must be skipped in favor of the lowercase one.
        self.assertEqual(_fallbackPinyin("葉"), "ye4")
        self.assertEqual(_fallbackPinyin("業"), "ye4")

    def test_multi_reading_character_shows_top_two_not_just_the_first_listed(self):
        # Real bug report: 行く (iku, "to go") showed Mandarin mnemonic "hang2" alone - CC-CEDICT's
        # own file order for 行 lists hang2 ("row/profession/bank") before xing2 ("to walk/go/OK",
        # by far the more relevant/common reading here), with heng2 (only used in one fixed
        # compound, "道行") in between. There's no real frequency data to pick a single "right"
        # answer, so both of the top two (ranked by attached sense count as a proxy for how
        # central the reading is) are shown instead of committing to whichever CEDICT happened to
        # list first - xing2 must appear, and the rarely-relevant heng2 must not crowd it out.
        pinyin = _fallbackPinyin("行")
        self.assertIn("xing2", pinyin)
        self.assertNotIn("heng2", pinyin)

    def test_single_reading_character_is_unaffected(self):
        # The vast majority of characters have only one common Mandarin reading - must still
        # render as a single plain pinyin, no stray "/" for a reading that doesn't exist.
        self.assertEqual(_fallbackPinyin("気"), "qi4")


class OnyomiRootPlusNativeSuffixBugTests(unittest.TestCase):
    """
    Regression test for a real bug: 切ない (setsunai, "heartrending") was misclassified as pure
    "onyomi" because 切's on'yomi せつ matches the start of the reading, and the classifier used
    to let the leftover "ない" pass through via a blanket literal-identity check with no
    requirement that it be legitimate okurigana for anything - real on'yomi (Sino-Japanese)
    compounds are always 100% Kanji with zero embedded kana (時間, 瞬間, 残業). This then made
    lookupChineseCognate()/getMandarinPinyin() run on the full lemma including the kana, pasting
    it literally into a "Chinese" pinyin string (e.g. "qie1 な い"). Fixed in _matchSegmentation()
    by removing the blanket kana pass-through entirely - only a Kanji's own declared kun'yomi
    okurigana may consume trailing kana now.
    """

    def test_setsunai_is_jukujigo_not_onyomi(self):
        entries = analyzeSelection("切ない", 0, 3)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["category"], "jukujigo")
        self.assertIsNone(entries[0]["chineseCognate"])

    def test_setsunai_mandarin_pinyin_has_no_leaked_kana(self):
        # 切 has two genuinely common Mandarin readings (qie1 "to cut", qie4 "eager/urgent/
        # definitely") with no reliable way to rank one as more common than the other - _charPinyin
        # now shows both instead of arbitrarily committing to one (see its own docstring: CEDICT's
        # file order isn't frequency order, confirmed by the 行/xing2-vs-hang2 case). The important
        # thing this test still verifies is what its name says: no leaked kana ("ない" must not
        # appear in the pinyin) - not which single reading "wins".
        entries = analyzeSelection("切ない", 0, 3)
        pinyin = entries[0]["mandarinPinyin"]
        self.assertEqual(pinyin["traditional"], "切")
        self.assertIn("qie1", pinyin["pinyin"])
        self.assertNotIn("ない", pinyin["pinyin"])

    def test_setsunai_still_gets_a_real_japanese_meaning(self):
        # japaneseMeaning is unconditional (Milestone 6) - unaffected by the category fix.
        entries = analyzeSelection("切ない", 0, 3)
        self.assertEqual(entries[0]["japaneseMeaning"]["status"], "found")
        self.assertIn("heartrending", entries[0]["japaneseMeaning"]["gloss"])

    def test_genuine_kunyomi_okurigana_words_are_unaffected(self):
        # 出会う/危ない/少ない all legitimately end in kana that IS declared okurigana for their
        # own kanji - must still classify correctly after removing the blanket pass-through.
        for text, expectedLemma in [("出会った瞬間", "出会う"), ("危ない橋", "危ない"), ("少ない量", "少ない")]:
            entries = analyzeSelection(text, 0, len(text))
            match = next(e for e in entries if e["lemma"] == expectedLemma)
            self.assertEqual(match["category"], "kunyomi")


class EnglishInJapaneseLyricTests(unittest.TestCase):
    """Japanese lyrics mix in English; those words must never become vocab entries."""

    def test_english_words_are_not_vocab(self):
        text = "How I am gonna find it どうやって? Oh let me know"
        lemmas = [e["lemma"] for e in analyzeSelection(text, 0, len(text))]
        for lemma in lemmas:
            self.assertTrue(any(ord(ch) > 0x3000 for ch in lemma), f"non-Japanese lemma {lemma!r}")

    def test_japanese_words_next_to_english_are_still_kept(self):
        text = "Baby 時間 is running"
        lemmas = [e["lemma"] for e in analyzeSelection(text, 0, len(text))]
        self.assertEqual(lemmas, ["時間"])


class AnalyzeSelectionCognateGatingTests(unittest.TestCase):
    """
    Chinese-cognate lookup must only run for onyomi/mixed words - never fabricated for a
    native Japanese (kunyomi) or idiomatic (jukujigo) word, per the plan's core requirement.
    """

    def test_onyomi_word_gets_a_cognate(self):
        entries = analyzeSelection("時間が経つのは早い", 0, 2)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["category"], "onyomi")
        self.assertIsNotNone(entries[0]["chineseCognate"])
        self.assertEqual(entries[0]["chineseCognate"]["status"], "confirmed")

    def test_kunyomi_word_gets_no_cognate(self):
        # 出会った (deatta) - conjugated kunyomi verb, classified via its lemma 出会う.
        entries = analyzeSelection("出会った瞬間から", 0, 3)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["category"], "kunyomi")
        self.assertIsNone(entries[0]["chineseCognate"])

    def test_jukujigo_word_gets_no_cognate(self):
        # 今日 (kyou) - idiomatic whole-word reading, not decomposable per character.
        entries = analyzeSelection("今日はいい天気", 0, 2)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["category"], "jukujigo")
        self.assertIsNone(entries[0]["chineseCognate"])


class WhitespaceOffsetDriftTests(unittest.TestCase):
    """
    Regression tests for a real bug: fugashi/MeCab never emits whitespace (spaces,
    newlines) as its own token - it's dropped from `surface` and only reported via the
    *next* token's `white_space` attribute. resolveContextReading() used to track token
    positions by naively summing `len(word.surface)`, which silently drifted out of sync
    with the real string offset after every skipped space/newline, corrupting the
    highlight-overlap check for everything that followed. Found via the real lyric field
    "情熱で溶ける甘い愛のChocolate Ah\n衝撃が走る" (saved_labels/TWICE/Funny Valentine_lyrics.json),
    where highlighting exactly "衝撃が走る" only returned 走る - 衝撃 (a real, CC-CEDICT-attested
    cognate meaning "impact/shock", same as Japanese) was silently dropped because the
    accumulated 2-character drift (1 dropped space + 1 dropped newline) pushed its computed
    token range just outside the highlighted range.
    """

    def test_newline_before_highlight_does_not_drop_the_word(self):
        fullText = "何かがある\n衝撃が走る"
        start = fullText.index("衝撃が走る")
        end = start + len("衝撃が走る")
        entries = analyzeSelection(fullText, start, end)
        surfaces = [e["surface"] for e in entries]
        self.assertIn("衝撃", surfaces)
        self.assertIn("走る", surfaces)

    def test_mixed_script_line_with_space_and_newline_before_highlight(self):
        # The exact real-world string that surfaced the bug.
        fullText = "情熱で溶ける甘い愛のChocolate Ah\n衝撃が走る"
        start = fullText.index("衝撃が走る")
        end = start + len("衝撃が走る")
        entries = analyzeSelection(fullText, start, end)
        surfaces = [e["surface"] for e in entries]
        self.assertIn("衝撃", surfaces)
        self.assertIn("走る", surfaces)

        shougeki = next(e for e in entries if e["surface"] == "衝撃")
        self.assertEqual(shougeki["category"], "onyomi")
        self.assertEqual(shougeki["chineseCognate"]["status"], "confirmed")

    def test_selection_tight_against_true_token_boundary_still_matches(self):
        # A dropped space ("abc def") then a dropped newline before "甘い" accumulates a
        # 2-character drift. Highlighting *exactly* 甘い's true offsets is the tightest
        # possible case - any leftover drift makes the overlap check fail outright.
        fullText = "abc def\n甘い"
        start = fullText.index("甘い")
        end = start + len("甘い")
        tokens = resolveContextReading(fullText, start, end)
        self.assertEqual([t.surface for t in tokens], ["甘い"])


class SongVocabPersistenceTests(unittest.TestCase):
    """
    Milestone 4: personal per-song vocab persistence (addWordToSongReference/loadSongVocab/
    renderSongReferenceHtml). Runs inside a throwaway temp directory (chdir'd into) so it never
    touches the real saved_labels/ tree - the paths built by _vocabJsonPath/_referenceHtmlPath
    are relative, mirroring LyricsEditor._lyricsJsonPath()'s own convention.
    """

    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def _entry(self, lemma, surface=None, category="onyomi", cognate=None):
        return {
            "surface": surface or lemma,
            "reading": "じかん",
            "lemma": lemma,
            "lemmaReading": "じかん",
            "category": category,
            "chineseCognate": cognate,
            "dateAdded": "2026-09-08",
            "sourceLine": "Nayeon: 時間がない",
        }

    def test_new_entry_is_appended_and_files_are_written(self):
        entry = self._entry("時間")
        entries = addWordToSongReference("TWICE", "TestSong", entry)

        self.assertEqual(entries, [entry])
        self.assertEqual(loadSongVocab("TWICE", "TestSong"), [entry])
        self.assertTrue(os.path.exists("saved_labels/TWICE/TestSong_kanji_vocab.json"))
        self.assertTrue(os.path.exists("saved_labels/TWICE/TestSong_kanji_reference.html"))

    def test_upsert_by_lemma_replaces_rather_than_duplicates(self):
        # Re-highlighting a different conjugation of the same word (both resolving to lemma
        # 出会う) must hit the same saved entry, not create a second one - same dedupe key
        # philosophy as classifyReading() itself.
        addWordToSongReference("TWICE", "TestSong", self._entry("出会う", surface="出会った", category="kunyomi"))
        entries = addWordToSongReference(
            "TWICE", "TestSong", self._entry("出会う", surface="出会う", category="kunyomi")
        )

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["surface"], "出会う")

    def test_loading_nonexistent_song_returns_empty_list(self):
        self.assertEqual(loadSongVocab("TWICE", "NeverSaved"), [])

    def test_rendered_html_contains_word_and_cognate(self):
        cognate = {"status": "confirmed", "traditional": "時間", "pinyin": "shi2 jian1", "gloss": ["time"]}
        path = renderSongReferenceHtml("TWICE", "TestSong", [self._entry("時間", cognate=cognate)])

        with codecs.open(path, "r", encoding="utf-8") as f:
            htmlText = f.read()
        self.assertIn("時間", htmlText)
        self.assertIn("shi2 jian1", htmlText)

    def test_rendered_html_omits_cognate_section_for_kunyomi(self):
        entry = self._entry("出会う", category="kunyomi", cognate=None)
        path = renderSongReferenceHtml("TWICE", "TestSong", [entry])

        with codecs.open(path, "r", encoding="utf-8") as f:
            htmlText = f.read()
        self.assertNotIn("confirmed", htmlText)
        self.assertNotIn("not attested", htmlText)


class CrossSongIndexTests(unittest.TestCase):
    """
    Milestone 5: automatic cross-song shared-vocabulary index, scanned directly from
    saved_labels/*/*_lyrics.json. Runs inside a throwaway temp directory with synthetic
    saved_labels/ fixtures - never touches the real data.
    """

    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def _writeLyricsFile(self, group, song, entries):
        path = f"saved_labels/{group}/{song}_lyrics.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with codecs.open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)

    def test_contains_kana_detects_hiragana_and_katakana_only(self):
        self.assertTrue(_containsKana("時間だ"))
        self.assertTrue(_containsKana("タイム"))
        self.assertFalse(_containsKana("時間"))
        self.assertFalse(_containsKana("안녕하세요"))

    def test_kana_detection_overrides_a_wrong_language_tag(self):
        # The real bug this session found: saved_labels/TWICE/Do Not Touch_lyrics.json had a
        # genuinely-Japanese verse (containing 瞬間) mistagged "language": "Korean".
        entry = {"language": "Korean", "korean": "瞬間だ"}
        self.assertTrue(_isJapaneseLyricEntry(entry))

    def test_pure_kanji_entry_still_counts_as_japanese_via_the_tag(self):
        # Regression for a real under-inclusion found while building this: kana-detection ALONE
        # would miss genuinely-Japanese pure-Kanji fragments with zero kana (saved_labels/TWICE/
        # Do Not Touch_lyrics.json's real "準備"/"時" entries) - the stored tag must still count
        # too, so the filter is a union of both signals, not a kana-only replacement.
        entry = {"language": "Japanese", "korean": "残業"}
        self.assertTrue(_isJapaneseLyricEntry(entry))

    def test_non_japanese_untagged_entry_is_excluded(self):
        entry = {"language": "Korean", "korean": "안녕하세요"}
        self.assertFalse(_isJapaneseLyricEntry(entry))

    def test_word_shared_across_two_songs_dedupes_within_a_song_but_not_across_songs(self):
        self._writeLyricsFile("TWICE", "SongA", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"], "lyricId": "a1"},
            {"language": "Japanese", "korean": "時間だ", "memberName": ["Nayeon"], "lyricId": "a2"},
        ])
        self._writeLyricsFile("TWICE", "SongB", [
            {"language": "Japanese", "korean": "時間だよ", "memberName": ["Momo"], "lyricId": "b1"},
        ])

        index = scanLyricsForKanjiVocab()

        self.assertIn("時間", index)
        occurrences = index["時間"]["occurrences"]
        self.assertEqual(len(occurrences), 2)  # SongA's two lines dedupe to one occurrence
        songs = {(o["group"], o["song"]) for o in occurrences}
        self.assertEqual(songs, {("TWICE", "SongA"), ("TWICE", "SongB")})

    def test_mislabeled_entry_is_still_scanned_via_kana_detection(self):
        self._writeLyricsFile("TWICE", "Mislabeled", [
            {"language": "Korean", "korean": "瞬間が来る", "memberName": ["Momo"], "lyricId": "m1"},
        ])
        index = scanLyricsForKanjiVocab()
        self.assertIn("瞬間", index)

    def test_pure_kanji_no_kana_entry_is_not_dropped(self):
        self._writeLyricsFile("TWICE", "PureKanji", [
            {"language": "Japanese", "korean": "残業", "memberName": ["Sana"], "lyricId": "p1"},
        ])
        index = scanLyricsForKanjiVocab()
        self.assertIn("残業", index)

    def test_non_japanese_song_contributes_nothing(self):
        self._writeLyricsFile("ITZY", "KoreanSong", [
            {"language": "Korean", "korean": "안녕하세요", "memberName": ["Yeji"], "lyricId": "k1"},
        ])
        index = scanLyricsForKanjiVocab()
        self.assertEqual(index, {})

    def test_build_kanji_reference_index_writes_both_outputs(self):
        self._writeLyricsFile("TWICE", "SongA", [
            {"language": "Japanese", "korean": "時間がない", "memberName": ["Nayeon"], "lyricId": "a1"},
        ])
        self._writeLyricsFile("TWICE", "SongB", [
            {"language": "Japanese", "korean": "時間だよ", "memberName": ["Momo"], "lyricId": "b1"},
        ])
        # A word that only appears in one song - must still land in word_index.json AND on its
        # song's panel in by_song.html (2026-09-08 redesign: every word per song, not just
        # words shared with another song - see renderWordIndexBySongHtml).
        self._writeLyricsFile("TWICE", "SongC", [
            {"language": "Japanese", "korean": "残業した", "memberName": ["Sana"], "lyricId": "c1"},
        ])

        buildKanjiReferenceIndex()

        self.assertTrue(os.path.exists("kanji_reference/word_index.json"))
        self.assertTrue(os.path.exists("kanji_reference/by_song.html"))

        with codecs.open("kanji_reference/word_index.json", "r", encoding="utf-8") as f:
            savedIndex = json.load(f)
        self.assertIn("時間", savedIndex)
        self.assertIn("残業", savedIndex)

        with codecs.open("kanji_reference/by_song.html", "r", encoding="utf-8") as f:
            htmlText = f.read()
        self.assertIn("時間", htmlText)
        self.assertIn("残業", htmlText)


class WordIndexBySongHtmlTests(unittest.TestCase):
    """
    Milestone 5 redesign (2026-09-08, per user request): the flat "shared_words.html" table was
    replaced with a per-song browsable page - a sidebar (grouped by artist group, sorted) lets
    you switch songs, and every word shows a "Found In" list of the *other* songs it recurs in,
    each a link to that song's panel. Every word in a song is listed, not just ones it shares
    with another song - a word unique to its song just gets an empty "Found In" cell.
    """

    def setUp(self):
        self._origCwd = os.getcwd()
        self._tmpDir = tempfile.mkdtemp()
        os.chdir(self._tmpDir)

    def tearDown(self):
        os.chdir(self._origCwd)
        shutil.rmtree(self._tmpDir, ignore_errors=True)

    def _wordEntry(self, reading, category, occurrences):
        return {"reading": reading, "category": category, "chineseCognate": None, "occurrences": occurrences}

    def _occ(self, group, song):
        return {"group": group, "song": song, "memberName": ["Someone"], "lyricId": "id"}

    def test_every_song_gets_its_own_panel(self):
        index = {
            "時間": self._wordEntry("じかん", "onyomi", [self._occ("TWICE", "SongA"), self._occ("TWICE", "SongB")]),
            "残業": self._wordEntry("ざんぎょう", "onyomi", [self._occ("TWICE", "SongC")]),
        }
        path = renderWordIndexBySongHtml(index)
        with codecs.open(path, "r", encoding="utf-8") as f:
            htmlText = f.read()

        self.assertIn(_songKey("TWICE", "SongA"), htmlText)
        self.assertIn(_songKey("TWICE", "SongB"), htmlText)
        self.assertIn(_songKey("TWICE", "SongC"), htmlText)
        self.assertIn("時間", htmlText)
        self.assertIn("残業", htmlText)

    def test_shared_word_links_to_the_other_song_not_itself(self):
        index = {
            "時間": self._wordEntry("じかん", "onyomi", [self._occ("TWICE", "SongA"), self._occ("TWICE", "SongB")]),
        }
        path = renderWordIndexBySongHtml(index)
        with codecs.open(path, "r", encoding="utf-8") as f:
            htmlText = f.read()

        keyA = _songKey("TWICE", "SongA")
        keyB = _songKey("TWICE", "SongB")
        panelA = htmlText[htmlText.index(f"id='{keyA}'"): htmlText.index(f"id='{keyB}'")]
        # SongA's row for 時間 must link to SongB (the *other* song), never to itself.
        self.assertIn(f"data-song='{keyB}'", panelA)
        self.assertNotIn(f"data-song='{keyA}'", panelA)

    def test_word_unique_to_one_song_has_no_found_in_link(self):
        index = {
            "残業": self._wordEntry("ざんぎょう", "onyomi", [self._occ("TWICE", "SongC")]),
        }
        path = renderWordIndexBySongHtml(index)
        with codecs.open(path, "r", encoding="utf-8") as f:
            htmlText = f.read()

        # The CSS always defines a ".found-in" rule (static per page), but no found-in <ul>
        # should actually be rendered when the word has no other occurrences.
        self.assertNotIn("<ul class='found-in'>", htmlText)
        # Only the sidebar's own nav entry for SongC should exist - no found-in link to itself
        # or anywhere else, since it's the only song with this word. (The JS also mentions
        # "song-link" as a CSS class selector, so match the actual rendered anchor markup.)
        self.assertEqual(htmlText.count("class='song-link'"), 1)

    def test_songs_are_grouped_and_sorted_by_group_then_song(self):
        index = {
            "気分": self._wordEntry("きぶん", "onyomi", [self._occ("WJSN", "Zebra")]),
            "時間": self._wordEntry("じかん", "onyomi", [self._occ("TWICE", "Alcohol-Free")]),
        }
        path = renderWordIndexBySongHtml(index)
        with codecs.open(path, "r", encoding="utf-8") as f:
            htmlText = f.read()

        # TWICE sorts before WJSN alphabetically, so its nav-group heading must appear first.
        self.assertLess(htmlText.index(">TWICE<"), htmlText.index(">WJSN<"))


class LemmaPosSuffixBugTests(unittest.TestCase):
    """
    Regression test for a real bug found while verifying Milestone 6 against real saved_labels
    data: unidic-lite bakes a "-<POS category>" disambiguator suffix directly into the lemma for
    some common homograph-prone words - 私 (BTS/Let Go) and 君 (TWICE/Funny Valentine) both came
    back as "私-代名詞"/"君-代名詞" ("-pronoun") instead of bare "私"/"君". This silently
    misclassified both as "jukujigo" instead of kunyomi, and made JMdict lookups fail outright
    since "私-代名詞" isn't a real kanji spelling in JMdict. Fixed in _analyzeToken() by
    stripping everything from the first hyphen onward (a literal ASCII hyphen never appears in a
    genuine Japanese lemma).
    """

    def test_watashi_pronoun_lemma_suffix_is_stripped(self):
        entries = analyzeSelection("私は元気です", 0, 1)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["lemma"], "私")
        self.assertEqual(entries[0]["category"], "kunyomi")
        self.assertEqual(entries[0]["japaneseMeaning"]["status"], "found")
        self.assertIn("I", entries[0]["japaneseMeaning"]["gloss"])

    def test_kimi_pronoun_lemma_suffix_is_stripped(self):
        entries = analyzeSelection("君は美しい", 0, 1)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["lemma"], "君")
        self.assertEqual(entries[0]["category"], "kunyomi")
        self.assertEqual(entries[0]["japaneseMeaning"]["status"], "found")


class JapaneseMeaningLookupTests(unittest.TestCase):
    """
    Milestone 6: English-meaning lookup via JMdict, run for every word regardless of on'yomi/
    kun'yomi classification - unlike lookupChineseCognate(), which only ever fires for
    onyomi/mixed. This is what makes a kunyomi word (no Chinese shortcut at all) actually
    useful, and the motivating real example was 離す, easily mistaken for "to leave" when it
    actually means "to separate/let go of" - a different word (離れる) means "to leave".
    """

    def test_kunyomi_word_gets_a_real_meaning(self):
        # 離す (hanasu) - a native Japanese kunyomi verb with zero Chinese-cognate coverage.
        result = lookupJapaneseMeaning("離す", "はなす")
        self.assertEqual(result["status"], "found")
        self.assertIn("to separate", result["gloss"])

    def test_homophone_same_reading_different_kanji_resolves_correctly(self):
        # 離す and 話す are both read はなす but mean completely different things ("to separate"
        # vs "to speak") - looking up by (kanji, reading) together, not reading alone, must
        # never conflate them.
        hanasu1 = lookupJapaneseMeaning("離す", "はなす")
        hanasu2 = lookupJapaneseMeaning("話す", "はなす")
        self.assertIn("to separate", hanasu1["gloss"])
        self.assertTrue(any("speak" in g or "talk" in g for g in hanasu2["gloss"]))
        self.assertNotEqual(hanasu1["gloss"], hanasu2["gloss"])

    def test_same_kanji_different_reading_gives_different_meaning(self):
        # 湯 read ゆ ("hot water", the everyday Japanese sense) vs the same kanji read タン
        # ("soup", borrowed specifically for the Chinese sense) - real motivating example for
        # why the "Japanese Meaning" column exists at all (気分/湯 can differ from the Chinese
        # meaning even when the character is shared).
        yu = lookupJapaneseMeaning("湯", "ゆ")
        tan = lookupJapaneseMeaning("湯", "タン")
        self.assertEqual(yu["status"], "found")
        self.assertEqual(tan["status"], "found")
        self.assertIn("hot water", yu["gloss"])
        self.assertIn("soup", tan["gloss"])

    def test_onyomi_word_also_gets_a_japanese_meaning(self):
        # Unlike chineseCognate, japaneseMeaning is NOT gated by category - an onyomi word gets
        # both a Chinese cognate AND its own Japanese meaning.
        result = lookupJapaneseMeaning("時間", "じかん")
        self.assertEqual(result["status"], "found")
        self.assertIn("time", result["gloss"])

    def test_unknown_kanji_spelling_reports_not_found(self):
        result = lookupJapaneseMeaning("XYZ_never_in_jmdict", "なし")
        self.assertEqual(result["status"], "not_found")

    def test_reading_mismatch_falls_back_to_first_entry_rather_than_crashing(self):
        # If fugashi's reading doesn't exactly match any JMdict entry for the kanji spelling,
        # fall back to the first entry rather than reporting not_found outright.
        result = lookupJapaneseMeaning("時間", "でたらめなよみ")
        self.assertEqual(result["status"], "found")

    def test_analyze_selection_populates_japanese_meaning_for_every_category(self):
        # 出会った (kunyomi, via lemma 出会う) - must get a real meaning despite having no
        # Chinese cognate at all (chineseCognate is None for kunyomi).
        entries = analyzeSelection("出会った瞬間から", 0, 3)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["chineseCognate"])
        self.assertEqual(entries[0]["japaneseMeaning"]["status"], "found")

        # 今日 (jukujigo) - also no Chinese cognate, but still gets a Japanese meaning.
        entries = analyzeSelection("今日はいい天気", 0, 2)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["chineseCognate"])
        self.assertEqual(entries[0]["japaneseMeaning"]["status"], "found")


class KanaOnlyWordIntakeTests(unittest.TestCase):
    """
    Real gap found via a live-vocab audit: analyzeSelection() used to keep Kanji-containing words
    only, which silently dropped every kana-only CONTENT word (ちょっと, とても, これ, ...) from
    vocab intake - not just particles/punctuation, which was the only exclusion actually intended.
    Fixed via core.kanji_reference.isContentWord()/_isKatakanaOnly() (Milestone 5 prerequisite,
    .claude/FLASHCARD_WEB_UPGRADE_PLAN.md).
    """

    def test_kana_only_adverb_is_now_captured(self):
        # ちょっと has no Kanji in its surface at all, but is a real, glossable adverb (citation
        # lemma 一寸) - must no longer be silently dropped.
        entries = analyzeSelection("ちょっと待って", 0, 4)
        chotto = next(e for e in entries if e["surface"] == "ちょっと")
        self.assertEqual(chotto["japaneseMeaning"]["status"], "found")

    def test_katakana_only_loanword_still_excluded(self):
        # コーヒー ("coffee") is a real content word by POS, but a katakana-only loanword -
        # deliberately still excluded (see isContentWord()'s docstring: Kanji-presence is
        # irrelevant to content-word status, but katakana-only surfaces stay out of scope here).
        entries = analyzeSelection("コーヒーを飲む", 0, 4)
        self.assertNotIn("コーヒー", [e["surface"] for e in entries])

    def test_particle_still_excluded(self):
        # は (topic marker) must not suddenly start being captured as "vocab" - the content/
        # function distinction still excludes real particles exactly as before.
        entries = analyzeSelection("これは本です", 0, 6)
        self.assertNotIn("は", [e["surface"] for e in entries])

    def test_kana_only_word_never_fabricates_a_chinese_cognate(self):
        # A kana-only word must classify as something other than onyomi/mixed (see
        # classifyReading()'s kanji-required base case) so chineseCognate stays None - never a
        # spurious "Chinese cognate" pasted together from bare kana.
        entries = analyzeSelection("とても嬉しい", 0, 3)
        totemo = next(e for e in entries if e["surface"] == "とても")
        self.assertIsNone(totemo["chineseCognate"])


class MandarinPinyinMnemonicTests(unittest.TestCase):
    """
    Milestone 7: Mandarin pinyin shown for EVERY word (not gated by category), as a pure
    phonetic memorization aid - distinct from chineseCognate, which stays gated to onyomi/mixed
    and makes a real lexical-attestation claim. Motivating real examples from the user: 夢
    (kunyomi "yume", but recalling Mandarin "meng4" via the shared character is a useful memory
    hook) and 履く (kunyomi "haku", "to put on footwear" - the character 履 means "shoe" in
    Chinese, an "archaic Chinese preserved in Japanese" case).
    """

    def test_kunyomi_word_gets_pinyin_even_with_no_chinese_cognate(self):
        # 夢 (yume, "dream") - kunyomi, so chineseCognate is None, but the character's own
        # Mandarin pinyin should still be shown as a memorization aid.
        entries = analyzeSelection("夢を見る", 0, 1)
        self.assertEqual(len(entries), 1)
        self.assertIsNone(entries[0]["chineseCognate"])
        self.assertEqual(entries[0]["mandarinPinyin"], {"traditional": "夢", "pinyin": "meng4"})

    def test_onyomi_word_reuses_the_real_cedict_pinyin_not_a_fresh_guess(self):
        # For an attested onyomi word, mandarinPinyin should reuse chineseCognate's own real
        # CC-CEDICT pinyin exactly, not recompute a naive per-character concatenation that could
        # in principle differ (tone sandhi, etc.) from the real attested reading.
        entries = analyzeSelection("時間が経つ", 0, 2)
        self.assertEqual(len(entries), 1)
        cognate = entries[0]["chineseCognate"]
        self.assertEqual(cognate["status"], "confirmed")
        self.assertEqual(entries[0]["mandarinPinyin"]["pinyin"], cognate["pinyin"])

    def test_not_attested_onyomi_word_reuses_the_constructed_fallback(self):
        entries = analyzeSelection("残業する", 0, 2)
        self.assertEqual(len(entries), 1)
        cognate = entries[0]["chineseCognate"]
        self.assertEqual(cognate["status"], "not_attested")
        self.assertEqual(entries[0]["mandarinPinyin"]["pinyin"], cognate["pinyinFallback"])

    def test_umlaut_u_colon_encoding_is_normalized_to_the_real_character(self):
        # Real bug found while testing 履 (lu:3 in the raw KANJIDIC2/CC-CEDICT source encoding) -
        # both sources encode u-with-umlaut as literal "u:", which must render as "ü".
        self.assertEqual(_normalizePinyin("lu:3"), "lü3")
        self.assertEqual(_normalizePinyin("nu:3"), "nü3")
        self.assertEqual(_normalizePinyin("hui4"), "hui4")  # no umlaut present - unchanged

    def test_haku_gets_the_real_shoe_character_pinyin(self):
        # 履く (haku, "to put on footwear") - kunyomi verb; lemma is 履く but only 履 is Kanji.
        entries = analyzeSelection("靴を履く", 0, 3)
        haku = next(e for e in entries if "履" in e["lemma"])
        self.assertEqual(haku["mandarinPinyin"]["pinyin"], "lü3")

    def test_get_mandarin_pinyin_reuses_cognate_when_given(self):
        cognate = {"status": "confirmed", "traditional": "瞬間", "pinyin": "shun4 jian1", "gloss": ["moment"]}
        result = getMandarinPinyin("瞬間", cognate)
        self.assertEqual(result, {"traditional": "瞬間", "pinyin": "shun4 jian1"})

    def test_get_mandarin_pinyin_computes_fresh_when_cognate_is_none(self):
        result = getMandarinPinyin("夢", None)
        self.assertEqual(result, {"traditional": "夢", "pinyin": "meng4"})


if __name__ == "__main__":
    unittest.main()
