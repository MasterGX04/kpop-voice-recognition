"""
Tests for core/label_runs.py. Pure functions, no disk I/O, so no tempdir needed unlike most of this
project's other core/test_*.py files.

Fixtures below are small synthetic labels/lyrics arrays modeled on real patterns found by scanning
every saved_labels/*_labels.json + *_lyrics.json pair in this repo (not copies of the live data
itself, which can change - see core/label_runs.py's module docstring for the empirical findings
that shaped this design):
    - TWICE "Doughnut", Tzuyu ~chunk 2456: one written line, sung across 3 pause-separated rows.
    - BTS "Stay Gold", J-Hope ~chunk 2635: two written lines back-to-back with barely a gap and an
      interleaved backing row from another member.
    - BTS "Stay Gold", Jungkook/RM/Jimin: overlapping/nested same-member rows (harmony doubling).
    - BTS "Stay Gold", Jimin ~chunk 4828: a co-sung tag line ("Stay Gold", Jungkook+Jimin together)
      whose linkedLabel.member ("Jimin") doesn't match the row's actual tag ("Jungkook") - the
      label lane only credits one member per row, so a lyric can legitimately point at a span
      tagged to its duet partner instead.

Run with: python -m unittest core.test_label_runs -v
"""

import unittest

from core.label_runs import (resolveLyricSpans, findBrokenLinks, inferLyricSpans, resolveAllSpans,
                             MAX_UNBOUNDED_CLIP_CHUNKS)


def _lyric(lyricId, member=None, start=None, end=None, endChunkOverride=None):
    linked = None
    if member is not None:
        linked = {"member": member, "startChunk": start, "endChunk": end}
    entry = {"lyricId": lyricId, "linkedLabel": linked}
    if endChunkOverride is not None:
        entry["endChunkOverride"] = endChunkOverride
    return entry


class ResolveLyricSpansTests(unittest.TestCase):
    def test_long_bounded_span_is_not_capped(self):
        # Doughnut/Sana shape: 5 rows, ~14 s total, bounded by the next linked lyric.
        labels = [
            ["Sana", 922, 936, False, False], ["Sana", 941, 1045, False, False],
            ["Sana", 1070, 1233, False, False], ["Sana", 1235, 1251, False, False],
            ["Sana", 1253, 1289, False, False], ["Dahyun", 1270, 1330, False, False],
        ]
        lyrics = [_lyric("sana", "Sana", 922, 936), _lyric("dahyun", "Dahyun", 1270, 1330)]
        self.assertEqual(resolveLyricSpans(labels, lyrics)["sana"], (922, 1289))

    def test_unbounded_last_lyric_is_capped(self):
        labels = [["A", 100, 110, False, False], ["A", 120, 5000, False, False]]
        start, end = resolveLyricSpans(labels, [_lyric("a", "A", 100, 110)])["a"]
        self.assertEqual(end - start, MAX_UNBOUNDED_CLIP_CHUNKS)

    def test_end_override_beats_cap(self):
        labels = [["A", 100, 110, False, False]]
        lyrics = [_lyric("a", "A", 100, 110, endChunkOverride=2000)]
        self.assertEqual(resolveLyricSpans(labels, lyrics)["a"], (100, 2000))

    def test_merges_pause_separated_rows_for_same_member(self):
        # Doughnut/Tzuyu shape: one line, three rows, real pauses in between, nothing else claims
        # rows 2 or 3, next linked lyric is a different member much further down.
        labels = [
            ["Tzuyu", 2456, 2473, False, False],
            ["Tzuyu", 2478, 2579, False, False],
            ["Tzuyu", 2605, 2828, False, False],
            ["Chaeyoung", 2806, 2867, False, False],
        ]
        lyrics = [
            _lyric("tzuyu-line", "Tzuyu", 2456, 2473),
            _lyric("chaeyoung-line", "Chaeyoung", 2806, 2867),
        ]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertEqual(spans["tzuyu-line"], (2456, 2828))

    def test_stops_at_next_linked_lyrics_row_despite_no_gap(self):
        # Stay Gold/J-Hope shape: two written lines, back-to-back, with a backing row from another
        # member sitting in the middle of the run. The first card must extend through the second
        # J-Hope row but stop before the third (which the next card claims).
        labels = [
            ["J-Hope", 2635, 2677, False, False],
            ["Jungkook", 2658, 2677, True, False],
            ["J-Hope", 2682, 2722, False, False],
            ["Jungkook", 2704, 2722, True, False],
            ["J-Hope", 2727, 2996, False, False],
        ]
        lyrics = [
            _lyric("jhope-line-1", "J-Hope", 2635, 2677),
            _lyric("jhope-line-2", "J-Hope", 2727, 2996),
        ]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertEqual(spans["jhope-line-1"], (2635, 2722))
        self.assertEqual(spans["jhope-line-2"], (2727, 2996))

    def test_overlapping_nested_same_member_row_does_not_shrink_end(self):
        # RM/Hooligan and Jungkook/Crystal Snow shape: a shorter row nested inside the previous
        # one (backing/harmony double-track). Merge must take max(end, row.end), not overwrite.
        labels = [
            ["RM", 1415, 1468, False, False],
            ["RM", 1425, 1435, False, True],
        ]
        lyrics = [_lyric("rm-line", "RM", 1415, 1468)]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertEqual(spans["rm-line"], (1415, 1468))

    def test_last_linked_lyric_consumes_remaining_same_member_rows(self):
        # No next linked lyric to bound against - should merge through to the end of the member's
        # own rows, skipping any interleaved other-member row, rather than stopping at row 1.
        labels = [
            ["Tzuyu", 5314, 5343, False, False],
            ["Nayeon", 5330, 5360, False, False],
            ["Tzuyu", 5360, 5400, False, False],
        ]
        lyrics = [_lyric("tzuyu-last", "Tzuyu", 5314, 5343)]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertEqual(spans["tzuyu-last"], (5314, 5400))

    def test_lyric_without_linked_label_is_omitted(self):
        labels = [["Tzuyu", 100, 200, False, False]]
        lyrics = [_lyric("unlinked")]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertNotIn("unlinked", spans)

    def test_co_sung_line_credited_to_duet_partner_resolves_via_unambiguous_span(self):
        # Stay Gold shape: Jimin's main line, then a "Stay Gold" tag line he co-sings with
        # Jungkook. Only Jungkook got a row for that stretch (label lane credits one member per
        # row), but the lyric's own linkedLabel.member says "Jimin" since that's who the card is
        # tracking. The span (4828, 4888) is unique in the file, so it should resolve anyway,
        # walking forward as "Jungkook" (the row's real tag) from that point on.
        labels = [
            ["Jimin", 4793, 4890, False, False],
            ["Jungkook", 4828, 4888, False, False],
            ["Jin", 4885, 4931, False, False],
        ]
        lyrics = [
            _lyric("jimin-main-line", "Jimin", 4793, 4890),
            _lyric("stay-gold-tag", "Jimin", 4828, 4888),
        ]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertIn("stay-gold-tag", spans)
        self.assertNotIn("stay-gold-tag", findBrokenLinks(labels, lyrics))
        self.assertEqual(spans["stay-gold-tag"], (4828, 4888))

    def test_duet_pair_with_both_members_tagged_is_not_confused(self):
        # Real duet rows (~150+ in this dataset) where BOTH members have their own row at the
        # identical span, e.g. Stay Gold (4107, 4130) tagged both "Jimin" and "V". The exact
        # (member, span) match must win here - never fall through to the span-only guess, which
        # could otherwise pick the wrong singer and merge along their unrelated later rows.
        labels = [
            ["Jimin", 4107, 4130, False, False],
            ["V", 4107, 4130, False, False],
            ["Jimin", 4135, 4180, False, False],
            ["V", 4200, 4260, False, False],
        ]
        lyrics = [_lyric("jimin-duet-line", "Jimin", 4107, 4130)]
        spans = resolveLyricSpans(labels, lyrics)
        # Must walk forward as Jimin (extending into 4135-4180), not as V (which would reach 4260).
        self.assertEqual(spans["jimin-duet-line"], (4107, 4180))

    def test_end_chunk_override_replaces_computed_end_without_affecting_neighbors(self):
        # Hand-fix for whenever the heuristic guesses wrong: overriding one line's end must not
        # change where the FOLLOWING line thinks it starts (still anchored on its own linkedLabel).
        labels = [
            ["Tzuyu", 2456, 2473, False, False],
            ["Tzuyu", 2478, 2579, False, False],
            ["Tzuyu", 2605, 2828, False, False],
            ["Chaeyoung", 2806, 2867, False, False],
        ]
        lyrics = [
            _lyric("tzuyu-line", "Tzuyu", 2456, 2473, endChunkOverride=2579),
            _lyric("chaeyoung-line", "Chaeyoung", 2806, 2867),
        ]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertEqual(spans["tzuyu-line"], (2456, 2579))
        self.assertEqual(spans["chaeyoung-line"], (2806, 2867))

    def test_ambiguous_span_with_no_member_match_is_left_broken(self):
        # If the exact span is shared by two DIFFERENT members and neither matches the lyric's own
        # linkedLabel.member, there's no safe unambiguous fallback - guessing risks merging along
        # the wrong singer's rows, so this must stay unresolved rather than picking one.
        labels = [
            ["Jimin", 4107, 4130, False, False],
            ["V", 4107, 4130, False, False],
        ]
        lyrics = [_lyric("mismatched", "Suga", 4107, 4130)]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertNotIn("mismatched", spans)
        self.assertEqual(findBrokenLinks(labels, lyrics), ["mismatched"])

    def test_single_row_line_with_no_pauses_is_unaffected(self):
        labels = [
            ["J-Hope", 2727, 2996, False, False],
            ["Jungkook", 3010, 3050, False, False],
        ]
        lyrics = [
            _lyric("jhope-solo", "J-Hope", 2727, 2996),
            _lyric("jungkook-next", "Jungkook", 3010, 3050),
        ]
        spans = resolveLyricSpans(labels, lyrics)
        self.assertEqual(spans["jhope-solo"], (2727, 2996))


def _plain(member, start, isAdLib=False, adLibDuration=0):
    return {"memberName": [member], "startChunk": start, "isAdLib": isAdLib, "adLibDuration": adLibDuration}


class InferLyricSpansTests(unittest.TestCase):
    def test_lyric_maps_to_row_starting_eleven_chunks_later(self):
        labels = [["Suga", 111, 200, False, False]]
        self.assertEqual(inferLyricSpans(labels, [_plain("Suga", 100)]), {0: (111, 200)})

    def test_nine_chunk_lead_in_is_accepted(self):
        labels = [["Athena", 109, 180, False, False]]
        self.assertEqual(inferLyricSpans(labels, [_plain("Athena", 100)]), {0: (109, 180)})

    def test_wrong_member_row_is_not_matched(self):
        labels = [["V", 111, 200, False, False]]
        self.assertEqual(inferLyricSpans(labels, [_plain("Suga", 100)]), {})

    def test_split_lyric_boundaries_come_from_the_lyrics_themselves(self):
        # One long row, written as three lyric cards; cards 2 and 3 start mid-row (BTS pattern).
        labels = [["RM", 111, 400, False, False]]
        lyrics = [_plain("RM", 100), _plain("RM", 189), _plain("RM", 289)]
        self.assertEqual(
            inferLyricSpans(labels, lyrics),
            {0: (111, 200), 1: (200, 300), 2: (300, 400)},
        )

    def test_pause_separated_rows_merge_until_next_lyric(self):
        labels = [["Tzuyu", 111, 150, False, False], ["Tzuyu", 160, 200, False, False],
                  ["Momo", 311, 350, False, False]]
        lyrics = [_plain("Tzuyu", 100), _plain("Momo", 300)]
        self.assertEqual(inferLyricSpans(labels, lyrics)[0], (111, 200))

    def test_other_members_rows_are_not_merged(self):
        labels = [["A", 111, 150, False, False], ["B", 155, 190, False, False],
                  ["A", 165, 220, False, False]]
        self.assertEqual(inferLyricSpans(labels, [_plain("A", 100)])[0], (111, 220))

    def test_adlib_row_is_never_a_match(self):
        labels = [["Rei", 111, 200, False, True]]
        self.assertEqual(inferLyricSpans(labels, [_plain("Rei", 100)]), {})

    def test_adlib_uses_start_and_duration_with_no_lead_in(self):
        labels = [["Suga", 121, 154, False, False]]
        lyrics = [_plain("Suga", 132, isAdLib=True, adLibDuration=50)]
        self.assertEqual(inferLyricSpans(labels, lyrics), {0: (132, 182)})

    def test_off_window_lead_in_falls_back_to_nearest_same_member_row(self):
        labels = [["Wonyoung", 117, 260, False, False]]  # 17 chunks - hand-timing slop
        self.assertEqual(inferLyricSpans(labels, [_plain("Wonyoung", 100)]), {0: (117, 260)})

    def test_unbounded_span_length_is_capped(self):
        # Short anchor row + a same-member row 200 chunks later-merged forward (gap rule allows a
        # chain of close rows); only the merged extension is capped, never the anchor row itself.
        labels = [["A", 111, 130, False, False]] + [["A", 131 + 20 * i, 146 + 20 * i, False, False] for i in range(60)]
        start, end = inferLyricSpans(labels, [_plain("A", 100)])[0]
        self.assertEqual(end - start, MAX_UNBOUNDED_CLIP_CHUNKS)

    def test_unbounded_anchor_row_longer_than_cap_is_not_cut(self):
        labels = [["A", 111, 5000, False, False]]
        self.assertEqual(inferLyricSpans(labels, [_plain("A", 100)])[0], (111, 5000))

    def test_bounded_long_span_is_not_capped(self):
        # 20 s line with a next lyric after it: the next lyric's start bounds it, not a blanket cap.
        labels = [["A", 111, 900, False, False], ["B", 950, 1000, False, False]]
        lyrics = [_plain("A", 100), _plain("B", 939)]
        self.assertEqual(inferLyricSpans(labels, lyrics)[0], (111, 900))

    def test_per_song_nine_chunk_lead_drives_mid_row_split(self):
        # Song's own lead-in is 9 (first lyric proves it), so the mid-row card starts at start+9.
        labels = [["A", 109, 400, False, False]]
        lyrics = [_plain("A", 100), _plain("A", 191)]
        self.assertEqual(inferLyricSpans(labels, lyrics), {0: (109, 200), 1: (200, 400)})

    def test_fill_unresolved_gives_adlib_row_lyric_a_default_clip(self):
        labels = [["Rei", 111, 200, False, True]]  # ad-lib row -> no match
        result = inferLyricSpans(labels, [_plain("Rei", 100)], fillUnresolved=True)
        self.assertEqual(result, {0: (111, 311)})

    def test_fill_unresolved_stops_at_next_lyric(self):
        labels = [["Rei", 111, 200, False, True]]
        lyrics = [_plain("Rei", 100), _plain("Rei", 150)]
        self.assertEqual(inferLyricSpans(labels, lyrics, fillUnresolved=True)[0], (111, 161))

    def test_fill_unresolved_is_off_by_default_and_never_overrides_a_match(self):
        labels = [["Suga", 111, 200, False, False], ["Rei", 311, 400, False, True]]
        lyrics = [_plain("Suga", 100), _plain("Rei", 300)]
        self.assertNotIn(1, inferLyricSpans(labels, lyrics))
        filled = inferLyricSpans(labels, lyrics, fillUnresolved=True)
        self.assertEqual(filled[0], (111, 200))
        self.assertIn(1, filled)


class RowFieldSemanticsTests(unittest.TestCase):
    """Label row = [member, start, end, isBacking, isAdLib]. Backing rows are real singing; only
    ad-lib rows are excluded (this used to be inverted)."""

    def test_backing_row_of_same_member_extends_the_span(self):
        labels = [["A", 100, 150, False, False], ["A", 155, 220, True, False]]
        self.assertEqual(resolveLyricSpans(labels, [_lyric("a", "A", 100, 150)])["a"], (100, 220))

    def test_adlib_row_of_same_member_does_not_extend_the_span(self):
        labels = [["A", 100, 150, False, False], ["A", 400, 420, False, True]]
        self.assertEqual(resolveLyricSpans(labels, [_lyric("a", "A", 100, 150)])["a"], (100, 150))

    def test_row_tied_with_next_cards_start_is_not_merged(self):
        # Film out shape: the previous singer's backing row starts exactly where the next card's row
        # does but sorts before it, so a row-index bound alone would swallow it.
        labels = [["Jimin", 100, 150, False, False], ["Jimin", 200, 280, True, False],
                  ["Jin", 200, 280, False, False]]
        lyrics = [_lyric("jimin", "Jimin", 100, 150), _lyric("jin", "Jin", 200, 280)]
        self.assertEqual(resolveLyricSpans(labels, lyrics)["jimin"], (100, 150))


class OverlappingSingerTests(unittest.TestCase):
    def test_different_singer_starting_inside_the_line_does_not_cut_it(self):
        labels = [["A", 111, 300, False, False], ["B", 261, 400, False, False]]
        lyrics = [_plain("A", 100), _plain("B", 250)]
        spans = inferLyricSpans(labels, lyrics)
        self.assertEqual(spans[0], (111, 300))     # B starts at 261 but A's row runs to 300
        self.assertEqual(spans[1], (261, 400))

    def test_same_singer_starting_mid_row_still_ends_the_previous_card(self):
        labels = [["A", 111, 400, False, False]]
        self.assertEqual(inferLyricSpans(labels, [_plain("A", 100), _plain("A", 189)])[0], (111, 200))

    def test_row_that_starts_after_the_next_card_is_still_not_merged(self):
        labels = [["A", 111, 150, False, False], ["B", 211, 260, False, False], ["A", 300, 350, False, False]]
        lyrics = [_plain("A", 100), _plain("B", 200)]
        self.assertEqual(inferLyricSpans(labels, lyrics)[0], (111, 150))

    def test_resolve_all_other_singers_unresolved_card_does_not_clip_a_linked_span(self):
        labels = [["A", 100, 150, False, False], ["A", 160, 300, False, False], ["B", 250, 400, False, False]]
        a = {"lyricId": "a", "startChunk": 89, "memberName": ["A"],
             "linkedLabel": {"member": "A", "startChunk": 100, "endChunk": 150}}
        b = {"lyricId": "b", "startChunk": 239, "memberName": ["B"],
             "linkedLabel": {"member": "B", "startChunk": 999, "endChunk": 1000}}   # broken link
        self.assertEqual(resolveAllSpans(labels, [a, b])[0], (100, 300))


class ResolveAllSpansTests(unittest.TestCase):
    # BTS "Stay Gold" J-Hope: card 20 (3 lines = 3 rows) is followed by card 21, whose linkedLabel
    # (2727, 2996) matches NO row (the real rows are 2727-2832 and 2886-2996), i.e. a broken link.
    LABELS = [
        ["J-Hope", 2635, 2677, False, False], ["Jungkook", 2658, 2677, True, False],
        ["J-Hope", 2682, 2722, False, False], ["Jungkook", 2704, 2722, True, False],
        ["J-Hope", 2727, 2832, False, False], ["Jungkook", 2752, 2832, True, False],
        ["J-Hope", 2832, 2837, False, True], ["J-Hope", 2838, 2880, False, False],
        ["Jungkook", 2838, 2880, True, False], ["J-Hope", 2879, 2888, False, True],
        ["Jungkook", 2886, 2992, True, False], ["J-Hope", 2886, 2996, False, False],
        ["Jungkook", 2992, 3039, False, False],
    ]

    def _lyrics(self):
        card20 = {"lyricId": "c20", "memberName": ["J-Hope"], "startChunk": 2624, "isAdLib": False,
                  "linkedLabel": {"member": "J-Hope", "startChunk": 2635, "endChunk": 2677}}
        card21 = {"lyricId": "c21", "memberName": ["J-Hope", "Jungkook"], "startChunk": 2787, "isAdLib": False,
                  "linkedLabel": {"member": "J-Hope", "startChunk": 2727, "endChunk": 2996}}
        card22 = {"lyricId": "c22", "memberName": ["Jungkook"], "startChunk": 2981, "isAdLib": False,
                  "linkedLabel": {"member": "Jungkook", "startChunk": 2992, "endChunk": 3039}}
        return [card20, card21, card22]

    def test_broken_link_uses_lead_in_not_the_stale_snapshot(self):
        spans = resolveAllSpans(self.LABELS, self._lyrics())
        self.assertEqual(spans[1][0], 2787 + 11)          # not the stale 2727

    def test_broken_link_card_bounds_the_previous_card(self):
        spans = resolveAllSpans(self.LABELS, self._lyrics())
        # was 2635-2996, swallowing card 21. Card 21 starts mid-row on purpose (its sung start 2798 =
        # startChunk 2787 + 11, inside the 2727-2832 row), so card 20 ends exactly there.
        self.assertEqual(spans[0], (2635, 2798))
        self.assertLessEqual(spans[0][1], spans[1][0])

    def test_broken_link_card_plays_its_whole_last_row_past_another_singers_start(self):
        spans = resolveAllSpans(self.LABELS, self._lyrics())
        # Jungkook's card 22 starts at 2992, over the last 4 chunks of J-Hope's 'now...' (row ends 2996):
        # a different singer starting never cuts the line, so card 21 plays all the way to 2996.
        self.assertEqual(spans[1], (2798, 2996))

    def test_adlib_card_inside_a_line_does_not_clip_it(self):
        lyrics = self._lyrics()
        lyrics.insert(1, {"lyricId": None, "memberName": ["J-Hope"], "startChunk": 2700,
                          "isAdLib": True, "adLibDuration": 20})
        self.assertEqual(resolveAllSpans(self.LABELS, lyrics)[0][1], 2798)  # ad-lib at 2700 would have clipped it to 2700

    def test_end_override_is_not_clipped(self):
        lyrics = self._lyrics()
        lyrics[0]["endChunkOverride"] = 2990
        self.assertEqual(resolveAllSpans(self.LABELS, lyrics)[0], (2635, 2990))

    def test_linked_card_starts_at_start_chunk_plus_lead_not_the_row_start(self):
        # Stay Gold 36 shape: the card's link points 36 chunks before its own sung start.
        labels = [["V", 4385, 4412, False, False], ["V", 4421, 4480, False, False]]
        lyric = {"lyricId": "x", "startChunk": 4410, "memberName": ["V"],
                 "linkedLabel": {"member": "V", "startChunk": 4385, "endChunk": 4412}}
        self.assertEqual(resolveAllSpans(labels, [lyric])[0][0], 4410 + 11)

    def test_adlib_card_start_gets_no_lead(self):
        labels = [["A", 100, 150, False, False]]
        lyric = {"lyricId": "x", "startChunk": 100, "memberName": ["A"], "isAdLib": True,
                 "adLibDuration": 20, "linkedLabel": {"member": "A", "startChunk": 100, "endChunk": 150}}
        self.assertEqual(resolveAllSpans(labels, [lyric])[0][0], 100)

    def test_fully_linked_song_is_unchanged(self):
        labels = [["A", 100, 150, False, False], ["B", 300, 400, False, False]]
        lyrics = [{"lyricId": "a", "startChunk": 89, "memberName": ["A"],
                   "linkedLabel": {"member": "A", "startChunk": 100, "endChunk": 150}},
                  {"lyricId": "b", "startChunk": 289, "memberName": ["B"],
                   "linkedLabel": {"member": "B", "startChunk": 300, "endChunk": 400}}]
        self.assertEqual(resolveAllSpans(labels, lyrics), {0: (100, 150), 1: (300, 400)})



class AdLibInterruptionTests(unittest.TestCase):
    # TWICE "Funny Valentine", Sana ~chunk 518: her last phrase (698-754) comes after Mina's "tick tock" ad-lib,
    # 27 chunks past her 640-671 row - just over the merge gap limit - and was cut off.
    LABELS = [
        ["Sana", 529, 584, False, False], ["Sana", 586, 613, False, False], ["Sana", 615, 638, False, False],
        ["Sana", 640, 671, False, False], ["Mina", 666, 698, False, False], ["Sana", 698, 754, False, False],
        ["Mina", 748, 840, False, False],
    ]

    def _lyrics(self, adLib):
        lyrics = [
            {"startChunk": 518, "memberName": ["Sana"], "isAdLib": False},
            {"startChunk": 655, "memberName": ["Mina"], "isAdLib": True, "adLibDuration": 50 if adLib else 0},
            {"startChunk": 737, "memberName": ["Mina"], "isAdLib": False},
        ]
        return lyrics

    def test_line_continues_past_an_adlib_that_covers_the_gap(self):
        self.assertEqual(inferLyricSpans(self.LABELS, self._lyrics(True))[0], (529, 754))

    def test_same_gap_without_an_adlib_is_still_a_break(self):
        self.assertEqual(inferLyricSpans(self.LABELS, self._lyrics(False))[0], (529, 671))

    def test_adlib_that_does_not_cover_the_gap_does_not_bridge_it(self):
        lyrics = self._lyrics(True)
        lyrics[1]["adLibDuration"] = 30          # 655-685 stops short of the next Sana row at 698
        self.assertEqual(inferLyricSpans(self.LABELS, lyrics)[0], (529, 671))


if __name__ == "__main__":
    unittest.main()
