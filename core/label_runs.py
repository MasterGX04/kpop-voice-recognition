"""
Resolve a linked lyric's real audio span - the whole singer's part, not just the one label row
`linkedLabel` happens to snapshot. See .claude/FLASHCARD_WEB_UPGRADE_PLAN.md karaoke-line-boundary
discussion.

Problem: a lyric's `linkedLabel` (gui/lyrics_editor.py) is a snapshot of exactly one row in
`<song>_labels.json`. A singer pausing mid-line splits what's conceptually one sung line into
several consecutive label rows for the same member - only the first of which the lyric remembers -
so naive playback of `linkedLabel.startChunk..endChunk` cuts off after the first sub-phrase.
Conversely a continuous rap can span one label row that several separate lyric cards all sit inside
(BTS-style), so "always merge to the end of the member's rows" would over-extend and swallow the
next card's audio too.

Empirically (see conversation/analysis, not restated here), pause length does NOT distinguish a
mid-line breath from a real line break - real boundaries are sometimes *shorter* than mid-line
pauses. The only reliable boundary signal already in the data is: where the *next* linked lyric's
`linkedLabel` claims a row starts. So: merge same-member label rows forward from a lyric's own
matched row, skipping other members' rows and ad-lib rows (not lyric text), stopping the instant
we'd reach the row the next linked lyric claims.

A run-scan across every saved_labels/*_labels.json + *_lyrics.json pair in this repo confirmed:
- label arrays are chronologically sorted by startChunk (no defensive re-sort needed here).
- linked-lyric order by `linkedLabel.startChunk` always matches order by the lyric's own display
  `startChunk` (no defensive re-sort needed there either).
- ~32 songs have overlapping/nested same-member rows (backing/harmony double-tracked at an
  overlapping timestamp) - merging must take max(end, row.end), never blindly overwrite, or a
  nested row would shrink the tracked end.
- one lyric (Stay Gold, linkedLabel.member="Jimin", span (4828, 4888)) has no row matching that
  exact (member, span) - but the span itself is real, tagged "Jungkook" (row 116). This is BTS
  co-singing a tag line together (confirmed intended by the user) where the label lane can only
  credit one member per row; the lyric's snapshot remembers the *other* singer. This dataset has
  ~150+ places where two different members share an identical (start, end) span (real duet-pair
  rows, e.g. Stay Gold (4107, 4130) tagged both "Jimin" and "V"), so blindly matching span alone
  and ignoring member would sometimes grab the wrong singer's row and merge along the wrong
  person's subsequent lines. The fix here: fall back to a span-only match ONLY when exactly one
  row anywhere has that exact span (unambiguous), and then walk forward using THAT row's real
  tagged member, not the lyric's stale one. Only a genuinely unresolvable case (no span match, or
  an ambiguous one shared by multiple members) is left broken.
"""

from core.util_functions import findLabelIndexBySpan


def _resolveAnchor(labels: list, member, start, end):
    """
    Returns (rowIndex, member) for the row a linkedLabel snapshot refers to, or (None, None) if it
    can't be resolved. Tries the exact (member, start, end) match first; falls back to an
    unambiguous span-only match (see module docstring) for the co-sung/duet-credit-mismatch case.
    """
    rowIndex = findLabelIndexBySpan(labels, start, end, member=member)
    if rowIndex is not None:
        return rowIndex, member

    spanMatches = [i for i, row in enumerate(labels) if row[1] == start and row[2] == end]
    if len(spanMatches) == 1:
        rowIndex = spanMatches[0]
        return rowIndex, labels[rowIndex][0]

    return None, None


def resolveLyricSpans(labels: list, lyrics: list) -> dict:
    """
    Returns {lyricId: (startChunk, endChunk)} for every lyric whose `linkedLabel` snapshot resolves
    to a row in `labels` (see _resolveAnchor). Lyrics with no `linkedLabel`, or one that's still
    unresolvable (see findBrokenLinks below), are omitted entirely; callers should fall back to the
    lyric's own bare `startChunk` for those, same as core.vocab_sync._resolveChunks already does.

    A lyric may also carry an `endChunkOverride` (sibling field to `linkedLabel`) to hand-correct
    the rare case this heuristic guesses wrong, without touching the label data itself. It replaces
    the computed end only - the lyric still participates in anchoring/bounding its neighbors
    normally, so overriding one line's end doesn't affect where its neighbors think it starts.
    """
    resolvable = []  # (rowIndex, member, lyricId, endChunkOverride)
    for lyric in lyrics:
        linked = lyric.get("linkedLabel")
        if not linked or linked.get("startChunk") is None:
            continue
        rowIndex, member = _resolveAnchor(
            labels, linked.get("member"), linked.get("startChunk"), linked.get("endChunk")
        )
        if rowIndex is not None:
            resolvable.append((rowIndex, member, lyric.get("lyricId"), lyric.get("endChunkOverride")))

    resolvable.sort(key=lambda entry: entry[0])

    spans = {}
    for i, (rowIndex, member, lyricId, override) in enumerate(resolvable):
        nextRowIndex = resolvable[i + 1][0] if i + 1 < len(resolvable) else None
        # Defensive: sorted + no-duplicate-span invariants (confirmed empirically) mean this
        # shouldn't happen, but never let a same-or-earlier "next" bound produce a corrupt merge.
        if nextRowIndex is not None and nextRowIndex <= rowIndex:
            nextRowIndex = None

        start = labels[rowIndex][1]
        end = labels[rowIndex][2]

        j = rowIndex + 1
        limit = len(labels) if nextRowIndex is None else nextRowIndex
        # Also bound by TIME: a row tied with the next card's row (e.g. a backing row at the same
        # start, sorted just before it) is the next card's audio even though its index is lower.
        nextStartChunk = None if nextRowIndex is None else labels[nextRowIndex][1]
        while j < limit:
            row = labels[j]
            if nextStartChunk is not None and row[1] >= nextStartChunk:
                j += 1
                continue
            if not _isAdLibRow(row) and row[0] == member:
                end = max(end, row[2])
            j += 1

        if override is not None:
            end = override
        elif nextRowIndex is None:
            # Cap only the merged-forward extension; never cut the lyric's own anchor row.
            end = min(end, max(labels[rowIndex][2], start + MAX_UNBOUNDED_CLIP_CHUNKS))
        spans[lyricId] = (start, end)

    return spans


def findBrokenLinks(labels: list, lyrics: list) -> list:
    """
    Diagnostic helper: lyricIds whose `linkedLabel` snapshot is genuinely unresolvable against
    `labels` - no row matches its (member, span) and no unambiguous span-only fallback exists
    either (see _resolveAnchor). Not used by resolveLyricSpans directly, but callers (e.g. a future
    "Compile Vocab" scan) can surface these so a human can re-link them.
    """
    broken = []
    for lyric in lyrics:
        linked = lyric.get("linkedLabel")
        if not linked or linked.get("startChunk") is None:
            continue
        rowIndex, _ = _resolveAnchor(
            labels, linked.get("member"), linked.get("startChunk"), linked.get("endChunk")
        )
        if rowIndex is None:
            broken.append(lyric.get("lyricId"))
    return broken


# ---------------------------------------------------------------------------------------------
# Inferring spans for lyrics that were never hand-linked (no `linkedLabel`).
#
# Empirical rule, from scanning every saved_labels/*_lyrics.json + *_labels.json pair (35 songs,
# 997 non-ad-lib lyrics): a lyric's `startChunk` sits a fixed lead-in BEFORE the audio it
# represents - 758 lyrics at exactly 11 chunks before a same-member label row's start, 123 at 9
# (Fifty Fifty songs), almost nothing in between. Cross-check: of the 221 hand-linked lyrics, 209
# link to a row exactly 11 chunks after their startChunk. So "lyric.startChunk + LEAD" is the real
# sung start, and a same-member row starting there IS the lyric's first row.
#
# Second pattern (~75 lyrics, mostly BTS): NO row starts at startChunk+LEAD, because a member's
# row is longer than one lyric box, so the person entering lyrics split the sung phrase across
# several lyric cards. The later cards' sung start (startChunk+LEAD) falls INSIDE a same-member
# row that an earlier card already claimed. Such a card owns [startChunk+LEAD, ...) up to the next
# lyric's sung start, and the earlier card ends where this one begins. Boundaries between split
# cards therefore come from the lyrics themselves, not from the labels.
#
# Ad-libs: their startChunk is already the sung start (no lead-in) and adLibDuration is the length.
# ---------------------------------------------------------------------------------------------

DEFAULT_LEAD_CHUNKS = 11
_LEAD_WINDOW = range(8, 13)         # observed 9 and 11; tolerate 8..12
_LOOSE_MAX_OFFSET = 20            # fallback search limit for off-window lead-ins
DEFAULT_CLIP_CHUNKS = 200           # 8 s - clip length when a lyric has no usable label row
MAX_CLIP_CHUNKS = 250              # 10 s at 40 ms/chunk - cap on an ad-lib's stated duration only
# Cap for a span with NO next-lyric boundary (the last linked lyric of a song): without a bound,
# same-member rows merge forward for minutes. Spans that ARE bounded by the next lyric are never
# capped - slow ballads / continuous raps legitimately run 10+ s (Doughnut Tzuyu/Sana, BTS JA raps
# were cut off by the old blanket 10 s cap). Tried 30 s for the unbounded case: it over-extended the
# three real last-lyric cases (BTS Film out, BTS Let Go, WJSN Save Me), so it stays at 10 s; use
# endChunkOverride on a lyric that genuinely runs longer.
MAX_UNBOUNDED_CLIP_CHUNKS = 250    # 10 s
_MAX_MERGE_GAP_CHUNKS = 25          # don't merge a same-member row further than ~1 s past the run


def _memberMatches(lyricMembers, rowMember):
    if not lyricMembers or "All" in lyricMembers:
        return True
    return rowMember in lyricMembers


def _asMemberList(memberName):
    if isinstance(memberName, list):
        return memberName
    return [memberName] if memberName else []


def _insideSameMemberRow(labels, chunk, members):
    return any(
        row[1] < chunk < row[2] and not _isAdLibRow(row) and _memberMatches(members, row[0])
        for row in labels
    )


def _carriesOn(member, nextLyric) -> bool:
    """Does `nextLyric` (the card that follows) include `member` among its singers? Unknown or "All"
    singers count as yes (the safe, old behaviour: stop where the next card starts)."""
    singers = _asMemberList(nextLyric.get("memberName"))
    return member is None or not singers or "All" in singers or member in singers


def _isAdLibRow(row):
    # Label row layout is [member, start, end, isBacking, isAdLib] (gui/audio_tester.py). Ad-lib rows
    # ("hn", "yeah" over the top) are not lyric text, so they never anchor or extend a card. Backing
    # rows ARE the member's real singing and count normally. (This used to read row[3] - isBacking -
    # as "isCut", which dropped backing vocals and let ad-libs through.)
    return len(row) > 4 and bool(row[4])


def _inferAnchors(labels: list, lyrics: list, lead: int):
    """Returns (adLibSpans, anchors): each non-ad-lib lyric that matches a row, as
    (sungStart, lyricIndex, member, rowIndex) sorted by sung start, plus ad-lib spans keyed by index.
    `member` is the singer of the row the lyric anchors on - the voice the card is actually about."""
    spans = {}
    anchors = []  # (sungStart, lyricIndex, member, rowIndex) for non-ad-lib resolved lyrics

    for idx, lyric in enumerate(lyrics):
        start = lyric.get("startChunk")
        if start is None:
            continue

        if lyric.get("isAdLib"):
            duration = lyric.get("adLibDuration") or 0
            if duration > 0:
                spans[idx] = (start, start + min(duration, MAX_CLIP_CHUNKS))
            continue

        members = _asMemberList(lyric.get("memberName"))

        # Case 1: a same-member row begins one lead-in after the lyric's startChunk.
        candidates = [
            (abs((row[1] - start) - lead), i, row)
            for i, row in enumerate(labels)
            if (row[1] - start) in _LEAD_WINDOW and not _isAdLibRow(row) and _memberMatches(members, row[0])
        ]
        if candidates:
            _, rowIndex, row = min(candidates, key=lambda c: (c[0], c[1]))
            anchors.append((row[1], idx, row[0], rowIndex))
            continue

        # Case 1b: hand-timing slop - same-member row starting a little outside the observed
        # 8..12 window (seen at 13-19 chunks). Nearest wins.
        loose = [
            (row[1] - start, i, row)
            for i, row in enumerate(labels)
            if _LEAD_WINDOW.stop <= (row[1] - start) <= _LOOSE_MAX_OFFSET
            and not _isAdLibRow(row) and _memberMatches(members, row[0])
        ]
        if loose and not _insideSameMemberRow(labels, start + lead, members):
            _, rowIndex, row = min(loose, key=lambda c: (c[0], c[1]))
            anchors.append((row[1], idx, row[0], rowIndex))
            continue

        # Case 2: split lyric - the expected sung start lands inside a same-member row.
        sungStart = start + lead
        containing = [
            (i, row) for i, row in enumerate(labels)
            if row[1] < sungStart < row[2] and not _isAdLibRow(row) and _memberMatches(members, row[0])
        ]
        if containing:
            rowIndex, row = containing[0]
            anchors.append((sungStart, idx, row[0], rowIndex))

    anchors.sort(key=lambda a: (a[0], a[2]))
    return spans, anchors


def inferLyricSpans(labels: list, lyrics: list, fillUnresolved: bool = False) -> dict:
    """
    Returns {lyricIndex: (startChunk, endChunk)} - keyed by the lyric's position in `lyrics`, since
    unlinked lyrics have no lyricId. Covers lyrics that resolveLyricSpans() can't (no linkedLabel),
    and can be used for linked ones as a cross-check. Lyrics that match nothing are omitted (caller
    keeps its old bare-startChunk fallback).

    See the section comment above for the lead-in rule, split-lyric (mid-row) handling and ad-libs.

    fillUnresolved=True gives every remaining lyric that has a startChunk (no matching row, or it
    points at an ad-lib row, or the member has no labels there) a default clip instead of omitting
    it: [startChunk + lead, +DEFAULT_CLIP_CHUNKS), stopping early where the next lyric's sung start
    begins. Ad-libs without a duration are filled too.
    """
    lead = _detectLead(labels, lyrics)
    spans, anchors = _inferAnchors(labels, lyrics, lead)


    for i, (sungStart, idx, member, rowIndex) in enumerate(anchors):
        nextAnchor = next((a for a in anchors[i + 1:] if a[0] > sungStart), None)
        nextStart = nextAnchor[0] if nextAnchor else None

        end = max(sungStart, labels[rowIndex][2])
        for row in labels[rowIndex + 1:]:
            if nextStart is not None and row[1] >= nextStart:
                break
            if row[1] - end > _MAX_MERGE_GAP_CHUNKS:
                break  # a real break in the singing, not a mid-line breath
            if not _isAdLibRow(row) and row[0] == member:
                end = max(end, row[2])

        # Clip at the next card's start only when this card's singer is carrying on in it (a card that
        # begins mid-row, e.g. Stay Gold J-Hope 20 -> 21, or SWIM Suga -> Suga+Jungkook). Another
        # singer starting over the end of this line never cuts it: the whole line plays, overlap and all.
        if nextAnchor is not None and _carriesOn(member, lyrics[nextAnchor[1]]):
            end = min(end, nextStart)
        if nextStart is None:
            end = min(end, max(labels[rowIndex][2], sungStart + MAX_UNBOUNDED_CLIP_CHUNKS))
        spans[idx] = (sungStart, end)

    if fillUnresolved:
        lyricStarts = sorted(
            l["startChunk"] + (0 if l.get("isAdLib") else lead)
            for l in lyrics if l.get("startChunk") is not None
        )
        for idx, lyric in enumerate(lyrics):
            start = lyric.get("startChunk")
            if start is None or idx in spans:
                continue
            sungStart = start + (0 if lyric.get("isAdLib") else lead)
            nextStart = next((t for t in lyricStarts if t > sungStart), None)
            end = sungStart + DEFAULT_CLIP_CHUNKS
            if nextStart is not None:
                end = min(end, nextStart)
            spans[idx] = (sungStart, end)

    return spans


def _detectLead(labels: list, lyrics: list) -> int:
    """Per-song lead-in: the most common start->row offset among unambiguous matches, else the
    global default. Songs made with a different tool version (9 vs 11) stay self-consistent."""
    counts = {}
    for lyric in lyrics:
        start = lyric.get("startChunk")
        if start is None or lyric.get("isAdLib"):
            continue
        members = _asMemberList(lyric.get("memberName"))
        offsets = {
            row[1] - start for row in labels
            if (row[1] - start) in _LEAD_WINDOW and _memberMatches(members, row[0])
        }
        if len(offsets) == 1:
            offset = offsets.pop()
            counts[offset] = counts.get(offset, 0) + 1
    if not counts:
        return DEFAULT_LEAD_CHUNKS
    return max(counts, key=counts.get)


def resolveAllSpans(labels: list, lyrics: list) -> dict:
    """
    Returns {lyricIndex: (startChunk, endChunk or None)} - the one span a lyric's audio should play,
    for every lyric with a startChunk. Combines the two resolvers above:

    - hand-linked and resolvable -> resolveLyricSpans (the whole merged part) started at startChunk +
      lead (the sung start; the lyric box animates in during the `lead` chunks before), BUT its end is clipped
      to the sung start of the next lyric that is NOT hand-linked-and-resolvable AND includes this card's singer. resolveLyricSpans
      only bounds by other *resolvable* links, so a broken/missing link next door used to let the
      previous card swallow it (Stay Gold: card 20 ran 2635-2996 over all of card 21).
    - hand-linked but BROKEN (no row matches the stored snapshot, i.e. the label was edited after
      linking) -> the inferred span, which applies the lead-in rule (startChunk + 11). The stale
      snapshot is only a last resort; replaying it starts at the wrong place.
    - unlinked -> the inferred span.
    Ad-lib cards never act as a boundary (they sit inside someone else's line by design).
    """
    hard = resolveLyricSpans(labels, lyrics)
    inferred = inferLyricSpans(labels, lyrics, fillUnresolved=True)
    lead = _detectLead(labels, lyrics)

    unresolvedStarts = []  # (sung start, lyric) of every lyric that isn't a resolved hand link
    for idx, lyric in enumerate(lyrics):
        if lyric.get("lyricId") in hard or lyric.get("isAdLib") or lyric.get("startChunk") is None:
            continue
        unresolvedStarts.append((inferred[idx][0] if idx in inferred else lyric["startChunk"] + lead, lyric))
    unresolvedStarts.sort(key=lambda t: t[0])

    spans = {}
    for idx, lyric in enumerate(lyrics):
        start = lyric.get("startChunk")
        if start is None:
            continue
        lyricId = lyric.get("lyricId")
        if lyricId in hard:
            spanStart, spanEnd = hard[lyricId]
            # A card's startChunk is when its lyric box starts animating in; the singer starts
            # `lead` chunks later (11 by default). Start there, not at the linked row's start, which
            # can disagree (a link to a different row, or hand-timing slop). Ad-libs have no lead-in.
            if not lyric.get("isAdLib"):
                if start + lead < spanEnd:
                    spanStart = start + lead
                elif idx in inferred:        # the link points somewhere else entirely
                    spans[idx] = inferred[idx]
                    continue
            member = (lyric.get("linkedLabel") or {}).get("member")
            # Only a card that includes this card's singer can end this span: it starts where its
            # author put it, often mid-row on purpose (common in BTS). Another singer starting over
            # the end of the line never cuts it.
            nextStart = next((t for t, nxt in unresolvedStarts
                              if spanStart < t < spanEnd and _carriesOn(member, nxt)), None)
            if nextStart is not None and lyric.get("endChunkOverride") is None:
                spanEnd = nextStart
            spans[idx] = (spanStart, spanEnd)
        elif idx in inferred:
            spans[idx] = inferred[idx]
        else:
            linked = lyric.get("linkedLabel")
            if linked and linked.get("startChunk") is not None:
                spans[idx] = (linked["startChunk"], linked.get("endChunk"))
    return spans
