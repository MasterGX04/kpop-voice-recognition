// Pure helpers for the flashcard's lyric line: split it into clickable words and estimate where in
// the clip each word starts. No DOM access, so it can be unit-tested under node (karaoke.test.js).
//
// The data has per-LINE timing only (no per-word timestamps), so every onset here is an ESTIMATE:
// proportional to how many sung characters precede the word. The UI says so. See
// .claude/KARAOKE_PLAN.md (A5 / K2).
(function (root) {
  "use strict";

  // Small kana fuse with the kana before them (きゃ = 1 mora), so they add no time of their own.
  const SMALL_KANA = /[ゃゅょぁぃぅぇぉゎャュョァィゥェォヮ]/g;
  const SUNG_CHAR = /[\p{L}\p{N}]/u;

  // "|" in stored lyrics only marks a member colour change in the Tk lyric box, and U+2063 is the
  // invisible pause marker (core/lyric_text.py) - neither is ever displayed.
  function cleanLine(text) {
    return String(text || "").replace(/[|\u2063]/g, "");
  }

  function weightOf(word) {
    let n = 0;
    for (const ch of word.replace(SMALL_KANA, "")) if (SUNG_CHAR.test(ch)) n += 1;
    return n;
  }

  function splitWords(text, locale) {
    if (typeof Intl !== "undefined" && Intl.Segmenter) {
      const seg = new Intl.Segmenter(locale, { granularity: "word" });
      return Array.from(seg.segment(text), (s) => ({ text: s.segment, isWord: !!s.isWordLike }));
    }
    // Fallback: one "word" per character (still clickable, just finer-grained).
    return Array.from(text, (ch) => ({ text: ch, isWord: SUNG_CHAR.test(ch) }));
  }

  // Returns [{text, isWord, fraction}] covering the whole cleaned line exactly (whitespace,
  // newlines and punctuation kept as non-word pieces). `fraction` (0..1) is set on words only.
  function segmentLine(rawLine, locale) {
    const pieces = splitWords(cleanLine(rawLine), locale)
      .map((p) => ({ ...p, isWord: p.isWord && weightOf(p.text) > 0 }));
    const total = pieces.reduce((sum, p) => sum + (p.isWord ? weightOf(p.text) : 0), 0);
    let before = 0;
    for (const p of pieces) {
      if (!p.isWord) continue;
      p.fraction = total > 0 ? before / total : 0;
      before += weightOf(p.text);
    }
    return pieces;
  }

  const api = { cleanLine, weightOf, segmentLine };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.Karaoke = api;
})(typeof window !== "undefined" ? window : globalThis);
