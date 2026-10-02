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

  // --- Playback clock (K1). The audio plays in the Python process, so the page keeps its own clock: `play` is
  // what playOccurrenceAudio returned ({clipStartChunk, offsetMs, playMs, chunkMs, latencyMs}) and `elapsedMs`
  // is the time since that call came back. On a slowed clip (`rate` < 1) song time advances `rate` x real time. Returns the absolute chunk now being sung (fractional), or null once
  // the clip is over.
  function chunkAt(play, elapsedMs) {
    if (!play || !(play.chunkMs > 0)) return null;
    const into = elapsedMs - (play.latencyMs || 0);
    if (elapsedMs >= play.playMs) return null;
    return play.clipStartChunk + (play.offsetMs + Math.max(0, into) * (play.rate || 1)) / play.chunkMs;
  }

  // --- Highlight (K2): a word is lit once the clock reaches its start and stays lit until the clip ends. A word
  // with no startChunk (the plain-estimate fallback) is never lit - no precise-looking guess.
  function isLit(startChunk, chunk) {
    return chunk !== null && typeof startChunk === "number" && chunk >= startChunk;
  }

  // --- Tap-along take (Part 5): one tap per word, in order. A tap records the clock chunk for the next word, skip
  // leaves it estimated, undo steps back one entry. No DOM, so it is unit-tested under node.
  function newTake(count) {
    const log = [];                                   // {index, chunk|null}
    return {
      count,
      get next() { return log.length; },
      get done() { return log.length >= count; },
      tap(chunk) {
        if (this.done || chunk === null || chunk === undefined) return false;
        log.push({ index: log.length, chunk });
        return true;
      },
      skip() {
        if (this.done) return false;
        log.push({ index: log.length, chunk: null });
        return true;
      },
      undo() { return log.pop() || null; },
      taps() {
        const out = {};
        for (const e of log) if (e.chunk !== null) out[e.index] = e.chunk;
        return out;
      },
      tappedCount() { return log.filter((e) => e.chunk !== null).length; },
      entries() { return log.slice(); },
    };
  }

  // The start chunk of every tap unit in tap order, from a getLineTiming / previewTaps result - the same walk the tap
  // dialog uses to lay out its chips: a syllable each (syllable mode), a kana part each (split kanji), else a word.
  function unitStarts(timing) {
    const out = [];
    for (const p of timing.pieces || []) {
      if (!p.isWord) continue;
      if (timing.unit === "syllable" && p.syllables) for (const s of p.syllables) out.push(s.startChunk);
      else if (p.parts && p.parts.length > 1) for (const part of p.parts) out.push(part.startChunk);
      else out.push(p.startChunk);
    }
    return out;
  }

  const api = { cleanLine, weightOf, segmentLine, chunkAt, isLit, newTake, unitStarts };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  root.Karaoke = api;
})(typeof window !== "undefined" ? window : globalThis);
