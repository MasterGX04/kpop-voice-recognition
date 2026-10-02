// Run with: node gui/web/vocab_review/karaoke.test.js
const assert = require("assert");
const { cleanLine, weightOf, segmentLine, chunkAt, isLit, newTake, unitStarts } = require("./karaoke.js");

assert.strictEqual(cleanLine("a|b"), "ab");
assert.strictEqual(cleanLine(null), "");
assert.strictEqual(cleanLine("君を\u2063優しく"), "君を優しく");   // invisible pause marker

// small kana add no time; ー and っ count as one mora each
assert.strictEqual(weightOf("きょう"), 2);
assert.strictEqual(weightOf("ラーメン"), 4);
assert.strictEqual(weightOf("、。 "), 0);

// Real Doughnut/Tzuyu lyric: segments must reassemble to the exact (cleaned) text, newlines kept.
const line = "そばにいなくても\n切れずにリンクしてるね\nMemory の余韻に浸っていたい (I, I, I, I)";
const segs = segmentLine(line, "ja");
assert.strictEqual(segs.map((s) => s.text).join(""), line);
const words = segs.filter((s) => s.isWord);
assert.ok(words.length >= 8, "expected several words, got " + words.length);
assert.strictEqual(words[0].fraction, 0);
for (let i = 1; i < words.length; i++) assert.ok(words[i].fraction >= words[i - 1].fraction);
assert.ok(words[words.length - 1].fraction < 1 && words[words.length - 1].fraction > 0.7);
assert.ok(segs.every((s) => s.isWord === (s.fraction !== undefined)));
assert.ok(segs.some((s) => !s.isWord && s.text.includes("\n")));

// Korean + a "|" colour marker that must vanish.
const ko = segmentLine("사랑해|요 너를", "ko");
assert.strictEqual(ko.map((s) => s.text).join(""), "사랑해요 너를");

// Punctuation-only / empty input is safe.
assert.deepStrictEqual(segmentLine("", "ja"), []);
assert.strictEqual(segmentLine("…", "ja").filter((s) => s.isWord).length, 0);

// Playback clock: whole clip from chunk 649, and a mid-clip start 1000 ms in.
const whole = { clipStartChunk: 649, offsetMs: 0, playMs: 4200, chunkMs: 40, latencyMs: 0 };
assert.strictEqual(chunkAt(whole, 0), 649);
assert.strictEqual(chunkAt(whole, 400), 659);
assert.strictEqual(chunkAt(whole, 4200), null);                      // clip over
assert.strictEqual(chunkAt({ ...whole, offsetMs: 1000, playMs: 3200 }, 400), 649 + 35);
assert.strictEqual(chunkAt({ ...whole, latencyMs: 80 }, 400), 649 + 8);   // latency delays the highlight
assert.strictEqual(chunkAt({ ...whole, latencyMs: 80 }, 40), 649);        // never before the clip start
assert.strictEqual(chunkAt(null, 10), null);
// A slowed clip (rate 0.5): song time advances half as fast as the wall clock; playMs is already real time.
const slow = { ...whole, rate: 0.5, playMs: 8400 };
assert.strictEqual(chunkAt(slow, 800), 649 + 10);
assert.strictEqual(chunkAt(slow, 8400), null);

// Highlight: lit from its start chunk on, never when there is no chunk info or no clock.
assert.strictEqual(isLit(660, 659.9), false);
assert.strictEqual(isLit(660, 660), true);
assert.strictEqual(isLit(undefined, 700), false);
assert.strictEqual(isLit(660, null), false);

// Tap take: taps fill words in order, skip leaves a hole, undo steps back one, nothing past the last word.
const take = newTake(3);
assert.strictEqual(take.tap(null), false);                 // no clock (clip over) -> ignored
assert.ok(take.tap(631.5) && take.skip() && take.tap(700));
assert.strictEqual(take.done, true);
assert.strictEqual(take.tap(800), false);
assert.deepStrictEqual(take.taps(), { 0: 631.5, 2: 700 });
assert.strictEqual(take.tappedCount(), 2);
assert.deepStrictEqual(take.undo(), { index: 2, chunk: 700 });
assert.strictEqual(take.next, 2);
assert.ok(take.tap(705));                                  // re-tap the undone word
assert.deepStrictEqual(take.taps(), { 0: 631.5, 2: 705 });
assert.strictEqual(newTake(2).undo(), null);               // undo on an empty take is safe

// Unit starts follow the chip order: syllables, kana parts of a split kanji, or whole words.
const syl = { unit: "syllable", pieces: [{ isWord: false, text: " " },
  { isWord: true, startChunk: 10, syllables: [{ startChunk: 10 }, { startChunk: 12 }] }, { isWord: true, startChunk: 15, syllables: [{ startChunk: 15 }] }] };
assert.deepStrictEqual(unitStarts(syl), [10, 12, 15]);
const wordMode = { unit: "word", pieces: [{ isWord: true, startChunk: 5, parts: [{ startChunk: 5 }, { startChunk: 9 }] }, { isWord: true, startChunk: 20 }] };
assert.deepStrictEqual(unitStarts(wordMode), [5, 9, 20]);
console.log("karaoke.test.js ok");
