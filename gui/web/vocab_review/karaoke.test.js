// Run with: node gui/web/vocab_review/karaoke.test.js
const assert = require("assert");
const { cleanLine, weightOf, segmentLine } = require("./karaoke.js");

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
console.log("karaoke.js: all checks passed");
