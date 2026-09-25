# Using This Tool to Learn Japanese

A personal study workflow for using the lyrics tool's Japanese features (reading auto-fill, and the
in-progress Kanji reference collector — see `.claude/KANJI_REFERENCE_PLAN.md`) to actually learn the
language, not just display it.

## The workflow

1. **Type or paste a verified Kanji/kana lyric line** into the Japanese Lyric field in the Lyrics Editor.
   Don't trust a scraped/OCR'd lyric blindly — a wrong character breaks every downstream step.
2. **Auto-fill the reading** with the "Convert Kanji → Reading" button. Use **Romaji** for a first pass
   at a new song (fastest to skim), and switch to **Hiragana** when you want to drill kana reading
   itself rather than lean on romanization.
3. **Highlight unfamiliar or interesting words** as you go and add them to the per-song Kanji reference
   (once built — Milestone 4+ in the technical plan). Don't try to capture every word; capture the ones
   you had to stop and think about.
4. **Periodically check `shared_words.html`** (once built — Milestone 5) for words that show up across
   multiple songs. Repetition across independently-written lyrics is a natural spaced-repetition signal:
   a word you've now seen in three different songs is worth actually memorizing; a word you saw once is
   lower priority.
5. **Use the on'yomi/kun'yomi tag to focus effort differently**:
   - **On'yomi words with a confirmed Chinese cognate** (e.g. 時間) are close to free vocabulary for you
     — the reading is new, but the meaning is already available from Traditional Chinese. Spend your
     time on the *sound*, not re-learning the meaning.
   - **On'yomi words with no attested Chinese cognate** (Japan-coined compounds, 和製漢語) are a middle
     case — the character meanings are still a hint, but the specific compound was coined in Japan, so
     don't assume a Chinese speaker would recognize it as-is.
   - **Kun'yomi words** (e.g. 出会う) get no help at all from Chinese — these are native Japanese
     vocabulary and need real rote memorization, same as a totally unrelated language. Budget more
     repetition here.

## Why a single Kanji doesn't have "the" reading

This tripped up testing directly, so it's worth internalizing: a Kanji's reading isn't a fixed
property of the character — it depends on the word it's part of. `時` alone, out of context,
gets read as **ジ** (its on'yomi); the same `時` in `時が止まる` correctly reads as **とき**
(toki, kun'yomi). Neither is "wrong" — 時 genuinely has both readings, used in different words
(時間=jikan uses ジ; 時=toki as a standalone noun uses とき). This is exactly why the Kanji
reference tool tokenizes the *whole lyric line* before classifying, rather than just whatever
substring you highlight — it needs the surrounding words to know which reading is actually in
play. Practically: if you highlight part of a word and get a reading that looks wrong, the tool
probably grabbed the reading of the token you were part of, not a slice of it (highlighting only
"時" inside "時間" still gives you 時間's reading, jikan — Kanji reference granularity is
per-word, not per-character). To pin down a single character's reading in isolation, highlight it
somewhere it's genuinely standing alone as its own word.

## Known limits to keep in mind

- The tokenizer (`fugashi`/unidic) occasionally picks an uncommon reading for an ambiguous word (e.g.
  わたくし instead of the more common わたし for 私) — treat auto-generated readings as a strong draft,
  not ground truth, and hand-correct when something looks off.
- Long vowels romanize as "ou"/"ei" by default (学校→gakkou, 先生→sensei) since that's right for the
  vast majority of words, especially Sino-Japanese on'yomi compounds — but a handful of words are
  genuinely spelled with a literal doubled vowel instead (大阪=Ōsaka, おおきい, ねえさん), and those
  will come out wrong ("大阪" romanizes as "ousaka" here, not "oosaka"). No way to tell these apart
  automatically from the tool's inputs alone — hand-correct if you spot one.
- The on'yomi/kun'yomi classifier can't resolve genuine jukujigo (idiomatic whole-word readings like
  今日=kyō, 大人=otona) — that's expected, not a bug; those just get flagged as "special reading" with no
  further breakdown attempted.
- CC-CEDICT coverage of Japan-coined compounds is inherently incomplete, so "not attested in Chinese"
  sometimes just means "not in this dictionary yet," not "definitely never used in Chinese."
- The Shinjitai→Traditional conversion (OpenCC's `jp2t.json`) is explicitly flagged upstream as
  exploratory — spot-check unfamiliar conversions against Wikipedia's Shinjitai tables if something
  looks wrong.

## What's next

The other open idea from planning discussions is a **grammar breakdown layer** — tokenizing a lyric line
into stem + particle + conjugation chain (e.g. 食べ-たく-ない = eat + want + negative) using the same
tagger, to make agglutinative conjugation visible instead of something you have to reverse-engineer by
eye. Not yet planned in detail or built; a natural next phase once the Kanji reference collector is
finished.
