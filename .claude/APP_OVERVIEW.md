# App Overview

A Tkinter desktop tool for building K-pop/J-pop lyric videos: label which member is singing at
each moment in a song, sync bilingual lyrics to the audio, and (newer) build a Japanese Kanji /
Korean Hanja vocabulary flashcard system mined from those lyrics. Everything is local-first - no
server, no cloud sync. State lives in JSON files under `saved_labels/` and, for the vocab feature,
a SQLite database at `data/vocab_srs.db`.

## Time unit: the "chunk"

Audio is sliced into fixed **40ms chunks** (`CHUNK_MS = 40` in `core/song_stats.py`, echoed in
`core/audio_processing.py`, `ml/model_predictor.py`, `util/phrase_extraction.py`). Every timestamp
in this codebase - label spans, lyric anchors, video export cursor - is a chunk index, not
milliseconds or seconds. Convert with `seconds = chunkIndex * 0.04`.

## Top-level windows

- **`gui/voice_recognition_gui.py` (`VoiceTrainerGUI`)** - the app's main/root window. Group and
  member management (`groups.json`, `group_icons/<group>/group.json` via `core/group_registry.py`),
  member image generation, media folder setup, album art. Opens a song's editor as a `Toplevel`.
- **`gui/audio_tester.py` (`VoiceDetectionApp`)** - "Line Distribution Labeler" / **Audio Tester**
  window, one per open song. Owns the waveform/label canvas, playback, video export, the member
  timeline, and the menu bar (`Record`/`Edit`/`Labels`/`Playback`/`View`/`Lyrics`/`Vocab`/`Tools`).
  `self.root` here is that song's own `Toplevel`, **not** the main app window - always pass
  `parent=self.root` to any dialog/messagebox opened from this class, or it silently attaches to
  the main window instead (a real bug, fixed once already - see VOCAB_SRS_PLAN.md).
- **`gui/lyrics_editor.py` (`LyricsEditor`)** - owned by `VoiceDetectionApp`. The add/edit/delete UI
  for lyric lines, persisted to `saved_labels/<group>/<song>_lyrics.json`. Has per-language tool
  buttons (Japanese: Convert Kanji→Reading, Add Kanji to Document, Grammar; Korean: Grammar, which
  merges grammar breakdown + Hanja lookup in one button).
- **`gui/lyrics_box.py` (`LyricBox`)** - renders one lyric line as canvas items, anchored via
  `self.parent.targetLyricsX` (set by `VoiceDetectionApp.syncLyricBoxToBackground()`).

## Data on disk

```
saved_labels/<group>/<song>_labels.json    # [[member, startChunk, endChunk, isCut, isBacking], ...]
saved_labels/<group>/<song>_lyrics.json    # [{lyricId, language, memberName:[...], korean, romanization,
                                            #   english, startChunk, isAdLib, adLibDuration,
                                            #   anchorMode, linkedLabel:{member,startChunk,endChunk}|null}, ...]
saved_labels/<group>/<song>_history.json   # undo/redo stack of full label-array snapshots
groups.json, group_icons/<group>/group.json  # group/member identity - name string is the only key,
                                              # no numeric/UUID ids anywhere in this app
data/vocab_srs.db                          # SQLite - the vocab/flashcard feature, see below
kanji_reference/                           # gitignored, fully regenerated - legacy cross-song HTML report
```

JSON write convention used everywhere: `utf-8`, `ensure_ascii=False`, `indent=4`. No atomic writes,
no backup files - `saved_labels/` files are written directly. Broad `try/except` around load, with
a safe empty-default fallback, is the norm (see `core/util_functions.py`, `core/group_registry.py`).

## The Japanese/Korean linguistic pipeline (`core/`)

Pre-existing, built up over several milestones documented in their own `.claude/*_PLAN.md` docs -
read those for the deep detail:

- **`core/kanji_reference.py`** (`.claude/KANJI_REFERENCE_PLAN.md`) - Japanese Kanji analysis:
  Shinjitai→Traditional conversion, on'yomi/kun'yomi/mixed/jukujigo classification, Chinese-cognate
  lookup via CC-CEDICT, Japanese-meaning lookup via JMdict, Mandarin pinyin mnemonic. Its
  `analyzeSelection(fullText, startOffset, endOffset)` is the main entry point and is reused
  directly by the vocab feature. It also has an **older, now-unused** JSON+HTML persistence path
  (`addWordToSongReference`, `scanLyricsForKanjiVocab`) - superseded by the SQLite vocab store, but
  deliberately left intact (still has 54 passing tests) rather than gutted.
- **`core/japanese_utils.py`** - shared `fugashi` tagger singleton + kana/romaji conversion.
- **`core/grammar_breakdown.py`** (`.claude/GRAMMAR_BREAKDOWN_PLAN.md`) - Japanese whole-line
  stem+particle+ending breakdown.
- **`core/korean_hanja.py`** (`.claude/KOREAN_HANJA_PLAN.md`) - Hangul→Hanja homophone
  disambiguation via a Wiktionary index (`data/ko_wiktionary/`). Unlike Japanese, one Hangul
  spelling can map to **several unrelated real Hanja** (e.g. 화 → 火/禍/和/化/畫/靴) - always
  returns a list, never guesses one.
- **`core/korean_dictionary.py`** - shared Wiktionary-index lookup (English gloss + POS) used by
  both `korean_hanja.py` and `korean_grammar_breakdown.py`.
- **`core/korean_grammar_breakdown.py`** (`.claude/KOREAN_GRAMMAR_BREAKDOWN_PLAN.md`) - Korean
  whole-line breakdown via `kiwipiepy`, merges in Hanja candidates per content word.
- **`core/korean_utils.py`** - shared `Kiwi()` tagger singleton.

## The vocab/flashcard feature (`core/vocab_*.py`, `core/korean_vocab.py`, `gui/vocab_review.py`)

Built in a later session on top of the pipeline above. **Full detail, architecture, decisions, and
the bug history live in `.claude/VOCAB_SRS_PLAN.md` - read that before touching any of it.** One-line
summary: a SQLite-backed spaced-repetition vocab store, fed by a "Compile Vocab" scan over
`saved_labels/*/*_lyrics.json`, browsable/editable via a "Vocab Review" window, with cross-language
(Japanese↔Korean) linking by shared Chinese-cognate root.

## Testing convention

Stdlib `unittest` only - no pytest, no test framework configured. Every `core/test_*.py` file is
runnable standalone: `python -m unittest core.test_<name> -v`. Tests that touch disk state
(`saved_labels/`, `data/vocab_srs.db`, `kanji_reference/`) `chdir()` into a `tempfile.mkdtemp()` in
`setUp`/restore in `tearDown`, since all these paths are plain-relative (not `__file__`-anchored),
so a chdir'd test gets a fully isolated copy for free. Follow this pattern for any new test file.
Current full suite: 97 tests (`test_kanji_reference` 54 + `test_vocab_*` 43), all green.

**Convention this project has consistently followed**: write real regression tests (not just
inline verification), and when a real bug is found, name the test after the bug and explain the
scenario in a comment/docstring - this codebase's test suite doubles as a bug history. Keep doing
that for anything new.

## Project-specific quirks worth knowing before changing anything

- **No numeric/UUID ids for group/member/song** - group name, member name, and song title are the
  keys everywhere (filenames, dict keys, JSON fields). Only the vocab DB and `lyricId`/`labelId`-ish
  concepts (`lyricId`, chunk-span matching for labels) introduce any kind of stable identity.
- **Labels have no stable id at all** - re-identified by `(member, startChunk, endChunk)` span match
  (`core.util_functions.findLabelIndexBySpan`). Lyrics *do* have a `lyricId` (added later).
- A resize/layout race already bit this codebase once (`targetLyricsX` not existing when lyrics
  loaded, because it's only set 120ms after a `<Configure>` event, on a flat 50ms timer) - fixed by
  calling `addBackgroundImage()` once from `startLayout()` (scheduled via `after_idle`, which
  reliably beats a fixed-delay `after(N)` timer). Keep this ordering sensitivity in mind if you ever
  touch `gui/audio_tester.py`'s init sequence.
- `.claude/settings.local.json` is just a Bash/Read permission allowlist, not project config.
