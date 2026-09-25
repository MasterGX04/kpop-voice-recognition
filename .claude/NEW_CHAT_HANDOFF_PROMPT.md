# Prompt for starting a new chat on this project

Copy everything below the line into a fresh Claude Code conversation in this repo to bring it up
to speed on the vocab/flashcard feature before asking for new work.

---

I'm continuing work on this project's Japanese Kanji / Korean Hanja vocab + spaced-repetition
flashcard feature. Before doing anything, please read these two files in full:

1. `.claude/APP_OVERVIEW.md` - what this whole app is, its main windows, data conventions, and the
   pre-existing Japanese/Korean linguistic analysis pipeline the vocab feature is built on top of.
2. `.claude/VOCAB_SRS_PLAN.md` - the detailed record of the vocab/flashcard feature itself: schema,
   module map, every design decision and why it was made, and a bug history section documenting
   real bugs that were found and fixed after it first shipped (including one serious one: manual
   edits in the Review screen used to get silently reverted by the next "Compile" scan - now fixed
   with `meaning_locked`/`hanja_locked` columns, but the exact same *category* of mistake is easy to
   reintroduce in a new feature if you're not aware of it).

Also skim the other `.claude/*_PLAN.md` docs (`KANJI_REFERENCE_PLAN.md`, `KOREAN_HANJA_PLAN.md`,
`GRAMMAR_BREAKDOWN_PLAN.md`, `KOREAN_GRAMMAR_BREAKDOWN_PLAN.md`) if the new work touches the
underlying Japanese/Korean analysis functions rather than just the vocab DB/GUI layer - they cover
`core/kanji_reference.py`, `core/korean_hanja.py`, `core/grammar_breakdown.py`,
`core/korean_grammar_breakdown.py` in more depth than the overview doc does.

Ground rules for how I want you to work on this specific feature, based on how the last session
went:

- **Verify empirically before and after changes**, not just by reading code. This codebase's own
  convention (and every bug fix in `VOCAB_SRS_PLAN.md`'s bug history) was verified two ways: the
  permanent `core/test_vocab_*.py` unittest suite, *and* a real Tkinter smoke test that actually
  drives the widgets (button `.invoke()`, checking rendered text) against seeded data in a
  throwaway tempdir - a plain unit test on the backend alone missed real bugs (an invisible
  `ttk.Combobox`, a messagebox with the wrong parent) that only showed up by actually interacting
  with the widgets.
- **Delete throwaway verification scripts once you're done with them** - they're not deliverables.
  Keep the permanent `core/test_*.py` suite as the lasting artifact.
- **Run the full relevant test suite before calling anything done**:
  `python -m unittest core.test_vocab_db core.test_vocab_store_ja core.test_vocab_store_ko core.test_vocab_link core.test_vocab_sync core.test_kanji_reference -v`
  (97 tests, currently all green - any new failure is a real regression, not flakiness).
- **If you add a new way to write to `vocab_ja`/`vocab_ko` from the GUI**, decide up front whether
  it needs to set `meaning_locked`/`hanja_locked` (or a new lock flag) so a future rescan doesn't
  silently undo it - this is the exact bug that already bit a real user once.
- **If you add a bulk/loop write operation**, thread a shared `conn` through it (see how
  `core/vocab_sync.py` calls `vocab_store_ja.upsertVocab(entry, conn=conn)`) rather than opening a
  connection per row - that mistake once made a compile-all operation take 60-85 seconds and freeze
  the whole Tkinter app.
- This project has **no PR/commit-per-feature discipline enforced** - work directly, but only
  create git commits if I explicitly ask for one.

What I want to build next: Flashcard Tool Upgrade:
You need a UI stack that can fluidly render typography, color-coded grammar blocks, and dynamic layouts without the rigid verbosity of pure Python GUI frameworks.

Adopt a Hybrid Desktop Framework: Choose a setup like Python + PyWebView, which allows you to run a local HTML/CSS/JS frontend that communicates directly with your existing Python backend.

Design the Flashcard Layout: Use HTML5, Tailwind CSS, or vanilla JavaScript to build the interactive flashcards and karaoke lyric displays.

Wire the Interaction Logic: Set up the JavaScript frontend to trigger local Python API calls (e.g., submit_review(word_id, rating)) when you interact with the flashcard buttons or hotkeys, updating the SQLite database instantly without an internet connection.

Help plan this out and let me make decisions out of sevveral options and outweight the costs because I don't know what to do