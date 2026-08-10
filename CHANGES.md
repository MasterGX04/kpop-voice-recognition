# Changes

Three commits, in order:

## 1. `d0df51b` — Committed pre-existing uncommitted work

The repo hadn't been committed in a long time. Reviewed and committed the existing
working-tree diff as-is (not part of this task) before touching anything else, so the
reorg below would be its own clean diff:

- Subgroup/active-member support in `group_registry.py` (subgroups like MISAMO no longer
  force every member to be "blank"/unused).
- New group icon sets: Fifty Fifty "Imperfect I'mperfect", NMIXX "Heavy Serenade", aespa
  "Savage", plus new groups WJSN and One Direction.
- New `util/` package (wavlm-based MLP training/prediction/embedding visualization),
  `datasets/group_configs.json` (per-member training hyperparameters), `history_manager.py`
  (label undo/redo), `add_labels_menu.py`.
- A batch of `saved_labels/`/`predicted_labels/` updates across BTS/IVE/NMIXX/aespa/etc.
- Removed two now-unused queued model checkpoints (`models/queue/IVE_muq_head*.pt`).
- Added `tmpdir/` to `.gitignore` instead of committing it — it's an ~85MB SpeechBrain
  pretrained-model cache (`classifier.ckpt`, `embedding_model.ckpt`, `hyperparams.yaml`),
  not original work, and re-downloadable.

## 2. `2f22430` — Reorganized root `.py` files into packages

The repo root had ~27 loose `.py` files sitting next to media/cache/build directories.
Moved them into four new packages, based on actual imports (checked, not guessed from
filenames):

| Package | Files |
|---|---|
| `gui/` | `voice_recognition_gui.py`, `audio_tester.py`, `add_labels_menu.py`, `line_distribution_panel.py`, `label_lanes.py`, `label_overlay.py`, `lyrics_box.py`, `lyrics_editor.py`, `navigation_arrows.py`, `zoom_functions.py`, `history_manager.py`, `cut_clip_manager.py`, `TrackItem.py`, `VideoTrack.py`, `thumbnail_functions.py`, `video_record.py` |
| `core/` | `group_registry.py`, `audio_processing.py`, `util_functions.py` |
| `ml/` | `train_kpop_singers.py`, `model_predictor.py`, `speaker_embedding.py`, `harmony_test.py`, `extract_adlibs.py`, `training_utils.py` |
| `media/` | `image_generator.py`, `LoopingBackground.py` |

`thumbnail_functions.py` and `video_record.py` ended up in `gui/` rather than `media/` (the
original plan's guess) because both are tkinter/GUI-only consumers (`ThumbnailManager` is
used by `audio_tester.py`; `video_record.py`'s screen-capture helper is only used by
`VideoTrack.py`). `training_utils.py` went to `ml/` instead of `core/` since nothing
actually imports it and its name/content (torch loss helpers) fits training scripts better.

Existing directories (`datasets/`, `model/`, `models/`, `util/`, `saved_labels/`,
`group_icons/`, `member_images/`, `cache_audio/`, etc.) were left untouched.

**Every cross-module import was updated** to the new package paths (e.g.
`from group_registry import GroupRegistry` → `from core.group_registry import GroupRegistry`
in `gui/voice_recognition_gui.py`).

**Found and fixed a real breakage the move would have caused**: several files compute a
"repo root" path from their own `__file__` location (`resourcePath()` in
`core/util_functions.py`, `gui/voice_recognition_gui.py`, `gui/audio_tester.py`,
`gui/VideoTrack.py`, `gui/line_distribution_panel.py`; `exeDir()` in
`gui/voice_recognition_gui.py`; `getAppRoot()` in `core/group_registry.py`). These all
assumed the file's own directory *was* the repo root — true when the files sat at the
root, false now that they're one level deeper. Each was fixed to walk up one extra
directory level. `getAppRoot()` in particular anchors `group_icons/` and `groups.json` —
without this fix, the app would have failed to find any group data at all.

**Entry point changed**: the app must now be launched as `python -m gui.voice_recognition_gui`
(training as `python -m ml.train_kpop_singers`) from the repo root, not
`python voice_recognition_gui.py`. Running a moved script by direct path puts the script's
own directory on `sys.path[0]` instead of the repo root, breaking absolute imports like
`from datasets.new_datasets import ...`. Updated `voice_recognition_gui.spec` (entry script
path + `pathex=['.']`) and `commands.txt`'s saved training commands accordingly.

**Verified**: import-smoke-tested every moved entry module (`gui.voice_recognition_gui`,
`core.group_registry`, `ml.train_kpop_singers`, `ml.model_predictor`) across the project's
two venvs (`.build_venv` lacks `torch`; `pyannote-env` has it), confirmed `GroupRegistry`
resolves `group_icons/`/`groups.json` to the true repo root and loads all 13 groups, and
launched the actual app (`python -m gui.voice_recognition_gui`) to confirm it opens without
crashing.

**Pre-existing bug found, not fixed (out of scope)**: `ml/model_predictor.py` imports
`PresenceHead` from `ml/train_kpop_singers.py`, but that name only ever appears in a
comment there (`# head2 = PresenceHead(...)`) — this import was already broken before the
move (confirmed via `git show` on the pre-reorg file). `model_predictor.py` cannot
currently be imported.

## 3. `66f8656` — Song picker: sort by Length / Fairness

New module **`core/song_stats.py`**:

- `computeSongLengthSeconds(labels)` — unweighted sum of every label's
  `(end_chunk - start_chunk + 1) * 40ms`, regardless of member/backing/adlib.
- `computeSongFairness(labels)` — per-member weighted seconds (backing lines ×`0.7`,
  ad-libs ×`1.0`), then `1 - pstdev/mean` across members with nonzero total. Mirrors
  `LineDistributionPanel.computeStats()` in `gui/line_distribution_panel.py` exactly
  (including its n=1 → `1.0` and n=0 → `None` edge cases), just computed from a raw
  `_labels.json` file instead of a live app timeline.
- `getGroupSongStats(group, songNames)` — reads/writes
  `saved_labels/{group}/_song_stats_cache.json` (gitignored, like `cache_audio/`), keyed
  by each labels file's mtime; only recomputes songs whose file changed.
- `invalidateSongStats(group, songName)` — recomputes one song immediately; called from
  `VoiceDetectionApp.saveLabels()` in `gui/audio_tester.py` right after it writes the
  labels JSON, so the cache is fresh the instant labels are saved rather than waiting for
  the next lazy mtime check.

**Weight values** (`BACKING_WEIGHT = 0.7`, `ADLIB_WEIGHT = 1.0`) were sanity-checked against
`saved_labels/BTS/Let Go_labels.json` (J-Hope has both backing overlap AND real lead/rap
blocks) vs `saved_labels/BTS/Spring Day Studio_labels.json` (J-Hope has *only* backing,
overlapping nearly the whole second half) — 0.7 keeps a backing-only performance from
counting as equal to a lead performance without zeroing it out. Kept separate from
`datasets/vocal_metadata.py`'s `backing_weight`/`adlib_weight` (`0.5`/`0.8`), which are
training-loss weights for a different purpose.

**Deviation from the original plan**: `commands.txt` had a note reading
`Fairness score: 1 - (std/mean x 2/sqrt(n))`, a different formula than what
`line_distribution_panel.py` implements. Asked and confirmed: use the existing
`1 - stdev/mean` formula; the `2/sqrt(n)` variant was "glitchy" and not useful for
within-group sorting. Removed that note from `commands.txt` (now reads
`Fairness score: 1 - (std/mean)`).

**GUI changes** in `gui/voice_recognition_gui.py`:

- `chooseSongWindow` gets two new `ttk.Combobox` controls in the top bar: **View**
  (`By Album` / `Flat List`, default `By Album`) and **Sort** (`Alphabetical` /
  `Song Length` / `Fairness`, default `Alphabetical`).
- `refreshSongPickerUI` computes `song_stats.getGroupSongStats(...)` once per refresh,
  builds a `_sortKey` that ranks by the selected stat (descending — longest/fairest
  first) with unlabeled songs always sorted last, and branches on View mode: `By Album`
  keeps the existing per-album grouping (just sorting within each album/unclaimed section
  by `_sortKey` instead of always alphabetically); `Flat List` skips album grouping
  entirely and ranks every song in the group together.
- `_addSongRow` now shows the active stat next to each song name (e.g. `2:34` for length,
  `Fairness 78%` for fairness); nothing shown for Alphabetical mode or unlabeled songs.

**Verified**: `computeSongFairness`/`computeSongLengthSeconds` against real `Let Go` (length
363s, fairness 0.48) vs `Spring Day Studio` (length 354s, fairness 0.42) labels;
`getGroupSongStats` across all 42 BTS songs (labeled and unlabeled) with a real disk cache
written; `invalidateSongStats` recomputes and overwrites a stale cache entry immediately;
Alphabetical-mode sort key reduces to the same ordering as the original hardcoded
`str.lower` sort (proven by construction — every song gets the same primary/secondary sort
tuple elements, so ties break on name); import-smoke-tested and launched the app with the
new controls wired in, confirmed no crash.

**Not independently verified by this pass**: the actual click-through behavior of the two
new dropdowns and the visual song-row stat labels — the app was left running for manual
visual confirmation rather than driven programmatically (no browser/UI-automation tool
available for a tkinter desktop app).
