# Source of these files

`JPShinjitaiCharacters.txt` and `JPShinjitaiPhrases.txt` are copied verbatim from the
[OpenCC project](https://github.com/BYVoid/OpenCC) (`data/dictionary/` in the `opencc` PyPI
source distribution, version 1.4.2), used there to back the `jp2t.json` conversion config
("New Japanese Kanji (Shinjitai) to Old Japanese Kanji (Kyūjitai)").

Licensed under the Apache License, Version 2.0 — see `LICENSE-OpenCC` in this folder.

Used here by `core/kanji_reference.py: toTraditional()` to convert Japanese Shinjitai
Kanji to their Traditional Chinese equivalent forms (e.g. 楽→樂, 会→會) before looking a
word up in CC-CEDICT, since Japanese's postwar simplified forms are a third character-form
standard distinct from both Simplified and Traditional Chinese.
