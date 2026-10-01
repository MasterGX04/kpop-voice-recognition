import os
import json
import codecs
import uuid
import tkinter as tk
from tkinter import messagebox
from gui.lyrics_box import LyricBox, LYRIC_LEAD_CHUNKS
from core.util_functions import ModalGuard, findLabelIndexBySpan
from core.lyric_text import (EDITOR_PAUSE_GLYPH, stripAll, stripAllWithSelection,
                             toEditorText, fromEditorText)

def _truncate(text: str, maxChars: int = 80) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= maxChars:
        return text
    return text[: maxChars - 1] + "…"

def _showCopyableInfo(title: str, text: str, parent=None):
    """
    Same purpose as messagebox.showinfo, but the body is a real (disabled, so read-only) Text
    widget instead of a native dialog label - so the generated analysis it displays (Kanji
    Analysis / Grammar Breakdown results) can actually be selected and copied out, plus an
    explicit "Copy All" button for one-click clipboard copy.
    """
    win = tk.Toplevel(parent)
    win.title(title)
    win.transient(parent)

    buttonFrame = tk.Frame(win)
    buttonFrame.pack(side="bottom", fill="x", pady=(0, 10))

    def copyAll():
        win.clipboard_clear()
        win.clipboard_append(text)

    tk.Button(buttonFrame, text="Copy All", command=copyAll).pack(side="left", padx=(10, 5))
    tk.Button(buttonFrame, text="OK", command=win.destroy).pack(side="left")

    scrollbar = tk.Scrollbar(win)
    scrollbar.pack(side="right", fill="y", pady=10)
    textWidget = tk.Text(win, wrap="word", width=80, height=20, yscrollcommand=scrollbar.set)
    textWidget.pack(side="left", fill="both", expand=True, padx=(10, 0), pady=10)
    scrollbar.config(command=textWidget.yview)

    textWidget.insert("1.0", text)
    textWidget.config(state="disabled")  # read-only, but still selectable/copyable
    textWidget.tag_add("sel", "1.0", "end")
    textWidget.focus_set()

    win.grab_set()
    win.wait_window()

    # Tk's local grab_set() isn't stack-based - destroying win just drops the grab
    # entirely, it does NOT restore whatever was grabbed before. Without this, nothing
    # keeps `parent` (the Lyric Editor) above its sibling Toplevels (e.g. the Lyrics
    # Manager list, also transient(app.root) rather than transient(parent)), so the
    # editor could end up hidden behind another already-open window once this closes.
    if parent is not None:
        try:
            parent.grab_set()
            parent.lift()
            parent.focus_force()
        except tk.TclError:
            pass


def _preview(text: str, maxLines: int = 2, maxCharsPerLine: int = 70) -> str:
    if not text:
        return ""
    lines = text.splitlines()  # preserves explicit line breaks
    out = []
    for line in lines[:maxLines]:
        line = line.rstrip()
        if len(line) > maxCharsPerLine:
            line = line[: maxCharsPerLine - 1] + "…"
        out.append(line)
    if len(lines) > maxLines:
        out.append("…")
    return "\n".join(out)

class LyricsEditor:
    """
    Owns the Lyrics add/edit/delete UI, and JSON persistence.

    It uses composition: it holds a reference to your main VoiceDetectionApp
    so it can reuse:
      - app.root / app.canvas
      - app.members / app.lyrics / app.images
      - app.disableRootKeybinds(), app.enableRootKeybinds()
      - app._getCircleImages(), app.rebuildLyricsAnimations()
      - app.selectedGroup, app.songName
    """

    def __init__(self, app):
        self.app = app
        
    def _lyricsJsonPath(self) -> str:
        return f"./saved_labels/{self.app.selectedGroup}/{self.app.songName}_lyrics.json"

    def _loadLyricsJsonList(self):
        path = self._lyricsJsonPath()
        if not os.path.exists(path):
            return []
        with codecs.open(path, "r", encoding="utf-8", errors="ignore") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
        return data if isinstance(data, list) else []

    def _saveLyricsJsonList(self, entries):
        path = self._lyricsJsonPath()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with codecs.open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=4)

    def _migrateLyricEntry(self, entry):
        """Backfill lyricId/linkedLabel on entries written before this feature existed."""
        changed = False
        if not entry.get("lyricId"):
            entry["lyricId"] = str(uuid.uuid4())
            changed = True
        if "linkedLabel" not in entry:
            entry["linkedLabel"] = None
            changed = True
        return entry, changed

    def _loadAndMigrateLyricsJsonList(self):
        entries = self._loadLyricsJsonList()
        anyChanged = False
        for i, e in enumerate(entries):
            entries[i], changed = self._migrateLyricEntry(e)
            anyChanged = anyChanged or changed
        if anyChanged:
            self._saveLyricsJsonList(entries)
        return entries

    def upsertLyricEntry(self, entry):
        """Insert or replace by lyricId (startChunk is no longer the identity)."""
        entries = self._loadLyricsJsonList()
        lyricId = entry["lyricId"]

        replaced = False
        for i, e in enumerate(entries):
            if e.get("lyricId") == lyricId:
                entries[i] = entry
                replaced = True
                break

        if not replaced:
            entries.append(entry)

        entries.sort(key=lambda x: int(x.get("startChunk", 0)))
        self._saveLyricsJsonList(entries)

    def deleteLyricEntry(self, lyricId: str):
        """
        Fully delete a lyric:
        - remove canvas items (destroy LyricBox)
        - remove from app.lyrics
        - rebuild animations
        - remove from JSON
        """
        app = self.app

        # 1. Remove runtime LyricBox + canvas items
        if lyricId in app.lyrics:
            lyricBox = app.lyrics.pop(lyricId)

            # Properly delete all canvas objects
            if hasattr(lyricBox, "destroy"):
                lyricBox.destroy()

        # 2. Rebuild animations so state is clean
        app.rebuildLyricsAnimations()

        # 3. Remove from persisted JSON
        entries = self._loadLyricsJsonList()
        entries = [
            e for e in entries
            if e.get("lyricId") != lyricId
        ]
        entries.sort(key=lambda x: int(x.get("startChunk", 0)))
        self._saveLyricsJsonList(entries)

    def _entryFromLyricBox(self, lyricId, lb):
        memberNames = lb.memberNames if isinstance(lb.memberNames, list) else [lb.memberNames]
        return {
            "lyricId": lyricId,
            "linkedLabel": getattr(lb, "linkedLabel", None),
            "language": lb.language,
            "memberName": memberNames,
            "korean": lb.koreanLyric,
            "romanization": lb.romanization,
            "english": lb.englishTrans,
            "startChunk": lb.startChunk,
            "isAdLib": lb.isAdLib,
            "adLibDuration": int(lb.adLibDuration) if lb.isAdLib else 0,
            "anchorMode": getattr(lb, "anchorMode", "startChunk"),
        }

    def resyncLinkedLyrics(self, member, oldStartChunk, oldEndChunk, newStartChunk, newEndChunk):
        """
        Called when a label's span changes (drag/keyboard move). Any lyric whose
        linkedLabel snapshot matches the label's OLD span gets its snapshot refreshed
        to the new span, and — only if the label's startChunk itself moved — its
        own startChunk resynced to newStartChunk - LYRIC_LEAD_CHUNKS.
        """
        app = self.app
        changedAny = False
        for lyricId, lyric in list(app.lyrics.items()):
            linked = getattr(lyric, "linkedLabel", None)
            if not linked:
                continue
            if (linked.get("member") != member
                    or linked.get("startChunk") != oldStartChunk
                    or linked.get("endChunk") != oldEndChunk):
                continue

            lyric.linkedLabel = {"member": member, "startChunk": newStartChunk, "endChunk": newEndChunk}
            if newStartChunk != oldStartChunk:
                lyric.startChunk = max(0, newStartChunk - LYRIC_LEAD_CHUNKS)

            self.upsertLyricEntry(self._entryFromLyricBox(lyricId, lyric))
            changedAny = True

        if changedAny:
            app.rebuildLyricsAnimations()

    # ---------- UI ----------
    def openLyricsEditor(self, mode="add", existingLyricId=None, startChunk=None, memberName=None, linkedLabel=None):
        """
        mode: "add" or "edit"
        existingLyricId: only used in edit mode to identify the lyric being edited
        """
        if not ModalGuard.try_open("lyrics_menu"):
            return
        try:
            self._openLyricsEditorImpl(
                mode=mode,
                existingLyricId=existingLyricId,
                startChunk=startChunk,
                memberName=memberName,
                linkedLabel=linkedLabel,
            )
        except Exception:
            # The window's own onClose() normally does this cleanup, but it
            # never runs if construction blows up before wait_window (e.g. bad
            # lyric data) - without this, the guard stays acquired forever and
            # root keybinds/zoom (disabled below) never come back until the
            # app is restarted.
            ModalGuard.close("lyrics_menu")
            self.app.enableRootKeybinds()
            self.app.videoTrackItem.setUiBusy(False)
            raise

    def _openLyricsEditorImpl(self, mode="add", existingLyricId=None, startChunk=None, memberName=None, linkedLabel=None):
        app = self.app
        app.videoTrackItem.setUiBusy(True)

        # Prefill from existing lyric if editing
        prefillMembers = []
        prefillLang = "Korean"
        prefillKorean = ""
        prefillRoman = ""
        prefillEnglish = ""
        prefillIsAdLib = False
        prefillStartChunk = startChunk
        prefillLinkedLabel = linkedLabel

        if mode == "edit":
            if existingLyricId is None:
                raise ValueError("existingLyricId required for edit mode")

            lyric = app.lyrics.get(existingLyricId)
            if lyric is None:
                # If missing, just fall back to add mode
                mode = "add"
                existingLyricId = None
                prefillIsAdLib = False
            else:
                prefillIsAdLib = bool(getattr(lyric, "isAdLib", False))
                prefillMembers = list(getattr(lyric, "memberNames", []))
                prefillLang = getattr(lyric, "language", "Korean")
                prefillKorean = getattr(lyric, "koreanLyric", "")
                prefillRoman = getattr(lyric, "romanization", "")
                prefillEnglish = getattr(lyric, "englishTrans", "")
                prefillStartChunk = getattr(lyric, "startChunk", startChunk)
                prefillLinkedLabel = getattr(lyric, "linkedLabel", None)

        # If caller passed memberName for convenience (labels menu), use it if no prefill members exist
        if memberName and not prefillMembers:
            prefillMembers = [memberName]

        def _resolveLinkStatus():
            if not prefillLinkedLabel:
                return None
            idx = findLabelIndexBySpan(
                app.labels, prefillLinkedLabel.get("startChunk"), prefillLinkedLabel.get("endChunk"),
                member=prefillLinkedLabel.get("member")
            )
            label = f"{prefillLinkedLabel.get('member')} ({prefillLinkedLabel.get('startChunk')}-{prefillLinkedLabel.get('endChunk')})"
            if idx is None:
                return (f"Linked line: {label} — label not found (moved or deleted)", "#b8860b")
            return (f"Linked line: {label}", "#006400")

        inputWindow = tk.Toplevel(app.root)
        inputWindow.title("Edit Lyrics Box" if mode == "edit" else "Add Lyrics Box")
        inputWindow.geometry("600x400")
        inputWindow.transient(app.root)
        inputWindow.grab_set()

        app.disableRootKeybinds()

        # Make the window scrollable
        canvas = tk.Canvas(inputWindow)
        scrollFrame = tk.Frame(canvas)
        scrollbar = tk.Scrollbar(inputWindow, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        innerWindowId = canvas.create_window((0, 0), window=scrollFrame, anchor="nw")

        def updateScrollRegion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollFrame.bind("<Configure>", updateScrollRegion)

        # Make inner frame match canvas width so bbox is sane
        def onCanvasConfigure(e):
            canvas.itemconfig(innerWindowId, width=e.width)

        canvas.bind("<Configure>", onCanvasConfigure)

        def clampCanvasYView():
            first, last = canvas.yview()
            if first < 0:
                canvas.yview_moveto(0)
            elif last > 1:
                canvas.yview_moveto(1 - (last - first))

        def onMouseWheel(evt):
            if not inputWindow.winfo_exists():
                return "break"
            if evt.delta:
                canvas.yview_scroll(-1 * int(evt.delta / 120), "units")
                clampCanvasYView()
            return "break"

        def onLinuxWheel(evt):
            if not inputWindow.winfo_exists():
                return "break"
            if evt.num == 4:
                canvas.yview_scroll(-1, "units")
            elif evt.num == 5:
                canvas.yview_scroll(1, "units")
            clampCanvasYView()
            return "break"

        def bindWheelToWidget(widget):
            widget.bind("<MouseWheel>", onMouseWheel)
            widget.bind("<Button-4>", onLinuxWheel)
            widget.bind("<Button-5>", onLinuxWheel)

        # Bind to the whole window + containers
        bindWheelToWidget(inputWindow)
        bindWheelToWidget(canvas)
        bindWheelToWidget(scrollFrame)

        memberMapping = {m["name"]: m for m in app.members}
        memberMapping["All"] = {"name": "All", "color": "#000000"}
        memberNames = list(memberMapping.keys())

        tk.Label(scrollFrame, text="Member Name:").pack(pady=5)

        membersFrame = tk.Frame(scrollFrame)
        membersFrame.pack(pady=5)

        memberVars = []
        memberFrames = []

        def addMemberDropdown(defaultName=None):
            if len(memberVars) > 3:
                return

            frame = tk.Frame(membersFrame)
            frame.pack(pady=2)

            name = "All" if defaultName == "Gang Vocal" else defaultName
            var = tk.StringVar(value=name if name else memberNames[0])

            dropdown = tk.OptionMenu(frame, var, *memberNames)
            dropdown.pack(side="left")

            def removeMember():
                frame.destroy()
                memberVars.remove(var)
                memberFrames.remove(frame)

            removeButton = tk.Button(frame, text="X", command=removeMember)
            removeButton.pack(side="left", padx=5)

            memberVars.append(var)
            memberFrames.append(frame)

        # Prefill member dropdowns
        if prefillMembers:
            for m in prefillMembers:
                addMemberDropdown(m)
        else:
            addMemberDropdown(None)

        tk.Button(scrollFrame, text="Add Member", command=addMemberDropdown).pack(pady=5)

        langVar = tk.StringVar(value=prefillLang)
        readingFormatVar = tk.StringVar(value="romaji")

        def switchLanguage():
            isJapanese = (langVar.get() == "Japanese")

            koreanLabel.config(
                text="Japanese Lyric (Kanji/Kana):" if isJapanese else "Korean Lyric:"
            )
            romanLabel.config(
                text="Reading (Hiragana/Romaji):" if isJapanese else "Romanization:"
            )

            if isJapanese:
                if not japaneseToolsFrame.winfo_ismapped():
                    japaneseToolsFrame.pack(fill="x", padx=10, pady=(0, 5))
            else:
                japaneseToolsFrame.pack_forget()

            if langVar.get() == "Korean":
                if not koreanToolsFrame.winfo_ismapped():
                    koreanToolsFrame.pack(fill="x", padx=10, pady=(0, 5))
            else:
                koreanToolsFrame.pack_forget()

            if langVar.get() == "English":
                koreanFrame.pack_forget()
                romanFrame.pack_forget()
            else:
                if not koreanFrame.winfo_ismapped():
                    koreanFrame.pack(fill="x", pady=5, before=engFrame)
                if not romanFrame.winfo_ismapped():
                    romanFrame.pack(fill="x", pady=5, before=engFrame)

        langFrame = tk.Frame(scrollFrame)
        langFrame.pack(fill="x", pady=5)
        tk.Radiobutton(langFrame, text="Korean", variable=langVar, value="Korean", command=switchLanguage).pack(side="left", padx=10)
        tk.Radiobutton(langFrame, text="Japanese", variable=langVar, value="Japanese", command=switchLanguage).pack(side="left", padx=10)
        tk.Radiobutton(langFrame, text="English", variable=langVar, value="English", command=switchLanguage).pack(side="left")

        # Duplicate dropdown (keep for add; optional for edit)
        tk.Label(scrollFrame, text="Duplicate Existing Lyrics:").pack(pady=5)
        duplicateVar = tk.StringVar(value="None")
        duplicateOptionsMap = {}
        for lid, lyric in app.lyrics.items():
            optionText = f"{lyric.memberNames} -> {lyric.startChunk}"
            duplicateOptionsMap[optionText] = lid
        lyricOptions = ["None"] + list(duplicateOptionsMap.keys())
        tk.OptionMenu(scrollFrame, duplicateVar, *lyricOptions).pack(padx=5)

        # Korean/Japanese Lyric Field
        koreanFrame = tk.Frame(scrollFrame)
        koreanFrame.pack(fill="x", pady=5)
        koreanLabel = tk.Label(koreanFrame, text="Korean Lyric:")
        koreanLabel.pack(anchor="w")
        koreanEntry = tk.Text(koreanFrame, height=4, wrap="word", undo=True, autoseparators=True, maxundo=-1)
        koreanEntry.pack(fill="x", padx=10)

        # The pause marker is an invisible character in storage (core.lyric_text), so the editor shows a
        # visible stand-in; it is converted on load/save and stripped before any analysis below.
        def insertPause(event=None):
            koreanEntry.insert("insert", EDITOR_PAUSE_GLYPH)
            koreanEntry.focus_set()
            return "break"

        koreanEntry.bind("<Control-space>", insertPause)
        pauseRow = tk.Frame(koreanFrame)
        pauseRow.pack(fill="x", padx=10, pady=(2, 0))
        tk.Button(pauseRow, text=f"Insert pause {EDITOR_PAUSE_GLYPH}  (Ctrl+Space)", command=insertPause).pack(side="left")
        tk.Label(pauseRow, text=f"{EDITOR_PAUSE_GLYPH} = the singer pauses here (karaoke timing only; not shown on the lyric card)",
                 fg="grey").pack(side="left", padx=8)

        # Romanization / Reading Field
        romanFrame = tk.Frame(scrollFrame)
        romanFrame.pack(fill="x", pady=5)
        romanLabel = tk.Label(romanFrame, text="Romanization:")
        romanLabel.pack(anchor="w")
        romanEntry = tk.Text(romanFrame, height=4, wrap="word", undo=True, autoseparators=True, maxundo=-1)
        romanEntry.pack(fill="x", padx=10)

        # Japanese-only tools: auto-fill the reading from the Kanji/Kana text above
        japaneseToolsFrame = tk.Frame(romanFrame)

        tk.Radiobutton(japaneseToolsFrame, text="Romaji", variable=readingFormatVar, value="romaji").pack(side="left")
        tk.Radiobutton(japaneseToolsFrame, text="Hiragana", variable=readingFormatVar, value="hiragana").pack(side="left", padx=(5, 10))

        def autoFillReading():
            from core.japanese_utils import kanjiTextToReading

            kanjiText = fromEditorText(koreanEntry.get("1.0", "end")).strip("\n")
            if not kanjiText.strip():
                messagebox.showwarning(
                    "Nothing to Convert", "Type the Japanese lyric above first.", parent=inputWindow
                )
                return

            existingReading = romanEntry.get("1.0", "end").strip()
            if existingReading and not messagebox.askyesno(
                "Overwrite Reading?", "This will replace the current Reading text. Continue?",
                parent=inputWindow
            ):
                return

            reading = kanjiTextToReading(kanjiText, readingFormatVar.get())
            romanEntry.delete("1.0", "end")
            romanEntry.insert("1.0", reading)

        tk.Button(japaneseToolsFrame, text="Convert Kanji → Reading", command=autoFillReading).pack(side="left")

        def addKanjiToDocument():
            from core.kanji_reference import analyzeSelection
            from core import vocab_store_ja
            from core.vocab_sync import _renderSongHtmlFromDb

            selRanges = koreanEntry.tag_ranges("sel")
            if not selRanges:
                messagebox.showwarning(
                    "Nothing Selected", "Highlight a Kanji word above first.", parent=inputWindow
                )
                return

            try:
                selStart, selEnd = selRanges

                # Text.count(a, b, "chars") returns None instead of (0,) when a == b (a
                # documented Tcl/Tk quirk) - which happens here whenever the highlight starts
                # at the very first character of the field ("1.0"), since that's compared
                # against itself. Without the "or (0,)" fallback, [0] on that None crashes
                # with "TypeError: 'NoneType' object is not subscriptable" every time the
                # beginning of the text is highlighted.
                fullText = fromEditorText(koreanEntry.get("1.0", "end-1c"))
                startOffset = (koreanEntry.count("1.0", selStart, "chars") or (0,))[0]
                endOffset = (koreanEntry.count("1.0", selEnd, "chars") or (0,))[0]
                # Analyse (and store as the occurrence's lyric text) the clean line: no pause marker,
                # no "|" colour split, with the highlighted range re-expressed in the clean text.
                fullText, startOffset, endOffset = stripAllWithSelection(fullText, startOffset, endOffset)

                results = analyzeSelection(fullText, startOffset, endOffset)

                # Persist every word just analyzed into the SQLite vocab/SRS store (core.vocab_db)
                # rather than the old per-song JSON file - "one click" saves everything the button
                # showed, matching its "Add Kanji to Document" wording. Occurrence linking uses
                # whichever lyric this popup is editing (existingLyricId is None for a brand-new,
                # not-yet-saved lyric - the occurrence is simply keyed on the text itself then) and
                # its linked audio-label span when one exists, same span this editor already tracks
                # for resyncLinkedLyrics().
                if results:
                    singerNames = [v.get() for v in memberVars if v.get()]
                    occStartChunk = prefillLinkedLabel.get("startChunk") if prefillLinkedLabel else None
                    occEndChunk = prefillLinkedLabel.get("endChunk") if prefillLinkedLabel else None
                    for result in results:
                        vocabId, _ = vocab_store_ja.upsertVocab(result)
                        vocab_store_ja.addOccurrence(
                            vocabId, app.selectedGroup, app.songName, singerNames, fullText,
                            existingLyricId, occStartChunk, occEndChunk,
                        )
                    _renderSongHtmlFromDb(app.selectedGroup, app.songName)

                blocks = []
                for result in results:
                    surface, category = result["surface"], result["category"]
                    block = [f"{surface} ({result['reading']}) — {category}"]
                    if result["lemma"] != surface:
                        block.append(f"  Dictionary form: {result['lemma']} ({result['lemmaReading']})")

                    # Shown for every word regardless of category (Milestone 7) - a purely
                    # phonetic memorization aid (Mandarin pinyin for the Kyuujitai/Traditional
                    # form), never a claim that the word is actually Chinese - see
                    # .claude/KANJI_REFERENCE_PLAN.md.
                    pinyin = result["mandarinPinyin"]
                    block.append(f"  Mandarin: {pinyin['traditional']} ({pinyin['pinyin']})")

                    # Shown for every word regardless of category (Milestone 6, JMdict) - this is
                    # the only meaning a kunyomi word ever gets, since it has no Chinese-cognate
                    # shortcut at all - see .claude/KANJI_REFERENCE_PLAN.md.
                    meaning = result["japaneseMeaning"]
                    if meaning["status"] == "found":
                        block.append(f"  Meaning: {'; '.join(meaning['gloss'][:3])}")

                    # Never shown for kunyomi/jukujigo (native Japanese words) - analyzeSelection()
                    # only populates this for onyomi/mixed - see .claude/KANJI_REFERENCE_PLAN.md.
                    cognate = result["chineseCognate"]
                    if cognate and cognate["status"] == "confirmed":
                        gloss = "; ".join(cognate["gloss"][:2])
                        block.append(f"  Chinese: {cognate['traditional']} ({cognate['pinyin']}) — {gloss}")
                    elif cognate and cognate["status"] == "not_attested":
                        block.append(
                            f"  Chinese: not attested (Japan-coined term) — "
                            f"constructed pinyin: {cognate['pinyinFallback']}"
                        )

                    blocks.append("\n".join(block))
            except Exception as exc:
                # This is a test harness, not a finished feature (see .claude/KANJI_REFERENCE_PLAN.md) -
                # surface unexpected failures instead of letting Tkinter swallow them into the console,
                # where they're easy to mistake for the button silently doing nothing.
                import traceback
                traceback.print_exc()
                messagebox.showerror(
                    "Kanji Analysis Failed", f"{type(exc).__name__}: {exc}", parent=inputWindow
                )
                return

            if not results:
                messagebox.showinfo(
                    "Kanji Analysis", "No Kanji words found in the highlighted text.", parent=inputWindow
                )
                return

            _showCopyableInfo("Kanji Analysis", "\n\n".join(blocks), parent=inputWindow)

        tk.Button(japaneseToolsFrame, text="Add Kanji to Document", command=addKanjiToDocument).pack(side="left", padx=(10, 0))

        def breakDownGrammar():
            from core.grammar_breakdown import breakdownLine, groupIntoChunks

            selRanges = koreanEntry.tag_ranges("sel")
            if selRanges:
                selStart, selEnd = selRanges
                fullText = koreanEntry.get("1.0", "end-1c")
                startOffset = (koreanEntry.count("1.0", selStart, "chars") or (0,))[0]
                endOffset = (koreanEntry.count("1.0", selEnd, "chars") or (0,))[0]
                text = fullText[startOffset:endOffset]
            else:
                text = koreanEntry.get("1.0", "end-1c")
            text = stripAll(fromEditorText(text))

            if not text.strip():
                messagebox.showwarning(
                    "Nothing to Break Down", "Type or highlight Japanese lyric text above first.",
                    parent=inputWindow
                )
                return

            try:
                entries = breakdownLine(text)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                messagebox.showerror(
                    "Grammar Breakdown Failed", f"{type(exc).__name__}: {exc}", parent=inputWindow
                )
                return

            if not entries:
                messagebox.showinfo("Grammar Breakdown", "No words found in the text.", parent=inputWindow)
                return

            # Grouped into bunsetsu-like chunks (a content word plus the particles/auxiliaries
            # that attach to it) instead of one flat list of tokens - a flat list reads as a
            # "discombobulated blob of vocab" with no visible structure once a line has more
            # than 3-4 words (real user feedback), same complaint the ように/んだ pattern layer
            # already solved for individual constructions, just at the whole-line level now.
            # Each chunk is also tagged with its syntactic role (Object/Topic/Subject/...) when
            # the attached particle has one clear, well-known role - a label, not a translation:
            # the real pieces and their glosses are still shown underneath, same as before.
            lines = []
            for chunk in groupIntoChunks(entries):
                header = f"[{chunk['label']}] {chunk['surface']}" if chunk["label"] else chunk["surface"]
                lines.append(header)
                head = chunk["head"]
                lines.append(f"    Base: {head['surface']} — {head['gloss'] or '(no gloss)'}")
                for tailEntry in chunk["tail"]:
                    lines.append(f"    + {tailEntry['surface']} — {tailEntry['gloss'] or '(no gloss)'}")
                lines.append("")

            _showCopyableInfo("Grammar Breakdown", "\n".join(lines).rstrip("\n"), parent=inputWindow)

        tk.Button(japaneseToolsFrame, text="Grammar", command=breakDownGrammar).pack(side="left", padx=(10, 0))

        # Korean-only tools: a single "Grammar" button that breaks down the Hangul text above AND
        # looks up Hanja candidates for every content word in the same pass - merged per direct
        # user request, since a separate per-word "Show Hanja" button that required highlighting
        # one word at a time first was too tedious to be useful in practice (see
        # .claude/KOREAN_HANJA_PLAN.md / KOREAN_GRAMMAR_BREAKDOWN_PLAN.md; the merge itself lives
        # in core/korean_grammar_breakdown.py: breakdownLine()'s "hanja" field).
        koreanToolsFrame = tk.Frame(romanFrame)

        def breakDownKoreanGrammar():
            from core.korean_grammar_breakdown import breakdownLine as breakdownKoreanLine

            selRanges = koreanEntry.tag_ranges("sel")
            if selRanges:
                selStart, selEnd = selRanges
                text = koreanEntry.get(selStart, selEnd)
            else:
                text = koreanEntry.get("1.0", "end-1c")
            text = stripAll(fromEditorText(text))

            if not text.strip():
                messagebox.showwarning(
                    "Nothing to Break Down", "Type or highlight Korean lyric text above first.",
                    parent=inputWindow
                )
                return

            try:
                entries = breakdownKoreanLine(text)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                messagebox.showerror(
                    "Grammar Breakdown Failed", f"{type(exc).__name__}: {exc}", parent=inputWindow
                )
                return

            if not entries:
                messagebox.showinfo("Grammar Breakdown", "No words found in the text.", parent=inputWindow)
                return

            # Phase 1 scope (.claude/KOREAN_GRAMMAR_BREAKDOWN_PLAN.md): a flat per-token list, not
            # grouped into chunks like the Japanese version - function words are just indented
            # under whichever content word precedes them for readability. A content word's Hanja
            # candidates (see core/korean_hanja.py: lookupHanja(), merged into breakdownLine()'s
            # "hanja" field) are only ever shown when real candidates were actually found - per
            # direct user request, nothing is printed for the (overwhelmingly common) case of a
            # native Korean word with no Sino-Korean origin at all.
            lines = []
            for entry in entries:
                gloss = entry["gloss"] or "(no gloss)"
                prefix = "  " if entry["role"] == "function" else ""
                lines.append(f"{prefix}{entry['surface']} ({entry['role']}) — {gloss}")
                for candidate in entry["hanja"] or []:
                    hanjaGloss = "; ".join(candidate["gloss"][:2])
                    lines.append(
                        f"{prefix}    Hanja: {candidate['hanja']} ({candidate['pinyin']}) — {hanjaGloss}"
                    )

            _showCopyableInfo("Grammar Breakdown", "\n".join(lines), parent=inputWindow)

        tk.Button(koreanToolsFrame, text="Grammar", command=breakDownKoreanGrammar).pack(side="left", padx=(10, 0))

        # English Translation Field
        engFrame = tk.Frame(scrollFrame)
        engFrame.pack(fill="x", pady=5)
        tk.Label(engFrame, text="English Translation:").pack(anchor="w")
        engEntry = tk.Text(engFrame, height=4, wrap="word", undo=True, autoseparators=True, maxundo=-1)
        engEntry.pack(fill="x", padx=10)

        # Starting Chunk Field
        chunkFrame = tk.Frame(scrollFrame)
        chunkFrame.pack(fill="x", pady=5)
        tk.Label(chunkFrame, text="Starting Chunk:").pack(anchor="w")
        chunkEntry = tk.Entry(chunkFrame)
        if prefillStartChunk is not None:
            chunkEntry.insert(0, str(prefillStartChunk))
        chunkEntry.pack(fill="x", padx=10)

        linkStatus = _resolveLinkStatus()
        if linkStatus is not None:
            statusText, statusColor = linkStatus
            tk.Label(scrollFrame, text=statusText, fg=statusColor).pack(anchor="w", padx=10, pady=(2, 0))

        # Prefill text fields
        koreanEntry.insert("1.0", toEditorText(prefillKorean))
        romanEntry.insert("1.0", prefillRoman)
        engEntry.insert("1.0", prefillEnglish)

        # Don't let Ctrl+Z on the very first keystroke wipe out the prefilled/loaded
        # text - the undo stack should only track edits the user makes from here on.
        koreanEntry.edit_reset()
        romanEntry.edit_reset()
        engEntry.edit_reset()

        switchLanguage()
        
        adLibVar = tk.StringVar(value="AdLib" if prefillIsAdLib else "Normal")

        adLibFrame = tk.Frame(scrollFrame)
        adLibFrame.pack(fill="x", pady=8)

        tk.Label(adLibFrame, text="Line Type:").pack(side="left", padx=(0, 10))

        tk.Radiobutton(adLibFrame, text="Normal", variable=adLibVar, value="Normal").pack(side="left", padx=5)
        tk.Radiobutton(adLibFrame, text="Ad-lib", variable=adLibVar, value="AdLib").pack(side="left", padx=5)

        durationFrame = tk.Frame(scrollFrame)
        durationFrame.pack(fill="x", pady=(0, 8))
        tk.Label(durationFrame, text="Ad-lib Duration (40 ms chunks):").pack(anchor="w")

        adLibDurationEntry = tk.Entry(durationFrame)
        adLibDurationEntry.pack(fill="x", padx=10)

        adLibDurationEntry.insert(
            0,
            str(int(getattr(app.lyrics.get(existingLyricId), "adLibDuration", 50) or 50))
            if existingLyricId is not None and app.lyrics.get(existingLyricId) is not None else "50"
        )

        def syncDurationEnabled(*_args):
            isAdLib = (adLibVar.get() == "AdLib")
            state = "normal" if isAdLib else "disabled"
            adLibDurationEntry.configure(state=state)

        adLibVar.trace_add("write", syncDurationEnabled)
        syncDurationEnabled()

        def fillFromDuplicate(*_args):
            selectedText = duplicateVar.get()
            if selectedText == "None":
                return
            selectedLyricId = duplicateOptionsMap.get(selectedText)
            selectedLyric = app.lyrics.get(selectedLyricId)
            if selectedLyric is None:
                return

            langVar.set(selectedLyric.language)
            switchLanguage()
            
            isAdLib = bool(getattr(selectedLyric, "isAdLib", False))
            adLibVar.set("AdLib" if isAdLib else "Normal")

            koreanEntry.delete("1.0", "end")
            koreanEntry.insert("1.0", toEditorText(selectedLyric.koreanLyric))

            romanEntry.delete("1.0", "end")
            romanEntry.insert("1.0", selectedLyric.romanization)

            engEntry.delete("1.0", "end")
            engEntry.insert("1.0", selectedLyric.englishTrans)
            
            adLibDurationEntry.configure(state="normal")
            adLibDurationEntry.delete(0, "end")
            adLibDurationEntry.insert(0, str(int(getattr(selectedLyric, "adLibDuration", 50) or 50)))
            syncDurationEnabled()

        duplicateVar.trace("w", fillFromDuplicate)

        def submit():
            selectedMembers = [v.get() for v in memberVars]

            if len(selectedMembers) != len(set(selectedMembers)):
                messagebox.showwarning("Duplicate Members", "Each member must be unique. Please select different members.")
                return

            try:
                newStartChunk = int(chunkEntry.get())
            except ValueError:
                messagebox.showwarning("Invalid Chunk", "Starting Chunk must be an integer.")
                return

            hasNativeText = langVar.get() in ("Korean", "Japanese")
            koreanLyric = fromEditorText(koreanEntry.get("1.0", "end")).strip() if hasNativeText else ""
            romanization = romanEntry.get("1.0", "end").strip() if hasNativeText else ""
            englishTrans = engEntry.get("1.0", "end").strip()

            isAdLib = (adLibVar.get() == "AdLib")
            adLibDuration = 50
            if isAdLib:
                try:
                    adLibDuration = int(adLibDurationEntry.get())
                except ValueError:
                    messagebox.showwarning("Invalid Duration", "Ad-lib duration must be an integer (seconds).")
                    return
            
            self._commitLyric(
                existingLyricId=existingLyricId if mode == "edit" else None,
                newStartChunk=newStartChunk,
                selectedMembers=selectedMembers,
                language=langVar.get(),
                koreanLyric=koreanLyric,
                romanization=romanization,
                englishTrans=englishTrans,
                isAdLib=isAdLib,
                adLibDuration=adLibDuration,
                anchorMode="startChunk",
                linkedLabel=prefillLinkedLabel,
            )
            app.enableRootKeybinds()
            onClose()

        submitFrame = tk.Frame(inputWindow)
        submitFrame.pack(side="bottom")
        tk.Button(submitFrame, text="Save" if mode == "edit" else "Submit", command=submit).pack(pady=10, fill="x")

        def onClose():
            app.enableRootKeybinds()
            ModalGuard.close("lyrics_menu")
            app.videoTrackItem.setUiBusy(False)
            inputWindow.destroy()

        inputWindow.protocol("WM_DELETE_WINDOW", onClose)
        app.root.wait_window(inputWindow)
    
    def openLyricsEditorMenu(self, event=None):
        if not ModalGuard.try_open("lyrics_edit_menu"):
            return
        try:
            self._openLyricsEditorMenuImpl()
        except Exception:
            # Same safety net as openLyricsEditor: the window's own onClose()
            # normally releases the guard and re-enables keybinds/zoom, but it
            # never runs if construction blows up before wait_window - without
            # this, 'L' would silently do nothing until the app is restarted.
            ModalGuard.close("lyrics_edit_menu")
            self.app.enableRootKeybinds()
            self.app.videoTrackItem.setUiBusy(False)
            raise

    def _openLyricsEditorMenuImpl(self):
        app = self.app
        app.videoTrackItem.setUiBusy(True)
        
        # Decide width = min(windowSize//2, rootWidth//2), with safe fallbacks
        rootW = app.root.winfo_width() or 1920
        rootH = app.root.winfo_height() or 1080
        windowSize = getattr(app, "windowSize", rootW)  # if you have a windowSize attribute
        maxWidth = max(520, min(rootW // 2, int(windowSize // 2)))
        height = max(450, int(rootH * 0.75))
        
        win = tk.Toplevel(app.root)
        win.title("Lyrics Editor")
        win.geometry(f"{maxWidth}x{height}")
        win.transient(app.root)
        win.grab_set()
        
        app.disableRootKeybinds()

        # Header
        header = tk.Frame(win)
        header.pack(fill="x", padx=10, pady=(10, 0))
        tk.Label(header, text="Lyrics Editor", font=("Arial", 14, "bold")).pack(side="left")
        
        # Scrollable body
        body = tk.Frame(win)
        body.pack(fill="both", expand=True, padx=10, pady=10)
        
        canvas = tk.Canvas(body, highlightthickness=0)
        scrollFrame = tk.Frame(canvas)
        scrollbar = tk.Scrollbar(body, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        innerWindowId = canvas.create_window((0, 0), window=scrollFrame, anchor="nw")
        
        def updateScrollRegion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def onMouseWheel(evt):
            if not win.winfo_exists():
                return "break"
            if evt.delta:
                canvas.yview_scroll(-1 * int(evt.delta / 120), "units")
            return "break"

        def onLinuxWheel(evt):
            if not win.winfo_exists():
                return "break"
            if evt.num == 4:
                canvas.yview_scroll(-1, "units")
            elif evt.num == 5:
                canvas.yview_scroll(1, "units")
            return "break"

        def bindWheel(widget):
            widget.bind("<MouseWheel>", onMouseWheel)
            widget.bind("<Button-4>", onLinuxWheel)
            widget.bind("<Button-5>", onLinuxWheel)

        scrollFrame.bind("<Configure>", updateScrollRegion)
        bindWheel(win)
        bindWheel(body)
        bindWheel(canvas)
        bindWheel(scrollFrame)
        
        # Render list + refresh helper
        def clearFrame(frame):
            for child in frame.winfo_children():
                child.destroy()

        refreshGeneration = [0]

        def refreshList():
            # Bump the generation so any still-pending batch from a PREVIOUS
            # refreshList() call (e.g. the user clicked Edit/Delete again
            # before the last chunked build finished) recognizes it's stale
            # and stops instead of adding rows into a frame this call is
            # about to clear.
            refreshGeneration[0] += 1
            myGen = refreshGeneration[0]

            clearFrame(scrollFrame)

            # Sort lyrics by startChunk
            items = sorted(app.lyrics.items(), key=lambda kv: kv[1].startChunk)

            if not items:
                tk.Label(
                    scrollFrame,
                    text="No lyrics yet. Click 'Add Lyrics' at the bottom to add one.",
                    anchor="w",
                    justify="left",
                    wraplength=maxWidth - 40
                ).pack(fill="x", pady=10)
                return

            buildItemsChunk(items, 0, myGen)

        def buildItemsChunk(items, startIdx, myGen, chunkSize=10):
            """
            Build rows in small batches instead of all at once. A typical
            song has 30-50 lyrics and each row builds a tagged Text widget,
            a preview Text widget, and two buttons - building them all in
            one synchronous call can run long enough that the Tk event loop
            never gets control back, freezing video/audio-sync (both driven
            by after() timers) while this menu is opening during playback.
            """
            if not win.winfo_exists():
                # Menu was closed while a batch was still pending.
                return

            if myGen != refreshGeneration[0]:
                # A newer refreshList() call superseded this one.
                return

            endIdx = min(startIdx + chunkSize, len(items))
            for lyricId, lyric in items[startIdx:endIdx]:
                startChunk = lyric.startChunk

                # Build member string like "Tzuyu/Mina"
                memberNames = getattr(lyric, "memberNames", [])
                if isinstance(memberNames, str):
                    memberNames = [memberNames]
                memberStr = "/".join(memberNames) if memberNames else "Unknown"

                # Row color for lyric text (first member)
                rowColor = "#000000"
                if memberNames:
                    rowColor = app.getMemberColor(memberNames[0], forLyrics=True) or "#000000"

                # Preview: Korean then English
                korean = toEditorText(getattr(lyric, "koreanLyric", "") or "")
                english = getattr(lyric, "englishTrans", "") or ""
                kPreview = _preview(korean, maxLines=2)
                ePreview = _preview(english, maxLines=2)

                previewText = ""
                if kPreview:
                    previewText += kPreview
                if ePreview:
                    previewText += ("\n" if previewText else "") + ePreview
                if not previewText:
                    previewText = "(no text)"

                # Row container
                row = tk.Frame(scrollFrame, bd=1, relief="solid")
                row.pack(fill="x", pady=6)
                bindWheel(row)

                # Top line: Start Chunk + Members (per-member name colors + prefix colored)
                titlePrefix = f"Start Chunk: {startChunk}  —  "

                titleFrame = tk.Frame(row)
                titleFrame.pack(fill="x", padx=10, pady=(8, 2))
                bindWheel(titleFrame)

                titleText = tk.Text(
                    titleFrame,
                    height=1,
                    wrap="none",
                    bd=0,
                    highlightthickness=0,
                    padx=0,
                    pady=0
                )
                titleText.pack(fill="x", expand=True)
                bindWheel(titleText)
                titleText.configure(font=("Arial", 11, "bold"))

                # Insert prefix with rowColor
                prefixTag = f"prefixColor_{lyricId}"
                titleText.insert("1.0", titlePrefix)
                titleText.tag_add(prefixTag, "1.0", titleText.index("end-1c"))
                titleText.tag_config(prefixTag, foreground=rowColor)

                # Insert member names with per-name color tags
                members = memberNames if isinstance(memberNames, (list, tuple)) else [memberStr]
                for idx, name in enumerate(members):
                    if idx > 0:
                        titleText.insert("end", "/")

                    startIndex = titleText.index("end-1c")
                    titleText.insert("end", name)
                    endIndex = titleText.index("end-1c")

                    color = app.getMemberColor(name, forLyrics=True) or rowColor
                    tagName = f"memberColor_{lyricId}_{idx}"
                    titleText.tag_add(tagName, startIndex, endIndex)
                    titleText.tag_config(tagName, foreground=color)

                titleText.configure(state="disabled")

                # Link status (if this lyric was created from a label row)
                linkedLabel = getattr(lyric, "linkedLabel", None)
                if linkedLabel:
                    idx = findLabelIndexBySpan(
                        app.labels, linkedLabel.get("startChunk"), linkedLabel.get("endChunk"),
                        member=linkedLabel.get("member")
                    )
                    labelDesc = f"{linkedLabel.get('member')} ({linkedLabel.get('startChunk')}-{linkedLabel.get('endChunk')})"
                    if idx is None:
                        linkText = f"Linked line: {labelDesc} — not found (moved or deleted)"
                        linkColor = "#b8860b"
                    else:
                        linkText = f"Linked line: {labelDesc}"
                        linkColor = "#006400"
                    tk.Label(
                        row, text=linkText, fg=linkColor, anchor="w", font=("Arial", 9)
                    ).pack(fill="x", padx=10, pady=(0, 2))

                # Preview (same rowColor)
                previewWidget = tk.Text(
                    row,
                    height=3,
                    wrap="word",
                    bd=0,
                    highlightthickness=0
                )
                previewWidget.pack(fill="x", padx=10, pady=(0, 8))
                previewWidget.insert("1.0", previewText)
                bindWheel(previewWidget)

                previewTag = f"previewColor_{lyricId}"
                previewWidget.tag_add(previewTag, "1.0", "end-1c")
                previewWidget.tag_config(previewTag, foreground=rowColor)
                previewWidget.configure(state="disabled")

                # Buttons
                btns = tk.Frame(row)
                btns.pack(fill="x", padx=10, pady=(0, 10))
                bindWheel(btns)

                def onEdit(lid=lyricId):
                    # Opens editor in edit mode
                    self.editLyricsBox(lid)
                    refreshList()
                    updateScrollRegion()

                def onDelete(lid=lyricId, sc=startChunk):
                    if not messagebox.askyesno(
                        "Delete Lyric",
                        f"Delete lyric at startChunk {sc}?\n\nThis cannot be undone.",
                        parent=self.app.root
                    ):
                        return

                    # Remove from runtime dict + canvas
                    if lid in app.lyrics:
                        old = app.lyrics.pop(lid)
                        if hasattr(old, "destroy"):
                            old.destroy()

                    # Remove from JSON
                    self.deleteLyricEntry(lid)

                    # Rebuild animations / redraw
                    app.rebuildLyricsAnimations()

                    # Refresh UI list
                    refreshList()
                    updateScrollRegion()

                tk.Button(btns, text="Edit Lyric", command=onEdit).pack(side="left")
                tk.Button(btns, text="Delete Lyric", command=onDelete).pack(side="left", padx=8)

            if endIdx < len(items):
                win.after(1, lambda: buildItemsChunk(items, endIdx, myGen, chunkSize))

        refreshList()
        updateScrollRegion()

        # Bottom bar with Add button
        bottom = tk.Frame(win)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        def onAdd():
            # Opens add mode
            self.addLyricBox()

        tk.Button(bottom, text="Add Lyrics", command=onAdd).pack(fill="x")

        def onClose():
            app.enableRootKeybinds()
            ModalGuard.close("lyrics_edit_menu")
            app.videoTrackItem.setUiBusy(False)
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", onClose)
        app.root.wait_window(win)
    
    def _commitLyric(
        self,
        existingLyricId,  # None if adding
        newStartChunk,
        selectedMembers,
        language,
        koreanLyric,
        romanization,
        englishTrans,
        isAdLib,
        adLibDuration=50,
        anchorMode="startChunk",
        linkedLabel=None,
    ):
        app = self.app
        newStartChunk = int(newStartChunk)
        lyricId = existingLyricId or str(uuid.uuid4())

        # ---- 1) Destroy the old canvas object if we're editing in place ----
        if existingLyricId is not None:
            oldBox = app.lyrics.get(existingLyricId)
            if oldBox is not None and hasattr(oldBox, "destroy"):
                oldBox.destroy()

        # ---- 2) Build new LyricBox (fresh object) ----
        circleImages = app._getCircleImages(selectedMembers)
        lyricBox = LyricBox(
            app.canvas, app, selectedMembers, circleImages,
            koreanLyric, romanization, englishTrans,
            newStartChunk, language, lyricId, isAdLib=isAdLib, adLibDuration=adLibDuration,
            linkedLabel=linkedLabel
        )
        lyricBox.anchorMode = anchorMode

        # ---- 3) Install + rebuild ----
        app.lyrics[lyricId] = lyricBox
        app.rebuildLyricsAnimations()

        # ---- 4) Persist JSON (upsert by lyricId) ----
        entry = {
            "lyricId": lyricId,
            "linkedLabel": linkedLabel,
            "language": language,
            "memberName": selectedMembers,
            "korean": koreanLyric,
            "romanization": romanization,
            "english": englishTrans,
            "startChunk": newStartChunk,
            "isAdLib": isAdLib,
            "adLibDuration": int(adLibDuration) if isAdLib else 0,
            "anchorMode": anchorMode,
        }
        self.upsertLyricEntry(entry)

    def addLyricBox(self, event=None, startChunk=None, memberName=None, linkedLabel=None):
        # Backwards-compatible wrapper for old call sites
        return self.openLyricsEditor(mode="add", startChunk=startChunk, memberName=memberName, linkedLabel=linkedLabel)

    def editLyricsBox(self, lyricId: str):
        # Called from your lyrics menu "Edit" button
        return self.openLyricsEditor(mode="edit", existingLyricId=lyricId)