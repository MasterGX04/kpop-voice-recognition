import os
import json
import codecs
import tkinter as tk
from tkinter import messagebox
from lyrics_box import LyricBox

def _truncate(text: str, maxChars: int = 80) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= maxChars:
        return text
    return text[: maxChars - 1] + "…"

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

    def upsertLyricEntry(self, entry):
        """
        Insert or replace by startChunk (so edits are easy later).
        Current addLyricBox behavior appends duplicates; upsert is safer for edit support.
        """
        entries = self._loadLyricsJsonList()
        startChunk = int(entry["startChunk"])

        replaced = False
        for i, e in enumerate(entries):
            if int(e.get("startChunk", -1)) == startChunk:
                entries[i] = entry
                replaced = True
                break

        if not replaced:
            entries.append(entry)

        entries.sort(key=lambda x: int(x.get("startChunk", 0)))
        self._saveLyricsJsonList(entries)

    def deleteLyricEntry(self, startChunk: int):
        """
        Fully delete a lyric:
        - remove canvas items (destroy LyricBox)
        - remove from app.lyrics
        - rebuild animations
        - remove from JSON
        """
        app = self.app
        startChunk = int(startChunk)

        # 1. Remove runtime LyricBox + canvas items
        if startChunk in app.lyrics:
            lyricBox = app.lyrics.pop(startChunk)

            # Properly delete all canvas objects
            if hasattr(lyricBox, "destroy"):
                lyricBox.destroy()

        # 2. Rebuild animations so state is clean
        app.rebuildLyricsAnimations()

        # 3. Remove from persisted JSON
        entries = self._loadLyricsJsonList()
        entries = [
            e for e in entries
            if int(e.get("startChunk", -1)) != startChunk
        ]
        entries.sort(key=lambda x: int(x.get("startChunk", 0)))
        self._saveLyricsJsonList(entries)

    # ---------- UI ----------
    def openLyricsEditor(self, mode="add", existingStartChunk=None, startChunk=None, memberName=None):
        """
        mode: "add" or "edit"
        existingStartChunk: only used in edit mode to identify the original lyric key
        """
        app = self.app

        # Prefill from existing lyric if editing
        prefillMembers = []
        prefillLang = "Korean"
        prefillKorean = ""
        prefillRoman = ""
        prefillEnglish = ""
        prefillIsAdLib = False
        prefillStartChunk = startChunk

        if mode == "edit":
            if existingStartChunk is None:
                raise ValueError("existingStartChunk required for edit mode")

            lyric = app.lyrics.get(int(existingStartChunk))
            if lyric is None:
                # If missing, just fall back to add mode at that chunk
                mode = "add"
                prefillStartChunk = existingStartChunk
                prefillIsAdLib = False
            else:
                prefillIsAdLib = bool(getattr(lyric, "isAdLib", False))
                prefillMembers = list(getattr(lyric, "memberNames", []))
                prefillLang = getattr(lyric, "language", "Korean")
                prefillKorean = getattr(lyric, "koreanLyric", "")
                prefillRoman = getattr(lyric, "romanization", "")
                prefillEnglish = getattr(lyric, "englishTrans", "")
                prefillStartChunk = getattr(lyric, "startChunk", existingStartChunk)

        # If caller passed memberName for convenience (labels menu), use it if no prefill members exist
        if memberName and not prefillMembers:
            prefillMembers = [memberName]

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

        def switchLanguage():
            if langVar.get() == "English":
                koreanFrame.pack_forget()
                romanFrame.pack_forget()
            else:
                koreanFrame.pack(fill="x", pady=5)
                romanFrame.pack(fill="x", pady=5)

        langFrame = tk.Frame(scrollFrame)
        langFrame.pack(fill="x", pady=5)
        tk.Radiobutton(langFrame, text="Korean", variable=langVar, value="Korean", command=switchLanguage).pack(side="left", padx=10)
        tk.Radiobutton(langFrame, text="English", variable=langVar, value="English", command=switchLanguage).pack(side="left")

        # Duplicate dropdown (keep for add; optional for edit)
        tk.Label(scrollFrame, text="Duplicate Existing Lyrics:").pack(pady=5)
        duplicateVar = tk.StringVar(value="None")
        lyricOptions = ["None"] + [f"{lyric.memberNames} -> {lyric.startChunk}" for lyric in app.lyrics.values()]
        tk.OptionMenu(scrollFrame, duplicateVar, *lyricOptions).pack(padx=5)

        # Korean Lyric Field
        koreanFrame = tk.Frame(scrollFrame)
        koreanFrame.pack(fill="x", pady=5)
        tk.Label(koreanFrame, text="Korean Lyric:").pack(anchor="w")
        koreanEntry = tk.Text(koreanFrame, height=4, wrap="word")
        koreanEntry.pack(fill="x", padx=10)

        # Romanization Field
        romanFrame = tk.Frame(scrollFrame)
        romanFrame.pack(fill="x", pady=5)
        tk.Label(romanFrame, text="Romanization:").pack(anchor="w")
        romanEntry = tk.Text(romanFrame, height=4, wrap="word")
        romanEntry.pack(fill="x", padx=10)

        # English Translation Field
        engFrame = tk.Frame(scrollFrame)
        engFrame.pack(fill="x", pady=5)
        tk.Label(engFrame, text="English Translation:").pack(anchor="w")
        engEntry = tk.Text(engFrame, height=4, wrap="word")
        engEntry.pack(fill="x", padx=10)

        # Starting Chunk Field
        chunkFrame = tk.Frame(scrollFrame)
        chunkFrame.pack(fill="x", pady=5)
        tk.Label(chunkFrame, text="Starting Chunk:").pack(anchor="w")
        chunkEntry = tk.Entry(chunkFrame)
        if prefillStartChunk is not None:
            chunkEntry.insert(0, str(prefillStartChunk))
        chunkEntry.pack(fill="x", padx=10)

        # Prefill text fields
        koreanEntry.insert("1.0", prefillKorean)
        romanEntry.insert("1.0", prefillRoman)
        engEntry.insert("1.0", prefillEnglish)
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
        adLibDurationEntry.insert(0, 50)

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
            selectedChunk = int(selectedText.split(" -> ")[1])
            selectedLyric = app.lyrics[selectedChunk]

            langVar.set(selectedLyric.language)
            switchLanguage()
            
            isAdLib = bool(getattr(selectedLyric, "isAdLib", False))
            adLibVar.set("AdLib" if isAdLib else "Normal")

            koreanEntry.delete("1.0", "end")
            koreanEntry.insert("1.0", selectedLyric.koreanLyric)

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

            koreanLyric = koreanEntry.get("1.0", "end").strip() if langVar.get() == "Korean" else ""
            romanization = romanEntry.get("1.0", "end").strip() if langVar.get() == "Korean" else ""
            englishTrans = engEntry.get("1.0", "end").strip()

            # If editing and the key changed, remove the old one first
            if mode == "edit" and existingStartChunk is not None:
                oldKey = int(existingStartChunk)
                
                # Case A: startChunk changed -> destroy oldKey
                if oldKey != newStartChunk and oldKey in app.lyrics:
                    oldLyric = app.lyrics.pop(oldKey)
                    if hasattr(oldLyric, "destroy"):
                        oldLyric.destroy()

                # Case B: startChunk unchanged -> destroy existing at that key before replacing
                if oldKey == newStartChunk and newStartChunk in app.lyrics:
                    oldLyric = app.lyrics[newStartChunk]
                    if hasattr(oldLyric, "destroy"):
                        oldLyric.destroy()

            isAdLib = (adLibVar.get() == "AdLib")
            adLibDuration = 50
            if isAdLib:
                try:
                    adLibDuration = int(adLibDurationEntry.get())
                except ValueError:
                    messagebox.showwarning("Invalid Duration", "Ad-lib duration must be an integer (seconds).")
                    return
            
            self._commitLyric(
                existingStartChunk=existingStartChunk if mode == "edit" else None,
                newStartChunk=newStartChunk,
                selectedMembers=selectedMembers,
                language=langVar.get(),
                koreanLyric=koreanLyric,
                romanization=romanization,
                englishTrans=englishTrans,
                isAdLib=isAdLib,
                adLibDuration=adLibDuration,
                anchorMode="startChunk",
            )
            app.enableRootKeybinds()
            inputWindow.destroy()

        submitFrame = tk.Frame(inputWindow)
        submitFrame.pack(side="bottom")
        tk.Button(submitFrame, text="Save" if mode == "edit" else "Submit", command=submit).pack(pady=10, fill="x")

        def onClose():
            app.enableRootKeybinds()
            inputWindow.destroy()

        inputWindow.protocol("WM_DELETE_WINDOW", onClose)
        app.root.wait_window(inputWindow)
    
    def openLyricsEditorMenu(self, event=None):
        app = self.app
        
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
        canvas.create_window((0, 0), window=scrollFrame, anchor="nw")
        
        def updateScrollRegion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def onMouseWheel(evt):
            if win.winfo_exists() and evt.delta:
                canvas.yview_scroll(-1 * int(evt.delta / 120), "units")

        def onLinuxWheel(evt):
            if not win.winfo_exists():
                return
            if evt.num == 4:
                canvas.yview_scroll(-1, "units")
            elif evt.num == 5:
                canvas.yview_scroll(1, "units")

        canvas.bind("<MouseWheel>", onMouseWheel)
        canvas.bind("<Button-4>", onLinuxWheel)
        canvas.bind("<Button-5>", onLinuxWheel)
        canvas.bind("<Enter>", lambda _e: canvas.focus_set())

        scrollFrame.bind("<Configure>", updateScrollRegion)
        
        # Render list + refresh helper
        def clearFrame(frame):
            for child in frame.winfo_children():
                child.destroy()

        def refreshList():
            clearFrame(scrollFrame)

            # Sort lyrics by startChunk
            items = sorted(app.lyrics.items(), key=lambda kv: int(kv[0]))

            if not items:
                tk.Label(
                    scrollFrame,
                    text="No lyrics yet. Click 'Add Lyrics' at the bottom to add one.",
                    anchor="w",
                    justify="left",
                    wraplength=maxWidth - 40
                ).pack(fill="x", pady=10)
                return

            for startChunk, lyric in items:
                startChunk = int(startChunk)

                # Build member string like "Tzuyu/Mina"
                memberNames = getattr(lyric, "memberNames", [])
                if isinstance(memberNames, str):
                    memberNames = [memberNames]
                memberStr = "/".join(memberNames) if memberNames else "Unknown"

                # Preview: Korean then English (or just English if Korean blank)
                korean = getattr(lyric, "koreanLyric", "") or ""
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

                # Top line: Start Chunk + Members (with per-member color)
                titlePrefix = f"Start Chunk: {startChunk}  —  "

                titleFrame = tk.Frame(row)
                titleFrame.pack(fill="x", padx=10, pady=(8, 2))

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

                # Make it look like a label
                titleText.configure(font=("Arial", 11, "bold"))

                # Insert prefix
                titleText.insert("1.0", titlePrefix)

                # Insert member names with per-name color tags
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
                korean = getattr(lyric, "koreanLyric", "") or ""
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

                # Top line: Start Chunk + Members (per-member name colors + prefix colored)
                titlePrefix = f"Start Chunk: {startChunk}  —  "

                titleFrame = tk.Frame(row)
                titleFrame.pack(fill="x", padx=10, pady=(8, 2))

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
                titleText.configure(font=("Arial", 11, "bold"))

                # Insert prefix with rowColor
                prefixTag = f"prefixColor_{startChunk}"
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
                    tagName = f"memberColor_{startChunk}_{idx}"
                    titleText.tag_add(tagName, startIndex, endIndex)
                    titleText.tag_config(tagName, foreground=color)

                titleText.configure(state="disabled")

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

                previewTag = f"previewColor_{startChunk}"
                previewWidget.tag_add(previewTag, "1.0", "end-1c")
                previewWidget.tag_config(previewTag, foreground=rowColor)
                previewWidget.configure(state="disabled")

                # Buttons
                btns = tk.Frame(row)
                btns.pack(fill="x", padx=10, pady=(0, 10))

                def onEdit(sc=startChunk):
                    # Opens editor in edit mode
                    self.editLyricsBox(sc)
                    refreshList()
                    updateScrollRegion()

                def onDelete(sc=startChunk):
                    if not messagebox.askyesno(
                        "Delete Lyric",
                        f"Delete lyric at startChunk {sc}?\n\nThis cannot be undone."
                    ):
                        return

                    # Remove from runtime dict + canvas
                    if sc in app.lyrics:
                        old = app.lyrics.pop(sc)
                        if hasattr(old, "destroy"):
                            old.destroy()

                    # Remove from JSON
                    self.deleteLyricEntry(sc)

                    # Rebuild animations / redraw
                    app.rebuildLyricsAnimations()

                    # Refresh UI list
                    refreshList()
                    updateScrollRegion()

                tk.Button(btns, text="Edit Lyric", command=onEdit).pack(side="left")
                tk.Button(btns, text="Delete Lyric", command=onDelete).pack(side="left", padx=8)

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
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", onClose)
        app.root.wait_window(win)
    
    def _commitLyric(
        self,
        existingStartChunk, # None if adding
        newStartChunk,
        selectedMembers,
        language,
        koreanLyric,
        romanization,
        englishTrans,
        isAdLib,
        adLibDuration=50,
        anchorMode="startChunk",
    ):
        app = self.app
        newStartChunk = int(newStartChunk)

        # If editing and the startChunk changed, remove the old key first
        if existingStartChunk is not None:
            existingStartChunk = int(existingStartChunk)
            if existingStartChunk != newStartChunk:
                if existingStartChunk in app.lyrics:
                    # also delete the canvas items if LyricBox exposes a destroy()
                    old = app.lyrics.pop(existingStartChunk)
                    if hasattr(old, "destroy"):
                        old.destroy()

        # Rebuild LyricBox instance (fresh object is easiest & safest)
        circleImages = app._getCircleImages(selectedMembers)
        lyricBox = LyricBox(
            app.canvas, app, selectedMembers, circleImages,
            koreanLyric, romanization, englishTrans,
            newStartChunk, language, isAdLib=isAdLib, adLibDuration=adLibDuration
        )

        # attach optional linkage metadata (new format)
        lyricBox.anchorMode = anchorMode

        app.lyrics[newStartChunk] = lyricBox
        app.rebuildLyricsAnimations()

        # Persist JSON in the new schema
        entry = {
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
        
    def addLyricBox(self, event=None, startChunk=None, memberName=None):
        app = self.app

        if startChunk is not None:
            try:
                startChunk = int(startChunk)
            except ValueError:
                # Let the editor handle invalid input if someone calls it weirdly
                pass
            else:
                if startChunk in app.lyrics:
                    messagebox.showerror(
                        "Lyric Already Exists",
                        f"A lyric already exists at startChunk {startChunk}.\n\n"
                        "Use Edit Lyrics to modify the existing lyric instead."
                    )
                    return
        # Backwards-compatible wrapper for old call sites
        return self.openLyricsEditor(mode="add", startChunk=startChunk, memberName=memberName)

    def editLyricsBox(self, startChunk: int):
        # Called from your lyrics menu "Edit" button
        return self.openLyricsEditor(mode="edit", existingStartChunk=int(startChunk))