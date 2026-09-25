import tkinter as tk
from tkinter import ttk
import copy
from core.util_functions import ModalGuard, findLabelIndexBySpan
from gui.lyrics_box import LYRIC_LEAD_CHUNKS

class AddLabelsMenu:
    def __init__(self, app):
        """
        'app' is the main VoiceDetectionApp instance.
        """
        self.app = app
        self.window = None
        
        # State variables (previously trapped inside the function)
        self.checkboxes = {}
        self.checkboxesByIndex = []
        self.backingVars = {}
        self.adLibVars = {}
        self.rowToLabelIndex = []
        self.labelKeys = []
        self.rowWidgets = {}
        self.deletedLabelIndices = set()
        self._pendingRows = []  # rows still to be built by _buildRowsChunk()

        # Shift-click state
        self.lastClicked = {"main": -1, "repeat": -1, "adlib": -1}
        self.shiftRange = {"main": (-1, -1), "repeat": (-1, -1), "adlib": (-1, -1)}
        
        self.memberVar = None

    def show(self):
        if not ModalGuard.try_open("labels_menu"):
            return
        try:
            self._showImpl()
        except Exception:
            # close_menu() normally does this cleanup, but it never runs if
            # construction blows up before the window can be interacted with -
            # without this, the guard stays acquired and keybinds/zoom stay
            # disabled forever, same failure mode fixed in close_menu() above.
            ModalGuard.close("labels_menu")
            self.app.enableRootKeybinds()
            self.app.videoTrackItem.setUiBusy(False)
            raise

    def _showImpl(self):
        self.app.disableRootKeybinds()
        self.app.videoTrackItem.setUiBusy(True)

        self.window = tk.Toplevel(self.app.root)
        self.window.title("Add labels")
        self.window.geometry("700x600")
        self.window.transient(self.app.root)
        self.window.grab_set()
        self.window.protocol("WM_DELETE_WINDOW", self.close_menu)
        
        # --- UI Setup ---
        checklistFrame = tk.Frame(self.window)
        checklistFrame.pack(pady=0, fill="both", expand=True)
        
        self.canvas = tk.Canvas(checklistFrame)
        self.scrollFrame = tk.Frame(self.canvas)
        scrollbar = tk.Scrollbar(checklistFrame, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.create_window((0, 0), window=self.scrollFrame, anchor="nw")
        
        self.scrollFrame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._bind_scrolling(checklistFrame)

        # Build the dynamic UI elements. _build_rows() builds in small
        # batches and calls _build_footer() itself once every row is done.
        self._build_rows()

    # ==========================================
    # SCROLLING LOGIC
    # ==========================================
    def _bind_scrolling(self, checklistFrame):
        def onMouseWheel(e):
            if not self.window.winfo_exists(): return
            if e.delta != 0:
                self.canvas.yview_scroll(-1 * int(e.delta / 120), "units")
            return "break"

        def onLinuxWheel(e):
            if not self.window.winfo_exists(): return
            if e.num == 4:
                self.canvas.yview_scroll(-1, "units")
            elif e.num == 5:
                self.canvas.yview_scroll(1, "units")
            return "break"
            
        self.canvas.bind("<MouseWheel>", onMouseWheel)
        self.canvas.bind("<Button-4>", onLinuxWheel)
        self.canvas.bind("<Button-5>", onLinuxWheel)
        self.canvas.bind("<Enter>", lambda _e: self.canvas.focus_set())
        
        for widget in (self.window, checklistFrame, self.canvas, self.scrollFrame):
            widget.bind("<MouseWheel>", onMouseWheel)
            widget.bind("<Button-4>", onLinuxWheel)
            widget.bind("<Button-5>", onLinuxWheel)

    # ==========================================
    # DATA UTILITIES
    # ==========================================
    def _findLabelIndexBySpan(self, startPoint, endPoint, member=None):
        return findLabelIndexBySpan(
            self.app.labels, startPoint, endPoint, member=member,
            excludeIndices=self.deletedLabelIndices
        )

    def _rebuildRowToLabelIndex(self):
        for i, (_m, s, e) in enumerate(self.labelKeys):
            self.rowToLabelIndex[i] = self._findLabelIndexBySpan(s, e, member=_m)

    def _refreshRowUI(self, i):
        labelIndex = self.rowToLabelIndex[i]
        _oldMember, startPoint, endPoint = self.labelKeys[i]

        member, isBacking, isAdLib = None, False, False
        if labelIndex is not None and labelIndex not in self.deletedLabelIndices:
            lab = self.app.labels[labelIndex]
            while len(lab) < 5: lab.append(False)
            member, _, _, isBacking, isAdLib = lab[0], lab[1], lab[2], lab[3], lab[4]

        memberText = f" -> {member}" if member else ""
        text = f"Start: {startPoint}, End: {endPoint}{memberText}"
        color = self.app.getMemberColor(member) if member else "black"

        self.rowWidgets[i]["labelCb"].configure(text=text, fg=color)
        self.backingVars[i].set(bool(isBacking))
        self.adLibVars[i].set(bool(isAdLib))

    # ==========================================
    # UI BUILDERS
    # ==========================================
    def _build_rows(self):
        """
        Build one row per label in small batches instead of all ~100+ at
        once. A typical song has 60-113 labels, and each row creates 3
        Checkbuttons + a Button with several bind() calls - building all of
        them in a single synchronous call can take long enough (multi-
        hundred-ms to seconds) that the Tk event loop never gets control
        back, which is what froze video/audio-sync (both driven by after()
        timers) while this menu was opening during playback. Yielding to the
        event loop between batches via after() lets those timers run.
        """
        self._pendingRows = list(enumerate(self.app.getLabels()))
        self._buildRowsChunk()

    def _buildRowsChunk(self, chunkSize=15):
        if not self.window.winfo_exists():
            # Menu was closed while a batch was still pending - stop instead
            # of building rows into a destroyed frame.
            return

        chunk = self._pendingRows[:chunkSize]
        self._pendingRows = self._pendingRows[chunkSize:]

        for i, (member, startPoint, endPoint, isBacking, isAdLib) in chunk:
            self._buildRow(i, member, startPoint, endPoint, isBacking, isAdLib)

        if self._pendingRows:
            self.window.after(1, self._buildRowsChunk)
        else:
            self._build_footer()

    def _buildRow(self, i, member, startPoint, endPoint, isBacking, isAdLib):
        var = tk.BooleanVar()
        backingVar = tk.BooleanVar(value=bool(isBacking))
        adLibVar = tk.BooleanVar(value=bool(isAdLib))

        self.checkboxesByIndex.append((var, backingVar, adLibVar))
        self.labelKeys.append((member, startPoint, endPoint))

        labelIndex = self._findLabelIndexBySpan(startPoint, endPoint, member=member)
        self.rowToLabelIndex.append(labelIndex)

        self.checkboxes[i] = var
        self.backingVars[i] = backingVar
        self.adLibVars[i] = adLibVar

        memberText = f" -> {member}" if member is not None else ""
        text = f"Start: {startPoint}, End: {endPoint}{memberText}"
        color = self.app.getMemberColor(member) if member else "black"

        labelCheckbox = tk.Checkbutton(
            self.scrollFrame, text=text, variable=var,
            anchor="w", bg="lightgray", fg=color, selectcolor="darkgrey"
        )
        labelCheckbox.grid(row=i, column=0, sticky="w", padx=5, pady=2)

        backingCheckbox = tk.Checkbutton(
            self.scrollFrame, text="Are they backing vocals?",
            variable=backingVar, anchor="w",
            bg="lightgray", fg="darkblue", selectcolor="darkgrey"
        )
        backingCheckbox.grid(row=i, column=1, padx=5, pady=2)

        adLibCheckBox = tk.Checkbutton(
            self.scrollFrame, text="Ad Lib",
            variable=adLibVar, anchor="w",
            bg="lightgray", fg="purple", selectcolor="darkgrey"
        )
        adLibCheckBox.grid(row=i, column=2, padx=5, pady=2)

        self.rowWidgets[i] = {"labelCb": labelCheckbox, "backCb": backingCheckbox, "adCb": adLibCheckBox}

        # Shift-click bindings
        labelCheckbox.bind("<Button-1>", lambda event, index=i: self._onCheckboxClick(event, index, "main"))
        backingCheckbox.bind("<Button-1>", lambda event, index=i: self._onCheckboxClick(event, index, "repeat"))
        adLibCheckBox.bind("<Button-1>", lambda event, index=i: self._onCheckboxClick(event, index, "adlib"))

        # Right click opens editor
        labelCheckbox.bind("<Button-3>", lambda event, index=i: (self._openEditDialog(index), "break"))

        if member:
            def createAddLyricsCallback(sp=startPoint, ep=endPoint, mn=member):
                if self.app.isExportingVideo: return
                linkedLabel = {"member": mn, "startChunk": sp, "endChunk": ep}
                return lambda: self.app.lyricsEditor.addLyricBox(
                    startChunk=max(0, sp - LYRIC_LEAD_CHUNKS), memberName=mn, linkedLabel=linkedLabel
                )

            addLyricButton = tk.Button(self.scrollFrame, text="Add Lyrics",
                                       command=createAddLyricsCallback(startPoint, endPoint, member), bg="lightblue")
            addLyricButton.grid(row=i, column=3, padx=5, pady=2)

    def _build_footer(self):
        memberLabel = tk.Label(self.window, text="Choose Member:")
        memberLabel.pack(pady=5)

        memberMapping = {member['name']: member for member in self.app.members}
        memberMapping["Gang Vocal"] = {"name": "Gang Vocal", "id": "gang"}
        memberMapping["Cut"] = {"name": "Cut", "id": 'cut"'}
        memberNames = list(memberMapping.keys())
        
        self.memberVar = tk.StringVar(value=memberNames[0] if memberNames else "")
        memberDropdown = ttk.Combobox(self.window, textvariable=self.memberVar, values=memberNames, state="readonly")
        memberDropdown.pack(pady=5)
        
        buttonFrame = tk.Frame(self.window)
        buttonFrame.pack(pady=10)
        
        tk.Button(buttonFrame, text="Save Labels", command=self._saveSelectedLabels).pack(side="left", padx=5)
        tk.Button(buttonFrame, text="Close", command=self.close_menu).pack(side="left", padx=5)

    # ==========================================
    # INTERACTION LOGIC
    # ==========================================
    def _onCheckboxClick(self, event, index, checkboxType):
        if event.state & 0x0001:  # Shift held
            if self.lastClicked[checkboxType] != -1:
                start = min(self.lastClicked[checkboxType], index)
                end = max(self.lastClicked[checkboxType], index)
                for k in range(start, end + 1):
                    varMain, varRepeat, varAdLib = self.checkboxesByIndex[k]
                    if checkboxType == "main": varMain.set(True)
                    elif checkboxType == "repeat": varRepeat.set(True)
                    elif checkboxType == "adlib": varAdLib.set(True)
                self.shiftRange[checkboxType] = (start, end)
            return "break"
        else:
            s, e = self.shiftRange[checkboxType]
            if s != -1 and e != -1:
                for k in range(s, e + 1):
                    varMain, varRepeat, varAdLib = self.checkboxesByIndex[k]
                    if checkboxType == "main": varMain.set(False)
                    elif checkboxType == "repeat": varRepeat.set(False)
                    elif checkboxType == "adlib": varAdLib.set(False)
                self.shiftRange[checkboxType] = (-1, -1)
            self.lastClicked[checkboxType] = index

    # ==========================================
    # SUB-MENUS & ACTIONS (WITH UNDO/REDO)
    # ==========================================
    def _openEditDialog(self, i):
        labelIndex = self.rowToLabelIndex[i]
        _oldMember, startPoint, endPoint = self.labelKeys[i]
        
        currentMember, currentBacking, currentAdLib = None, False, False
        if labelIndex is not None and labelIndex not in self.deletedLabelIndices:
            lab = self.app.labels[labelIndex]
            while len(lab) < 5: lab.append(False)
            currentMember, currentBacking, currentAdLib = lab[0], bool(lab[3]), bool(lab[4])
        
        editWin = tk.Toplevel(self.window)
        editWin.title(f"Edit label ({startPoint}–{endPoint})")
        editWin.transient(self.window)
        editWin.grab_set()
        editWin.geometry("360x200")
        
        tk.Label(editWin, text=f"Start: {startPoint}   End: {endPoint}").pack(pady=8)

        memberNames = [m['name'] for m in self.app.members] + ["Gang Vocal", "Cut"]
        memberVarRow = tk.StringVar(value=currentMember if currentMember in memberNames else (memberNames[0] if memberNames else ""))
        ttk.Combobox(editWin, textvariable=memberVarRow, values=memberNames, state="readonly").pack(pady=5)

        backingVarRow = tk.BooleanVar(value=currentBacking)
        adLibVarRow = tk.BooleanVar(value=currentAdLib)
        tk.Checkbutton(editWin, text="Backing vocals", variable=backingVarRow).pack(pady=3)
        tk.Checkbutton(editWin, text="Ad Lib", variable=adLibVarRow).pack(pady=3)
    
        def applyEdit():
            nonlocal labelIndex
            
            # --- HISTORY PUSH ---
            unsaved_starts, unsaved_ends = self.app.getUnsavedPoints()
            self.app.history_manager.pushUndoState(
                labels=copy.deepcopy(self.app.labels),
                unsaved_starts=copy.deepcopy(unsaved_starts),
                unsaved_ends=copy.deepcopy(unsaved_ends),
                description=f"Menu Edit: {startPoint}-{endPoint}"
            )
            # --------------------

            chosenMember = memberVarRow.get()
            oldMember = None

            if labelIndex is None or labelIndex in self.deletedLabelIndices:
                labelIndex = self._findLabelIndexBySpan(startPoint, endPoint, member=chosenMember)

            if labelIndex is None:
                newLabel = [chosenMember, startPoint, endPoint, backingVarRow.get(), adLibVarRow.get()]
                self.app.labels.append(newLabel)
                self.rowToLabelIndex[i] = len(self.app.labels) - 1
            else:
                lab = self.app.labels[labelIndex]
                while len(lab) < 5: lab.append(False)
                
                if oldMember is None: oldMember = lab[0]
                lab[0], lab[3], lab[4] = chosenMember, backingVarRow.get(), adLibVarRow.get()
                self.rowToLabelIndex[i] = labelIndex

            self.app.clipManager.rebuild(self.app.labels, len(self.app.chunks)) 
            
            membersToUpdate = set()
            if oldMember and oldMember != chosenMember and oldMember not in self.app.bannedNames:
                membersToUpdate.add(oldMember)
            if chosenMember and chosenMember not in self.app.bannedNames:
                membersToUpdate.add(chosenMember)
                
            for m in membersToUpdate:
                trackItem = self.app.memberImages.get(m)
                if trackItem:
                    trackItem.initializeTimeline(includeBacking=self.app.includeBacking)
                
            self._refreshRowUI(i)
            self.app.onLabelsChanged()
            self.app.saveLabels(self.app.selectedGroup, True) 
            editWin.destroy()

        def deleteLabel():
            nonlocal labelIndex
            if labelIndex is None:
                self.rowToLabelIndex[i] = None
                self._refreshRowUI(i)
                editWin.destroy()
                return

            # --- HISTORY PUSH ---
            unsaved_starts, unsaved_ends = self.app.getUnsavedPoints()
            self.app.history_manager.pushUndoState(
                labels=copy.deepcopy(self.app.labels),
                unsaved_starts=copy.deepcopy(unsaved_starts),
                unsaved_ends=copy.deepcopy(unsaved_ends),
                description=f"Menu Delete: {startPoint}-{endPoint}"
            )
            # --------------------

            lab = self.app.labels[labelIndex]
            oldMember = lab[0] if lab and len(lab) >= 1 else None

            del self.app.labels[labelIndex]
            
            self.rowToLabelIndex[i] = None
            self.app.clipManager.rebuild(self.app.labels, len(self.app.chunks)) 
            self._refreshRowUI(i)
            
            if oldMember and oldMember not in self.app.bannedNames:
                trackItem = self.app.memberImages.get(oldMember)
                if trackItem:
                    trackItem.initializeTimeline(includeBacking=self.app.includeBacking)

            self.app.saveLabels(self.app.selectedGroup, True)
            self.app.onLabelsChanged() 
            editWin.destroy()

        btnFrame = tk.Frame(editWin)
        btnFrame.pack(pady=10)
        tk.Button(btnFrame, text="Apply", command=applyEdit).pack(side="left", padx=6)
        tk.Button(btnFrame, text="Delete", command=deleteLabel).pack(side="left", padx=6)
        tk.Button(btnFrame, text="Cancel", command=editWin.destroy).pack(side="left", padx=6)

    def _saveSelectedLabels(self):
        # --- HISTORY PUSH ---
        unsaved_starts, unsaved_ends = self.app.getUnsavedPoints()
        self.app.history_manager.pushUndoState(
            labels=copy.deepcopy(self.app.labels),
            unsaved_starts=copy.deepcopy(unsaved_starts),
            unsaved_ends=copy.deepcopy(unsaved_ends),
            description="Menu: Bulk save labels"
        )
        # --------------------

        if self.deletedLabelIndices:
            self.app.labels = [lab for idx, lab in enumerate(self.app.labels) if idx not in self.deletedLabelIndices]
            self.deletedLabelIndices.clear()
            self._rebuildRowToLabelIndex()

        membersToUpdate = set()
        anyMain = any(var.get() for var in self.checkboxes.values())
        
        if anyMain:
            for i, var in self.checkboxes.items():
                if not var.get(): continue

                chosenMember = self.memberVar.get()
                _, startPoint, endPoint = self.labelKeys[i]
                isBacking = self.backingVars[i].get()
                isAdLib = self.adLibVars[i].get()

                idx = self._findLabelIndexBySpan(startPoint, endPoint, member=chosenMember)
                if idx is None:
                    self.app.labels.append([chosenMember, startPoint, endPoint, isBacking, isAdLib])
                    self.rowToLabelIndex[i] = len(self.app.labels) - 1
                else:
                    lab = self.app.labels[idx]
                    while len(lab) < 5: lab.append(False)
                    oldMember = lab[0]
                    lab[0], lab[3], lab[4] = chosenMember, isBacking, isAdLib
                    self.rowToLabelIndex[i] = idx

                    if oldMember and oldMember != chosenMember and oldMember not in self.app.bannedNames:
                        membersToUpdate.add(oldMember)
                
                self.app.clipManager.rebuild(self.app.labels, len(self.app.chunks)) 
                self._refreshRowUI(i)

                if chosenMember and chosenMember not in self.app.bannedNames:
                    membersToUpdate.add(chosenMember)
                    
            for m in membersToUpdate:
                trackItem = self.app.memberImages.get(m)
                if trackItem:
                    trackItem.initializeTimeline(includeBacking=self.app.includeBacking)
            
            self.app.saveLabels(self.app.selectedGroup, True)
            
        else:
            changed = 0
            for i in range(len(self.labelKeys)):
                idx = self.rowToLabelIndex[i]
                if idx is None: continue
                
                lab = self.app.labels[idx]
                while len(lab) < 5: lab.append(False)

                oldB, oldA = lab[3], lab[4]
                newB, newA = self.backingVars[i].get(), self.adLibVars[i].get()
                lab[3], lab[4] = newB, newA

                if (oldB, oldA) != (newB, newA): changed += 1

            if changed > 0:
                self.app.saveLabels(self.app.selectedGroup, True)

        # onLabelsChanged() also refreshes which members are visible on the
        # canvas (and re-maxes their scale for the new count) - this bulk
        # save path is how a member commonly gets their first label in a
        # song, so without this they'd stay off-screen until some other
        # action happened to trigger a refresh.
        self.app.onLabelsChanged()

        self.app.selectedMarker = None
        self.app.selectedLabel = None
        self.app.originalLabel = None
        self.close_menu()
    
    def close_menu(self):
        try:
            self.window.grab_release()
        except Exception: pass
        try:
            self.window.destroy()
        except Exception: pass

        # These three must always run, even if grab_release/destroy above
        # raised - show() unconditionally disabled keybinds/zoom and marked
        # the guard open, so skipping any of them here (as ModalGuard.close
        # used to, nested inside the destroy() try) permanently breaks 'e'
        # and every other canvas shortcut plus zoom until the app restarts.
        ModalGuard.close("labels_menu")
        self.app.enableRootKeybinds()
        self.app.videoTrackItem.setUiBusy(False)
        self.app.enableRootKeybinds()