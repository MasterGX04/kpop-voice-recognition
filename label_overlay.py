import tkinter as tk

class LabelOverlayController:
    def __init__(self, root, canvas, getLabelsFn, members):
        """
        root: Tk root (for Toplevel tooltip)
        canvas: main timeline canvas
        getLabelsFn: callable returning self.labels from parent
        """
        self.root = root
        self.canvas = canvas
        self.getMatchedPoints = getLabelsFn

        self._tooltipWin = None
        self._tooltipLabel = None
        self._visible = False
        
        self._markerInfo = {}
        
        # name -> hex color
        self.memberColorByName = {}
        for m in (members or []):
            name = m.get("name")
            color = m.get("color")
            if name and color:
                self.memberColorByName[name] = color
        
    # ---------- Tooltip core ----------
    def _ensureTooltip(self):
        if self._tooltipWin:
            return

        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.withdraw()

        lbl = tk.Label(
            win,
            bg="#111",
            fg="white",
            font=("Arial", 10),
            padx=8,
            pady=4,
            relief="solid",
            bd=1
        )
        lbl.pack()

        self._tooltipWin = win
        self._tooltipLabel = lbl
        
    def _show(self, event, text, bg=None):
        self._ensureTooltip()
        
        # Dynamic background
        bg = bg or "#111"
        self._tooltipLabel.config(text=text, bg=bg)
        try:
            self._tooltipWin.config(bg=bg)
        except tk.TclError:
            pass
        
        self._tooltipWin.geometry(f"+{event.x_root+12}+{event.y_root+12}")
        self._tooltipWin.deiconify()
        self._visible = True
        
    def _move(self, event):
        if self._visible:
            self._tooltipWin.geometry(f"+{event.x_root+12}+{event.y_root+12}")

    def _hide(self, event=None):
        if self._tooltipWin:
            self._tooltipWin.withdraw()
            self._visible = False
    
    # ---------- Label lookup ----------
    def _labelsForBoundary(self, chunkIndex, markerType):
        """
        Uses matchedPoints: (member, start, end, isBacking, isAdlib)
        Returns all entries whose start/end matches boundary.
        """
        out = []
        for member, start, end, isBacking, isAdlib in self.getMatchedPoints():
            if markerType == "start" and start == chunkIndex:
                out.append((member, start, end, isBacking, isAdlib))
            elif markerType == "end" and end == chunkIndex:
                out.append((member, start, end, isBacking, isAdlib))
        return out
    
    def _durationSecs(self, start, end):
        # Each chunk is 40ms
        return round(((end - start) * 40) / 1000.0, 2)
    
    def formatText(self, chunkIndex, markerType):
        matches = self._labelsForBoundary(chunkIndex, markerType)
        if not matches:
            return f"{markerType.upper()} @ {chunkIndex}\n(unlabeled)", "#111"

        # Determine background color:
        # - if exactly one unique member in matches and we have a color, use it
        uniqueMembers = list({m for m, *_ in matches})
        bg = "#111"
        if len(uniqueMembers) == 1:
            bg = self.memberColorByName.get(uniqueMembers[0], "#111")

        lines = []
        for m, s, e, b, a in matches:
            dur = self._durationSecs(s, e)
            lines.append(
                f"{m}: {s} → {e}  ({dur:.2f} secs)"
                f"{' (Backing)' if b else ''}"
                f"{' (Adlib)' if a else ''}"
            )

        text = f"{markerType.upper()} @ {chunkIndex}\n" + "\n".join(lines)
        return text, bg

    # ---------- Public API ----------
    def bindBoundaryMarker(self, markerId, chunkIndex, markerType):
        # Store initial info
        self._markerInfo[markerId] = (chunkIndex, markerType)

        def on_enter(e):
            self._activeMarkerId = markerId
            # Look up the latest chunkIndex at hover time (no stale closure)
            ci, mt = self._markerInfo.get(markerId, (chunkIndex, markerType))
            txt, bg = self.formatText(ci, mt)
            self._show(e, txt, bg=bg)
            
        def on_leave(e):
            # only clear if we're leaving the same marker
            if getattr(self, "_activeMarkerId", None) == markerId:
                self._activeMarkerId = None
            self._hide(e)

        self.canvas.tag_bind(markerId, "<Enter>", on_enter)
        self.canvas.tag_bind(markerId, "<Motion>", self._move)
        self.canvas.tag_bind(markerId, "<Leave>", on_leave)
        
    def updateBoundaryMarker(self, markerId, chunkIndex=None, markerType=None):
        old = self._markerInfo.get(markerId)
        if not old:
            return
        oldChunk, oldType = old
        self._markerInfo[markerId] = (
            oldChunk if chunkIndex is None else chunkIndex,
            oldType if markerType is None else markerType
        )

    # Optional cleanup if you delete markers
    def forgetMarker(self, markerId):
        if getattr(self, "_activeMarkerId", None) == markerId:
            self._hide(None)
            self._activeMarkerId = None
        self._markerInfo.pop(markerId, None)