import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

class ThumbnailManager:
    """
    Adds a 'Thumbnail' menu and manages a safe 'thumbnail mode' overlay.

    Key idea: DO NOT delete video/lyrics canvas items (deleting often breaks update loops).
    Instead: temporarily hide them, show a thumbnail image behind, and then restore.
    """

    SUPPORTED_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".svg")
    
    def __init__(self, parent, menubar=None):
        """
        parent is your main AudioTester / App object.
        Expects:
          - parent.root (Tk)
          - parent.canvas (Canvas)
          - parent.videoTrackItem (optional, with .videoFrameId)
          - parent.lyricsBackgroundId (optional)
          - parent.hideAllLyrics() (optional)
          - parent.showAllLyrics() or parent.renderLyrics()/redraw (optional; we handle best-effort)
          - parent.selectedGroup, parent.songName (optional for default save path)
        """
        self.parent = parent
        self.root = parent.root
        self.menubar = menubar
        self.canvas = parent.canvas

        self.thumbnailMode = False
        self.thumbnailImagePath = None

        self.thumbnailTk = None
        self.thumbnailCanvasId = None
        
        self._hiddenItems = []  # list of canvas item ids we set to hidden
        self._tempExitBindings = []  # list of (widget, sequence, funcid)

        self._ensureMenu()
        
    # -------------------------
    # Menu wiring
    # -------------------------
    def _ensureMenu(self):
        # If you already have a menubar, reuse it. Otherwise create one.
        menubar = self.menubar
        if menubar is None:
            print(f"no menu bar can be found")
            menubar = tk.Menu(self.root)
            self.root.config(menu=menubar)
            self.parent.menubar = menubar
            
        self.thumbnailMenu = tk.Menu(menubar, tearoff=0)
        self.thumbnailMenu.add_command(label="Select Image…", command=self.selectThumbnailImage)
        self.thumbnailMenu.add_command(label="Use background.* from song folder", command=self.tryAutoPickBackground)
        self.thumbnailMenu.add_separator()
        self.thumbnailMenu.add_command(label="Enter Thumbnail Mode", command=self.enterThumbnailMode)
        self.thumbnailMenu.add_command(label="Exit Thumbnail Mode", command=self.exitThumbnailMode)
        self.thumbnailMenu.add_command(label="Toggle Thumbnail Mode", command=self.toggleThumbnailMode)
        self.thumbnailMenu.add_separator()
        self.thumbnailMenu.add_command(label="Capture Thumbnail Screenshot: Ctrl-T", command=self.captureThumbnailScreenshot)

        menubar.add_cascade(label="Thumbnail", menu=self.thumbnailMenu)

        # Hotkey
        self.root.bind_all("<Control-t>", self.captureThumbnailScreenshot)
        
    # -------------------------
    # Image selection
    # -------------------------
    def selectThumbnailImage(self):
        path = filedialog.askopenfilename(
            title="Select thumbnail image",
            filetypes=[
                ("Images", "*.png *.jpg *.jpeg *.svg"),
                ("PNG", "*.png"),
                ("JPEG", "*.jpg *.jpeg"),
                ("SVG", "*.svg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        if not self._isSupported(path):
            messagebox.showwarning(
                "Unsupported", 
                "Please select a png/jpg/jpeg/svg file.",
                parent=self.root
            )
            return

        self.thumbnailImagePath = path

        # If already in thumbnail mode, reload it immediately
        if self.thumbnailMode:
            self._applyThumbnailOverlay()

    def tryAutoPickBackground(self):
        """
        Your old behavior: look in the song directory for background.(png/jpg/jpeg/svg).
        """
        basePath = self._getSongBaseDir()
        if not basePath:
            messagebox.showwarning(
                "Not found", 
                "Couldn't determine the current song folder.",
                parent=self.root
            )
            return

        # Try common names first
        candidates = []
        for ext in self.SUPPORTED_IMAGE_EXTS:
            candidates.append(os.path.join(basePath, f"background{ext}"))
            candidates.append(os.path.join(basePath, f"Background{ext}"))

        picked = None
        for c in candidates:
            if os.path.exists(c):
                picked = c
                break

        if not picked:
            # fallback: any image in folder
            for name in os.listdir(basePath):
                if name.lower().endswith(self.SUPPORTED_IMAGE_EXTS):
                    picked = os.path.join(basePath, name)
                    break

        if not picked:
            messagebox.showwarning(
                "Not found", 
                "No background image found in the song folder.",
                parent=self.root
                )
            return

        self.thumbnailImagePath = picked
        if self.thumbnailMode:
            self._applyThumbnailOverlay()
        else:
            messagebox.showinfo(
                "Selected", 
                f"Using: {os.path.basename(picked)}",
                parent=self.root
            )

    def _isSupported(self, path: str) -> bool:
        return path.lower().endswith(self.SUPPORTED_IMAGE_EXTS)
    
    # -------------------------
    # Mode control
    # -------------------------
    def toggleThumbnailMode(self):
        if self.thumbnailMode:
            self.exitThumbnailMode()
        else:
            self.enterThumbnailMode()

    def enterThumbnailMode(self):
        if self.thumbnailMode:
            return

        # Need an image path
        if not self.thumbnailImagePath:
            # Try auto-pick silently; if still none, prompt
            self.tryAutoPickBackground()
            if not self.thumbnailImagePath:
                self.selectThumbnailImage()
                if not self.thumbnailImagePath:
                    return

        self.thumbnailMode = True

        # Hide video/lyrics items safely
        self._hideVideoAndLyrics()

        # Put thumbnail image behind everything else
        self._applyThumbnailOverlay()

        # Ensure any click returns to normal mode (including clicking buttons)
        self._bindExitOnAnyClick()
        
        self.parent.setUIHidden(True)        

    def exitThumbnailMode(self):
        if not self.thumbnailMode:
            return

        self.thumbnailMode = False

        # Remove thumbnail overlay
        if self.thumbnailCanvasId is not None:
            try:
                self.canvas.delete(self.thumbnailCanvasId)
            except Exception:
                pass
            self.thumbnailCanvasId = None
        self.thumbnailTk = None

        # Restore hidden items
        self._restoreHiddenItems()
        
        self._restoreLyricsAtCurrentChunk()

        # Remove temporary bindings
        self._unbindTempBindings()
        
        self.parent.setUIHidden(False)

    # -------------------------
    # Implementation details
    # -------------------------
    def _hideVideoAndLyrics(self):
        """
        Hide (not delete) items that cause glitches if removed.
        """
        self._hiddenItems.clear()

        # Video frame id
        vti = getattr(self.parent, "videoTrackItem", None)
        if vti is not None and hasattr(vti, "videoFrameId"):
            vid = getattr(vti, "videoFrameId", None)
            if vid:
                self._hideCanvasItem(vid)

        # Lyrics background
        lbid = self.parent.lyricsBackgroundId
        if lbid:
            self._hideCanvasItem(lbid)

        # Call hideAllLyrics if present.
        self.parent.hideAllLyrics()

    def _hideCanvasItem(self, itemId):
        try:
            # Only hide if it exists and isn't already hidden
            self.canvas.itemconfigure(itemId, state="hidden")
            self._hiddenItems.append(itemId)
        except Exception:
            pass

    def _restoreHiddenItems(self):
        for itemId in self._hiddenItems:
            try:
                self.canvas.itemconfigure(itemId, state="normal")
            except Exception:
                pass
        self._hiddenItems.clear()

    def _applyThumbnailOverlay(self):
        """
        Loads thumbnail image, resizes to current canvas WxH, and places at (0,0).
        Keeps it behind other items.
        """
        if not self.thumbnailImagePath or not os.path.exists(self.thumbnailImagePath):
            return

        # Remove previous overlay if any
        if self.thumbnailCanvasId is not None:
            try:
                self.canvas.delete(self.thumbnailCanvasId)
            except Exception:
                pass
            self.thumbnailCanvasId = None
            self.thumbnailTk = None

        try:
            if self.thumbnailImagePath.lower().endswith(".svg"):
                messagebox.showwarning(
                    "SVG support",
                    "SVG needs conversion (e.g., cairosvg). Please choose png/jpg/jpeg for now.",
                    parent=self.root
                )
                return

            # Make sure geometry is up to date so winfo_width/height are correct
            self.root.update_idletasks()

            canvasW = self.canvas.winfo_width()
            canvasH = self.canvas.winfo_height()

            # Safety fallback if Tk reports tiny values (common right after startup)
            if canvasW <= 2 or canvasH <= 2:
                canvasW = getattr(self.parent, "baseWidth", 1280)
                canvasH = getattr(self.parent, "baseHeight", 720)

            img = Image.open(self.thumbnailImagePath).convert("RGB")

            img = self._resizeCover(img, canvasW, canvasH)

            self.thumbnailTk = ImageTk.PhotoImage(img)

            self.thumbnailCanvasId = self.canvas.create_image(
                0, 0, anchor="nw", image=self.thumbnailTk
            )
            self.canvas.tag_lower(self.thumbnailCanvasId)

        except Exception as e:
            messagebox.showerror(
                "Thumbnail error", 
                f"Could not load thumbnail:\n{e}",
                parent=self.root
            )
    
    def _resizeCover(self, img, targetW, targetH):
        # Preserve aspect ratio, crop center to fill entire target
        srcW, srcH = img.size
        scale = max(targetW / srcW, targetH / srcH)
        newW = int(srcW * scale)
        newH = int(srcH * scale)
        resized = img.resize((newW, newH), Image.Resampling.LANCZOS)

        left = (newW - targetW) // 2
        top = (newH - targetH) // 2
        return resized.crop((left, top, left + targetW, top + targetH))
    
    def _bindExitOnAnyClick(self):
        """
        Bind root + common widgets so any click exits thumbnail mode.
        Using bind_all catches most, but some widgets swallow events;
        so we also bind root explicitly.
        """
        self._unbindTempBindings()

        def _exit(_evt=None):
            # if already back, ignore
            if self.thumbnailMode:
                self.exitThumbnailMode()

        # bind_all catches clicks anywhere in the app
        funcid1 = self.root.bind_all("<Button-1>", _exit, add="+")
        funcid2 = self.root.bind_all("<Button-2>", _exit, add="+")
        funcid3 = self.root.bind_all("<Button-3>", _exit, add="+")
        self._tempExitBindings.append(("bind_all", "<Button-1>", funcid1))
        self._tempExitBindings.append(("bind_all", "<Button-2>", funcid2))
        self._tempExitBindings.append(("bind_all", "<Button-3>", funcid3))

    def _unbindTempBindings(self):
        # Tkinter returns a string funcid for bind_all; unbind requires that funcid.
        for kind, seq, funcid in self._tempExitBindings:
            try:
                if kind == "bind_all":
                    self.root.unbind_all(seq)  # simplest: remove all handlers for that seq
            except Exception:
                pass
        self._tempExitBindings.clear()

    def captureThumbnailScreenshot(self, event=None):
        wasInMode = self.thumbnailMode

        if not wasInMode:
            self.enterThumbnailMode()
            if not self.thumbnailMode:
                return

        owner = self.canvas.winfo_toplevel()
        # Let Tk finish layout / redraw before grabbing
        owner.after(50, lambda: self._doThumbnailScreenshot(wasInMode, owner))
    
    def _doThumbnailScreenshot(self, wasInMode, ownerWindow):
        try:
            from PIL import ImageGrab
        except Exception:
            messagebox.showerror(
                "Screenshot not available",
                "PIL.ImageGrab isn't available in this environment.",
                parent=self.root
            )
            if not wasInMode:
                self.exitThumbnailMode()
            return

        # Force final geometry sync
        ownerWindow.update_idletasks()
        ownerWindow.update() 

        x0 = self.canvas.winfo_rootx()
        y0 = self.canvas.winfo_rooty()
        x1 = x0 + self.canvas.winfo_width()
        y1 = y0 + self.canvas.winfo_height()

        try:
            shot = ImageGrab.grab(bbox=(x0, y0, x1, y1))
            shot = shot.convert("RGB")

            savePath = self._defaultThumbnailSavePath()
            os.makedirs(os.path.dirname(savePath), exist_ok=True)
            shot.save(savePath, format="PNG")

            messagebox.showinfo(
                "Saved",
                f"Thumbnail saved:\n{savePath}",
                parent=self.root
            )
        except Exception as e:
            messagebox.showerror(
                "Capture failed",
                str(e),
                parent=self.root
            )
        finally:
            if not wasInMode:
                # Important: exit AFTER capture
                self.exitThumbnailMode() 

    def _defaultThumbnailSavePath(self):
        """
        Default: ./training_data/{selectedGroup}/{songName}_thumbnail.png
        Falls back to current directory if info missing.
        """
        selectedGroup = getattr(self.parent, "selectedGroup", None)
        songName = getattr(self.parent, "songName", None)

        if selectedGroup and songName:
            return os.path.join(".", "thumbnails", selectedGroup, f"{songName}_thumbnail.png")

        return os.path.join(".", "thumbnail.png")

    def _getSongBaseDir(self):
        """
        Matches your old logic: base path derived from self.testSongPath.
        If your new code uses something else, adapt this function only.
        """
        testSongPath = getattr(self.parent, "testSongPath", None)
        if testSongPath and isinstance(testSongPath, str):
            return os.path.dirname(testSongPath)

        # Fall back: if you store audio path as currentAudioPath, use that
        currentAudioPath = getattr(self.parent, "currentAudioPath", None)
        if currentAudioPath and isinstance(currentAudioPath, str):
            return os.path.dirname(currentAudioPath)

        return None
    
    def _enterLyricFreeze(self):
        """
        Hide all LyricBoxes and remember enough to restore later.
        We DON'T want renderLyrics to keep mutating while in thumbnail mode.
        """
        self._savedLyricsState = None

        # Only do this if these exist (your app does)
        if hasattr(self.parent, "activeLyricIds") and hasattr(self.parent, "lastChunkSeen"):
            try:
                self._savedLyricsState = {
                    "activeLyricIds": set(self.parent.activeLyricIds),
                    "lastChunkSeen": getattr(self.parent, "lastChunkSeen", -1),
                }
            except Exception:
                self._savedLyricsState = None

        # Hide all lyric boxes (best effort)
        lyricsDict = getattr(self.parent, "lyrics", None)
        if isinstance(lyricsDict, dict):
            for lb in lyricsDict.values():
                try:
                    lb.hide()
                except Exception:
                    pass

    def _restoreLyricsAtCurrentChunk(self):
        """
        Restore lyric state so renderLyrics works again.

        For your stateful renderLyrics:
        - clear activeLyricIds
        - set lastChunkSeen to (chunkIndex - 1)
        - call renderLyrics(chunkIndex) once to rebuild on-screen state
        """
        if not hasattr(self.parent, "renderLyrics"):
            return

        self.parent.unsuppressLyricsAndRefresh()