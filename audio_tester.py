import os, traceback, hashlib, sys
from thumbnail_functions import ThumbnailManager
import time
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import threading
from lyrics_editor import LyricsEditor
from pydub import AudioSegment
from TrackItem import TrackItem
from collections import defaultdict, Counter
import pygame
from VideoTrack import VideoTrackItem
from navigation_arrows import NavigationArrows
import json, copy
import codecs
from lyrics_box import LyricBox
from zoom_functions import ZoomManager, ProgressBarHandle
from util_functions import ensureReadableOnBackground, getCached720pVideo, ModalGuard
from label_overlay import LabelOverlayController
from label_lanes import LabelLaneRenderer
from cut_clip_manager import CutClipManager
from line_distribution_panel import LineDistributionPanel

def resourcePath(*parts: str) -> str:
    # When packaged (PyInstaller), sys._MEIPASS points to the temp extracted dir
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)

ffmpegPath = resourcePath("ffmpeg.exe")
ffprobePath = resourcePath("ffprobe.exe")

AudioSegment.converter = ffmpegPath
AudioSegment.ffmpeg = ffmpegPath
AudioSegment.ffprobe = ffprobePath

def cacheKeyForPath(path: str) -> str:
    # key changes if the file changes (mtime + size)
    st = os.stat(path)
    s = f"{os.path.abspath(path)}|{st.st_mtime_ns}|{st.st_size}"
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def getOrCreateMenubar(root: tk.Tk) -> tk.Menu:
        """
        Returns the existing menubar if root already has one, otherwise creates it.
        """
        existing = root.cget("menu")
        if existing:
            try:
                return root.nametowidget(existing)
            except Exception:
                pass

        menubar = tk.Menu(root)
        root.config(menu=menubar)
        return menubar

def ensureAudioForPlayback(path: str, cacheDir: str = "cache_audio", targetSr: int = 22050):
    """
    Returns a cached path that is resampled to targetSr and stored as mp3 for size.
    """
    os.makedirs(cacheDir, exist_ok=True)
    key = cacheKeyForPath(path)
    outPath = os.path.join(cacheDir, f"{key}_sr{targetSr}.mp3")

    if os.path.exists(outPath):
        return outPath, False

    audio = AudioSegment.from_file(path)  # wav/mp3/etc
    audio = audio.set_frame_rate(targetSr)
    # optional: force mono for smaller file + consistent timing
    audio.export(outPath, format="mp3")
    return outPath, True

def normalizeLabel(label):
            # supports [member, start, end] or [member, start, end, isBacking, isAdLib]
            member, start, end = label[0], int(label[1]), int(label[2])
            isBacking = bool(label[3]) if len(label) > 3 else False
            isAdLib  = bool(label[4]) if len(label) > 4 else False
            if end < start:
                start, end = end, start  # or raise, depending on your UI constraints
            return [member, start, end, isBacking, isAdLib]

def labelKey(label):
    # identity key (your compromise rule)
    labelKey = (label[0], int(label[1]))

    return  labelKey # (member, start)\
        
def dedupeLabelsByKey(labels):
        # overwrite strategy: last one wins (or choose max end, your call)
        byKey = {}
        for lbl in labels:
            nl = normalizeLabel(lbl)
            k = labelKey(nl)
            if k not in byKey or nl[2] > byKey[k][2]:
                byKey[k] = nl
        out = list(byKey.values())
        out.sort(key=lambda l: (l[1], l[2], l[0]))
        return out
    
def getVideoDurationMs(videoPath: str) -> int:
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0 or frames <= 0:
        return None

    return int((frames / fps) * 1000)

def isMatchingMusicVideoMs(videoMs: int, audioMs: int, toleranceMs=750) -> bool:
    return abs(videoMs - audioMs) <= toleranceMs

class VoiceDetectionApp:
    def __init__(
            self, *, root, members, modelPath, 
            images, testSongPath, vocalsOnlyPath, vocalsLeadPath, 
            vocalsBackingPath, videoPath, selectedGroup, songDir
        ):
        self.root = root
        
        self.initChunkIndexInTitle(self.root)
        self.members = members
        self.images = images
        self.playbackThread = None
        self.labels = []
        self.selectedMarker = None
        self.songName = os.path.splitext(os.path.basename(testSongPath))[0]
        self.bannedNames = ["Gang Vocal", "Cut"]
        self.uiHidden = False
        self.isExportingVideo = False
        self.exportStopEvent = threading.Event()
        self.root.bind_all("<Escape>", self._onCancelExport) 
        self.songDir = songDir
        self.originalSongPath = testSongPath
        
        # names for leading and backing vocal files
        cachedMix, _ = ensureAudioForPlayback(testSongPath)
        cachedVocals, _ = ensureAudioForPlayback(vocalsOnlyPath)
        self._buildActiveLyricIds = []
        self.lyricsLayerDirty = False
        self.lyricsSuppressed = False
        self.slotHeightBase = 0

        self.testSongPath = cachedMix
        self.vocalsOnlyPath = cachedVocals
        self.vocalsLeadPath = vocalsLeadPath
        self.vocalsBackingPath = vocalsBackingPath
            
        self.selectedGroup = selectedGroup
        # Which audio file pyame should currently play
        self.currentAudioPath = self.testSongPath
        self.audioMode = "mix" # Or vocals
        self.leadEnabled = True      # default: lead on
        self.backEnabled = False     # default: backing off
        self.panMode = "split"

        self.baseWidth = 1920
        self.baseHeight = 1080
        
        self.scaleX = 1.0
        self.scaleY = 1.0
        
        self.viewportOffsetX = 0
        self.viewportOffsetY = 0
        
        if os.path.exists(self.testSongPath):
            self.audio = AudioSegment.from_file(self.testSongPath)
        else:
            print("Test song path does not exist:", self.testSongPath)
        self.chunk_duration = 40
        self.totalDurationMs = len(self.audio)
        self.chunks = [self.audio[i:i + self.chunk_duration] for i in range(0, len(self.audio), int(self.chunk_duration))]
        self.detectionResults = []
        self.currentChunkIndex = 0  # Track current playback position
        self.playbackOffset = 0
        self.previousX = 0
        self.isPlaying = False
        self.isPaused = False
        self.isProcessed = False
        self.isManualUpdate = False
        self.skipNextAutoUpdate = False
        self.labelMarkers = {}
        self.lyrics = {}
        pygame.mixer.init()
            
        self.startPoints = []
        self.endPoints = []            
            
        self.startPointMarkers = {}
        self.openStartChunk = None
        self.endPointMarkers = {}
        
        self.canvasFrame = tk.Frame(root, bd=0, highlightthickness=0, relief="flat")
        self.canvasFrame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(
            self.canvasFrame,
            bg="white",
            highlightthickness=0,  
            bd=0,
            relief="flat"
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", self.onCanvasResize)
        
        self.memberImages = {}
        self.slotMap = {}
        self.memberImageIds = {}
        self.progressBarWidth = int(1280 * 0.75)
        self.testOrVideo = "Video"

        self.timeDisplayVar = tk.StringVar(value="00:00:000") # Display time iin MM:SS:milliseconds
        self.zoomManager = ZoomManager(self.canvas, self, None, self.totalDurationMs, self.chunk_duration, pygame)
        self.progressBarCanvas = tk.Canvas(self.canvas, width=self.progressBarWidth, height=20, bg="black")
        self.progressBarCanvas.place(relx=0.5, rely=0.9, anchor="center")
        self.navigationArrows = NavigationArrows(self.canvas, self, self.progressBarCanvas)
        
        self.initStatusMessage()
        self.currentSectionIndex = 0
        self.progressBarCanvas.place(relx=0.5, rely=0.9, anchor="center") # creates horizontal scale widget
        
        self.zoomManager.progressBar = self.progressBarCanvas
        self.lineDistPanel = LineDistributionPanel(self, parent=self.canvas)
        self.lineDistPanel.hide()
        
        self.timeDisplayLabel = tk.Label(self.canvas, textvariable=self.timeDisplayVar, bg="black", fg="white", font=("Arial", 12))
        self.timeDisplayLabel.place(relx=0.5, rely=0.95, anchor="center")
        
        self.progressBarCanvas.bind("<ButtonPress-1>", self.onProgressBarClick)
        self.progressBarCanvas.bind("<B1-Motion>", self.onDragHandle)    
        self.progressBarCanvas.bind("<ButtonRelease-1>", self.onProgressBarRelease)
        
        self.progressBarHandle = ProgressBarHandle(self.progressBarCanvas, self, self.progressBarWidth, self.chunk_duration)
        self.includeBacking = True

        def get720pVideo(videoPath):
            try:
                return getCached720pVideo(videoPath)
            except Exception as e:
                messagebox.showerror(f"Error processing video for 720p cache: {e}", parent=self.root)
                return None
        
        # Initialize VideoTrack
        safeVideoPath = None
        isMusicVideo = False
        # 1) User-provided video takes priority
        if videoPath and os.path.exists(videoPath):
            self.videoPath = videoPath
            safeVideoPath = get720pVideo(videoPath)
        
        # 2) Song-named fallback
        if not safeVideoPath:
            candidate = os.path.join(self.songDir, f"{self.songName}.mp4")
            if os.path.exists(candidate):
                safeVideoPath = get720pVideo(candidate)
                self.videoPath = candidate

        # 3) Final fallback
        if not safeVideoPath:
            print("No valid video found, using default looping background.")
            safeVideoPath = resourcePath("looping_background.mp4")
            self.videoPath = safeVideoPath
            isMusicVideo = False
        else:
            videoMs = getVideoDurationMs(safeVideoPath)
            audioMs = self.totalDurationMs  # <-- from len(self.audio)

            if videoMs is None:
                print("Could not read video duration, treating as background.")
                isMusicVideo = False
            else:
                isMusicVideo = isMatchingMusicVideoMs(videoMs, audioMs)

                if not isMusicVideo:
                    print(
                        f"Video/audio length mismatch "
                        f"(video={videoMs}ms, audio={audioMs}ms). "
                        f"Using looping background behavior."
                    )
                    
        print(f"Original videoPath: {self.videoPath}")

        self.videoTrackItem = VideoTrackItem(
            self.canvas,
            self,
            safeVideoPath,
            scale=100,
            scaleX=self.scaleX,
            baseHeight=720,
            isMusicVideo=isMusicVideo
        )
        # Voice detection results
        memberList = [member['name'] for member in members] + ['Gang Vocal']
        # print(f"Member list: {memberList}")
        if os.path.exists(modelPath):
            os.makedirs("./predictions", exist_ok=True)
            try:
                from model_predictor import predict_song_selective
                model2Path = f"./models/{self.selectedGroup}_muq_head_phase2.pt"
                labels = predict_song_selective(
                    group_name=self.selectedGroup, 
                    song_name=self.songName, 
                    encoder_path="OpenMuQ/MuQ-large-msd-iter",
                    head1_path=modelPath,
                    head2_path=model2Path,
                    member_names=memberList
                )
                labels40 = labels["labels_40ms"]
            except Exception as e:
                print("❌ Error while running predict_40ms:")
                print(f"   {e}")                  # short message
                traceback.print_exc()             # full stack trace
                labels40 = []                     # fallback so UI does not crash
        else:
            labels40 = []

        self.voiceDetectionResults = labels40
        print("Detection results:", labels40[280:460])
        
        self.labelOverlay = LabelOverlayController(
            root=self.root,
            canvas=self.canvas,
            getLabelsFn=self.getLabels,
            members=self.members
        )
            
        self.labels = self.loadSavedLabels() # Store labels (member, start, end)
        if self.labels == [] and len(labels40) > 0: 
            self.labels = self.createLabelsFromPredictions(labels40)
        
        self.labelLaneRenderer = LabelLaneRenderer(
            canvas=self.canvas,
            zoomManager=self.zoomManager,
            progressBarCanvas=self.progressBarCanvas,
            getLabelsFn=lambda: self.labels,
            getMemberColorFn=self.getMemberColor,  # you already have this
            maxLanes=4
        )
        self.clipManager = CutClipManager(self.chunk_duration, jumpCallback=self.jumpToMs)
        self.clipManager.rebuild(self.labels, len(self.chunks))
        self.root.after_idle(self.startLayout)
        self.lyricsEditor = LyricsEditor(self)
        self.lastChunkSeen = -1
        self.root.after(100, self.drawTimeMarkers)
        
        self.startEvents = {}
        self.activeLyricIds = set()
        def handleLyrics():
            self.loadLyricsFromFile()
            self.buildLyricStartEvents()
            
        self.root.after(50, handleLyrics)
        
        if len(self.voiceDetectionResults) > 0:
            self.evaluateVoiceDetectionAccuracy()
            
        self.lastKeyPressTime = 0
        self.enableRootKeybinds()
        self.canvas.focus_set() 
        self.cropVideoVar = tk.BooleanVar(value=False)
        self.setupMenubar(self.root)
        
        # Dragging state
        self.selectedLabel = None
        self.isDraggingMarker = False
        self.isPendingDrag = False
        self.pendingDrag = False
        self.dragStartXY = None
        self.dragThreshold = 4 # px
        
        # undo/redo stack for convenience
        self.undoStack = []
        self.redoStack = []
        self.dragStartLabels = None # Snapshot before drag starts
        
        # Splitting labels into two parts status
        self.splitGapActive = False
        self.splitGapStartChunk = None
        
        # Temporary marker shown when setting the start of a gap-split
        self.splitGapMarkerId = None
        
        self.root.protocol("WM_DELETE_WINDOW", self.onClose)
        
        self.root.bind("<Control-h>", self.toggleUIElements)
        self.root.bind("<Control-s>", self.resetLabels)
        self.root.bind("<Control-Shift-B>", self.changeMode)
        self.root.bind("<Control-b>", lambda e: self.cropVideoVar.set(
            not self.cropVideoVar.get()
        ) or self.toggleVideoCrop())
        
        # self.root.after(75, self.enforceCanvasLayering)
        self.thumbnailManager = ThumbnailManager(self, menubar=self.menubar)
        
        self.loadVocalPresence()
    # end init

    def onClose(self):
        """Cleanly stop audio/video playback when this window is closed."""
        self.isPlaying = False
        self.isPaused = False
        self.isManualUpdate = True
        ModalGuard.close("voice_app")
        if self.isExportingVideo:
            self._onCancelExport()  # Ensure export is stopped if window is closed during export
        
        # Stop music if it's playing
        try:
            if pygame.mixer.get_init():
                try:
                    pygame.mixer.music.stop()
                except Exception:
                    pass

                # pygame 2.x has unload(); this is what actually releases the file handle
                try:
                    pygame.mixer.music.unload()
                except Exception:
                    # unload may not exist on very old pygame
                    pass

                # Quit mixer to be extra sure no handle remains
                try:
                    pygame.mixer.quit()
                except Exception:
                    pass
        except Exception as e:
            print("Error stopping/closing mixer on close:", e)

        # Stop video if present
        if hasattr(self, "videoTrackItem") and self.videoTrackItem:
            try:
                self.videoTrackItem.pause()
                self.videoTrackItem.stop()
            except Exception as e:
                print("Error stopping video on close:", e)
        
        # Destroy just this window (the UI window), not the whole app
        ModalGuard.close("lyrics_menu")
        ModalGuard.close("labels_menu")
        self.root.destroy()
    
    def loadVocalPresence(self):
        """
        Loads precomputed 40ms vocal presence JSON and stores it in:
            self.vocalPresence  (np.ndarray, dtype=bool)

        Uses:
            ./training_data/{group}/{songName}_vocals_40ms_activity.json
        """
        if not hasattr(self, "selectedGroup") or not hasattr(self, "songName"):
            print("[VocalPresence] Missing selectedGroup or songName.")
            self.vocalPresence = None
            return

        jsonPath = os.path.join(
            ".",
            "training_data",
            self.selectedGroup,
            f"{self.songName}_vocals_40ms_activity.json"
        )

        if not os.path.isfile(jsonPath):
            print(f"[VocalPresence] JSON not found: {os.path.abspath(jsonPath)}")
            self.vocalPresence = None
            return

        try:
            with open(jsonPath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "isSilence" not in data:
                print("[VocalPresence] 'isSilence' not found in JSON.")
                self.vocalPresence = None
                return

            # Convert to boolean numpy array
            self.vocalPresence = np.array(data["isSilence"], dtype=bool)

            print(f"[VocalPresence] Loaded {len(self.vocalPresence)} chunks.")

        except Exception as e:
            print(f"[VocalPresence] Failed to load JSON: {e}")
            self.vocalPresence = None
    
    def initStatusMessage(self):
        self._statusVisible = False
        self.statusMessageId = self.canvas.create_text(
            10, self.canvas.winfo_height() - 20,
            anchor="sw",
            text="",
            fill="#cccccc",
            font=("Helvetica", 12),
            state="hidden"
        )
        self._statusClearJob = None

        # keep it pinned on resize
        def _reposition(_=None):
            h = self.canvas.winfo_height()
            if h <= 2:
                return
            self.canvas.coords(self.statusMessageId, 10, h - 10)

        self.canvas.bind("<Configure>", _reposition, add="+")
        self.root.after(0, _reposition)
        
    def showStatus(self, message, level="info", timeout=2000):
        """
        Show a non-intrusive status message on the canvas.
        level: info | warn | error
        """
        color = {
            "info": "#cfd8dc",
            "warn": "#ffcc80",
            "error": "#ef9a9a"
        }.get(level, "#cfd8dc")
        
        self._statusVisible = True

        self.canvas.itemconfig(
            self.statusMessageId,
            text=message,
            fill=color,
            state="normal"
        )

        # cancel previous clear
        if self._statusClearJob:
            try:
                self.root.after_cancel(self._statusClearJob)
            except Exception:
                pass

        # auto-clear
        self._statusClearJob = self.root.after(timeout, self._clearStatus)

    def _clearStatus(self):
        self._statusVisible = False
        self.canvas.itemconfig(self.statusMessageId, state="hidden")
        self._statusClearJob = None
    
    def resetLabels(self, event):
        self.labels = []
        self.selectedLabel = None
        self.startPoints = []
        self.endPoints = []
        self.startPointMarkers = {}
        self.endPointMarkers = {}
        self.labels = self.loadSavedLabels()
        for trackItem in self.memberImages.values():
            if trackItem:
                trackItem.initializeTimeline(includeBacking=self.includeBacking)
        
        self.initializePositions()
    
    def _getHistoryFilePath(self):
        fileNameWithoutExtension = self.songName
        return f"./saved_labels/{self.selectedGroup}/{fileNameWithoutExtension}_history.json"
    
    def toggleVideoCrop(self):
        enabled = self.cropVideoVar.get()
        if hasattr(self, "videoTrackItem"):
            self.videoTrackItem.toggleCrop(enabled)
          
    def setPredictedPointsFromMask(self, binaryMask, minSingingLength=3):
        """
        Use a binaryMask (0=silence, 1=singing) to populate self.labels, self.startPoints, and self.endPoints.
        """
        inSegment = False
        start = None
        
        for i, val in enumerate(binaryMask):
            if val == 1:
                if not inSegment:
                    # Only start a new segment if not already inside one
                    if not self.endPoints or i > self.endPoints[-1]:
                        inSegment = True
                        start = i
            else:
                if inSegment:
                    # Only end a segment if inside one
                    end = i - 1
                    if start is not None and end >= start:
                        segmentLength = end - start + 1
                        if segmentLength >= minSingingLength:
                            self.startPoints.append(start)
                            self.endPoints.append(end)
                    inSegment = False
                    start = None

        # Handle case where last segment reaches the very end
        if inSegment and start is not None:
            end = len(binaryMask) - 1
            if end >= start:
                segmentLength = end - start + 1
                if segmentLength >= minSingingLength:
                    self.startPoints.append(start)
                    self.endPoints.append(end)

        self.updateLabelMarkersDict()
        self.drawTimeMarkers()
        self.canvas.update()
        self.root.update_idletasks()   
        
    def buildLyricStartEvents(self):
        self.startEvents = {}
        self.activeLyricIds = set()

        for startChunk in self.lyrics.keys():
            self.startEvents.setdefault(startChunk, []).append(startChunk)  
    
    
    def setUIHidden(self, hidden: bool):
        """
        Explicitly set UI visibility state.
        hidden=True  -> UI hidden
        hidden=False -> UI visible
        """
        self.uiHidden = hidden
        newState = "hidden" if hidden else "normal"
        
        if hidden: 
            self.setPanelVisibility(not hidden)
            
        # --- Navigation arrows ---
        self.navigationArrows.updateArrows(self.progressBarCanvas)
        for arrow in self.navigationArrows.arrows.values():
            self.canvas.itemconfig(arrow, state=newState)

        # --- Progress bar handle ---
        self.canvas.itemconfig(self.progressBarHandle.handle, state=newState)

        # --- Progress bar canvas ---
        if hidden:
            self.progressBarCanvas.place_forget()
        else:
            self.progressBarCanvas.place(relx=0.5, rely=0.9, anchor="center")

        # --- Label lane renderer ---
        if hidden:
            self.labelLaneRenderer.hide()
        else:
            self.labelLaneRenderer.show()
        
        self.labelLaneRenderer.drawSection(
            self.currentSectionIndex,
            self.progressBarWidth
        )

        # --- Time markers ---
        self.canvas.itemconfig("time_marker", state=newState)

        # --- Time display label ---
        if hidden:
            self.timeDisplayLabel.place_forget()
        else:
            self.timeDisplayLabel.place(relx=0.5, rely=0.95, anchor="center")

        # --- Zoom UI ---
        self.zoomManager.setVisibility(hidden)

        # --- Label markers ---
        if hidden:
            self.clearAllMarkers()
        else:
            self.updateLabelMarkersDict()
            
        # --- Status message ---
        self.canvas.itemconfig(
            self.statusMessageId,
            state="normal" if self._statusVisible else "hidden"
        )
    
    def toggleUIElements(self, event=None):
        self.setUIHidden(not self.uiHidden)
    
    def startLayout(self):
        self.initializeMemberImages()
        self.initializePositions()
        self.slotHeightPx = int(round(self.slotHeightBase * self.scaleY))
        for t in self.memberImages.values():
            t.rescalePositionTimeline(self.scaleY)
        self.updateElementPositions() 
        self.lineDistPanel.update()
        self.enforceCanvasLayering()
    
    def startExportVideo(self, event=None):
        print("Video record function called!")
        if self.isExportingVideo:
            return
        
        if self.thumbnailManager.thumbnailMode:
            self.thumbnailManager.exitThumbnailMode()
        
        videoPath = self.videoPath

        originalAudioPath = self.originalSongPath
        
        savedChunk = int(self.currentChunkIndex)
        savedSection = int(self.currentSectionIndex)
        savedUIHidden = bool(self.uiHidden)
        
        self.isExportingVideo = True
        
        def _logExportError(msg: str):
            os.makedirs("./logs", exist_ok=True)
            with open("./logs/export_debug.log", "a", encoding="utf-8") as f:
                f.write(f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
                f.write(msg + "\n")
                
        try:
            # If you want UI hidden during export, do it deterministically:
            self.setUIHidden(True)

            # IMPORTANT: ensure export starts from current chunk
            # (processVideoAndSave will use self.currentChunkIndex after the change above)
            self.videoTrackItem.processVideoAndSave(
                songName=self.songName,
                originalAudioPath=originalAudioPath,
                originalVideoPath=videoPath,
                fpsCap=(0 if self.videoTrackItem.isMusicVideo else 24),
            )
        except Exception:
            tb = traceback.format_exc()
            _logExportError(tb)
            messagebox.showerror(
                "Export Error",
                "Export crashed. Full traceback saved to ./logs/export_debug.log",
                parent=self.root
            )
            
        finally:
            self.isExportingVideo = False

            # restore UI + position exactly
            self.setUIHidden(savedUIHidden)
            self.currentSectionIndex = savedSection
            self.seekToChunk(savedChunk)
     
    def _onCancelExport(self, event=None):
        if getattr(self, "isExportingVideo", False):
            print("🛑 Export cancel requested...")
            self.exportStopEvent.set()
            self.setUIHidden(False)
        
    def _finishExportVideo(self):
        self.isExportingVideo = False
        self.exportStopEvent.clear()
        self.activeLyricIds.clear()
        self.enableRootKeybinds()
        self.toggleUIElements()
                        
    def createThumbnail(self):
        basePath = self.testSongPath.rsplit('\\', 1)[0]
        thumbnailImagePath = os.path.join(basePath, "background.jpg")
        
        try:
            thumbnailImage = Image.open(thumbnailImagePath)
            thumbnailImage = thumbnailImage.resize((1280, 720), Image.Resampling.LANCZOS)
            thumbnailTk = ImageTk.PhotoImage(thumbnailImage)
            
            self.thumbnailId = self.canvas.create_image(0, 0, anchor="nw", image=thumbnailTk)
            self.canvas.tag_lower(self.thumbnailId)
            self.thumbnailTk = thumbnailTk    
        except FileNotFoundError:
            print(f"Error: {thumbnailImagePath} not found.")    
    
    def enforceCanvasLayering(self):
        c = self.canvas

        order = [
            "lyrics_bg",
            "lyrics_card_bg",
            "lyrics",
            "time_marker",
            "label_bar",
            "marker",
        ]

        # Safely raise nav arrows on the progress bar canvas (if they exist there)
        try:
            if self.progressBarCanvas.find_withtag("nav_arrow"):
                self.progressBarCanvas.tag_raise("nav_arrow")
        except Exception:
            pass

        # Safely enforce layering on main canvas
        for below, above in zip(order, order[1:]):
            if c.find_withtag(below) and c.find_withtag(above):
                c.tag_raise(above, below)
            
    def setThumbnail(self, event=None):
        # safe toggle instead of deleting canvas items
        self.thumbnailManager.toggleThumbnailMode()
        
    def changeMode(self, event):
        if self.testOrVideo == "Test":
            self.testOrVideo = "Video"
            print("Mode switched to Video!")
        elif self.testOrVideo == "Video":
            self.testOrVideo = "Test"
            print("Mode switched to Test!")
    
    def addBackgroundImage(self):
        # Remove previous background if it exists
        if hasattr(self, "lyricsBackgroundId") and self.lyricsBackgroundId:
            self.canvas.delete(self.lyricsBackgroundId)
            self.lyricsBackgroundId = None

        # Same positioning as before
        x = 750 * self.scaleX
        y = 0

        # Figure out how big the plane should be
        canvasW = self.canvas.winfo_width() or int(self.baseWidth * self.scaleX)
        canvasH = self.canvas.winfo_height() or int(self.baseHeight * self.scaleY)

        # Fill from x to the right edge, and from top to bottom
        x2 = canvasW
        y2 = canvasH

        # Create the light gray plane
        self.lyricsBackgroundId = self.canvas.create_rectangle(
            x, y, x2, y2,
            fill="#f6f6f6",
            outline="",
            tags="lyrics_bg"
        )

        # Raise to top (same behavior you had)
        self.syncLyricBoxToBackground()
        self.enforceCanvasLayering()
        
    def syncLyricBoxToBackground(self):
        def getLyricsBackgroundLeftX():
            bgId = getattr(self, "lyricsBackgroundId", None)
            if not bgId:
                return None

            bbox = self.canvas.bbox(bgId)  # (x1, y1, x2, y2) or None
            if not bbox:
                return None

            x1, _, _, _ = bbox
            return x1

        leftX = getLyricsBackgroundLeftX()
        #print(f"Left x for white: {leftX}")

        padding = int(10 * self.scaleX)  # optional, tweak to taste
        targetCanvasX = leftX + padding
        
        offX = getattr(self, "viewportOffsetX", 0)
        scaleX = getattr(self, "scaleX", 1.0)
        if scaleX <= 0:
            scaleX = 1.0
    
        targetBaseX = int(round((targetCanvasX - offX) / scaleX))

        self.targetLyricsX = targetBaseX
    
    def evaluateVoiceDetectionAccuracy(self):
        if not hasattr(self, "labels") or not self.labels:
            print("No labels available for evaluation.")
            return None
        
        if not hasattr(self, "voiceDetectionResults") or not self.voiceDetectionResults:
            print("⚠️ No predictions available in self.voiceDetectionResults.")
            return None
        
        numChunks = len(self.voiceDetectionResults)
        groundTruth = [set() for _ in range(numChunks)] # List of sets of true singers per chunk
        
        # Create ground truth chunk-level label array
        for label in self.labels:
            if len(label) < 3: continue
            member, start, end = label[:3]
            for i in range(start, end + 1):
                if 0 <= i < numChunks:
                    groundTruth[i].add(member)
                    
        # Evaluate predictions
        correct = total = 0
        
        # For member-specific stats
        memberTP = defaultdict(int)
        memberFP = defaultdict(int)
        memberFN = defaultdict(int)
        
        for i in range(numChunks):
            frameRoles = self.voiceDetectionResults[i]
            # union of all heads for accuracy
            main_names = frameRoles.get("main", []) or []
            harm_names = frameRoles.get("harmony", []) or []
            adlib_names = frameRoles.get("adlib", []) or []
            
            predicted = set()
            for m in main_names:
                predicted.add(m)
            for m in harm_names:
                predicted.add(m)
            for m in adlib_names:
                predicted.add(m)
            
            actual = groundTruth[i]
 
            # Update metrics
            correct += len(predicted & actual)
            total += len(predicted | actual)
            
            for member in predicted:
                if member in actual:
                    memberTP[member] += 1
                else:
                    memberFP[member] += 1
            
            for member in actual:
                if member not in predicted:
                    memberFN[member] += 1
            
        accuracy = correct / total if total > 0 else 0
        print(f"\nOverall Accuracy: {accuracy:.4f}")
        
        print("Member-wise Metrics:")
        for member in sorted(set(memberTP.keys()) | set(memberFN.keys())):
            tp = memberTP[member]
            fp = memberFP[member]
            fn = memberFN[member]

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print(f"  {member:10s} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

        return accuracy
        
    def moveMarkerLeft(self, event):
        """
        Move the selected marker left by one chunkIndex.
        """
        if self.selectedLabel:
            self.pushUndoState("marker move left")
        self.moveMarker(-1)

    def moveMarkerRight(self, event):
        """
        Move the selected marker right by one chunkIndex.
        """
        if self.selectedLabel:
            self.pushUndoState("marker move right")
        self.moveMarker(1)

    def selectMarker(self, chunkIndex, markerType):
        print(f"Marker selected at {chunkIndex} with type {markerType}")
        self.setSelectedMarker(chunkIndex, markerType)
       
    def deleteLabelAndMarkers(self, label):
        """
        Given a full label [member, startChunk, endChunk, isBacking, isAdlib],
        delete ONE start marker instance + ONE end marker instance for that label,
        update arrays, and update JSON.
        """
        member, startChunk, endChunk, _, _ = label

        def _deleteOneMarkerAtChunk(markerDict, chunk, defaultColorType):
            """
            Remove and delete exactly one canvas markerId from markerDict[chunk] (multiset-safe).
            Returns the deleted markerId or None.
            """
            ids = self._getMarkerIdsAtChunk(markerDict, chunk)
            if not ids:
                return None

            # pick one to delete (last drawn is usually topmost; any single is fine)
            markerId = ids.pop()  # removes ONE instance
            self._setMarkerIdsAtChunk(markerDict, chunk, ids)

            # remove overlay tracking for that one id
            if getattr(self, "labelOverlay", None):
                if hasattr(self.labelOverlay, "forgetMarker"):
                    self.labelOverlay.forgetMarker(markerId)

            # delete actual canvas item
            try:
                self.canvas.delete(markerId)
            except tk.TclError:
                pass

            return markerId

        # --- Remove exactly one start marker instance ---
        _deleteOneMarkerAtChunk(self.startPointMarkers, startChunk, "start")
        if startChunk in self.startPoints:
            # remove ONE occurrence
            self.startPoints.remove(startChunk)

        # --- Remove exactly one end marker instance ---
        _deleteOneMarkerAtChunk(self.endPointMarkers, endChunk, "end")
        if endChunk in self.endPoints:
            self.endPoints.remove(endChunk)

        # --- Update internal labels list (remove exactly this label) ---
        removed = False
        newLabels = []
        for l in self.labels:
            if (not removed) and (l[0] == member and l[1] == startChunk and l[2] == endChunk):
                removed = True
                continue
            newLabels.append(l)
        self.labels = newLabels

        # --- Rebuild marker timeline dict from labels (recommended) ---
        self.updateLabelMarkersDict()

        # --- Save updated labels back to JSON ---
        labelFilePath = f"./saved_labels/{self.selectedGroup}/{self.songName}_labels.json"
        try:
            sortedLabels = sorted(self.labels, key=lambda lab: lab[1])
            with open(labelFilePath, "w") as file:
                json.dump(sortedLabels, file, indent=4)
        except Exception as e:
            print(f"Error updating labels in {labelFilePath}: {e}")

        # Optional: refresh timelines now that labels changed
        for trackItem in self.memberImages.values():
            trackItem.initializeTimeline(includeBacking=self.includeBacking)
      
    def deleteSelectedMarker(self, event=None):
        """
        Delete the selected marker.

        Rules:
        - If a label is currently selected (self.selectedLabel), delete that label and its boundary markers.
        - Otherwise, delete ONLY the selected markerId (stray / not tied to selectedLabel).
        """
        if not self.selectedMarker:
            return

        markerId   = self.selectedMarker.get("id")          # MUST be a single canvas id (int)
        chunkIndex = self.selectedMarker.get("chunkIndex")
        markerType = self.selectedMarker.get("type")

        if markerId is None:
            # nothing safe to delete
            return

        # ---- Overlay cleanup (always by markerId) ----
        self.labelOverlay.hide()
        self.labelOverlay.forgetMarker(markerId)

        # ---- Case 1: a label is selected -> delete the whole label (preferred, avoids ambiguity) ----
        if self.selectedLabel is not None:
            self.pushUndoState("delete-label")
            # Your existing function should remove the label from self.labels and delete BOTH boundary markers
            # (and update JSON, timeMarkers, etc.)
            self.deleteLabelAndMarkers(self.selectedLabel)

            self.selectedMarker = None
            self.selectedLabel = None
            self.originalLabel = None
            return

        # ---- Case 2: no selectedLabel -> delete ONLY the selected markerId ----
        self.pushUndoState("delete stray marker")

        # Remove markerId from the chunk's id list (multiset-safe)
        if markerType == "start":
            ids = self._getMarkerIdsAtChunk(self.startPointMarkers, chunkIndex)
            if markerId in ids:
                ids.remove(markerId)
            self._setMarkerIdsAtChunk(self.startPointMarkers, chunkIndex, ids)

            # remove ONE occurrence in startPoints (multiset-safe)
            if chunkIndex in self.startPoints:
                self.startPoints.remove(chunkIndex)

            if getattr(self, "openStartChunk", None) == chunkIndex:
                self.openStartChunk = None

        elif markerType == "end":
            ids = self._getMarkerIdsAtChunk(self.endPointMarkers, chunkIndex)
            if markerId in ids:
                ids.remove(markerId)
            self._setMarkerIdsAtChunk(self.endPointMarkers, chunkIndex, ids)

            if chunkIndex in self.endPoints:
                self.endPoints.remove(chunkIndex)

        # Delete the actual canvas item
        try:
            self.canvas.delete(markerId)
        except tk.TclError:
            pass
        
        self.openStartChunk = None
        # re-stack visuals at this chunk (optional but usually correct)
        self.restackMarkersAtChunk(chunkIndex)

        # Clear selection
        self.selectedMarker = None
        self.selectedLabel = None
        self.originalLabel = None
    
    def resetMarkerColor(self):
        """
        Reset the color of the previously selected marker, if any.
        """
        if not self.selectedMarker:
            return

        markerId = self.selectedMarker.get("id")
        markerType = self.selectedMarker.get("type")

        if markerId is not None:
            try:
                self.canvas.itemconfig(markerId, fill=self._defaultMarkerColor(markerType))
            except tk.TclError:
                # marker might have been deleted/redrawn
                pass

        self.selectedMarker = None
    
    def _getClickedMarkerId(self, event):
        # Don't use find_closest by itself; it can "select" something even if you clicked empty space.
        # Use a small hitbox around the cursor.
        hit = self.canvas.find_overlapping(event.x-3, event.y-3, event.x+3, event.y+3)
        if not hit:
            return None
        # Prefer topmost item
        return hit[-1]
    
    def onMarkerClick(self, event):
        """
        Detect if a marker is clicked and darken its color.
        """
        self.resetMarkerColor()
        self.isDraggingMarker = False
        self.pendingDrag = False
        self.dragStartXY = (event.x, event.y)
        self.dragStartLabels = None
        self.selectedMarker = None
       
        clickedId = self._getClickedMarkerId(event)
        if clickedId is None:
            self.selectedMarker = None
            self.selectedLabel = None
            self.originalLabel = None
            return
        
        # --- START markers ---
        for chunkIndex, markerVal in self.startPointMarkers.items():
            ids = markerVal if isinstance(markerVal, list) else [markerVal]
            if clickedId in ids:
                self.setSelectedMarker(chunkIndex, "start", markerId=clickedId)
                self.prepareLabelUpdate(chunkIndex, "start")
                self.pendingDrag = True
                self.dragStartLabels = copy.deepcopy(self.labels)
                return

        # --- END markers ---
        for chunkIndex, markerVal in self.endPointMarkers.items():
            ids = markerVal if isinstance(markerVal, list) else [markerVal]
            if clickedId in ids:
                self.setSelectedMarker(chunkIndex, "end", markerId=clickedId)
                self.prepareLabelUpdate(chunkIndex, "end")
                self.pendingDrag = True
                self.dragStartLabels = copy.deepcopy(self.labels)
                return
            
        self.selectedMarker = None 
        self.originalLabel = None
        self.isDraggingMarker = False
        
    def onMarkerMotion(self, event):
        # Not interacting with a marker → ignore
        if not self.pendingDrag or not self.selectedMarker or not self.dragStartXY:
            return

        dx = event.x - self.dragStartXY[0]
        dy = event.y - self.dragStartXY[1]
        if not self.isDraggingMarker:
            if (dx*dx + dy*dy) < (self.dragThreshold * self.dragThreshold):
                return
            self.isDraggingMarker = True  # <-- becomes a real drag only after threshold

        # Now run your existing drag logic
        self.onMarkerDrag(event)
        
    def onMarkerDrag(self, event):
        """
        Drag the currently selected marker horizontally and update its chunk index.
        The progress bar handle + chunk counter follow the drag.
        """
        if not self.isDraggingMarker or not self.selectedMarker:
            return
        
        markerType = self.selectedMarker['type']
        oldChunkIndex = self.selectedMarker['chunkIndex']
        chunksInView = self.zoomManager.currentChunksInView
        markerSectionIndex = oldChunkIndex // chunksInView
        
        # Progress bar geometry (in canvas coordinates)
        barX = self.progressBarCanvas.winfo_x()
        barY = self.progressBarCanvas.winfo_y()
        barWidth = self.progressBarWidth
        
        # Constrain x within progress bar region
        x = max(barX, min(event.x, barX + barWidth))
        
        # Convert x to chunk offset within current section
        relative = (x - barX) / float(barWidth)
        chunkOffset = int(round(relative * (chunksInView - 1)))
        newChunkIndex = markerSectionIndex * chunksInView + chunkOffset
        
        # Clamp to valid range 
        newChunkIndex = max(0, min(len(self.chunks) - 1, newChunkIndex))
        
        # Enforce start <= end to avoid inverted labels
        if self.selectedLabel:
            startIdx, endIdx = self.selectedLabel[1], self.selectedLabel[2]
            if markerType == "start":
                # Don't let start go past end-1
                newChunkIndex = min(newChunkIndex, endIdx - 1) if endIdx > 0 else 0
            elif markerType == "end":
                # Don't let end go before start + 1
                newChunkIndex = max(newChunkIndex, startIdx + 1)
                
        if newChunkIndex == oldChunkIndex:
            # No effective movement
            return

        if markerType == "start":
            pointsList = self.startPoints
            markerDict = self.startPointMarkers
        else:
            pointsList = self.endPoints
            markerDict = self.endPointMarkers

        if oldChunkIndex in pointsList:
            pointsList.remove(oldChunkIndex)
        pointsList.append(newChunkIndex)

        self.selectedMarker["chunkIndex"] = newChunkIndex
        
        if self.selectedLabel:
            for label in self.labels:
                if label == self.selectedLabel:
                    if markerType == "start":
                        label[1] = newChunkIndex
                    else:
                        label[2] = newChunkIndex
                    self.selectedLabel = label  # keep ref up-to-date
                    break
        
        # --- Update timeMarkers only for old & new sections ---
        oldSection = oldChunkIndex // chunksInView
        newSection = newChunkIndex // chunksInView
        
        if hasattr(self, "labelMarkers"):
            # Remove tuple from oldSection
            if oldSection in self.labelMarkers:
                try:
                    self.labelMarkers[oldSection].remove((markerType, oldChunkIndex))
                    if not self.labelMarkers[oldSection]:
                        del self.labelMarkers[oldSection]
                except ValueError:
                    pass  # out of sync? ignore
                
                # Add tuple to newSection
            self.labelMarkers.setdefault(newSection, []).append((markerType, newChunkIndex))
        
        # --- Move the actual canvas line to the new X (before stacking) ---
        markerId = self.selectedMarker.get("id")
        if markerId is not None:
            # Compute new X for newChunkIndex
            relativeX = barX + (newChunkIndex % chunksInView / chunksInView) * barWidth
            x_new = self.canvas.canvasx(relativeX)
            baseY = barY
            height = 20

            # Temporarily place at base position
            self.canvas.coords(
                markerId,
                x_new, baseY - height,
                x_new, baseY
            )

            # Update dict key
            del markerDict[oldChunkIndex]
            markerDict[newChunkIndex] = markerId
        else:
            # Fallback: if marker not found, you *could* regenerate via updateLabelMarkers,
            # but this should normally not happen.
            print(f"Warning: canvas maifrker for {markerType} at {oldChunkIndex} not found.")
        
        # --- Re-stack only at the old and new chunks ---
        self.restackMarkersAtChunk(oldChunkIndex)
        self.restackMarkersAtChunk(newChunkIndex)
        
        # --- Sync the rest of the UI (chunk index, time, progress handle, video) ---
        self.selectedMarker["chunkIndex"] = newChunkIndex
        self.currentChunkIndex = newChunkIndex
        self.updateChunkIndexDisplay(newChunkIndex)

        newTimeMs = newChunkIndex * self.chunk_duration
    
        # --- Sync the rest of the UI (chunk index, time, progress handle, video) ---
        self.currentChunkIndex = newChunkIndex
        self.updateChunkIndexDisplay(newChunkIndex)
        newTimeMs = newChunkIndex * self.chunk_duration

        # Make the handle follow the drag and show time
        self.currentSectionIndex = markerSectionIndex
        self.updateDisplayedTime(newTimeMs)
        self.updateProgressBarHandle(newTimeMs)

        if hasattr(self, "videoTrackItem"):
            self.videoTrackItem.seek(newTimeMs)

        self.labelOverlay.updateBoundaryMarker(markerId, chunkIndex=newChunkIndex)
        self.isManualUpdate = True
        
    def onMarkerRelease(self, event):
        """
        When the user releases the mouse after dragging a marker,
        we save labels once to JSON to avoid per-frame lag.
        """
        # If we never crossed threshold, it's a click-select only
        if self.pendingDrag and not self.isDraggingMarker:
            self.pendingDrag = False
            self.dragStartXY = None
            # keep selection, but do NOT save JSON/undo/etc.
            return

        if not self.isDraggingMarker:
            return
        
        self.pendingDrag = False
        self.isDraggingMarker = False
        self.dragStartXY = None
        
        # only save if marker corresponds to an actual label
        if self.selectedLabel:
            # record previous state into undo stack
            if self.dragStartLabels is not None:
                snapshot = {
                    "labels": self.dragStartLabels,
                    "description": "marker drag",
                }
                self.undoStack.append(snapshot)
                self.redoStack.clear()
                self.appendHistoryToFile(snapshot)
                labelMember = self.selectedLabel[0]
                trackItem = self.memberImages.get(labelMember)
                if trackItem:
                    trackItem.initializeTimeline(includeBacking=self.includeBacking)
                self.dragStartLabels = None

            # now save the new labels to JSON, update timelines, etc.
            self.updateLabelInJSON()

    def appendHistoryToFile(self, state):
        """
        Append an action state to a JSON history file for long-term storage.
        This does NOT affect undo/redo after restart; it's mainly for inspection / debugging.
        """
        historyPath = self._getHistoryFilePath()
        try:
            if os.path.exists(historyPath):
                with open(historyPath, "r") as f:
                    history = json.load(f)
            else:
                history = []
            history.append(state)
            with open(historyPath, "w") as f:
                json.dump(history, f, separators=(",", ":"))
        except Exception as e:
            print(f"Could not append history: {e}")
            
    def pushUndoState(self, description=""):
        """
        Save the current labels into the undo stack.
        Clears the redo stack because a new action invalidates the forward history.
        """
        snapshot = {
            "labels": copy.deepcopy(self.labels),
            "description": description,
        }
        self.undoStack.append(snapshot)
        self.redoStack.clear()
        self.appendHistoryToFile(snapshot)
    
    def applyLabelsState(self, labels):
        """
        Replace self.labels with provided labels and refresh all marker-related state.
        """
        self.labels = copy.deepcopy(labels)
        self.onLabelsChanged(redrawSection=self.currentSectionIndex) 
        
        self.clipManager.rebuild(self.labels, len(self.chunks))       
        # Sync internal marker structures and redraw
        self.drawTimeMarkers()

        # Refresh member timelines / positions so everything stays consistent
        for trackItem in self.memberImages.values():
            if trackItem:
                trackItem.initializeTimeline(includeBacking=self.includeBacking)
        self.initializePositions()

        # Also overwrite the main labels JSON so it matches this state
        labelFilePath = f"./saved_labels/{self.selectedGroup}/{self.songName}_labels.json"
        try:
            with open(labelFilePath, "w") as f:
                json.dump(self.labels, f, separators=(",", ":"))
        except Exception as e:
            print(f"Error writing labels during undo/redo: {e}")
            
    def undo(self, event=None):
        if not self.undoStack:
            print("Nothing to undo.")
            return

        # Save current state into redo stack
        current = {
            "labels": copy.deepcopy(self.labels),
            "description": "auto-redo-snapshot",
        }
        self.redoStack.append(current)
        
        # Restore last undo state
        state = self.undoStack.pop()
        self.applyLabelsState(state["labels"])
        print("Undo:", state.get("description", ""))
    
    def redo(self, event=None):
        if not self.redoStack:
            print("Nothing to redo.")
            return
        
        # Save current state into undo stack
        current = {
            "labels": copy.deepcopy(self.labels),
            "description": "auto-undo-snapshot",
        }
        self.undoStack.append(current)
        
        # Restore last redo state
        state = self.redoStack.pop()
        self.applyLabelsState(state["labels"])
        print("Redo:", state.get("description", ""))

    def prepareLabelUpdate(self, chunkIndex, markerType):
        """
        Check if the selected marker belongs to a saved label and prepare for updates.
        """
        for label in self.labels:
            _, start, end = label[:3]
            if (markerType == "start" and start == chunkIndex) or (markerType == "end" and end == chunkIndex):
                self.selectedLabel = label
                self.originalLabel = label.copy()
                return
        self.selectedLabel = None
        self.originalLabel = None
    # end
    
    def updateLabelInJSON(self, event=None):
        """
        Save updated labels to JSON file
        """
        if not self.selectedLabel:
            return
        
        print("Labels have been updated!")
        labelFilePath = f"./saved_labels/{self.selectedGroup}/{self.songName}_labels.json"
        
        try:
            self.labels = dedupeLabelsByKey(self.labels)  # critical
            self.clipManager.rebuild(self.labels, len(self.chunks)) 
            with open(labelFilePath, "w") as file:
                json.dump(self.labels, file, separators=(",", ":"))

            self.updateLabelMarkersDict()
            self.initializePositions()

        except Exception as e:
            print(f"Error saving labels to {labelFilePath}: {e}")
             
    def restackMarkersAtChunk(self, chunkIndex):
        """
        Reposition markers at a single chunkIndex so that overlapping
        start/end markers are 'stacked' vertically instead of sitting on top
        of each other.

        This is O(1): only touches that one chunk, no full redraw.
        """  
        chunksInView = self.zoomManager.currentChunksInView
        if chunksInView <= 0:
            return
        
        if chunkIndex < 0 or chunkIndex >= len(self.chunks):
            return
        
        def _asIdList(v):
            if not v:
                return []
            if isinstance(v, (list, tuple)):
                return list(v)
            return [v]  # single id
        
        startIds = _asIdList(self.startPointMarkers.get(chunkIndex))
        endIds = _asIdList(self.endPointMarkers.get(chunkIndex))

        if not startIds and not endIds:
            return
        
        # Compute x position for this chunk
        barX = self.progressBarCanvas.winfo_x()
        barY = self.progressBarCanvas.winfo_y()
        barWidth = self.progressBarWidth
        
        relativeX = barX + ((chunkIndex % chunksInView) / chunksInView) * barWidth
        x = self.canvas.canvasx(relativeX)
        
        baseY = barY
        height = 20
        stackOffset = 20
        
        allIdsInOrder = startIds + endIds
        
        for stackIndex, markerId in enumerate(allIdsInOrder):
            yTop = baseY - height - (stackIndex * stackOffset)
            yBot = baseY - (stackIndex * stackOffset)
            self.canvas.coords(markerId, x, yTop, x, yBot)
     
    def _getMarkerIdsAtChunk(self, markerDict, chunkIndex):
        v = markerDict.get(chunkIndex)
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]  # backward compatibility if old code stored an int
    
    def _setMarkerIdsAtChunk(self, markerDict, chunkIndex, ids):
        if not ids:
            markerDict.pop(chunkIndex, None)
        else:
            markerDict[chunkIndex] = ids
    
    def _replaceOneOccurrence(self, pointsList, oldValue, newValue):
        """
        Replace ONE occurrence of oldValue with newValue in pointsList.
        If oldValue isn't found (shouldn't happen, but stay resilient), append newValue.
        This preserves the multiset counts without deleting "other" points.
        """
        try:
            i = pointsList.index(oldValue)  # O(n), fine at your scale
            pointsList[i] = newValue
        except ValueError:
            pointsList.append(newValue)
                           
    def moveMarker(self, direction):
        """
        Move the selected marker left (-1) or right (+1) by one chunkIndex.
        """
        def calculateX(chunkIndex):
            chunksInView = self.zoomManager.currentChunksInView
            return (
                self.progressBarCanvas.winfo_x() + 
                (chunkIndex % chunksInView / chunksInView) * self.progressBarWidth
            )
        # end calculateX 
        
        if not self.selectedMarker:
            # print("No marker selected.")
            return
        
        self.labelOverlay.hide()
        oldChunkIndex = self.selectedMarker["chunkIndex"]
        markerType = self.selectedMarker["type"]
        
        chunksInView = self.zoomManager.currentChunksInView
        markerSectionIndex = oldChunkIndex // chunksInView
        
        visibleSectionIndex = self.currentSectionIndex
        
        # If marker isn't in the current drawn section, jump UI first
        if markerSectionIndex != visibleSectionIndex:
            self.jumpToSection(markerSectionIndex)
            
            # after redraw, canvas ids changed; refresh markerId from dict
            if markerType == "start":
                ids = self._getMarkerIdsAtChunk(self.startPointMarkers, oldChunkIndex)
            else:
                ids = self._getMarkerIdsAtChunk(self.endPointMarkers, oldChunkIndex)
                
            # If still missing, bail safely (keeps selection but prevents coruption)
            # Choose an id deterministically.
            # If you don't yet track which stacked lane was selected, choose the topmost/last-drawn.
            self.selectedMarker["id"] = ids[-1] if ids else None

            if self.selectedMarker["id"] is None:
                print("Marker exists logically but isn't drawable in this section right now.")
                return
          
        markerId = self.selectedMarker.get("id")  
        newChunkIndex = oldChunkIndex + direction
        # print(f"Old chunk index: {chunkIndex}, New: {newChunkIndex}")
        if newChunkIndex < 0 or newChunkIndex >= len(self.chunks):
            print("Cannot move marker beyond bounds.")
            return
        
        if self.selectedLabel:
            startIdx, endIdx = self.selectedLabel[1], self.selectedLabel[2]
            if markerType == "start":
                newChunkIndex = min(newChunkIndex, endIdx - 1)
            else:
                newChunkIndex = max(newChunkIndex, startIdx + 1)

        if newChunkIndex == oldChunkIndex:
            return

        if markerType == "start" and getattr(self, "openStartChunk", None) == oldChunkIndex:
            self.openStartChunk = newChunkIndex
    
        y = self.progressBarCanvas.winfo_y()
        
        # Find markerId if missing (fallback, keeps code resilient)
        if markerId is None:
            if markerType == "start":
                markerId = self.startPointMarkers.get(oldChunkIndex)
            else:
                markerId = self.endPointMarkers.get(oldChunkIndex)

        if markerId is None:
            print(f"Warning: {markerType} marker at chunkIndex {oldChunkIndex} not found.")
            return
        
        # ---- Update canvas item position WITHOUT deleting it ----
        xNew = calculateX(newChunkIndex)
        self.canvas.coords(markerId, xNew, y - 20, xNew, y)
        
        # --- Remove old marker + index from the appropriate structures -
        if markerType == "start":
            # remove markerId from old chunk list
            oldIds = self._getMarkerIdsAtChunk(self.startPointMarkers, oldChunkIndex)
            if markerId in oldIds:
                oldIds.remove(markerId)
            self._setMarkerIdsAtChunk(self.startPointMarkers, oldChunkIndex, oldIds)

            # add markerId to new chunk list
            newIds = self._getMarkerIdsAtChunk(self.startPointMarkers, newChunkIndex)
            newIds.append(markerId)
            self._setMarkerIdsAtChunk(self.startPointMarkers, newChunkIndex, newIds)

            # multiset list update: remove ONE occurrence, add ONE occurrence
            self._replaceOneOccurrence(self.startPoints, oldChunkIndex, newChunkIndex)
            
        elif markerType == "end":
            oldIds = self._getMarkerIdsAtChunk(self.endPointMarkers, oldChunkIndex)
            if markerId in oldIds:
                oldIds.remove(markerId)
            self._setMarkerIdsAtChunk(self.endPointMarkers, oldChunkIndex, oldIds)

            newIds = self._getMarkerIdsAtChunk(self.endPointMarkers, newChunkIndex)
            newIds.append(markerId)
            self._setMarkerIdsAtChunk(self.endPointMarkers, newChunkIndex, newIds)

            self._replaceOneOccurrence(self.endPoints, oldChunkIndex, newChunkIndex)
        # Update label in self.labels if applicable
        if self.selectedLabel:
            # Update the label directly in self.labels if it's stored as a list
            for label in self.labels:
                if label == self.selectedLabel:
                    if markerType == "start":
                        label[1] = newChunkIndex  # Update the start index
                    elif markerType == "end":
                        label[2] = newChunkIndex  # Update the end index
                    # self.upsertLabel(label, self.selectedLabel)
                    self.selectedLabel = label  # Update the reference to the modified label
                    break
        
        # Restack locally at old and new chunk Positions
        self.restackMarkersAtChunk(oldChunkIndex)
        self.restackMarkersAtChunk(newChunkIndex)    
        
        # Update selectedMarker state
        self.selectedMarker["chunkIndex"] = newChunkIndex
        self.selectedMarker["id"] = markerId
        
        # Not working
        self.labelOverlay.updateBoundaryMarker(markerId, chunkIndex=newChunkIndex)
        self.updateLabelInJSON()
    
    def jumpToSection(self, sectionIndex: int):
        sectionIndex = max(0, sectionIndex)
        
        self.currentSectionIndex = sectionIndex
        self.progressBarHandle.currentSectionIndex = sectionIndex

        # Redraw only what depends on the section
        self.drawLabelMarkers(sectionIndex)
        self.drawTimeMarkers()

    def setupMenubar(self, root: tk.Tk):
        # This replaces addControls entirely (no bottom frame!)
        self.menubar = getOrCreateMenubar(root)
        
        # =========================
        # RECORD MENU (DEDICATED)
        # =========================
        recordMenu = tk.Menu(self.menubar, tearoff=0)

        recordMenu.add_command(
            label="Start Recording / Export…",
            accelerator="Ctrl+G",
            command=self.startExportVideo
        )

        # Optional future-proof items (safe to keep commented for now)
        # recordMenu.add_separator()
        # recordMenu.add_command(label="Open Finished Videos Folder", command=self.openFinishedVideosFolder)
        # recordMenu.add_command(label="Record Settings…", command=self.openRecordSettings)

        self.menubar.add_cascade(label="Record", menu=recordMenu)
        # =========================
        # EDIT MENU (MOST IMPORTANT)
        # =========================
        # --- Editing / Labels menu ---
        self.editingEnabledVar = tk.BooleanVar(value=getattr(self.clipManager, "enabled", True))
        labelsMenu = tk.Menu(self.menubar, tearoff=0)
            
        editMenu = tk.Menu(self.menubar, tearoff=0)

        editMenu.add_command(
            label="Undo",
            accelerator="Ctrl+Z",
            command=self.undo
        )
        editMenu.add_command(
            label="Redo",
            accelerator="Ctrl+Y",
            command=self.redo
        )

        editMenu.add_separator()

        editMenu.add_checkbutton(
            label="Enable Editing",
            onvalue=True,
            offvalue=False,
            variable=self.editingEnabledVar,
            command=lambda: (
                self.clipManager.toggleEnabled(),
                self.updateChunkIndexDisplay(self.currentChunkIndex)
            )
        )

        self.menubar.add_cascade(label="Edit", menu=editMenu)

        labelsMenu.add_separator()

        labelsMenu.add_command(
            label="Set Start Point",
            accelerator="Q",
            command=self.addStartPoint
        )
        labelsMenu.add_command(
            label="Set End Point",
            accelerator="W",
            command=self.addEndPoint
        )
        labelsMenu.add_command(
            label="Add Labels…",
            accelerator="E",
            command=lambda: self.showAddLabelsMenu(event=None)
        )
        
        labelsMenu.add_command(
            label="Toggle Pie Chart",
            accelerator="P",
            command=self.togglePanelVisibility
        )

        self.menubar.add_cascade(label="Labels", menu=labelsMenu)

        # =========================
        # PLAYBACK MENU
        # =========================
        playbackMenu = tk.Menu(self.menubar, tearoff=0)

        playbackMenu.add_command(
            label="Play / Pause",
            accelerator="Space",
            command=self.togglePlayPause
        )
        playbackMenu.add_command(
            label="Toggle Audio Mode",
            accelerator="V",
            command=self.toggleAudioMode
        )

        self.menubar.add_cascade(label="Playback", menu=playbackMenu)

        # =========================
        # VIEW MENU
        # =========================
        viewMenu = tk.Menu(self.menubar, tearoff=0)

        viewMenu.add_command(
            label="Toggle UI",
            accelerator="Ctrl+H",
            command=self.toggleUIElements
        )
        
        viewMenu.add_separator()

        viewMenu.add_checkbutton(
            label="Remove Black Bars (Crop Letterbox)",
            accelerator="Ctrl+B",
            variable=self.cropVideoVar,
            onvalue=True,
            offvalue=False,
            command=self.toggleVideoCrop
        )

        self.menubar.add_cascade(label="View", menu=viewMenu)
        
        # --- Lyrics menu ---
        lyricsMenu = tk.Menu(self.menubar, tearoff=0)
        lyricsMenu.add_command(
            label="Open Lyrics Menu…",
            accelerator="L",
            command=self.lyricsEditor.openLyricsEditorMenu
        )
        self.menubar.add_cascade(label="Lyrics", menu=lyricsMenu)

        # --- Tools menu (your “count backing” + reset positions) ---
        toolsMenu = tk.Menu(self.menubar, tearoff=0)
        toolsMenu.add_command(label="Reset Positions: Ctrl-R", command=lambda: self.countBacking(switch=False))
        toolsMenu.add_command(label="Count Backing", command=lambda: self.countBacking(switch=True))
        self.menubar.add_cascade(label="Tools", menu=toolsMenu)
        
        self.updateChunkIndexDisplay(self.currentChunkIndex)
    
    def initChunkIndexInTitle(self, root: tk.Tk):
        self._titleRoot = root
        self._baseTitle = root.title() or "Line Distribution Creator"
        self.menubar = None

    def updateChunkIndexDisplay(self, chunkIndex):
        # Call this whenever currentChunkIndex changes
        editing = "ON" if getattr(self.clipManager, "enabled", True) else "OFF"
        self._titleRoot.title(f"{self._baseTitle} |  Chunk {chunkIndex}  |  Editing {editing}")
    
    def refreshLyricsLayout(self):
        # 1) Rebuild each lyric box’s canvas items at the new scale
        for lyricBox in self.lyrics.values():
            lyricBox.rebuildForResize()
        
        self.rebuildLyricsAnimations()
        self.resetLyricsToChunk(self.currentChunkIndex)
        self.renderLyrics(self.currentChunkIndex)
    
    def onCanvasResize(self, event):
        aspectRatio = self.baseWidth / self.baseHeight 
        newWidth = int(self.canvas.winfo_width() * 0.75)
        self.progressBarWidth = newWidth
        
        if event.width / event.height > aspectRatio:
            # Width is too large, adjust based on height
            newHeight = event.height
            newWidth = int(newHeight * aspectRatio)
        else:
            # Height is too large, adjust based on width
            newWidth = event.width
            newHeight = int(newWidth / aspectRatio)
        
        self.viewportOffsetX = int((event.width  - newWidth) / 2)
        self.viewportOffsetY = int((event.height - newHeight) / 2)
        
        self.scaleX = newWidth / self.baseWidth
        self.scaleY = newHeight / self.baseHeight
        
        # 1) Rescale member timelines (vertical animation positions)
        self.slotHeightPx = int(round(self.slotHeightBase * self.scaleY))
        for trackItem in self.memberImages.values():
            trackItem.rescalePositionTimeline(self.scaleY)
        
        self.addBackgroundImage()
        self.progressBarCanvas.config(width=self.progressBarWidth)
        self.navigationArrows.updateArrows(self.progressBarCanvas)
        self.updateElementPositions()
        
        # Reset lyrics layout for new scale
        if hasattr(self, "lyrics") and self.lyrics:
            self.refreshLyricsLayout()
        
        self.progressBarHandle.progressBarWidth = newWidth
        self.updateLabelMarkersDict()
        self.root.after(50, self.drawTimeMarkers)
        
        if hasattr(self, "videoTrackItem"):
            # Adjust video height to fit canvas and maintain aspect ratio
            self.videoTrackItem.resize(newHeight)
        self.lineDistPanel.onResize(newWidth, newHeight)
        self.enforceCanvasLayering()
    
    def updateElementPositions(self):
        """Update the position and size of all canvas elements based on the new scale."""
        currentChunk =  self.currentChunkIndex
        
        for member, trackItem in self.memberImages.items():
            # Resize portrait
            effectiveScale = trackItem.scale * min(self.scaleX, self.scaleY)
            trackItem.resizeImages(effectiveScale)
            
            if 0 <= currentChunk < len(trackItem.positionTimeline):
                newY = trackItem.positionTimeline[currentChunk]
                
            imageId = self.memberImageIds[member]
            imageKey = trackItem.currentImageKey
            self.canvas.itemconfig(imageId, image=trackItem.sourceImages[imageKey])
            self.canvas.coords(imageId, 0, newY)
        
    def initializePositions(self):
        """Initializes the positions each member should be at for a specific chunk index"""
        n = len(self.chunks)
        # Run swap/animation logic only where it's safe to do so
        for currentChunk in range(n):
            for trackItem in self.memberImages.values():
                if currentChunk < len(self.chunks) - 4:
                    trackItem.checkAndSwap(currentChunk)
                    trackItem.updateAnimations(currentChunk)
                else:
                    base_idx = len(self.chunks) - 5
                    trackItem.positionTimeline[currentChunk] = trackItem.positionTimeline[base_idx]
        
        for trackItem in self.memberImages.values():
            trackItem.basePositionTimeline = trackItem.positionTimeline.copy()
        
    def initializeMemberImages(self):
        groupMembers = list(self.images.keys())
        numMembers = len(groupMembers)
        
        # Base window dimensions
        canvasHeight = self.baseHeight
        maxScale = 45
        
        # Estimate initial image height at max scale
        sampleImg = next(iter(self.images.values()))["dark"]
        imgBaseHeight = sampleImg.height
 
        # Step 1: calculate max scaled height per member
        canvasHeight = self.baseHeight - 10
        maxScaledHeightBase = int(imgBaseHeight * maxScale / 100)
        totalStackedHeight = numMembers * maxScaledHeightBase
        
        # Adjust scale if it exceeds canvas height
        if totalStackedHeight > canvasHeight:
            scale = (canvasHeight / (imgBaseHeight * numMembers)) * 100
            scale = min(scale, maxScale)
        else:
            scale = maxScale
            
        # Compute actual scaled image height and initial Y offset
        scaledPixelHeight = round(imgBaseHeight * scale / 100)
        self.slotHeightBase = round(imgBaseHeight * scale / 100)

        # Step 4: Compute where to start stacking so last member lands exactly at bottom
        self.memberImages = {}
        self.memberImageIds = {}
        memberTimes = []
        
        # --- build first trackItem just to lock in the true pixel height ---
        firstName = groupMembers[0]
        firstTrack = TrackItem(
            scale=scale,
            sourceImages={
                "dark": self.images[firstName]["dark"],
                "light": self.images[firstName]["light"],
            },
            animations=[],
            parent=self,
            trackMember=firstName,
        )
        firstTrack.initializeTimeline(self.includeBacking)
        firstTrack.resizeImages(scale)

        # ✅ authoritative step: whatever Tk is actually going to draw
        scaledPixelHeight = firstTrack.sourceImages["dark"].height()
        
        self.slotHeightBase = scaledPixelHeight

        # Now place first
        self.slotMap[firstName] = 0
        firstTrack.currentSlotIndex = 0
        imageId = self.canvas.create_image(0, 0, image=firstTrack.sourceImages["dark"], anchor="nw")
        firstTrack.setImageId(imageId)
        firstTrack.initializeProgressBar()
        self.memberImages[firstName] = firstTrack
        self.memberImageIds[firstName] = imageId
        memberTimes.append(firstTrack.timeline[len(self.chunks) - 1])
        
        for index, memberName in enumerate(groupMembers[1:], start=1):
            self.slotMap[memberName] = index

            trackItem = TrackItem(
                scale=scale,
                sourceImages={
                    "dark": self.images[memberName]["dark"],
                    "light": self.images[memberName]["light"],
                },
                animations=[],
                parent=self,
                trackMember=memberName,
            )
            trackItem.initializeTimeline(self.includeBacking)
            trackItem.resizeImages(scale)
            trackItem.currentSlotIndex = index

            # 🔒 enforce exact height match (optional assertion)
            h = trackItem.sourceImages["dark"].height()
            if h != scaledPixelHeight:
                print(f"WARNING: {memberName} resized height {h} != {scaledPixelHeight}")

            y = index * scaledPixelHeight
            imageId = self.canvas.create_image(0, y, image=trackItem.sourceImages["dark"], anchor="nw")
            trackItem.setImageId(imageId)
            trackItem.initializeProgressBar()

            self.memberImages[memberName] = trackItem
            self.memberImageIds[memberName] = imageId
            memberTimes.append(trackItem.timeline[len(self.chunks) - 1])

        for trackItem in self.memberImages.values():
            trackItem.setMaxTime(max(memberTimes))
    #end initializeMemberImages 
     
    def updateLabelMarkersDict(self):
        """
        Rebuild labelMarkers from self.labels (source of truth) + stray markers.
        Avoid double-adding stray markers that sit on committed label boundaries.
        """
        if hasattr(self, "uiHidden") and self.uiHidden:
            return
        
        self.labelMarkers = {}
        chunksInView = self.zoomManager.currentChunksInView

        def add(markerType, chunkIndex):
            sectionIndex = chunkIndex // chunksInView
            self.labelMarkers.setdefault(sectionIndex, []).append((markerType, chunkIndex))

        # 1) Committed labels + record their boundaries
        committed = set()  # (markerType, chunkIndex)
        for lab in self.labels:
            if len(lab) < 3:
                continue
            _, startChunk, endChunk = lab[:3]
            add("start", startChunk)
            add("end", endChunk)
            committed.add(("start", startChunk))
            committed.add(("end", endChunk))

        # 2) Pending / stray markers: only add if NOT already a committed boundary
        for chunkIndex in getattr(self, "startPoints", []):
            if ("start", chunkIndex) not in committed:
                add("start", chunkIndex)

        for chunkIndex in getattr(self, "endPoints", []):
            if ("end", chunkIndex) not in committed:
                add("end", chunkIndex)

        self.drawLabelMarkers(self.currentSectionIndex)
        
    def loadSavedLabels(self):
        """Load saved labels from a JSON file and update markers"""
        labelFilePath = f"./saved_labels/{self.selectedGroup}/{self.songName}_labels.json"
        
        if not os.path.exists(labelFilePath):
            print(f"No saved labels found at {labelFilePath}.") 
            return []
        
        # Load json file
        try:
            with open(labelFilePath, "r") as file:
                savedLabels = json.load(file)
                
                for label in savedLabels:
                    start = label[1]
                    end = label[2]
                    self.startPoints.append(start)
                    self.endPoints.append(end)
                    
                # Update startPoints, endPoints, and markers
                self.updateLabelMarkersDict()
                self.drawLabelMarkers(self.currentSectionIndex)
                
                self.canvas.update()
                self.root.update_idletasks()
                return savedLabels
        except Exception as e:
            print(f"Error loading labels from {labelFilePath}: {e}")
            return []
            
    # end loadSavedLabels        
    
    def progressBarValueToTime(self, value):
        """Convert progress bar value to actual song time"""
        visibleDuration = self.zoomManager.currentChunksInView * self.chunk_duration
        return (self.currentSectionIndex * visibleDuration) + (value * self.chunk_duration)
    
    def timeToProgressBarValue(self, timeMs):
        """Convert actual song time to progress bar value."""
        visibleDuration = self.zoomManager.currentChunksInView * self.chunk_duration
        self.currentSectionIndex = timeMs // visibleDuration
        localTimeMs = timeMs % visibleDuration
        return localTimeMs // self.chunk_duration
    
    # Works properly
    def updateProgressBarHandle(self, timeMs): 
        """Update the progress bar handle position based on the current time."""
        visibleDuration = self.zoomManager.currentChunksInView * self.chunk_duration
        totalDuration = len(self.chunks) * self.chunk_duration
        
        if visibleDuration <= 0 or totalDuration <= 0:
            return
            
        eps = 1e-6
        maxTime = max(0.0, totalDuration - eps)
        if timeMs >= maxTime:
            timeMs = maxTime
        elif timeMs < 0:
            timeMs = 0.0
            
        maxSectionIndex = int(maxTime // visibleDuration)
        newPlayheadSection = int(timeMs // visibleDuration)
        if newPlayheadSection > maxSectionIndex:
            newPlayheadSection = maxSectionIndex
        elif newPlayheadSection < 0:
            newPlayheadSection = 0
        
        oldPlayheadSection = self.progressBarHandle.currentSectionIndex
        viewSection = self.currentSectionIndex
        
        # update playhead section always
        self.progressBarHandle.currentSectionIndex = newPlayheadSection

        # "sticky follow": only advance the view if the user was following BEFORE the boundary change
        if newPlayheadSection != oldPlayheadSection:
            if viewSection == oldPlayheadSection:
                self.currentSectionIndex = newPlayheadSection
                self.drawLabelMarkers(self.currentSectionIndex)
                self.drawTimeMarkers()
            else:
                # user browsed away, do not jump view
                pass

        # compute x within the playhead section
        timeInSection = timeMs - (newPlayheadSection * visibleDuration)
        progressRatio = (timeInSection / visibleDuration) if visibleDuration > 0 else 0.0
        x = progressRatio * self.progressBarWidth
        
        if x < 0:
            x = 0.0
        elif x > self.progressBarWidth:
            x = float(self.progressBarWidth)

        # show/hide handle based on whether view matches playhead
        self.progressBarHandle.move(x, self.currentSectionIndex)
        self.previousX = x
    
    # Works properly
    def onDragHandle(self, event):
        """Handle dragging progress bar handle"""
        # Constrict x to bounds of progress bar
        x = max(0, min(event.x, self.progressBarWidth))
        self.progressBarHandle.jump(x, self.currentSectionIndex)
        
        pygame.mixer.music.pause()
        if hasattr(self, "videoTrackItem"):
            self.videoTrackItem.pause()
        
        visibleDuration = self.zoomManager.currentChunksInView * self.chunk_duration
        
        progressRatio = x / self.progressBarWidth
        #print(f"Current progressRatio: {progressRatio }")
        newTimeMs = int(visibleDuration * (self.currentSectionIndex + progressRatio))
        self.currentChunkIndex = min(
            int(newTimeMs / self.chunk_duration),
            len(self.chunks) - 1
        )       
        self.updateChunkIndexDisplay(self.currentChunkIndex)
        self.updateProgressBarHandle(newTimeMs)
        self.updateDisplayedTime(newTimeMs)
        
        if hasattr(self, "videoTrackItem"):
            self.videoTrackItem.seek(newTimeMs)
        
        self.isManualUpdate = True
        
    def updateProgressBar(self):
        """Redraw progress bar based on visible range"""
        playbackTime = self.playbackOffset + pygame.mixer.music.get_pos()
        self.updateLabelMarkersDict()
        self.updateProgressBarHandle(playbackTime)
                    
    def drawTimeMarkers(self):
        """Draw time markers for current section"""
        if hasattr(self, "uiHidden") and self.uiHidden:
            return  # Skip drawing if UI is hidden
        self.canvas.delete("time_marker")
        
        visibleDuration = self.zoomManager.currentChunksInView * self.chunk_duration
        # Determines start of current section
        startTimeMs = self.currentSectionIndex * visibleDuration
        
        progressBarX = self.progressBarCanvas.winfo_x()
        progressBarY = self.progressBarCanvas.winfo_y()
        progressBarWidth = self.progressBarWidth  # Use the updated width
        markerIntervalMs = visibleDuration // 10  # Interval in milliseconds

        for i in range(11):
            # Calculate x position and time
            x = progressBarX + (i / 10) * progressBarWidth
            timeMs = startTimeMs + (i * markerIntervalMs)
            minutes = timeMs // 60000
            seconds = (timeMs % 60000) // 1000
            milliseconds = round((timeMs % 1000) / 10)

            if milliseconds == 100:  # Handle overflow
                seconds += 1
                milliseconds -= 100
            if seconds == 60:  # Handle minute overflow
                minutes += 1
                seconds = 0

            # Draw the time marker line
            self.canvas.create_line(
                x, progressBarY - (50 * self.scaleY),
                x, progressBarY,
                fill="gray",
                tags="time_marker"
            )
            # Draw the timestamp
            timestamp = f"{seconds:02}:{milliseconds:02}" if minutes == 0 else f"{minutes:01}:{seconds:02}:{milliseconds:02}"
            self.canvas.create_text(
                x,
                progressBarY - (60 * self.scaleY),
                text=timestamp,
                fill="blue",
                font=("Arial", 8),
                tags="time_marker"
            )
            
    def getMemberColor(self, name, forLyrics=False):
        for member in self.members:
            if member['name'] == name:
                base = member["color"]
                if forLyrics:
                    return ensureReadableOnBackground(base, bgHex="#ffffff", minContrast=3.0)
                else:
                    return base
        return None
    # end getMemberColor
    
    def createLabelsFromPredictions(self, detectionResults):
        """
        Build self.labels from a list of per-chunk predictions (multi-head).

        detectionResults: list[dict]
        Each element corresponds to a 40 ms chunk and has:
            {
            "main":    List[str],  # member name or [] for silence
            "harmony": List[str],  # member names (can be empty)
            "adlib":   List[str],  # member names (can be empty)
            }

        Rules:
        - "" (blank main) is treated as silence.
        - For each member and each contiguous segment of activity, create:
            [member, startChunk, endChunk, isLead, isAdlib]
        where:
            isLead  = True  if segment is from the main head
            isAdlib = True  if segment is from the ad-lib head
            harmony = isLead == False and isAdlib == False
        - Avoid double-counting roles: if a member appears in multiple heads
        in the same chunk, we enforce a precedence:
            main > adlib > harmony
        so each member has at most one role per chunk.
        """
        def addNewLabel():
            if endChunk >= segStart:
                isLead = (segRole == "main")
                isAdlib = (segRole == "adlib")
                self.startPoints.append(segStart)
                self.endPoints.append(endChunk)
                labels.append([member, segStart, endChunk, isLead, isAdlib])
                
        if not detectionResults:
            return []
    
        # all known members from self.members
        memberNames = [m['name'] for m in self.members]
        numChunks = len(detectionResults)
        
        self.startPoints = []
        self.endPoints = []
        
        # For each member, we will store a per-chunk role:
        # "main", "harmony", "adlib", or "none"
        memberRoles = {member: ["none"] * numChunks for member in memberNames}
        
        # ---- 1) Fill memberRoles with per-chunk roles (no overlaps per member) ----
        for i, frame in enumerate(detectionResults):
            main_names = set(frame.get("main", []) or []) 
            harm_names = set(frame.get("harmony", []) or [])
            adlib_names = set(frame.get("adlib", []) or [])

            # Prevent cross-head duplicates:

            # If someone shows up in both harmony and adlib (shouldn't normally),
            # prefer adlib over harmony.
            both = harm_names & adlib_names
            if both:
                harm_names -= both  # keep them only in adlib_names

            for member in memberNames:
                role = "none"
                if member in main_names:
                    role = "main"
                elif member in adlib_names:
                    role = "adlib"
                elif member in harm_names:
                    role = "harmony"
                memberRoles[member][i] = role
        
        # ---- 2) Convert per-member roles into contiguous segments with flags ----
        labels = []    
        
        for member in memberNames:
            inSegment = False
            segStart = None
            segRole = None
            
            roles = memberRoles[member]
            
            for i, role in enumerate(roles):
                if role != "none":
                    if not inSegment:
                        # Start new segment
                        inSegment = True
                        segStart = i
                        segRole = role
                    elif role != segRole:
                        # Role changed -> close previous segment, start new one
                        endChunk = i - 1
                        if endChunk >= segStart:
                            isLead = (segRole == "main")
                            isAdlib = (segRole == "adlib")
                            self.startPoints.append(segStart)
                            self.endPoints.append(endChunk)
                            labels.append([member, segStart, endChunk, isLead, isAdlib])
                        
                        segStart = i
                        segRole = role
                else:
                    if inSegment:
                        # Close previous segment
                        endChunk = i - 1
                        addNewLabel()
                        inSegment = False
                        segStart = None
                        segRole = None
                            
        if inSegment and segStart is not None:
            endChunk = numChunks - 1
            addNewLabel()                
               
        # Sort labels by start chunk for consistency
        labels.sort(key=lambda lab: lab[1])
        
        return labels
    
    def onLabelsChanged(self, redrawSection=None):
        # 1) keep marker state sane
        self._syncPointsFromLabels()
        self._recomputeOpenStartChunk()

        # 2) make sure geometry is valid for x/y math
        self.root.update_idletasks()

        # 3) boundary markers
        self.updateLabelMarkersDict()  # ends by drawing current section in your code :contentReference[oaicite:5]{index=5}
        self.drawLabelMarkers(self.currentSectionIndex)
        
        # 4) label lanes
        if getattr(self, "labelLaneRenderer", None):
            sec = self.currentSectionIndex if redrawSection is None else redrawSection
            self.labelLaneRenderer.drawSection(sec, self.progressBarWidth)
        
    def showAddLabelsMenu(self, event=None):
        if not ModalGuard.try_open("labels_menu"):
            return  # another modal is open
        
        self.disableRootKeybinds()  
        self.videoTrackItem.setUiBusy(True)      
        # Create menu window
        labelMenu = tk.Toplevel(self.root)
        labelMenu.title("Add labels")
        labelMenu.geometry("700x600")
        labelMenu.transient(self.root)  # Make it a child of the root window
        labelMenu.grab_set()
        
        # Frame for checklist
        checklistFrame = tk.Frame(labelMenu)
        checklistFrame.pack(pady=0, fill="both", expand=True)
        
        canvas = tk.Canvas(checklistFrame)
        scrollFrame = tk.Frame(canvas)
        scrollbar = tk.Scrollbar(checklistFrame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.create_window((0, 0), window=scrollFrame, anchor="nw")
        
        def updateScrollRegion(_event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        scrollFrame.bind("<Configure>", updateScrollRegion)

        def onMouseWheel(e):
            # Scroll the menu list, not the main app zoom
            if not labelMenu.winfo_exists():
                return
            if e.delta != 0:
                canvas.yview_scroll(-1 * int(e.delta / 120), "units")
            return "break"

        def onLinuxWheel(e):
            if not labelMenu.winfo_exists():
                return
            if e.num == 4:
                canvas.yview_scroll(-1, "units")
            elif e.num == 5:
                canvas.yview_scroll(1, "units")
                
            return "break"
                
        canvas.bind("<MouseWheel>", onMouseWheel)
        canvas.bind("<Button-4>", onLinuxWheel)
        canvas.bind("<Button-5>", onLinuxWheel)

        # Helpful: make sure wheel events route here when mouse enters
        canvas.bind("<Enter>", lambda _e: canvas.focus_set())
        
        # Update scroll region
        def bindWheelToWidget(widget):
            widget.bind("<MouseWheel>", onMouseWheel)   # Windows/macOS
            widget.bind("<Button-4>", onLinuxWheel)     # Linux
            widget.bind("<Button-5>", onLinuxWheel)
        bindWheelToWidget(labelMenu)
        bindWheelToWidget(checklistFrame)
        bindWheelToWidget(canvas)
        bindWheelToWidget(scrollFrame)
        
        checkboxes = {}
        checkboxesByIndex = []  # Index-based access
        backingVars = {}
        adLibVars = {}
        
        rowToLabelIndex = [] 
        labelKeys = []  # index -> (memberFromGetLabels, start, end)
        
        # Keep references to row widgets so we update their UI immediately
        rowWidgets = {} # i -> {"labelCb": widget, "backCb": widget, "adCb": widget}
        # Mark deletions safely without shifting indices mid-session
        deletedLabelIndices = set()
        
        # Utility: find the labelIndex in self.labels for this (start,end) pair (ignores member)
        def findLabelIndexBySpan(startPoint, endPoint, member=None):
            for j, lab in enumerate(self.labels):
                if j in deletedLabelIndices:
                    continue
                if len(lab) >= 3 and lab[1] == startPoint and lab[2] == endPoint:
                    if member is None or lab[0] == member:
                        return j
            return None
        
        # Utility: after any delete that would shift indices, we re-scan and rebuild rowToLabelIndex
        def rebuildRowToLabelIndex():
            for i, (_m, s, e) in enumerate(labelKeys):
                rowToLabelIndex[i] = findLabelIndexBySpan(s, e, member=_m)
                
        # Utility: refresh the displayed text/color for a row
        def refreshRowUI(i):
            labelIndex = rowToLabelIndex[i]
            _oldMember, startPoint, endPoint = labelKeys[i]

            member = None
            isBacking = False
            isAdLib = False
            if labelIndex is not None and labelIndex not in deletedLabelIndices:
                lab = self.labels[labelIndex]
                while len(lab) < 5:
                    lab.append(False)
                member, _, _, isBacking, isAdLib = lab[0], lab[1], lab[2], lab[3], lab[4]

            memberText = f" -> {member}" if member else ""
            text = f"Start: {startPoint}, End: {endPoint}{memberText}"
            color = self.getMemberColor(member) if member else "black"

            rowWidgets[i]["labelCb"].configure(text=text, fg=color)

            backingVars[i].set(bool(isBacking))
            adLibVars[i].set(bool(isAdLib))
            
        # per-row edit dialog (double-click a row)
        def openEditDialog(i):
            labelIndex = rowToLabelIndex[i]
            _oldMember, startPoint, endPoint = labelKeys[i]
            
            currentMember = None
            currentBacking = False
            currentAdLib = False
            if labelIndex is not None and labelIndex not in deletedLabelIndices:
                lab = self.labels[labelIndex]
                while len(lab) < 5:
                    lab.append(False)
                currentMember = lab[0]
                currentBacking = bool(lab[3])
                currentAdLib = bool(lab[4])
            
            editWin = tk.Toplevel(labelMenu)
            editWin.title(f"Edit label ({startPoint}–{endPoint})")
            editWin.transient(labelMenu)
            editWin.grab_set()
            editWin.geometry("360x200")
            
            tk.Label(editWin, text=f"Start: {startPoint}   End: {endPoint}").pack(pady=8)

            memberMapping = {m['name']: m for m in self.members}
            memberMapping["Gang Vocal"] = {"name": "Gang Vocal", "id": "gang"}
            memberMapping["Cut"] = {"name": "Cut", "id": 'cut"'}
            memberNames = list(memberMapping.keys())
            
            memberVarRow = tk.StringVar(value=currentMember if currentMember in memberMapping else (memberNames[0] if memberNames else ""))
            ttk.Combobox(editWin, textvariable=memberVarRow, values=memberNames, state="readonly").pack(pady=5)

            backingVarRow = tk.BooleanVar(value=currentBacking)
            adLibVarRow = tk.BooleanVar(value=currentAdLib)

            tk.Checkbutton(editWin, text="Backing vocals", variable=backingVarRow).pack(pady=3)
            tk.Checkbutton(editWin, text="Ad Lib", variable=adLibVarRow).pack(pady=3)
        
            def applyEdit():
                nonlocal labelIndex
                chosenMember = memberVarRow.get()

                # Track oldMember if we’re editing an existing label
                oldMember = None
                # If this span already exists (even if member differs), update it instead of appending duplicates.
                if labelIndex is None or labelIndex in deletedLabelIndices:
                    labelIndex = findLabelIndexBySpan(startPoint, endPoint, member=chosenMember)

                if labelIndex is None:
                    newLabel = [chosenMember, startPoint, endPoint, backingVarRow.get(), adLibVarRow.get()]
                    self.labels.append(newLabel)
                    rowToLabelIndex[i] = len(self.labels) - 1
                else:
                    lab = self.labels[labelIndex]
                    while len(lab) < 5:
                        lab.append(False)
                    
                    if oldMember is None:
                        oldMember = lab[0]
                    lab[0] = chosenMember
                    lab[3] = backingVarRow.get()
                    lab[4] = adLibVarRow.get()
                    rowToLabelIndex[i] = labelIndex

                self.clipManager.rebuild(self.labels, len(self.chunks)) 
                
                # Refresh both old and new members’ timelines
                membersToUpdate = set()
                if oldMember and oldMember != chosenMember and oldMember not in self.bannedNames:
                    membersToUpdate.add(oldMember)
                if chosenMember and chosenMember not in self.bannedNames:
                    membersToUpdate.add(chosenMember)
                    
                for m in membersToUpdate:
                    trackItem = self.memberImages.get(m)
                    if trackItem:
                        trackItem.initializeTimeline(includeBacking=self.includeBacking)
                    
                refreshRowUI(i)
                self.onLabelsChanged(redrawSection=self.currentSectionIndex)
                self.saveLabels(self.selectedGroup, True) 
                editWin.destroy()

            def deleteLabel():
                nonlocal labelIndex
                if labelIndex is None:
                    # Nothing to delete; just clear row UI.
                    rowToLabelIndex[i] = None
                    refreshRowUI(i)
                    editWin.destroy()
                    return

                lab = self.labels[labelIndex]
                oldMember = lab[0] if lab and len(lab) >= 1 else None

                # Remove the label from the list
                del self.labels[labelIndex]
                
                rowToLabelIndex[i] = None
                self.clipManager.rebuild(self.labels, len(self.chunks)) 
                refreshRowUI(i)
                
                # Update that member's timeline
                if oldMember and oldMember not in self.bannedNames:
                    trackItem = self.memberImages.get(oldMember)
                    if trackItem:
                        trackItem.initializeTimeline(includeBacking=self.includeBacking)

                # Save new label set to JSON
                self.saveLabels(self.selectedGroup, True)
                self.onLabelsChanged(self.currentSectionIndex) 
                editWin.destroy()

            btnFrame = tk.Frame(editWin)
            btnFrame.pack(pady=10)

            tk.Button(btnFrame, text="Apply", command=applyEdit).pack(side="left", padx=6)
            tk.Button(btnFrame, text="Delete", command=deleteLabel).pack(side="left", padx=6)
            tk.Button(btnFrame, text="Cancel", command=editWin.destroy).pack(side="left", padx=6)

        # Build rows
        for i, (member, startPoint, endPoint, isBacking, isAdLib) in enumerate(self.getLabels()):
            var = tk.BooleanVar()
            backingVar = tk.BooleanVar(value=bool(isBacking))
            adLibVar = tk.BooleanVar(value=bool(isAdLib))
            checkboxesByIndex.append((var, backingVar, adLibVar))
            labelKeys.append((member, startPoint, endPoint))

            labelIndex = findLabelIndexBySpan(startPoint, endPoint, member=member)
            rowToLabelIndex.append(labelIndex)

            checkboxes[i] = var
            backingVars[i] = backingVar
            adLibVars[i] = adLibVar

            memberText = f" -> {member}" if member is not None else ""
            text = f"Start: {startPoint}, End: {endPoint}{memberText}"
            color = self.getMemberColor(member) if member else "black"

            labelCheckbox = tk.Checkbutton(
                scrollFrame, text=text, variable=var,
                anchor="w", bg="lightgray", fg=color, selectcolor="darkgrey"
            )
            labelCheckbox.grid(row=i, column=0, sticky="w", padx=5, pady=2)

            backingCheckbox = tk.Checkbutton(
                scrollFrame, text="Are they backing vocals?",
                variable=backingVar, anchor="w",
                bg="lightgray", fg="darkblue", selectcolor="darkgrey"
            )
            backingCheckbox.grid(row=i, column=1, padx=5, pady=2)

            adLibCheckBox = tk.Checkbutton(
                scrollFrame, text="Ad Lib",
                variable=adLibVar, anchor="w",
                bg="lightgray", fg="purple", selectcolor="darkgrey"
            )
            adLibCheckBox.grid(row=i, column=2, padx=5, pady=2)

            rowWidgets[i] = {"labelCb": labelCheckbox, "backCb": backingCheckbox, "adCb": adLibCheckBox}

            # Shift-click selection logic hooks
            labelCheckbox.bind("<Button-1>", lambda event, index=i: onCheckboxClick(event, index, "main"))
            backingCheckbox.bind("<Button-1>", lambda event, index=i: onCheckboxClick(event, index, "repeat"))
            adLibCheckBox.bind("<Button-1>", lambda event, index=i: onCheckboxClick(event, index, "adlib"))

            # Right click opens editor (much harder to do accidentally than single click)
            labelCheckbox.bind("<Button-3>", lambda event, index=i: (openEditDialog(index), "break"))

            if member:
                def createAddLyricsCallback(startPoint=startPoint, memberName=member):
                    if self.isExportingVideo:
                        return
                    return lambda: self.lyricsEditor.addLyricBox(
                        startChunk=max(0, startPoint - 11), 
                        memberName=memberName
                    )

                addLyricButton = tk.Button(scrollFrame, text="Add Lyrics",
                                        command=createAddLyricsCallback(startPoint, member),
                                        bg="lightblue")
                addLyricButton.grid(row=i, column=3, padx=5, pady=2)

        memberLabel = tk.Label(labelMenu, text="Choose Member:")
        memberLabel.pack(pady=5)

        memberMapping = {member['name']: member for member in self.members}
        memberMapping["Gang Vocal"] = {"name": "Gang Vocal", "id": "gang"}
        memberMapping["Cut"] = {"name": "Cut", "id": 'cut"'}
        memberNames = list(memberMapping.keys())
        memberVar = tk.StringVar(value=memberNames[0] if memberNames else "")
        memberDropdown = ttk.Combobox(labelMenu, textvariable=memberVar, values=memberNames, state="readonly")
        memberDropdown.pack(pady=5)

        # Track shift-click logic (your existing code; unchanged)
        lastClicked = {"main": -1, "repeat": -1, "adlib": -1}
        shiftRange = {"main": (-1, -1), "repeat": (-1, -1), "adlib": (-1, -1)}
        def onCheckboxClick(event, index, checkboxType):
            if event.state & 0x0001:  # Shift held
                if lastClicked[checkboxType] != -1:
                    start = min(lastClicked[checkboxType], index)
                    end = max(lastClicked[checkboxType], index)
                    for k in range(start, end + 1):
                        varMain, varRepeat, varAdLib = checkboxesByIndex[k]
                        if checkboxType == "main":
                            varMain.set(True)
                        elif checkboxType == "repeat":
                            varRepeat.set(True)
                        elif checkboxType == "adlib":
                            varAdLib.set(True)
                    shiftRange[checkboxType] = (start, end)
                return "break"
            else:
                s, e = shiftRange[checkboxType]
                if s != -1 and e != -1:
                    for k in range(s, e + 1):
                        varMain, varRepeat, varAdLib = checkboxesByIndex[k]
                        if checkboxType == "main":
                            varMain.set(False)
                        elif checkboxType == "repeat":
                            varRepeat.set(False)
                        elif checkboxType == "adlib":
                            varAdLib.set(False)
                    shiftRange[checkboxType] = (-1, -1)
                lastClicked[checkboxType] = index
        
        def saveSelectedLabels():
            # Apply deletes first (and avoid index chaos by rebuilding at the end)
            if deletedLabelIndices:
                self.labels = [lab for idx, lab in enumerate(self.labels) if idx not in deletedLabelIndices]
                deletedLabelIndices.clear()
                rebuildRowToLabelIndex()

            membersToUpdate = set()
            
            # Now handle "main checkbox" adds exactly like before, but never duplicate spans
            anyMain = any(var.get() for var in checkboxes.values())
            if anyMain:
                for i, var in checkboxes.items():
                    if not var.get():
                        continue

                    chosenMember = memberVar.get()
                    _, startPoint, endPoint = labelKeys[i]
                    isBacking = backingVars[i].get()
                    isAdLib = adLibVars[i].get()

                    # If span exists, update; else append
                    idx = findLabelIndexBySpan(startPoint, endPoint, member=chosenMember)
                    if idx is None:
                        self.labels.append([chosenMember, startPoint, endPoint, isBacking, isAdLib])
                        rowToLabelIndex[i] = len(self.labels) - 1
                    else:
                        lab = self.labels[idx]
                        while len(lab) < 5:
                            lab.append(False)

                        oldMember = lab[0]
                        # In the current logic oldMember == chosenMember,
                        # but we still handle the general case cleanly.
                        lab[0] = chosenMember
                        lab[3] = isBacking
                        lab[4] = isAdLib
                        rowToLabelIndex[i] = idx

                        if oldMember and oldMember != chosenMember and oldMember not in self.bannedNames:
                            membersToUpdate.add(oldMember)
                    self.clipManager.rebuild(self.labels, len(self.chunks)) 
                    refreshRowUI(i)

                    if chosenMember and chosenMember not in self.bannedNames:
                        membersToUpdate.add(chosenMember)
                        
                # Now refresh all affected members' timelines once
                for m in membersToUpdate:
                    trackItem = self.memberImages.get(m)
                    if trackItem:
                        trackItem.initializeTimeline(includeBacking=self.includeBacking)
                self.saveLabels(self.selectedGroup, True)
                
            else:
                # No main checkbox selected: only update backing/adlib flags for existing rows
                changed = 0
                for i in range(len(labelKeys)):
                    idx = rowToLabelIndex[i]
                    if idx is None:
                        continue
                    lab = self.labels[idx]
                    while len(lab) < 5:
                        lab.append(False)

                    oldB, oldA = lab[3], lab[4]
                    newB, newA = backingVars[i].get(), adLibVars[i].get()
                    lab[3], lab[4] = newB, newA

                    if (oldB, oldA) != (newB, newA):
                        changed += 1

                if changed > 0:
                    self.saveLabels(self.selectedGroup, True)

            self._recomputeOpenStartChunk()
            self.updateLabelMarkersDict()
            
            self.selectedMarker = None
            self.selectedLabel = None
            self.originalLabel = None
            closeMenu()
        
        def closeMenu():
            try:
                labelMenu.grab_release()
            except Exception:
                pass
            try:
                labelMenu.destroy()
                ModalGuard.close("labels_menu")
            except Exception:
                pass
            self.videoTrackItem.setUiBusy(False)
            self.enableRootKeybinds()
            
        buttonFrame = tk.Frame(labelMenu)
        buttonFrame.pack(pady=10)
        
        tk.Button(buttonFrame, text="Save Labels", command=saveSelectedLabels).pack(side="left", padx=5)
        tk.Button(buttonFrame, text="Close", command=closeMenu).pack(side="left", padx=5)
        labelMenu.protocol("WM_DELETE_WINDOW", closeMenu)
    
    def _getCircleImages(self, selectedMembers):
        circleImages = [
            self.images[member]["circle"]
            for member in selectedMembers
            if member in self.images and "circle" in self.images[member]
        ]
        
        return circleImages

    def rebuildLyricsAnimations(self):
        # Clear runtime structures
        self.startEvents = {}
        self.activeLyricIds = set()

        # Clear all lyric animations/cursors
        for lb in self.lyrics.values():
            lb.animations = []
            if hasattr(lb, "resetAnimCursor"):
                lb.resetAnimCursor()

        # Rebuild startEvents from lyrics dict
        # startEvents[chunk] = [startChunkId, ...]
        startEvents = {}
        for startChunk in self.lyrics.keys():
            startEvents.setdefault(startChunk, []).append(startChunk)
        self.startEvents = startEvents

        # Now rebuild the stacking/push-down logic in chronological order.
        # We simulate “building” from earliest to latest.
        sortedIds = sorted(self.lyrics.keys())
        self._buildActiveLyricIds = []  # ordered list of ids already placed

        def getActiveLyricBoxesAtChunk_builder(_chunk):
            # during rebuild, “active” means “already inserted before this lyric”
            return [
                self.lyrics[lid]
                for lid in self._buildActiveLyricIds
                if lid in self.lyrics and not getattr(self.lyrics[lid], "isAdLib", False)
            ]

        # Temporarily expose the builder method expected by LyricBox.initializeLyricPosition
        self.getActiveLyricBoxesAtChunk = getActiveLyricBoxesAtChunk_builder

        for lid in sortedIds:
            lb = self.lyrics[lid]
            
            if not lb.isAdLib:
                lb.initializeLyricPosition()
                self._buildActiveLyricIds.append(lid)
            else:
                lb.rebuildAdLibAnimation()
    
    def disableRootKeybinds(self):
        """Temporarily unbind all root-level keybindings while user is typing."""
        self.canvas.unbind("<Button-1>")
        self.canvas.unbind("<KeyPress-a>")
        self.canvas.unbind("<KeyPress-d>")
        self.canvas.unbind("<KeyPress-s>")
        self.canvas.unbind("<KeyPress-q>")
        self.canvas.unbind("<KeyPress-w>")
        self.canvas.unbind("<KeyPress-p>")
        self.canvas.unbind("<KeyPress-l>")
        self.canvas.unbind("<KeyPress-x>")
        self.canvas.unbind("<KeyPress-v>")
        self.canvas.unbind("<Return>")
        self.canvas.unbind("<Control-z>")
        self.canvas.unbind("<Control-y>")
        self.canvas.unbind("<Control-r>")
        self.canvas.unbind("<Control-g>")
        self.root.unbind_all("<space>")
        self.zoomManager.disableScrollZoom(self.root)

    def enableRootKeybinds(self):
        """Rebind all root-level keybindings after the lyric window is closed."""
        
        # Drag / release for markers on the main canvas
        self.canvas.bind("<ButtonPress-1>", self.onMarkerClick)
        self.canvas.bind("<B1-Motion>", self.onMarkerMotion)
        self.canvas.bind("<ButtonRelease-1>", self.onMarkerRelease)
        
        self.canvas.bind("<KeyPress-a>", self.moveMarkerLeft)
        self.canvas.bind("<KeyPress-d>", self.moveMarkerRight)
        self.canvas.bind("<Return>", self.updateLabelInJSON)
        self.canvas.bind("<KeyPress-e>", self.showAddLabelsMenu)
        self.canvas.bind("<KeyPress-q>", self.addStartPoint)
        self.canvas.bind("<KeyPress-w>", self.addEndPoint)
        self.canvas.bind("<KeyPress-p>", self.togglePanelVisibility)
        self.canvas.bind("<KeyPress-l>", self.lyricsEditor.openLyricsEditorMenu)
        self.root.bind_all("<space>", self.togglePlayPause)
        self.root.bind("<KeyPress-v>", self.toggleAudioMode)
        
        self.canvas.bind("<Shift-L>", lambda e: self.toggleLeadBacking("lead", e))
        self.canvas.bind("<Shift-B>", lambda e: self.toggleLeadBacking("back", e))
        self.canvas.bind("<Shift-P>", self.togglePanMode)
        
        # split current label at current chunk
        self.canvas.bind("<KeyPress-x>", self.handleSplitGapKey)
        self.canvas.bind("<Escape>", self.cancelSplitGap)
        self.canvas.bind("<Control-r>", lambda event: self.countBacking(switch=False))
        self.root.bind("<Control-g>", self.startExportVideo)
        
        # Undo / Redo
        self.canvas.bind("<Control-z>", self.undo)
        self.canvas.bind("<Control-y>", self.redo)
        self.zoomManager.enableScrollZoom(self.root)
     
    def togglePanelVisibility(self, event=None):
        # visible should be the opposite of hidden
        shouldShow = not self.lineDistPanel._placed
        self.setPanelVisibility(shouldShow)

        if shouldShow:
            self.timeDisplayLabel.place_forget()
        else:
            self.timeDisplayLabel.place(relx=0.5, rely=0.95, anchor="center")


    def setPanelVisibility(self, isVisible: bool):
        if isVisible:
            self.lineDistPanel.show()   # or self.lineDistPanel.show() if you want spin
        else:
            self.lineDistPanel.hide()
        
    def loadLyricsFromFile(self):
        """Loads lyrics from a JSON file and adds them to self.lyrics."""
        lyricsFilePath = f"./saved_labels/{self.selectedGroup}/{self.songName}_lyrics.json"
        
        if not os.path.exists(lyricsFilePath):
            print(f"Lyrics file not found: {lyricsFilePath}")
            return
        
        try:
            with codecs.open(lyricsFilePath, "r", encoding="utf-8", errors="ignore") as file:
                lyricsData = json.load(file)
        except json.JSONDecodeError:
            print(f"Error loading JSON file: {lyricsFilePath}")
            return

        for lyric in lyricsData:
            language = lyric["language"]
            startChunk = lyric["startChunk"]
            memberName = lyric["memberName"]
            koreanLyric = lyric.get("korean", "")
            romanization = lyric.get("romanization", "")
            englishTrans = lyric.get("english", "")
            isAdLib = lyric.get("isAdLib", False)
            adLibDuration= lyric.get("adLibDuration", 50)
            
            circleImages = self._getCircleImages(memberName)
            
            lyricBox = LyricBox(canvas=self.canvas, parent=self, memberNames=memberName, circleImages=circleImages, koreanLyric=koreanLyric, romanization=romanization, englishTrans=englishTrans, startChunk=startChunk, language=language, isAdLib=isAdLib, adLibDuration=adLibDuration)
            self.lyrics[startChunk] = lyricBox
    
    def resetLyricsToChunkStart(self, startChunk: int):
        startChunk = max(0, int(startChunk))

        # Parent trackers
        self.activeLyricIds.clear()
        self.lastChunkSeen = startChunk - 1   # safe even if startChunk==0 => -1

        # Reset every lyric's internal cursor so getBaseYAt() works from the start
        for lb in self.lyrics.values():
            lb.resetAnimCursor()              # <-- THIS is the missing reset :contentReference[oaicite:1]{index=1}
            lb.hide()  
            
    def hideAllLyrics(self, lyricsSurpressed=True):
        """Hides all lyric box objects stored in self.lyrics."""
        self.lyricsSuppressed = lyricsSurpressed
        for _, lyricBox in self.lyrics.items():
            lyricBox.hide()
    
    def unsuppressLyricsAndRefresh(self):
        self.lyricsSuppressed = False
        # reset state so activation works
        if hasattr(self, "activeLyricIds"):
            self.activeLyricIds.clear()
        self.lastChunkSeen = self.currentChunkIndex - 1
        self.renderLyrics(self.currentChunkIndex)
           
    def getActiveLyricBoxesAtChunk(self, chunkIndex):
        boxes = []
        for sc, lb in self.lyrics.items():
            if sc == chunkIndex:
                # this is usually the new lyric starting now; skip
                continue
            baseY = lb.getBaseYAt(chunkIndex)
            if baseY is not None:
                boxes.append(lb)
        return boxes  
            
    def renderLyrics(self, chunkIndex):
        if self.lyricsSuppressed:
            # Keep them hidden even if renderLyrics is being called every tick
            for lb in self.lyrics.values():
                lb.hide()
            self.lyricsLayerDirty = False
            return
        
        # Activate any new lyrics starting now
        last = getattr(self, "lastChunkSeen", -1)
        if chunkIndex > last:
            for c in range(last + 1, chunkIndex + 1):
                for lid in self.startEvents.get(c, []):
                    self.activeLyricIds.add(lid)
            self.lastChunkSeen = chunkIndex
        elif chunkIndex < last:
            # Seeking backwards / restart: safest is rebuild state
            self.lastChunkSeen = chunkIndex
            # optional: full recompute if you allow backwards seeking during export
            # self.rebuildActiveLyrics(chunkIndex)

        scaleY = getattr(self, "scaleY", 1.0)
        offY = getattr(self, "viewportOffsetY", 0)
        canvasH = self.canvas.winfo_height()

        toRemove = []

        for lid in list(self.activeLyricIds):
            lb = self.lyrics.get(lid)
            if not lb:
                toRemove.append(lid)
                continue

            baseY = lb.getBaseYAt(chunkIndex)
            if baseY is None:
                # not started yet or no timeline -> hide but keep active if you want
                lb.hide()
                continue

            canvasY = offY + baseY * scaleY
            top = canvasY
            if getattr(lb, "isAdLib", False):
                # store this in createAdLibDisplay(): lb.adLibTextHeightBase
                heightBase = getattr(lb, "adLibTextHeightBase", None)
                if heightBase is None:
                    # fallback: approximate using current canvas height / scale
                    heightCanvas = getattr(lb, "totalHeight", 0)
                    heightBase = (heightCanvas / scaleY) if scaleY else heightCanvas

                bottom = offY + (baseY + heightBase) * scaleY

                # Hide once bottom is above top of frame
                if (baseY + heightBase) < 0:
                    lb.hide()
                    toRemove.append(lid)  # stop tracking once it’s gone
                    continue

            else:
                # Normal lyrics: keep your existing behavior
                bottom = canvasY + lb.totalHeight

            onScreen = (bottom > 0) and (top < canvasH)

            if onScreen:
                lb.show()
                if getattr(lb, "isAdLib", False):
                    lb.setAdLibPosition(baseY)
                else:
                    lb.setPosition(baseY)
            else:
                lb.hide()
                # If it's below screen for good, you can stop tracking it
                if top >= canvasH and not getattr(self, "isExportingVideo", False):
                    toRemove.append(lid)
        for lid in toRemove:
            self.activeLyricIds.discard(lid)
        
        self.lyricsLayerDirty = False
        self.enforceCanvasLayering()
        
    def updateCanvasForCurrentPosition(self, chunkIndex):
        """Highlight the corresponding member's image if their voice matches the current time."""
        safeChunkIndex = min(max(chunkIndex, 0), len(self.chunks) - 1)
        
        if self.testOrVideo == "Video":
            membersCurrentlySinging = {}  # member -> (isBacking, isAdlib)

            for label in self.labels:
                member, start, end, isBacking, isAdlib = label
                if start <= safeChunkIndex <= end:
                    membersCurrentlySinging[member] = (isBacking, isAdlib)
                    
            # Update canvas for each member
            for member, trackItem in self.memberImages.items():
                imageId = self.memberImageIds[member]
                trackItem.updateAndDrawTimer(safeChunkIndex)
                if safeChunkIndex > trackItem.lastUpdateChunk:
                    trackItem.switchImage("clear")
                    trackItem.currentRole = "none"
                elif member in membersCurrentlySinging:
                    isBacking, isAdlib = membersCurrentlySinging[member]
                    
                    if not isBacking and not isAdlib:
                        trackItem.currentRole = "main"
                    elif isAdlib:
                        trackItem.currentRole = "adlib"
                    else:
                        trackItem.currentRole = "harmony"
                        
                    trackItem.switchImage("light")
                else:
                    trackItem.switchImage("dark")
                    trackItem.currentRole = "none"
                    
                self.canvas.itemconfig(imageId, image=trackItem.sourceImages[trackItem.currentImageKey]) 
            
            self.renderLyrics(safeChunkIndex)
        else:
            # voiceDetectionResults[chunkIndex] is now a dict with heads
            if self.vocalPresence is not None:
                firstMember, trackItem = list(self.memberImages.items())[0]
                imageId = self.memberImageIds[firstMember]
                if self.vocalPresence[safeChunkIndex] == 0:
                    trackItem.switchImage("light")
                    trackItem.currentRole = "main"
                else:
                    trackItem.switchImage("dark")
                    trackItem.currentRole = "none"
                    
                self.canvas.itemconfig(imageId, image=trackItem.sourceImages[trackItem.currentImageKey])
            
        self.canvas.update_idletasks()
    # end
    
    def updateMemberTestingCanvas(self, chunkIndex):
        frameRoles = self.voiceDetectionResults[chunkIndex] if self.voiceDetectionResults else {}
            
            # main is a single name or ""
        main_names = frameRoles.get("main", []) or []
        # harmony/adlib are lists
        harmony_names = frameRoles.get("harmony", []) or []
        adlib_names = frameRoles.get("adlib", []) or []

        # ✅ Update each member's image based on detection
        for member, trackItem in self.memberImages.items():
            imageId = self.memberImageIds[member]
            trackItem.updateAndDrawTimer(chunkIndex)
            
            if member in main_names:
                role = "main"
            elif member in harmony_names:
                role = "harmony"
            elif member in adlib_names:
                role = "adlib"
            else:
                role = "none"
                
            trackItem.currentRole = role
            
            if role == "none":
                trackItem.switchImage("dark")
            else:
                trackItem.switchImage("light")

            self.canvas.itemconfig(imageId, image=trackItem.sourceImages[trackItem.currentImageKey])
    
    def _restartAudioAtTime(self, newTimeMs):
        try: 
            if not self.isPaused and pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.music.play(start=newTimeMs / 1000.0)
                self.playbackOffset = newTimeMs  # keep video sync source correct
                self.isPlaying = True
                self.isManualUpdate = False  # you're done seeking
        except:
            return
    
    def onProgressBarClick(self, event):
        self.isManualUpdate = True
    
    # Fix conflict with onDragHandle
    def onProgressBarRelease(self, event):
        """Unset manual update flag when user stops interacting with the progress bar."""
        x = max(0, min(event.x, self.progressBarWidth))  # Constrain x within the canvas bounds
        self.progressBarHandle.jump(x, self.currentSectionIndex)
        # Calculate the new start chunk based on the handle's x position
        visibleDuration = self.zoomManager.currentChunksInView * self.chunk_duration
        progressRatio = x / self.progressBarWidth
        newTimeMs = int(visibleDuration * (self.currentSectionIndex + progressRatio))
        
        # Updates the chunk index
        newChunkIndex = min(
            int(newTimeMs // self.chunk_duration),
            len(self.chunks) - 1
        )
        self.seekToChunk(newChunkIndex)
        self.updateChunkIndexDisplay(self.currentChunkIndex)
        # print(f"Released at {newTimeMs}")
        
        self.updateCurrentTime(newTimeMs)
        # self.updateProgressBarHandle(newTimeMs)
        # Restart playback at the new position => This is normal
        self._restartAudioAtTime(newTimeMs)
        if hasattr(self, "videoTrackItem"):
            self.videoTrackItem.seek(newTimeMs)
            
        if not self.isPaused:
            self.playWithSavedResults(newTimeMs) # Annoying issue
        # Sync music playback with the new chunk index
      
    def resetLyricsToChunk(self, chunkIndex: int):
        # 1) Hide everything immediately (prevents the “stacking on top” visual garbage)
        for lb in self.lyrics.values():
            lb.hide()

        # 2) Reset active set
        self.activeLyricIds = set()

        # 3) Reset each lyric’s animation cursor so getBaseYAt works when time moves backwards
        for lb in self.lyrics.values():
            lb.resetAnimCursor()

        # 4) Rebuild which lyrics should be “active” at this chunk.
        # Rule of thumb: anything with startChunk <= chunkIndex is eligible to be active.
        # (renderLyrics will still hide it if it’s off-screen.)
        for startChunk in self.lyrics.keys():
            if startChunk <= chunkIndex:
                self.activeLyricIds.add(startChunk)
      
    def seekToChunk(self, newChunkIndex):
        # reset per-lyric cursors so getBaseYAt works from scratch at this chunk
        self.currentChunkIndex = newChunkIndex

        self.resetLyricsToChunk(self.currentChunkIndex)

        # render immediately at the new chunk
        self.renderLyrics(self.currentChunkIndex)
        
    def moveBackwardByChunks(self, event):
        """Move backward by five frames."""
        currentTime = int(time.time() * 1000)
        if currentTime - self.lastKeyPressTime < 250: return
        
        self.lastKeyPressTime = currentTime
        
        newPlaybackTime = max(0, self.playbackOffset - 5000)
        
        self.currentChunkIndex = int(newPlaybackTime / self.chunk_duration)
        self.playbackOffset = newPlaybackTime
        print(f"Moved backward to chunk index: {self.currentChunkIndex}, Playback time: {newPlaybackTime}ms")
        self.updateProgressBarHandle(newPlaybackTime)
        self.updateProgressBar(newPlaybackTime)
        self.updateCanvasForCurrentPosition()
        
        if self.isPaused:
            return
        
        # Update playback position
        pygame.mixer.music.stop()
        pygame.mixer.music.play(start=newPlaybackTime / 1000)

    def moveForwardByChunks(self, event):
        """Move forward by five chunks."""
        currentTime = int(time.time() * 1000)
        if currentTime - self.lastKeyPressTime < 250: return
        
        self.lastKeyPressTime = currentTime
        
        newPlaybackTime = min(self.totalDurationMs, self.playbackOffset + 5000)
        
        # Calculate the playback time
        self.currentChunkIndex = int(newPlaybackTime / self.chunk_duration)
        self.playbackOffset = newPlaybackTime
        print(f"Moved forward to chunk index: {self.currentChunkIndex}, Playback time: {newPlaybackTime}ms")

        # Update UI
        self.updateProgressBar(newPlaybackTime)
        self.updateProgressBarHandle(newPlaybackTime)
        self.updateCanvasForCurrentPosition()
        
        if self.isPaused: return
        
        # Update playback position
        pygame.mixer.music.stop()
        pygame.mixer.music.play(start=newPlaybackTime / 1000)
    
    def getLabels(self):
        matchedPoints = []

        sortedStartPoints = sorted(self.startPoints)
        sortedEndPoints = sorted(self.endPoints)

        startCounts = Counter(sortedStartPoints)
        endCounts = Counter(sortedEndPoints)

        # Pairs that exist in real labels (regardless of member)
        realPairs = set()

        # Step 1) Add all real labels (allow same (start,end) with different members)
        for label in self.labels:
            member, labelStart, labelEnd = label[:3]
            isBacking = label[3] if len(label) > 3 else False
            isAdLib = label[4] if len(label) > 4 else False

            matchedPoints.append((member, labelStart, labelEnd, isBacking, isAdLib))
            realPairs.add((labelStart, labelEnd))

            # Consume one start/end marker occurrence per label instance
            if startCounts[labelStart] > 0:
                startCounts[labelStart] -= 1
            if endCounts[labelEnd] > 0:
                endCounts[labelEnd] -= 1

        remainingStarts = sorted([s for s, c in startCounts.items() for _ in range(c)])
        remainingEnds = sorted([e for e, c in endCounts.items() for _ in range(c)])

        # Step 2) Pair leftover markers into placeholder labels (None member)
        endIdx = 0
        for startPoint in remainingStarts:
            while endIdx < len(remainingEnds) and remainingEnds[endIdx] <= startPoint:
                endIdx += 1
            if endIdx >= len(remainingEnds):
                break

            endPoint = remainingEnds[endIdx]
            endIdx += 1

            # IMPORTANT: only prevent placeholders that duplicate an existing real pair
            if (startPoint, endPoint) in realPairs:
                continue

            matchedPoints.append((None, startPoint, endPoint, False, False))
            # no need to track placeholder pairs unless you can generate same placeholder twice

        # Stable sort:
        # - by start
        # - then end
        # - then real labels before placeholders
        matchedPoints.sort(key=lambda x: (x[1], x[2], 1 if x[0] is None else 0))
        return matchedPoints

    # Helper function to check if chunk is in any area
    def isInStartOrEnd(self):
        if self.currentChunkIndex in self.startPoints or self.currentChunkIndex in self.endPoints:
            return True
        
        return False
    # End isInStartOrEnd
    
    def togglePlayPause(self, event=None):
        if self.isPlaying and not self.isPaused:
            self.pause()
        else:
            self.play()
    
    def updateDisplayedTime(self, timeMs):
        """Update time display based on milliseconds"""
        minutes = timeMs // 60000
        seconds = (timeMs % 60000) // 1000
        milliseconds = timeMs % 1000
        self.timeDisplayVar.set(f"{minutes:02}:{seconds:02}.{milliseconds:03}")
    
    def _syncPointsFromLabels(self):
        self.startPoints = []
        self.endPoints = []
        for lab in self.labels:
            if len(lab) < 3:
                continue
            self.startPoints.append(lab[1])
            self.endPoints.append(lab[2])

        # you just committed, so no “open” marker should remain
        self.openStartChunk = None
    
    def _recomputeOpenStartChunk(self):
        """
        Recompute openStartChunk from current marker lists.
        This keeps the gate sane after deletes/undo/reset/etc.
        """
        if len(self.startPoints) > len(self.endPoints):
            # treat the most recently added start as the open one
            self.openStartChunk = self.startPoints[-1] if self.startPoints else None
        else:
            self.openStartChunk = None

    def addStartPoint(self, event=None):
        if self.openStartChunk is not None or len(self.startPoints) > len(self.endPoints):
            self.showStatus(
                "You already have an open start point. Add an end point first.",
                level="warn"
            )
            return

        self.pushUndoState("add start marker")
        self.startPoints.append(self.currentChunkIndex)
        self.openStartChunk = self.currentChunkIndex
        self.addMarkerToSection(self.currentChunkIndex, "start")

        self.showStatus(
            f"Start point set at chunk {self.currentChunkIndex}.",
            level="info"
        )

    def addEndPoint(self, event=None):
        if self.openStartChunk is None and len(self.startPoints) <= len(self.endPoints):
            self.showStatus(
                "You need to add a start point before adding an end point.",
                level="warn"
            )
            return

        if self.openStartChunk == self.currentChunkIndex:
            self.showStatus(
                "End point cannot be the same chunk as the start point.",
                level="error"
            )
            return

        self.pushUndoState("add end marker")
        self.endPoints.append(self.currentChunkIndex)
        self.addMarkerToSection(self.currentChunkIndex, "end")
        self.openStartChunk = None

        self.showStatus(
            f"End point set at chunk {self.currentChunkIndex}.",
            level="info"
        )
    
    def clearAllMarkers(self):
        """
        Clear all start and end markers from the canvas and reset marker dictionaries.
        """
        self.labelOverlay.hide()
        self.canvas.delete("marker")

        # Reset state
        self.startPointMarkers.clear()
        self.endPointMarkers.clear()
        
    def drawLabelMarkers(self, sectionIndex):
        """
        Draw start/end markers for the current section.

        If multiple markers share the same chunkIndex (e.g., start + end at the
        same point, or multiple labels with same boundary), we "stack" them
        vertically so they don't hide each other.

        Up to 3 markers are stacked per chunkIndex.
        """
        self.clearAllMarkers()
        if hasattr(self, "uiHidden") and self.uiHidden:
            return
        
        if hasattr(self, "labelLaneRenderer") and self.labelLaneRenderer:
            self.labelLaneRenderer.drawSection(sectionIndex, self.progressBarWidth)

        if sectionIndex not in self.labelMarkers:
            return

        # Group markers by chunkIndex for this section
        markersByChunk = {}
        for markerType, chunkIndex in self.labelMarkers[sectionIndex]:
            markersByChunk.setdefault(chunkIndex, []).append(markerType)

        chunksInView = self.zoomManager.currentChunksInView

        selectedId = self.selectedMarker.get("id") if self.selectedMarker else None

        for chunkIndex, typeList in markersByChunk.items():
            relativeX = (
                self.progressBarCanvas.winfo_x()
                + (chunkIndex % chunksInView / chunksInView) * self.progressBarWidth
            )
            x = self.canvas.canvasx(relativeX)
            baseY = self.progressBarCanvas.winfo_y()

            if x < 0 or x > self.canvas.winfo_width():
                continue

            maxStack = 3
            for stackIndex, markerType in enumerate(typeList[:maxStack]):
                stackOffset = stackIndex * 20
                yTop = baseY - 20 - stackOffset
                yBottom = baseY - stackOffset

                # default color first
                defaultColor = "green" if markerType == "start" else "red"

                markerId = self.canvas.create_line(
                    x, yTop,
                    x, yBottom,
                    fill=defaultColor,
                    width=4,
                    tags=("marker", "start_marker" if markerType == "start" else "end_marker")
                )
                self.canvas.addtag_withtag("ui", markerId)

                # record in multiset dict
                if markerType == "start":
                    ids = self._getMarkerIdsAtChunk(self.startPointMarkers, chunkIndex)
                    ids.append(markerId)
                    self._setMarkerIdsAtChunk(self.startPointMarkers, chunkIndex, ids)
                    self.labelOverlay.bindBoundaryMarker(markerId, chunkIndex, "start")
                else:
                    ids = self._getMarkerIdsAtChunk(self.endPointMarkers, chunkIndex)
                    ids.append(markerId)
                    self._setMarkerIdsAtChunk(self.endPointMarkers, chunkIndex, ids)
                    self.labelOverlay.bindBoundaryMarker(markerId, chunkIndex, "end")

                # apply highlight ONLY if this exact id is selected
                if selectedId is not None and markerId == selectedId:
                    self.canvas.itemconfig(markerId, fill=("turquoise" if markerType == "start" else "pink"))
                    # keep chunk/type consistent after redraw
                    self.selectedMarker["chunkIndex"] = chunkIndex
                    self.selectedMarker["type"] = markerType
    # end drawLabelMarkers  
    
    def updateCurrentTime(self, newTimeMs):
        """Update the current time based on the progress bar value."""
        if not self.isManualUpdate: return
        
        self.playbackOffset = newTimeMs
        
        self.skipNextAutoUpdate = True
        self.updateDisplayedTime(newTimeMs)
        self.updateProgressBarHandle(newTimeMs)
        self.updateCanvasForCurrentPosition(int(newTimeMs / self.chunk_duration))
    # end updateCurrentTime
    
    def showSplitGapMarker(self, chunkIndex):
        """
        Draw a temporary purple marker at the given chunk index to indicate
        the start of a split-with-gap.
        """
        # Remove any previous temp marker
        if self.splitGapMarkerId is not None:
            self.canvas.delete(self.splitGapMarkerId)
            self.splitGapMarkerId = None
        
        chunksInView = self.zoomManager.currentChunksInView
        sectionIndex = self.progressBarHandle.currentSectionIndex
        
        # Compute x similar to drawMarkers
        relativeX = (
            self.progressBarCanvas.winfo_x()
            + (chunkIndex % chunksInView / chunksInView) * self.progressBarWidth
        )
        x = self.canvas.canvasx(relativeX)
        y = self.progressBarCanvas.winfo_y()
        
        self.splitGapMarkerId = self.canvas.create_line(
            x, y - 20,
            x, y,
            fill="#c080ff",  # soft purple
            width=3,
            dash=(3, 2)
        )
    
    def clearSplitGapMarker(self):
        """
        Remove the temporary split-gap marker if it exists.
        """
        if self.splitGapMarkerId is not None:
            self.canvas.delete(self.splitGapMarkerId)
            self.splitGapMarkerId = None
    
    def handleSplitGapKey(self, event=None):
        """
        Press X once to mark the start of a gap (breath),
        press X again at the end of the gap to perform the split.
        Esc cancels.
        """
        if not self.splitGapActive:
            self.splitGapActive = True
            self.splitGapStartChunk = self.currentChunkIndex
            self.showSplitGapMarker(self.splitGapStartChunk)
            print(f"[SplitGap] Start set at chunk {self.splitGapStartChunk}. Move to end and press X again, or Esc to cancel.")
            return
        
        # Second X: finish and apply split
        gapStart = self.splitGapStartChunk
        gapEnd = self.currentChunkIndex
        
        # Clear visual marker regardless of outcome
        self.clearSplitGapMarker() 
        
        # Reset mode right away (even if nothing happens)
        self.splitGapActive = False
        self.splitGapStartChunk = None
        
        if gapStart == gapEnd:
            print("[SplitGap] Start and end are the same chunk; nothing to split.")
            return
        
        # Normalize order (user may drag backwards)
        if gapEnd < gapStart:
            gapStart, gapEnd = gapEnd, gapStart
        
        self.performSplitWithGap(gapStart, gapEnd)
        
    def cancelSplitGap(self, event=None):
        """
        Cancel split-gap mode when Esc is pressed.
        """
        if self.splitGapActive:
            print(f"[SplitGap] Canceled (start at chunk {self.splitGapStartChunk} discarded).")
        
        self.splitGapActive = False
        self.splitGapStartChunk = None
        self.clearSplitGapMarker()
        
    def chooseLabelForGap(self, candidates):
        """
        Ask the user which label to split if multiple overlap the gap.
        `candidates` is a list of (index, label) pairs.
        Returns the chosen index in self.labels, or None if canceled.
        """
        if not candidates:
            return None
        
        # Build readable prompt
        lines = []
        for i, (idx, label) in enumerate(candidates):
            member, start, end = label[:3]
            lines.append(f"{i}: {member} [{start} - {end}]")
        
        prompt = "Multiple labels overlap this gap.\n\n" + "\n".join(lines) + \
             "\n\nEnter the number of the label to split (or Cancel):"

        choice = simpledialog.askinteger(
            "Choose label to split",
            prompt,
            minvalue=0,
            maxvalue=len(candidates) - 1
        )
        
        if choice is None:
            print("[SplitGap] User canceled label selection.")
            return None

        chosenIdx = candidates[choice][0]
        return chosenIdx
    
    def performSplitWithGap(self, gapStart, gapEnd):
        """
        Given a gap [gapStart, gapEnd] (inclusive in chunks),
        split one label that fully covers this segment into two labels,
        leaving the gap unlabeled.

        Example:
        label: [Yujin, 0, 100, ...]
        gap:   40..50

        -> [Yujin, 0, 39, ...] and [Yujin, 51, 100, ...]

        (We subtract 1 and add 1 so the gap itself has no label.)
        """
        # Find labels that fully cover the gap
        candidates = []
        for idx, label in enumerate(self.labels):
            if len(label) < 3:
                continue
            member, start, end = label[:3]
            if start <= gapStart and end >= gapEnd:
                candidates.append((idx, label))
                
        if not candidates:
            print(f"[SplitGap] No label fully covers gap [{gapStart}, {gapEnd}]. Nothing to split.")
            return
        
        if len(candidates) == 1:
            targetIndex = candidates[0][0]
        else:
            # Ask user which overlapping vocal to modify
            targetIndex = self.chooseLabelForGap(candidates)
            if targetIndex is None:
                return
        
        original = self.labels[targetIndex]
        member, start, end, isBacking, isAdLib = original
        
        # Compute new ranges, leaving a gap
        leftStart = start
        leftEnd = gapStart # last chunk before the breath
        rightStart = gapEnd  # first chunk after the breath
        rightEnd = end
        
        newLabels = []
        
        for idx, label in enumerate(self.labels):
            if idx != targetIndex:
                newLabels.append(label)
            else:
                # Replace this label with up to two shorter labels
                if leftEnd >= leftStart:
                    newLabels.append([member, leftStart, leftEnd, isBacking, isAdLib])
                if rightEnd >= rightStart:
                    newLabels.append([member, rightStart, rightEnd, isBacking, isAdLib])
            
        newLabels.sort(key=lambda lab: lab[1])

        print(f"[SplitGap] Split label {original} into:")
        for lab in newLabels:
            if lab[0] == member and (lab[1] >= start and lab[2] <= end):
                print("   ", lab)

        # Take snapshot BEFORE we overwrite labels
        self.pushUndoState(f"split gap {gapStart}-{gapEnd}")

        # Use the unified helper
        self.applyLabelsState(newLabels)
        
    def applyNewLabelsState(self, newLabels):
        """
        Replace self.labels with newLabels, rebuild start/end points,
        refresh markers and write labels to JSON.
        """
        self.labels = newLabels

        # Rebuild startPoints and endPoints
        self.onLabelsChanged(redrawSection=self.currentSectionIndex) 

        # Update marker structures and redraw
        self.updateLabelMarkersDict()   # this also calls drawMarkers(...)
        self.canvas.update()
        self.root.update_idletasks()

        # Save to the same labels JSON you already use=
        labelFilePath = f"./saved_labels/{self.selectedGroup}/{self.songName}_labels.json"
        try:
            with open(labelFilePath, "w") as f:
                json.dump(self.labels, f, separators=(",", ":"))
            print(f"[SplitGap] Labels saved to {labelFilePath}.")
        except Exception as e:
            print(f"[SplitGap] Error saving labels: {e}")
    
    def jumpToMs(self, targetMs):
        if targetMs is None:
            self.pause()
            return
        
        self.playbackOffset = targetMs

        try:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.currentAudioPath)
            pygame.mixer.music.play(start=targetMs / 1000.0)
        except Exception as e:
            print("Jump failed:", e)
            return

        if hasattr(self, "videoTrackItem") and self.videoTrackItem:
            self.videoTrackItem.seek(targetMs)
            
        self.syncVisualsToTime(targetMs)
            
    def play(self):
        # Play from saved detection results
        if not self.isPlaying and self.currentChunkIndex >= len(self.chunks) - 1:
            self.playbackOffset = 0
            self.currentChunkIndex = 0
            self.updateProgressBarHandle(0)
            self.updateDisplayedTime(0)

        if self.playbackOffset < 0:
            self.playbackOffset = 0
        
        if self.isPlaying:
            if self.isPaused:
                pygame.mixer.music.unpause()
                # print(f"Play Playback time: {playbackTime}\n Current chunk: {self.currentChunkIndex}")
                self.isPaused = False
                
                if hasattr(self, "videoTrackItem"):
                    self.videoTrackItem.play()
                self.playWithSavedResults(self.currentChunkIndex * self.chunk_duration)
            return
        else:  
            if hasattr(self, "videoTrackItem"):
                self.videoTrackItem.seek(self.playbackOffset)
                self.videoTrackItem.play()
            self.playWithSavedResults(self.playbackOffset)
        
    def pause(self):
        if self.isPlaying and not self.isPaused:
            self.isPaused = True
            #self.playbackOffset = self.currentChunkIndex * self.chunk_duration
            pygame.mixer.music.pause()
            if hasattr(self, "videoTrackItem"):
                self.videoTrackItem.pause()
     
    def countBacking(self, switch=True):
        """
        Toggle whether backing-only labels should contribute to timelines,
        then rebuild member images + timelines + position timelines.
        """
        if not self.isPlaying:
            return
        if switch:
            self.includeBacking = not getattr(self, "includeBacking", True)
        
        # Clean up  exisitng member UI
        for trackItem in self.memberImages.values():
            # Delete member image
            self.canvas.delete(trackItem.imageId)
            self.canvas.delete(trackItem.timerTextId)
            self.canvas.delete(trackItem.progressBarCanvasImage)
        
        self.memberImages = {}
        self.memberImageIds = {}
        self.slotMap = {}
        
        # Recreate TrackItems and their widgets (images + progress bars)
        self.startLayout()
        
    def forward(self):
        """Skip forward by one second (one chunk)."""
        self.currentChunkIndex = min(len(self.chunks) - 1, self.currentChunkIndex + 1)
        self.updateCanvasForCurrentPosition()
    
    def syncVisualsToTime(self, timeMs: int):
        """
        Force UI state (chunk index, section index, progress bar, labels)
        to reflect a given playback time, without touching audio/video playback.
        """
        if self.isExportingVideo:
            return
        timeMs = max(0, min(timeMs, (len(self.chunks) - 1) * self.chunk_duration))

        newChunkIndex = min(
            int(timeMs // self.chunk_duration),
            len(self.chunks) - 1
        )
        self.seekToChunk(newChunkIndex)

        # These are the SAME calls updateChunk() normally does
        self.updateChunkIndexDisplay(self.currentChunkIndex)
        self.updateProgressBarHandle(timeMs)
        self.updateDisplayedTime(timeMs)
        self.updateCanvasForCurrentPosition(self.currentChunkIndex)
    
    def playWithSavedResults(self, startTimeMs):
        """Replay the audio with saved detection results synced to the audio."""
        if self.isPaused and not self.isManualUpdate: 
            return
        
        if not self.isPlaying or self.isManualUpdate:
            try:
                if not self.isPlaying:
                    pygame.mixer.music.load(self.currentAudioPath)
                pygame.mixer.music.play(start=startTimeMs / 1000)
            except pygame.error as e:
                self.showStatus(f"Error loading audio file: {e}")
                self.isPlaying = False
                return

            self.playbackOffset = startTimeMs
            self.currentChunkIndex = min(int(startTimeMs / self.chunk_duration), len(self.chunks) - 1)
            self.isPlaying = True
            self.isPaused = False
            self.isManualUpdate = False
        
        def updateChunk():
            if not self.isPlaying or self.isManualUpdate: 
                return
        
            # Get current playback position in milliseconds
            playbackPos = pygame.mixer.music.get_pos()
            if playbackPos == -1:
                #print("Playback not started or stopped unexpectedly.")
                self.isPlaying = False
                return 
            
            playbackTime = self.playbackOffset + playbackPos
            self.currentChunkIndex = min(
                int(playbackTime / self.chunk_duration),
                len(self.chunks) - 1
            )
            
            # ✅ CLIP SKIP HERE (playback-only)
            if hasattr(self, "clipManager") and self.clipManager.enabled:
                # Use maybeSkipNext for Skip cut
                jumped = self.clipManager.maybeSkipNext(self.currentChunkIndex)
                if jumped:
                    self.root.after(self.chunk_duration, updateChunk)
                    return
                
            self.syncVisualsToTime(playbackTime)
            
            # Update UI for voice detection
            if len(self.detectionResults) > 0:
                for member, trackItem in self.memberImages.items():
                    isVoiceDetected = self.detectionResults[self.currentChunkIndex].get(member, False)
                    if isVoiceDetected:
                        trackItem.currentImageKey = "light"
                    else:
                        trackItem.currentImageKey = "dark"
                    
                    # Update the canvas with the current image
                    imageId = self.memberImageIds[member]
                    self.canvas.itemconfig(imageId, image=trackItem.sourceImages[trackItem.currentImageKey])
                
            if self.currentChunkIndex >= len(self.chunks):
                self.pause()

            # Schedule the next chunk update
            self.root.after(self.chunk_duration, updateChunk)

        # Start updating chunks
        updateChunk()
    # end playWIthSavedResults
    
    def getPlaybackTimeMs(self):
        if self.isPlaying and not self.isPaused and pygame.mixer.get_init():
            pos = pygame.mixer.music.get_pos()
            if pos < 0: pos = 0
            return self.playbackOffset + pos
        return self.currentChunkIndex * self.chunk_duration
    
    def _submixKey(self, leadOn: bool, backOn: bool, panMode: str) -> str:
        # include source paths in key so changing files changes cache
        leadPath = self.vocalsLeadPath or ""
        backPath = self.vocalsBackingPath or ""
        base = f"{leadPath}|{backPath}|lead={leadOn}|back={backOn}|pan={panMode}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()
    
    def buildVocalsSubmixPath(self, leadOn: bool, backOn: bool, panMode: str = "mono",
                          cacheDir: str = "cache_audio", targetSr: int = 22050) -> str:
        os.makedirs(cacheDir, exist_ok=True)

        if not leadOn and not backOn:
            return self.vocalsOnlyPath

        key = self._submixKey(leadOn, backOn, panMode)
        # include effective sr in filename so cache stays correct
        # we'll compute it after loading stems
        # outPath = ...

        lead = AudioSegment.from_file(self.vocalsLeadPath) if leadOn else None
        back = AudioSegment.from_file(self.vocalsBackingPath) if backOn else None

        # Decide an effective SR that won't downsample below the weakest stem
        stemSrs = [a.frame_rate for a in (lead, back) if a is not None]
        minStemSr = min(stemSrs) if stemSrs else targetSr
        effectiveSr = max(targetSr, minStemSr)

        outPath = os.path.join(cacheDir, f"submix_{key}_sr{effectiveSr}.mp3")
        if os.path.exists(outPath):
            return outPath

        def norm(a):
            # only resample if needed
            if a.frame_rate != effectiveSr:
                a = a.set_frame_rate(effectiveSr)
            return a.set_sample_width(2)

        if lead: lead = norm(lead)
        if back: back = norm(back)

        maxLen = max(len(x) for x in [lead, back] if x is not None)
        if lead and len(lead) < maxLen:
            lead += AudioSegment.silent(duration=maxLen - len(lead), frame_rate=effectiveSr)
        if back and len(back) < maxLen:
            back += AudioSegment.silent(duration=maxLen - len(back), frame_rate=effectiveSr)

        if panMode == "split":
            mix = AudioSegment.silent(duration=maxLen, frame_rate=effectiveSr).set_channels(2)
            if lead:
                mix = mix.overlay(lead.set_channels(2).pan(-1.0))
            if back:
                mix = mix.overlay(back.set_channels(2).pan(+1.0))
        else:
            if lead and back: 
                mix = lead.overlay(back)
            else:
                mix = lead if lead else back
            mix = mix.set_channels(1)

        mix.export(outPath, format="mp3")
        return outPath

    def switchAudioPathPreserveTime(self, newPath: str, playbackTimeMs: int):
        self.currentAudioPath = newPath
        if self.isPlaying and not self.isPaused:
            pygame.mixer.music.stop()
            pygame.mixer.music.load(self.currentAudioPath)
            pygame.mixer.music.play(start=playbackTimeMs / 1000.0)
            self.playbackOffset = playbackTimeMs

        # UI sync
        self.currentChunkIndex = min(int(playbackTimeMs / self.chunk_duration), len(self.chunks) - 1)
        self.updateChunkIndexDisplay(self.currentChunkIndex)
        self.updateProgressBarHandle(playbackTimeMs)
        self.updateDisplayedTime(playbackTimeMs)
        self.updateCanvasForCurrentPosition(self.currentChunkIndex)
        
    def toggleLeadBacking(self, which: str, event=None):
        """
        which = "lead" or "back"
        Only works when audioMode == "vocals".
        """
        if self.audioMode != "vocals":
            self.showStatus("ℹ️ Lead/back toggles only work in VOCALS mode (press V first).")
            return

        if not self.vocalsLeadPath or not os.path.exists(self.vocalsLeadPath):
            self.showStatus("⚠️ Missing lead-only vocals file; cannot toggle lead/back.")
            return
        if not self.vocalsBackingPath or not os.path.exists(self.vocalsBackingPath):
            self.showStatus("⚠️ Missing backing-only vocals file; cannot toggle lead/back.")
            return

        playbackTime = self.getPlaybackTimeMs()

        if which == "lead":
            self.leadEnabled = not self.leadEnabled
        elif which == "back":
            self.backEnabled = not self.backEnabled
        else:
            return

        # Don't allow both off
        if not self.leadEnabled and not self.backEnabled:
            # if user toggled one off, keep the other on
            if which == "lead":
                self.backEnabled = True
            else:
                self.leadEnabled = True

        submixPath = self.buildVocalsSubmixPath(
            leadOn=self.leadEnabled,
            backOn=self.backEnabled,
            panMode=self.panMode
        )

        self.switchAudioPathPreserveTime(submixPath, playbackTime)

        # Debug message
        mode = []
        if self.leadEnabled: mode.append("LEAD")
        if self.backEnabled: mode.append("BACK")
        self.showStatus(f"🎛️ Vocals submix: {'+'.join(mode)} | pan={self.panMode}")
    
    def togglePanMode(self, event=None):
        if self.audioMode != "vocals":
            self.showStatus("ℹ️ Pan mode only works in VOCALS mode.")
            return
        self.panMode = "split" if self.panMode == "mono" else "mono"
        self.showStatus(f"🎧 Pan mode: {self.panMode}")

        # Rebuild current submix immediately
        playbackTime = self.getPlaybackTimeMs()
        submixPath = self.buildVocalsSubmixPath(self.leadEnabled, self.backEnabled, self.panMode)
        self.switchAudioPathPreserveTime(submixPath, playbackTime)
    
    def toggleAudioMode(self, event=None):
        """
        Toggle between full mix (self.testSongPath) and vocals-only (self.vocalsOnlyPath)
        while preserving playback position. Bound to 'V'.
        """
        # If we don't have a vocals-only file, just bail
        if not os.path.exists(self.vocalsOnlyPath):
            self.showStatus("⚠️ Vocals-only file not found; cannot toggle audio mode.")
            return
        
        # Figure out where we are in the song (in ms)
        if self.isPlaying and not self.isPaused and pygame.mixer.get_init():
            pos = pygame.mixer.music.get_pos()
            if pos < 0:
                pos = 0
            playbackTime = self.playbackOffset + pos
        else:
            # Fallback: use current chunk index
            playbackTime = self.currentChunkIndex * self.chunk_duration

        # Toggle mode + path
        if self.currentAudioPath == self.testSongPath:
            self.currentAudioPath = self.vocalsOnlyPath
            self.audioMode = "vocals"
            self.showStatus("🔊 Switched to *vocals-only* audio.")
        else:
            self.currentAudioPath = self.testSongPath
            self.audioMode = "mix"
            self.showStatus("🎵 Switched to *full mix* audio.")

        # If we're currently playing (and not paused), restart playback on the new source
        if self.isPlaying and not self.isPaused:
            try:
                pygame.mixer.music.stop()
                pygame.mixer.music.load(self.currentAudioPath)
                pygame.mixer.music.play(start=playbackTime / 1000.0)

                # Keep our offset consistent with this new start
                self.playbackOffset = playbackTime
            except pygame.error as e:
                self.showStatus(f"Error switching audio source: {e}")
                return

            # Keep UI in sync
            self.currentChunkIndex = min(
                int(playbackTime / self.chunk_duration),
                len(self.chunks) - 1
            )
            self.updateChunkIndexDisplay(self.currentChunkIndex)
            self.updateProgressBarHandle(playbackTime)
            self.updateDisplayedTime(playbackTime)
            self.updateCanvasForCurrentPosition(self.currentChunkIndex)
    
    def _defaultMarkerColor(self, markerType: str) -> str:
        return "green" if markerType == "start" else "red"

    def _selectedMarkerColor(self, markerType: str) -> str:
        return "turquoise" if markerType == "start" else "pink"
    
    def setSelectedMarker(self, chunkIndex, markerType, markerId=None):
        # 1) Unhighlight old selection
        if self.selectedMarker and self.selectedMarker.get("id") is not None:
            oldId = self.selectedMarker["id"]
            oldType = self.selectedMarker["type"]
            try:
                self.canvas.itemconfig(oldId, fill=self._defaultMarkerColor(oldType))
            except tk.TclError:
                # marker might have been deleted/redrawn
                pass

        # 2) Resolve markerId if caller didn't provide it
        if markerId is None:
            if markerType == "start":
                ids = self._getMarkerIdsAtChunk(self.startPointMarkers, chunkIndex)
            else:
                ids = self._getMarkerIdsAtChunk(self.endPointMarkers, chunkIndex)

            markerId = ids[-1] if ids else None

        if markerId is None:
            self.showStatus(f"[WARN] setSelectedMarker: markerId not found for {markerType}@{chunkIndex}")
            self.selectedMarker = None
            return

        # 3) Highlight new selection + store it
        self.canvas.itemconfig(markerId, fill=self._selectedMarkerColor(markerType))
        self.selectedMarker = {"chunkIndex": chunkIndex, "type": markerType, "id": markerId}
        self.canvas.bind("<Delete>", self.deleteSelectedMarker)
        self.canvas.bind("<BackSpace>", self.deleteSelectedMarker)
    
    def addMarkerToSection(self, chunkIndex, markerType):
        """
        Add a single marker to the appropriate sectionIndex key in timeMarkers and update marker dictionaries.
        """
        sectionIndex = chunkIndex // self.zoomManager.currentChunksInView
        if sectionIndex not in self.labelMarkers:
            self.labelMarkers[sectionIndex] = []
            
        self.labelMarkers[sectionIndex].append((markerType, chunkIndex))
        markerId = None
        
        x = self.progressBarCanvas.winfo_x() + (
            (chunkIndex % self.zoomManager.currentChunksInView)
            / self.zoomManager.currentChunksInView
        ) * self.progressBarWidth
        yTop = self.progressBarCanvas.winfo_y() - 20
        yBottom = self.progressBarCanvas.winfo_y()

        if markerType == "start":
            markerId = self.canvas.create_line(
                x, yTop, x, yBottom,
                fill=self._defaultMarkerColor(markerType),
                width=4,
                tags=("marker", "start_marker")
            )
            self.labelOverlay.bindBoundaryMarker(markerId, chunkIndex, "start")

            ids = self._getMarkerIdsAtChunk(self.startPointMarkers, chunkIndex)
            ids.append(markerId)
            self._setMarkerIdsAtChunk(self.startPointMarkers, chunkIndex, ids)

        elif markerType == "end":
            markerId = self.canvas.create_line(
                x, yTop, x, yBottom,
                fill=self._defaultMarkerColor(markerType),
                width=4,
                tags=("marker", "end_marker")
            )
            self.labelOverlay.bindBoundaryMarker(markerId, chunkIndex, "end")

            ids = self._getMarkerIdsAtChunk(self.endPointMarkers, chunkIndex)
            ids.append(markerId)
            self._setMarkerIdsAtChunk(self.endPointMarkers, chunkIndex, ids)
         
        self.selectedLabel = None
        self.setSelectedMarker(chunkIndex, markerType, markerId=markerId)
        self.restackMarkersAtChunk(chunkIndex)
    
    def upsertLabel(self, updatedLabel, originalLabel=None):    
        updatedLabel = normalizeLabel(updatedLabel)
        updatedK = labelKey(updatedLabel)

        # If the edit changed the start chunk, you must delete the old key
        if originalLabel is not None:
            originalLabel = normalizeLabel(originalLabel)
            originalK = labelKey(originalLabel)
            if originalK != updatedK:
                self.labels = [
                    lbl for lbl in self.labels
                    if labelKey(normalizeLabel(lbl)) != originalK
                ]

        # Now overwrite by updated key
        out = []
        replaced = False
        for lbl in self.labels:
            nl = normalizeLabel(lbl)
            if labelKey(nl) == updatedK:
                out.append(updatedLabel)
                replaced = True
            else:
                out.append(nl)

        if not replaced:
            out.append(updatedLabel)

        self.labels = sorted(out, key=lambda l: (l[1], l[2], l[0]))
    
    def saveLabels(self, selectedGroup: str, clearExisting: bool = False) -> None: 
        # Get file name without extension        
        labelFilePath = f"./saved_labels/{selectedGroup}/{self.songName}_labels.json"
        directory = os.path.dirname(labelFilePath)
        os.makedirs(directory, exist_ok=True)

        existing = []
        if (not clearExisting) and os.path.exists(labelFilePath):
            with open(labelFilePath, "r") as f:
                existing = json.load(f)

        # normalize + dedupe by identity, existing first then overwrite with current session labels
        byKey = {}
        for lbl in existing:
            nl = normalizeLabel(lbl)
            byKey[labelKey(nl)] = nl

        for lbl in self.labels:
            nl = normalizeLabel(lbl)
            byKey[labelKey(nl)] = nl

        combined = list(byKey.values())
        combined.sort(key=lambda l: (l[1], l[2], l[0]))

        self.clipManager.rebuild(self.labels, len(self.chunks)) 
        with open(labelFilePath, "w") as f:
            json.dump(combined, f, separators=(",", ":"))
            
        if hasattr(self, "labelLaneRenderer") and self.labelLaneRenderer:
            self.labelLaneRenderer.drawSection(self.currentSectionIndex, self.progressBarWidth)