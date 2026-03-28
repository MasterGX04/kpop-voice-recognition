import cv2, wave
import os, sys
import time
import tkinter as tk
from PIL import Image, ImageTk
import threading
from TrackItem import TrackItem
import numpy as np
import subprocess
from bisect import bisect_right
import pygame
import queue
from video_record import captureWindowClientRGBA

def resourcePath(*parts: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)

writerError = {"err": None}

class VideoTrackItem(TrackItem):
    def __init__(self, canvas, parent, videoPath, scale=100, scaleX=1.0, baseHeight=720, isMusicVideo=True):
        super().__init__(scale, sourceImages={}, animations=[], type="video")
        self.canvas = canvas
            
        self.videoPath = videoPath
        self.parent = parent
        self.cap = cv2.VideoCapture(self.videoPath)
        self.cap_lock = threading.Lock()
        self.scale = scale
        self.scaleX = scaleX
        self.videoFrameId = None
        self.isPlaying = False
        self.isPaused = False
        self.thread = None
        self.baseHeight = baseHeight
        self.currentFrame = None
        
        self.frameQueue = queue.Queue(maxsize=2) # keep only freshest frames
        self.renderAfterId = None
        self.decodeStop = threading.Event()
        self.lastSeekFrameIndex = 0
        
        # Get video dimensions
        if self.cap.isOpened():
            self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            raise FileNotFoundError(f"Video file not found: {videoPath}")
        
        self.cropEnabled = False
        self.cropInsets = (0, 0, 0, 0)  # (top, bottom, left, right) in source pixels
        self.contentWidth = self.frameWidth
        self.contentHeight = self.frameHeight
        
        # Auto-detect once
        self.autoDetectAndSetCrop()
        
        self.adjustScale(baseHeight)
        self.position = self.setPosition()
        # Sets video fps
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            self.effective_fps = 30
        else:
            self.effective_fps = min(fps, 30) # cap to 30
        
        self.tkImg = None          # persistent ImageTk.PhotoImage
        self._lastPos = None       # avoid coords spam
        self._renderNextT = None   # drift-correct scheduling

        self.isMusicVideo = isMusicVideo
        self.uiBusy = False
        self.lastImg = None   # keep a stable reference (don’t rely on canvas.image)
        self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.isOpened() else 0
        self.videoDurationMs = 0
        if self.totalFrames > 0 and self.effective_fps > 0:
            self.videoDurationMs = int((self.totalFrames / float(self.effective_fps)) * 1000)
        
    def adjustScale(self, currentHeight):
        """Adjust the video dimensions and scale based on the current height."""
        self.scale = (currentHeight / self.baseHeight) * 100
        self.newHeight = int(currentHeight)

        # Use active aspect (cropped or full)
        aw = getattr(self, "_activeAspectW", self.frameWidth)
        ah = getattr(self, "_activeAspectH", self.frameHeight)
        if ah <= 0:
            ah = self.frameHeight

        self.newWidth = int(self.newHeight * (aw / ah))
        print(f"New height: {self.newHeight}, New width: {self.newWidth}")
    
    def _estimateCropInsetsFromFrame(
        self,
        frameBgr: np.ndarray,
        blackLumaThreshold: int = 18,
        rowBlackFrac: float = 0.985,
        colBlackFrac: float = 0.985,
        maxCropFrac: float = 0.40,
        minContentHeight: int = 120,
        minContentWidth: int = 160,
    ):
        """
        Returns (top, bottom, left, right) insets for detected black bars.
        bottom/right are "insets" (not coordinates).
        """
        if frameBgr is None or frameBgr.size == 0:
            return (0, 0, 0, 0)

        h, w = frameBgr.shape[:2]

        # Luma approx from BGR
        # Y ~= 0.114B + 0.587G + 0.299R
        b = frameBgr[:, :, 0].astype(np.float32)
        g = frameBgr[:, :, 1].astype(np.float32)
        r = frameBgr[:, :, 2].astype(np.float32)
        y = (0.114 * b + 0.587 * g + 0.299 * r).astype(np.uint8)

        black = (y <= blackLumaThreshold)

        # Row/col black ratios
        rowFrac = black.mean(axis=1)  # (h,)
        colFrac = black.mean(axis=0)  # (w,)

        # Find top inset: first row that is NOT "mostly black"
        top = 0
        while top < h and rowFrac[top] >= rowBlackFrac:
            top += 1

        # Find bottom inset similarly from bottom
        bottomCoord = h - 1
        while bottomCoord >= 0 and rowFrac[bottomCoord] >= rowBlackFrac:
            bottomCoord -= 1
        bottomInset = (h - 1) - bottomCoord  # number of rows to drop from bottom

        # Left/right (pillarbox) detection
        left = 0
        while left < w and colFrac[left] >= colBlackFrac:
            left += 1

        rightCoord = w - 1
        while rightCoord >= 0 and colFrac[rightCoord] >= colBlackFrac:
            rightCoord -= 1
        rightInset = (w - 1) - rightCoord

        # Clamp crazy crops (fade to black, scene cuts, etc.)
        maxTop = int(h * maxCropFrac)
        maxBottom = int(h * maxCropFrac)
        maxLeft = int(w * maxCropFrac)
        maxRight = int(w * maxCropFrac)

        top = min(top, maxTop)
        bottomInset = min(bottomInset, maxBottom)
        left = min(left, maxLeft)
        rightInset = min(rightInset, maxRight)

        # Ensure some content remains
        contentH = h - top - bottomInset
        contentW = w - left - rightInset
        if contentH < minContentHeight or contentW < minContentWidth:
            return (0, 0, 0, 0)

        return (top, bottomInset, left, rightInset)
    
    def _applyCropInsets(self, frameBgr: np.ndarray, insets):
        top, bottomInset, left, rightInset = insets
        if not frameBgr is None:
            h, w = frameBgr.shape[:2]
            y0 = max(0, top)
            y1 = max(y0 + 1, h - bottomInset)
            x0 = max(0, left)
            x1 = max(x0 + 1, w - rightInset)
            return frameBgr[y0:y1, x0:x1]
        return frameBgr
    
    def autoDetectAndSetCrop(self, sampleCount: int = 5):
        """
        Samples a few frames across the video, estimates crop insets per frame,
        and takes the median to stabilize against fades/cuts.
        """
        if not self.cap or not self.cap.isOpened():
            self.cropInsets = (0, 0, 0, 0)
            self.contentWidth = self.frameWidth
            self.contentHeight = self.frameHeight
            return

        totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if totalFrames <= 0:
            # Try first frame only
            with self.cap_lock:
                cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) or 0
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur)
            if not ret:
                return
            insets = self._estimateCropInsetsFromFrame(frame)
            self._setCropInsets(insets)
            return

        # Pick sample frame indices (avoid absolute beginning to dodge fade-in black)
        picks = []
        for i in range(sampleCount):
            t = (i + 1) / (sampleCount + 2)  # stays away from 0 and 1
            picks.append(int(t * (totalFrames - 1)))

        insetsList = []
        with self.cap_lock:
            cur = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) or 0
            for idx in picks:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    insetsList.append(self._estimateCropInsetsFromFrame(frame))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, cur)

        if not insetsList:
            self._setCropInsets((0, 0, 0, 0))
            return

        arr = np.array(insetsList, dtype=np.int32)
        med = np.median(arr, axis=0).astype(int)
        self._setCropInsets(tuple(int(x) for x in med))

    def _getNormalizedCropInsets(self):
        """
        Return crop insets as ratios relative to the preview video's source size.
        """
        if self.frameWidth <= 0 or self.frameHeight <= 0:
            return (0.0, 0.0, 0.0, 0.0)

        top, bottomInset, left, rightInset = self.cropInsets
        return (
            top / self.frameHeight,
            bottomInset / self.frameHeight,
            left / self.frameWidth,
            rightInset / self.frameWidth,
        )

    def _getScaledCropInsetsForFrame(self, frameBgr: np.ndarray):
        """
        Convert normalized crop insets to pixel insets for the current frame size.
        This is what you should use for HQ export frames.
        """
        if frameBgr is None or frameBgr.size == 0:
            return (0, 0, 0, 0)

        h, w = frameBgr.shape[:2]
        topR, bottomR, leftR, rightR = self._getNormalizedCropInsets()

        top = int(round(topR * h))
        bottomInset = int(round(bottomR * h))
        left = int(round(leftR * w))
        rightInset = int(round(rightR * w))

        # Clamp just in case
        top = max(0, min(top, h - 1))
        bottomInset = max(0, min(bottomInset, h - 1))
        left = max(0, min(left, w - 1))
        rightInset = max(0, min(rightInset, w - 1))

        # Prevent collapsing the frame
        if h - top - bottomInset < 2 or w - left - rightInset < 2:
            return (0, 0, 0, 0)

        return (top, bottomInset, left, rightInset)
    
    def _setCropInsets(self, insets):
        self.cropInsets = insets
        top, bottomInset, left, rightInset = insets
        self.contentWidth = max(1, self.frameWidth - left - rightInset)
        self.contentHeight = max(1, self.frameHeight - top - bottomInset)
        
    def toggleCrop(self, enabled: bool = None, currentHeight: int = None):
        """
        Toggle between normal (full frame) and cropped (black bars removed).
        Pass enabled=None to flip.
        currentHeight: if None, uses current self.newHeight (keeps perceived size stable).
        """
        if enabled is None:
            enabled = not self.cropEnabled
        self.cropEnabled = bool(enabled)

        # Update derived aspect based on content vs full frame
        if self.cropEnabled:
            cw, ch = self.contentWidth, self.contentHeight
            if cw <= 1 or ch <= 1:
                # If crop detection failed, don't enable
                self.cropEnabled = False
                cw, ch = self.frameWidth, self.frameHeight
        else:
            cw, ch = self.frameWidth, self.frameHeight

        # Recompute width/height mapping while keeping the chosen display height
        if currentHeight is None:
            currentHeight = getattr(self, "newHeight", self.baseHeight)

        self._activeAspectW = cw
        self._activeAspectH = ch
        self.resize(currentHeight)
        
    def detectCropInsetsForVideoPath(self, videoPath: str, sampleCount: int = 5):
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            return self.cropInsets

        try:
            totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            if totalFrames <= 0:
                ret, frame = cap.read()
                if not ret or frame is None:
                    return self.cropInsets
                return self._estimateCropInsetsFromFrame(frame)

            picks = []
            for i in range(sampleCount):
                t = (i + 1) / (sampleCount + 2)
                picks.append(int(t * (totalFrames - 1)))

            insetsList = []
            for idx in picks:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    insetsList.append(self._estimateCropInsetsFromFrame(frame))

            if not insetsList:
                return self.cropInsets

            arr = np.array(insetsList, dtype=np.int32)
            med = np.median(arr, axis=0).astype(int)
            return tuple(int(x) for x in med)
        finally:
            cap.release()
    
    def play(self):
        self.isPlaying = True
        self.isPaused = False
        self.decodeStop.clear()
        
        # Start decode thread once
        if not self.thread or not self.thread.is_alive():  # Check if the thread is not already running
            self.thread = threading.Thread(target=self._decodeLoop, daemon=True)
            self.thread.start()
        
        # Start render loop on Tk main thread
        if self.renderAfterId is None:
            self._renderTick()
        
    def pause(self):
        self.isPaused = True
        
    def stop(self):
        self.isPlaying = False
        self.isPaused = False
        self.decodeStop.set()

        if self.renderAfterId is not None:
            try:
                self.canvas.after_cancel(self.renderAfterId)
            except Exception:
                pass
            self.renderAfterId = None
    
    def resize(self, currentHeight):
        """Resize video dimensions dynamically."""
        self.adjustScale(currentHeight)

        # Instead of deleting the canvas item, invalidate the backing image so next render recreates properly.
        self.tkImg = None
        self._lastPos = None

    def _audioTimeMs(self) -> int:
        # pygame get_pos() is ms since playback/unpause; can be -1 briefly.
        pos = pygame.mixer.music.get_pos()
        if pos < 0:
            pos = 0
        return int(self.parent.playbackOffset + pos)
    
    def _loopFrameIndex(self, frameIndex: int) -> int:
        if self.totalFrames and self.totalFrames > 0:
            return frameIndex % self.totalFrames
        return frameIndex
    
    def _pushLatestFrame(self, frame):
        try:
            self.frameQueue.put_nowait(frame)
        except queue.Full:
            # drop one old frame, then insert newest
            try:
                self.frameQueue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.frameQueue.put_nowait(frame)
            except queue.Full:
                pass
        
    def _decodeLoop(self):
        fps = self.effective_fps if self.effective_fps > 0 else 30.0
        target_dt = 1.0 / fps
        next_t = time.perf_counter()

        # initial seek: only for looping backgrounds (optional), OR allow MV to start at audio time
        if self.totalFrames > 0:
            audio_ms = self._audioTimeMs()
            start_frame = int((audio_ms / 1000.0) * fps)
            if not self.isMusicVideo:
                start_frame = self._loopFrameIndex(start_frame)
            else:
                # clamp for MV so we don't mod-wrap
                if start_frame < 0:
                    start_frame = 0
                if start_frame >= self.totalFrames:
                    start_frame = self.totalFrames - 1

            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while self.isPlaying and self.cap.isOpened() and not self.decodeStop.is_set():
            if self.isPaused:
                time.sleep(0.01)
                continue
            
            if self.uiBusy:
                time.sleep(0.03)  # or even target_dt
                continue

            with self.cap_lock:
                ret, frame = self.cap.read()

                if not ret:
                    if not self.isMusicVideo and self.totalFrames and self.totalFrames > 0:
                        # looping background: wrap
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = self.cap.read()
                    else:
                        # music video: STOP instead of looping
                        self.isPlaying = False
                        break

            if not ret:
                time.sleep(0.01)
                continue

            if self.cropEnabled:
                scaledInsets = self._getScaledCropInsetsForFrame(frame)
                frame = self._applyCropInsets(frame, scaledInsets)
                
            frame = cv2.resize(frame, (self.newWidth, self.newHeight), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # keep only newest
            self._pushLatestFrame(frame)

            next_t += target_dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
                
    def setUiBusy(self, busy: bool, autoClearMs: int = 0):
        self.uiBusy = bool(busy)

        if getattr(self, "_busyAfterId", None) is not None:
            try:
                self.canvas.after_cancel(self._busyAfterId)
            except Exception:
                pass
            self._busyAfterId = None

        if self.uiBusy and autoClearMs > 0:
            self._busyAfterId = self.canvas.after(
                autoClearMs,
                lambda: setattr(self, "uiBusy", False)
            )
    
    def _renderTick(self):
        if not self.isPlaying:
            self.renderAfterId = None
            self._renderNextT = None
            return

        # If busy/paused, just slow down; do NOT touch the queue.
        if self.uiBusy or self.isPaused:
            try:
                while True:
                    self.frameQueue.get_nowait()
            except Exception:
                pass
            
            self.renderAfterId = self.canvas.after(120, self._renderTick)
            return

        frame = None
        try:
            frame = self.frameQueue.get_nowait()
        except Exception:
            pass

        if frame is not None:
            try:
                pilImg = Image.fromarray(frame)
                if self.tkImg is None:
                    # Create ONCE
                    self.tkImg = ImageTk.PhotoImage(pilImg)
                    self.videoFrameId = self.canvas.create_image(
                        self.position[0], self.position[1],
                        image=self.tkImg, anchor="nw", tags="layer_video"
                    )
                    self.canvas.tag_lower(self.videoFrameId)
                    self._lastPos = (self.position[0], self.position[1])
                else:
                    # Update pixels IN PLACE (huge speedup)
                    self.tkImg.paste(pilImg)

                # Only move if position actually changed
                pos = (self.position[0], self.position[1])
                if self._lastPos != pos and self.videoFrameId:
                    self.canvas.coords(self.videoFrameId, pos[0], pos[1])
                    self._lastPos = pos
            except tk.TclError:
                pass

        fps = self.effective_fps if self.effective_fps > 0 else 30.0
        period = 1.0 / fps
        now = time.perf_counter()
        if self._renderNextT is None:
            self._renderNextT = now + period
        else:
            self._renderNextT += period
            # If we're behind, don't try to "repay" debt forever
            if self._renderNextT < now:
                self._renderNextT = now + period

        delay_ms = max(1, int((self._renderNextT - now) * 1000.0))
        self.renderAfterId = self.canvas.after(delay_ms, self._renderTick)
    
    def setPosition(self):
        x = 300 / 1920 * 1920 * self.scaleX - (self.newWidth / 2)
        return (x, 0)
            
    def seek(self, timeMs):
        """Calculate the frame index based on the time in milliseconds"""
        if self.effective_fps > 0:
            frameIndex = int((timeMs / 1000.0) * self.effective_fps)
            if frameIndex < 0:
                frameIndex = 0
            if self.isMusicVideo:
                if self.totalFrames > 0:
                    frameIndex = min(frameIndex, self.totalFrames - 1)
            else:
                frameIndex = self._loopFrameIndex(frameIndex)
            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
                
        # flush queued frames
        try:
            while True:
                self.frameQueue.get_nowait()
        except Exception:
            pass
    
    def captureCanvas(self, canvas):
        canvas.update_idletasks()
        canvas.update()
        
        root = self.parent.root  # adjust if your root is stored elsewhere
        hwnd = root.winfo_id()
        windowImg = captureWindowClientRGBA(hwnd)  # PIL RGBA

        rootX = root.winfo_rootx()
        rootY = root.winfo_rooty()
        
        canvasX = canvas.winfo_rootx() - rootX
        canvasY = canvas.winfo_rooty() - rootY
        w = canvas.winfo_width()
        h = canvas.winfo_height()

        cropped = windowImg.crop((canvasX, canvasY, canvasX + w, canvasY + h))
        return cropped
      
    def getFirstKeepChunk(self, startChunk: int = 0) -> int:
        """
        Returns the first chunk index >= startChunk that is NOT inside a Cut range.
        Assumes self.cutRanges/self.cutStarts are already built.
        """
        chunk = max(0, int(startChunk))

        if not getattr(self, "cutRanges", None):
            return chunk

        # Walk forward through cut blocks if we land inside one.
        while True:
            i = bisect_right(self.cutStarts, chunk) - 1
            if i < 0:
                return chunk
            s, e = self.cutRanges[i]
            if s <= chunk <= e:
                chunk = e + 1
                continue
            return chunk
    
    def _resetExportUiToChunk(self, chunkIndex: int):
        """
        Hard reset the canvas + parent state so export always starts from the first keep chunk,
        regardless of where the user is currently scrubbing/paused (e.g., chunk 1044).
        """
        chunkIndex = max(0, int(chunkIndex))
        startTimeMs = int(chunkIndex * self.parent.chunk_duration)

        # 1) Stop/neutralize live playback influence (export will own visuals)
        self.parent.isPaused = False

        # 2) Reset parent time/index state to baseline
        self.parent.currentChunkIndex = chunkIndex
        self.parent.currentSectionIndex = 0
        self.parent.playbackOffset = startTimeMs
        
        self.parent.resetLyricsToChunkStart(chunkIndex)

        # 3) Reset lyric incremental state WITHOUT deleting canvas items
        self.parent.hideAllLyrics(False)
        
        try:
            self.position = self.setPosition()
        except Exception:
            pass

        # 6) One full redraw at the starting chunk
        self.parent.updateCanvasForCurrentPosition(chunkIndex)
        self.canvas.update_idletasks()
        
    def processFrame(self, frame, currentTimeMs, currentChunkIndex, exportCropInsets=None):
        newChunkIndex = int(currentTimeMs / self.parent.chunk_duration)

        if currentChunkIndex is None or newChunkIndex != currentChunkIndex:
            self.parent.updateCanvasForCurrentPosition(newChunkIndex)
            currentChunkIndex = newChunkIndex

        self.position = self.setPosition()
        video_x, video_y = int(self.position[0]), int(self.position[1])

        if self.cropEnabled:
            if exportCropInsets is None:
                scaledInsets = self._getScaledCropInsetsForFrame(frame)
            else:
                scaledInsets = exportCropInsets
            frame = self._applyCropInsets(frame, scaledInsets)

        frame = cv2.resize(frame, (self.newWidth, self.newHeight))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self._exportImgRef = img

        if self.videoFrameId:
            self.canvas.itemconfig(self.videoFrameId, image=img)
            self.canvas.coords(self.videoFrameId, video_x, video_y)
        else:
            self.videoFrameId = self.canvas.create_image(
                video_x, video_y, image=img, anchor="nw", tags="layer_video"
            )
            self.canvas.tag_lower(self.videoFrameId)

        self.canvas.update_idletasks()

        videoFrame = self.captureCanvas(self.canvas)
        finalFrame = np.array(videoFrame.convert("RGB"), dtype=np.uint8)

        if finalFrame.size == 0:
            return None, currentChunkIndex

        return finalFrame, currentChunkIndex
    
    def buildCutRanges(self):
        """
        Build merged, non-overlapping inclusive chunk ranges for labels with name == 'Cut'.
        Returns:
            cutRanges: List[Tuple[int,int]]  e.g. [(100, 500), (900, 920)]
            cutStarts: List[int]             e.g. [100, 900]  (for bisect)
        """
        labels = getattr(self.parent, "labels", None) or []
        ranges = []

        for label in labels:
            try:
                if label and label[0] == "Cut":
                    startChunk = int(label[1])
                    endChunk = int(label[2])
                    if endChunk < startChunk:
                        startChunk, endChunk = endChunk, startChunk
                    ranges.append((startChunk, endChunk))
            except Exception:
                continue

        if not ranges:
            return [], []

        ranges.sort(key=lambda x: x[0])

        merged = []
        curS, curE = ranges[0]
        for s, e in ranges[1:]:
            # merge overlapping OR touching ranges since inclusive
            if s <= curE + 1:
                curE = max(curE, e)
            else:
                merged.append((curS, curE))
                curS, curE = s, e
        merged.append((curS, curE))

        starts = [s for s, _ in merged]
        return merged, starts
    
    def isChunkInCutRanges(self, chunkIndex: int) -> bool:
        """
        O(log k) membership check against merged inclusive ranges.
        Requires self.cutRanges + self.cutStarts to be set.
        """
        if not self.cutRanges:
            return False

        # find rightmost range start <= chunkIndex
        i = bisect_right(self.cutStarts, chunkIndex) - 1
        if i < 0:
            return False
        s, e = self.cutRanges[i]
        return s <= chunkIndex <= e
    
    def _openVideoWriter(self, path, fps, width, height):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for: {path}")
        return out

    def _writerThreadLoop(self, frameQueue, tempVideoPath, fps, width, height, writerStop, writerError):
        try:
            out = self._openVideoWriter(tempVideoPath, fps, width, height)
            while not writerStop.is_set():
                frame = frameQueue.get()
                if frame is None:
                    break
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
        except Exception as e:
            writerError["err"] = e
            writerStop.set()
            
    def _isValidRgbFrame(self, frame, width, height):
        return (
            frame is not None and
            isinstance(frame, np.ndarray) and
            frame.ndim == 3 and
            frame.shape[2] == 3 and
            frame.shape[0] > 0 and frame.shape[1] > 0
        )
    
    def _getWavDurationMs(self, wavPath: str) -> int:
        # Fast, no ffprobe required (works for .wav)
        with wave.open(wavPath, "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate <= 0:
                return 0
            return int((frames / float(rate)) * 1000.0)
    
    def _restoreLiveVideoBindingAfterExport(self):
        # If export replaced the canvas image with _exportImgRef, point it back to tkImg.
        try:
            if self.videoFrameId and self.tkImg is not None:
                self.canvas.itemconfig(self.videoFrameId, image=self.tkImg)
            # Drop the export ref so it can GC and so you don't accidentally keep showing it.
            if hasattr(self, "_exportImgRef"):
                self._exportImgRef = None
        except Exception:
            pass
      
    def getLastAvailableChunk(self, audioEndMs: int) -> int:
        """Last chunk index within audio that is NOT cut. Returns 0 if everything is cut."""
        # last chunk that exists in the audio timeline
        eps = 1e-3
        lastRenderTimeMs = max(0.0, float(audioEndMs) - eps)
        lastAudioChunk = int(lastRenderTimeMs / self.parent.chunk_duration)

        # also clamp by chunks array length, just in case
        lastAudioChunk = min(lastAudioChunk, len(self.parent.chunks) - 1)

        # walk backward to find a keep chunk
        k = lastAudioChunk
        while k >= 0 and self.isChunkInCutRanges(k):
            k -= 1

        return max(0, k)
    
    def hasEndCut(self, maxChunkInclusive: int) -> bool:
        if not getattr(self, "cutRanges", None):
            return False
        # cutRanges are inclusive (s,e)
        lastS, lastE = self.cutRanges[-1]
        return lastE >= maxChunkInclusive
    
    def processVideoAndSave(self, songName, originalVideoPath=None, originalAudioPath=None, fpsCap=0):
        """
        Export a line-distribution video by rendering the UI onto the canvas and capturing frames.

        Key idea:
        - exportMs: output video timeline (continuous)
        - rawMs: original timeline for labels/UI (skips cut ranges)

        Cuts are defined in RAW chunk indices (inclusive).
        We build keep ranges in RAW milliseconds, then walk them sequentially for export.
        """
        import os, uuid, queue, threading
        import cv2
        import numpy as np
        from tqdm import tqdm
        from PIL import Image

        os.makedirs("./finished_videos", exist_ok=True)
        finalOutputPath = os.path.join("./finished_videos", f"{songName}_line_distribution.mp4")

        try:
            self.parent.exportStopEvent.clear()
        except Exception:
            pass

        # Build cut ranges once (RAW chunk indices, inclusive)
        self.cutRanges, self.cutStarts = self.buildCutRanges()

        cap = cv2.VideoCapture(originalVideoPath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found / failed to open: {originalVideoPath}")

        # FPS from source video
        srcFps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if srcFps <= 0:
            srcFps = 30.0

        effectiveFps = min(srcFps, fpsCap) if fpsCap and fpsCap > 0 else srcFps
        if effectiveFps <= 0:
            effectiveFps = 30.0

        frameDtMs = 1000.0 / float(effectiveFps)

        # Canvas dimensions
        self.canvas.update_idletasks()
        width = int(self.canvas.winfo_width())
        height = int(self.canvas.winfo_height())

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        isMusicVideo = bool(getattr(self, "isMusicVideo", True))
        
        exportCropInsets = self.cropInsets
        if self.cropEnabled:
            exportCropInsets = self.detectCropInsetsForVideoPath(originalVideoPath)

        # Determine RAW audio duration (pre-cuts) in ms
        rawAudioMs = 0
        if originalAudioPath:
            try:
                rawAudioMs = int(self._getWavDurationMs(originalAudioPath))
            except Exception:
                rawAudioMs = 0

        if rawAudioMs <= 0:
            # Fallback: video duration
            if totalFrames > 0:
                rawAudioMs = int(totalFrames * (1000.0 / float(srcFps)))
            else:
                rawAudioMs = 0

        if rawAudioMs <= 0:
            cap.release()
            raise RuntimeError("Could not determine export duration (audio and video duration both unavailable).")

        # Overlay rules
        overlayMs = 8000
        fadeInMs = 1000
        eps = 1e-3

        # Chunk parameters
        chunkMs = float(self.parent.chunk_duration)

        # RAW max chunk index (inclusive)
        maxChunkInclusive = int(max(0.0, float(rawAudioMs) - eps) / chunkMs)
        maxChunkInclusive = min(maxChunkInclusive, len(self.parent.chunks) - 1)

        # Find last RAW chunk that is NOT cut (for stable freeze state)
        lastAvailableChunk = maxChunkInclusive
        while lastAvailableChunk >= 0 and self.isChunkInCutRanges(lastAvailableChunk):
            lastAvailableChunk -= 1
        if lastAvailableChunk < 0:
            lastAvailableChunk = 0

        # Convert cutRanges (RAW chunks) -> cut ranges in RAW ms [startMs, endMs)
        cutMsRanges = []
        for s, e in (self.cutRanges or []):
            sMs = float(s) * chunkMs
            eMs = float(e + 1) * chunkMs
            cutMsRanges.append((sMs, eMs))
        cutMsRanges.sort(key=lambda x: x[0])

        # Build keep ranges in RAW ms [startMs, endMs)
        rawEndMs = float(maxChunkInclusive + 1) * chunkMs
        keepMsRanges = []
        cur = 0.0
        for sMs, eMs in cutMsRanges:
            if cur < sMs:
                keepMsRanges.append((cur, sMs))
            if eMs > cur:
                cur = eMs
        if cur < rawEndMs:
            keepMsRanges.append((cur, rawEndMs))

        # Total kept duration in EXPORT ms (this is "song end" in the export)
        keptMs = float(sum(e - s for s, e in keepMsRanges))

        # If everything is cut, we can still output just the overlay over a frozen UI state
        if keptMs < 1:
            keptMs = 0.0

        # Detect if the RAW end is cut (last cut range reaches rawEndMs)
        endCut = False
        if cutMsRanges:
            endCut = (cutMsRanges[-1][1] >= rawEndMs - 1e-6)

        # Define when overlay starts and how much extra to append
        # - No end cut: overlay starts AFTER kept content ends; append full 8s
        # - End cut: use tail (kept content already ends earlier than raw song),
        #            overlay ends at kept end if possible; append remainder to make 8s
        if not endCut:
            pieStartMs = keptMs
            appendMs = float(overlayMs)
        else:
            # content ends before kept end; tail exists
            # Here we define "content end" as the last non-cut boundary (lastAvailableChunk+1) mapped into export timeline.
            # To do that, we compute how much kept time occurs before that raw boundary.
            freezeRawMs = float(lastAvailableChunk + 1) * chunkMs

            # Convert a RAW ms boundary -> EXPORT ms by summing keep segments up to that boundary
            acc = 0.0
            for sMs, eMs in keepMsRanges:
                if freezeRawMs <= sMs:
                    break
                if freezeRawMs >= eMs:
                    acc += (eMs - sMs)
                else:
                    acc += (freezeRawMs - sMs)
                    break
            contentEndExportMs = max(0.0, min(acc, keptMs))

            tailMs = max(0.0, keptMs - contentEndExportMs)
            pieStartMs = max(contentEndExportMs, keptMs - float(overlayMs))
            appendMs = max(0.0, float(overlayMs) - tailMs)

        exportEndMs = keptMs + appendMs
        totalExportFrames = int(exportEndMs / frameDtMs) if exportEndMs > 0 else int(overlayMs / frameDtMs)

        # Freeze UI rendering at last available RAW time (stable state)
        freezeUiRawMs = max(0.0, float(lastAvailableChunk + 1) * chunkMs - eps)

        # Build the panel image from that frozen state
        self.parent.updateCanvasForCurrentPosition(lastAvailableChunk)
        self.parent.lineDistPanel.update(redrawOnly=False)
        panelImg = self.parent.lineDistPanel.renderPanelRgba()

        # Writer thread setup
        tempVideoPath = os.path.join("./finished_videos", f"temp_video_{uuid.uuid4().hex}.mp4")
        frameQueue = queue.Queue(maxsize=16)
        writerStop = threading.Event()

        global writerError
        writerError["err"] = None

        writerThread = threading.Thread(
            target=self._writerThreadLoop,
            args=(frameQueue, tempVideoPath, effectiveFps, width, height, writerStop, writerError),
            daemon=True
        )
        writerThread.start()

        # Prepare UI baseline at first kept chunk (or 0 if none)
        self.parent.isPaused = False
        firstKeepChunk = self.getFirstKeepChunk(0)
        self._resetExportUiToChunk(firstKeepChunk)

        self.parent.activeLyricIds.clear()
        self.parent.updateCanvasForCurrentPosition(firstKeepChunk)
        self.canvas.update_idletasks()
        self.canvas.update()

        # Map exportMs -> rawMs by walking keepMsRanges sequentially
        def rawMsAtExportMs(exportMs: float) -> float:
            if not keepMsRanges:
                return freezeUiRawMs
            rem = float(exportMs)
            for sMs, eMs in keepMsRanges:
                seg = eMs - sMs
                if rem <= seg:
                    return sMs + rem
                rem -= seg
            return keepMsRanges[-1][1]

        # MV freeze robustness
        lastGoodVideoFrame = None

        # State
        currentChunkIndex = None
        frameIndex = 0
        goodFrames = 0
        maxChunkSeen = 0

        progressBar = tqdm(
            total=totalExportFrames,
            desc="Processing Video",
            unit="frame",
            leave=True,
            position=0,
            dynamic_ncols=True
        )

        try:
            exportMs = 0.0
            while cap.isOpened() and frameIndex < totalExportFrames:
                if self.parent.exportStopEvent.is_set():
                    break
                if writerError.get("err") is not None:
                    raise RuntimeError(f"Writer thread failed: {writerError['err']}")

                inAppended = (exportMs >= keptMs)

                # rawMs drives the UI
                if not inAppended:
                    rawMs = rawMsAtExportMs(exportMs)
                else:
                    rawMs = freezeUiRawMs

                # Keep maxChunkSeen in RAW chunk indices for audio cut inversion
                rawChunk = int(rawMs / chunkMs)
                if rawChunk > maxChunkSeen:
                    maxChunkSeen = rawChunk
                    if maxChunkSeen > lastAvailableChunk:
                        maxChunkSeen = lastAvailableChunk

                # Pick video frame for exportMs
                frame = None
                videoTimeMs = rawMs if isMusicVideo else exportMs
                if totalFrames > 0:
                    # If we're in appended tail:
                    # - MV: freeze on last good frame
                    # - Background: keep looping
                    if isMusicVideo and inAppended and lastGoodVideoFrame is not None:
                        frame = lastGoodVideoFrame
                    else:
                        desiredFrame = int((videoTimeMs / 1000.0) * float(srcFps))

                        if isMusicVideo:
                            # MV should NOT loop; clamp to last frame
                            if desiredFrame < 0:
                                desiredFrame = 0
                            elif desiredFrame >= totalFrames:
                                desiredFrame = totalFrames - 1
                        else:
                            # Background loops forever
                            desiredFrame = desiredFrame % totalFrames

                        cap.set(cv2.CAP_PROP_POS_FRAMES, desiredFrame)
                        ret, frame = cap.read()

                        if not ret or frame is None:
                            # Fallback read
                            fallback = (totalFrames - 1) if isMusicVideo else 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, fallback)
                            ret, frame = cap.read()
                            if not ret or frame is None:
                                exportMs += frameDtMs
                                progressBar.update(1)
                                frameIndex += 1
                                continue

                        # Cache last good frame (important for MV freeze during appended tail)
                        lastGoodVideoFrame = frame

                else:
                    # No frame count available; sequential read + loop
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            exportMs += frameDtMs
                            progressBar.update(1)
                            frameIndex += 1
                            continue
                    lastGoodVideoFrame = frame

                # Render + capture
                try:
                    rgbFrame, currentChunkIndex = self.processFrame(
                        frame,
                        int(rawMs),
                        currentChunkIndex,
                        exportCropInsets=exportCropInsets
                    )
                    if self._isValidRgbFrame(rgbFrame, width, height):
                        if exportMs >= pieStartMs:
                            a = (exportMs - pieStartMs) / float(fadeInMs)
                            if a < 0: a = 0.0
                            if a > 1: a = 1.0

                            base = Image.fromarray(rgbFrame, mode="RGB").convert("RGBA")
                            overlay = panelImg.resize((width, height), Image.LANCZOS)

                            if a < 1.0:
                                r, g, b, alpha = overlay.split()
                                alpha = alpha.point(lambda p: int(p * a))
                                overlay = Image.merge("RGBA", (r, g, b, alpha))

                            out = Image.alpha_composite(base, overlay).convert("RGB")
                            rgbFrame = np.array(out, dtype=np.uint8)

                        frameQueue.put(rgbFrame)
                        goodFrames += 1

                except Exception as e:
                    print(f"⚠️ Frame {frameIndex} @ {int(exportMs)}ms failed: {e}")

                exportMs += frameDtMs
                progressBar.update(1)
                frameIndex += 1

                if frameIndex % 5 == 0:
                    try:
                        self.canvas.update_idletasks()
                        self.canvas.update()
                    except Exception:
                        pass

        finally:
            cap.release()
            progressBar.close()

            # Stop writer
            try:
                frameQueue.put(None)
            except Exception:
                pass
            writerStop.set()
            writerThread.join(timeout=30)

            try:
                self.parent._finishExportVideo()
            except Exception:
                pass

            try:
                self._restoreLiveVideoBindingAfterExport()
            except Exception:
                pass

        if writerError.get("err") is not None:
            raise RuntimeError(f"Writer thread failed: {writerError['err']}")

        if self.parent.exportStopEvent.is_set() or goodFrames <= 0:
            print("⛔ Export aborted or no frames produced. Skipping audio mux.")
            try:
                if os.path.exists(tempVideoPath):
                    os.remove(tempVideoPath)
            except Exception:
                pass
            return None

        # Mux audio with cuts + pad silence for the appended overlay time
        self.addAudioToVideo(
            tempVideoPath,
            originalAudioPath,
            goodFrames,
            effectiveFps,
            finalOutputPath,
            maxChunkInclusive=maxChunkInclusive,
            postrollMs=int(appendMs)  # only pad what we appended
        )

        try:
            os.remove(tempVideoPath)
        except Exception:
            pass

        print(f"✅ Saved: {finalOutputPath}")
        return finalOutputPath
    
    def buildKeepRangesSeconds(self, maxChunkInclusive: int):
        """
        Invert self.cutRanges (inclusive chunk ranges) into KEEP ranges in seconds.
        Assumes chunk i covers [i*chunkSec, (i+1)*chunkSec).
        Returns list of (startSec, endSec) with endSec > startSec.
        """
        chunkSec = self.parent.chunk_duration / 1000.0

        # If no cuts, keep whole thing
        if not getattr(self, "cutRanges", None):
            return [(0.0, (maxChunkInclusive + 1) * chunkSec)]

        keep = []
        cur = 0  # first chunk index to keep

        for s, e in self.cutRanges:
            if cur <= s - 1:
                keepStart = cur * chunkSec
                keepEnd = (s) * chunkSec  # start of cut
                if keepEnd > keepStart:
                    keep.append((keepStart, keepEnd))
            cur = max(cur, e + 1)

        # tail after last cut
        if cur <= maxChunkInclusive:
            keepStart = cur * chunkSec
            keepEnd = (maxChunkInclusive + 1) * chunkSec
            if keepEnd > keepStart:
                keep.append((keepStart, keepEnd))

        return keep

    def addAudioToVideo(self, videoPath, audioPath, totalFrames, fps, outputPath, maxChunkInclusive=None, postrollMs=8000):
        """
        Merge MP4 with audio, applying the same Cut removals to audio so it stays synced
        when video time is collapsed.

        maxChunkInclusive: last chunk index in the exported timeline (needed to invert cuts).
                        If None, we approximate from totalFrames/fps.
        """
        # If you didn’t pass chunk count, approximate from exported video duration
        if maxChunkInclusive is None:
            chunkSec = self.parent.chunk_duration / 1000.0
            videoDur = totalFrames / float(fps)
            maxChunkInclusive = int(videoDur / chunkSec)

        # If no cuts, simple mux
        ffmpegPath = resourcePath("ffmpeg.exe")
        padSec = max(0.0, float(postrollMs) / 1000.0)

        if not getattr(self, "cutRanges", None):
            cmd = [
                ffmpegPath, "-y",
                "-i", videoPath,
                "-i", audioPath,
            ]
            if padSec > 0:
                cmd += ["-af", f"apad=pad_dur={padSec}"]

            cmd += [
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                outputPath
            ]
            subprocess.run(cmd, check=True)
            return

        keepRanges = self.buildKeepRangesSeconds(maxChunkInclusive)

        if not keepRanges:
            subprocess.run([
                ffmpegPath, "-y",
                "-i", videoPath,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                outputPath
            ], check=True)
            return

        parts = []
        concatInputs = []
        for i, (s, e) in enumerate(keepRanges):
            parts.append(f"[1:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
            concatInputs.append(f"[a{i}]")

        # concat -> optionally apad -> [aout]
        filterComplex = (
            ";".join(parts)
            + ";"
            + "".join(concatInputs)
            + f"concat=n={len(keepRanges)}:v=0:a=1[acat]"
        )
        if padSec > 0:
            filterComplex += f";[acat]apad=pad_dur={padSec}[aout]"
            aMap = "[aout]"
        else:
            aMap = "[acat]"

        subprocess.run([
            ffmpegPath, "-y",
            "-i", videoPath,     # 0:v
            "-i", audioPath,     # 1:a
            "-filter_complex", filterComplex,
            "-map", "0:v:0",
            "-map", aMap,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            outputPath
        ], check=True)
    #end addAudioToVideo