import cv2, wave
import time, uuid
from PIL import Image, ImageTk
import threading
from TrackItem import TrackItem
import numpy as np
import os
import subprocess
from tqdm import tqdm
from bisect import bisect_right
import pygame
import queue
from video_record import captureWindowClientRGBA

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
        
        self.frameQueue = queue.Queue(maxsize=3) # keep only freshest frames
        self.renderAfterId = None
        self.decodeStop = threading.Event()
        self.lastSeekFrameIndex = 0
        
        # Get video dimensions
        if self.cap.isOpened():
            self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            raise FileNotFoundError(f"Video file not found: {videoPath}")
        
        self.adjustScale(baseHeight)
        self.position = self.setPosition()
        # Sets video fps
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"Current fps: {fps}")
        if fps <= 0:
            self.effective_fps = 30
        else:
            self.effective_fps = min(fps, 45) # cap to 30
            
        self.isMusicVideo = isMusicVideo
        self.totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap.isOpened() else 0
        self.videoDurationMs = 0
        if self.totalFrames > 0 and self.effective_fps > 0:
            self.videoDurationMs = int((self.totalFrames / float(self.effective_fps)) * 1000)
        
    def adjustScale(self, currentHeight):
        """Adjust the video dimensions and scale based on the current height."""
        # Calculate the new scale as a percentage
        self.scale = (currentHeight / self.baseHeight) * 100
        
        self.newHeight = currentHeight
        self.newWidth = int(self.newHeight * (self.frameWidth / self.frameHeight))
        print(f"New height: {self.newHeight}, New width: {self.newWidth}")
        # self.canvas.config(width=self.newWidth, height=self.newHeight)
      
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
        if self.videoFrameId:
            self.canvas.delete(self.videoFrameId)
            self.videoFrameId = None

    def _playVideo(self):
        # Get video frame rate (frames per second        
        while self.isPlaying and self.cap.isOpened():
            if self.isPaused:
                time.sleep(0.02)  # Wait briefly while paused
                continue
            
            # Get current audio playback time (ms)
            audio_pos_ms = pygame.mixer.music.get_pos()
            
            playback_ms = self.parent.playbackOffset + audio_pos_ms
            
            # Compute which video frame should be visible
            frame_index = int((playback_ms / 1000.0) * self.effective_fps)
            if frame_index < 0:
                frame_index = 0
                
            # Protect VideoCapture operations
            with self.cap_lock:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self.cap.read()
            
            if not ret:
                break
            
            # DRAW FRAME
            frame = cv2.resize(frame, (self.newWidth, self.newHeight))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            
            if self.videoFrameId:
                self.canvas.itemconfig(self.videoFrameId, image=img)
            else:
                self.videoFrameId = self.canvas.create_image(self.position[0], self.position[1], image=img, anchor="nw", tags="layer_video")
                  # Keep a reference to avoid garbage collection
                # Push the video to the back layer
                self.canvas.tag_lower(self.videoFrameId)
            self.canvas.image = img
            self.canvas.coords(self.videoFrameId, self.position[0], self.position[1])
            self.canvas.update()
            
            time.sleep(0.01)
            
        self.isPlaying = False

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

            frame = cv2.resize(frame, (self.newWidth, self.newHeight), interpolation=cv2.INTER_AREA)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # keep only newest
            try:
                while True:
                    self.frameQueue.get_nowait()
            except Exception:
                pass

            try:
                self.frameQueue.put_nowait(frame)
            except Exception:
                pass

            next_t += target_dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.perf_counter()
               
    def _renderTick(self):
        # Run ONLY on the Tk main thread via after()
        if not self.isPlaying:
            self.renderAfterId = None
            return
        
        if not self.isPaused:
            try:
                frame = self.frameQueue.get_nowait()
            except Exception:
                frame = None
            
            if frame is not None:
                img = ImageTk.PhotoImage(image=Image.fromarray(frame))

                if self.videoFrameId:
                    self.canvas.itemconfig(self.videoFrameId, image=img)
                else:
                    self.videoFrameId = self.canvas.create_image(
                        self.position[0], self.position[1], image=img, anchor="nw", tags="layer_video"
                    )
                    self.canvas.tag_lower(self.videoFrameId)

                self.canvas.coords(self.videoFrameId, self.position[0], self.position[1])

                # IMPORTANT: store reference so Tk doesn't GC it
                self.canvas.image = img
        
        self.renderAfterId = self.canvas.after(33, self._renderTick)
    
    def setPosition(self):
        x = 300 / 1920 * 1920 * self.scaleX - (self.newWidth / 2)
        return (x, 0)
            
    def seek(self, timeMs):
        """Calculate the frame index based on the time in milliseconds"""
        if self.effective_fps > 0:
            frameIndex = int((timeMs / 1000.0) * self.effective_fps)
            if frameIndex < 0:
                frameIndex = 0
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
        
    def processFrame(self, frame, currentTimeMs, currentChunkIndex):
        newChunkIndex = int(currentTimeMs / self.parent.chunk_duration)

        if currentChunkIndex is None or newChunkIndex != currentChunkIndex:
            self.parent.updateCanvasForCurrentPosition(newChunkIndex)
            currentChunkIndex = newChunkIndex

        self.position = self.setPosition()
        video_x, video_y = int(self.position[0]), int(self.position[1])

        frame = cv2.resize(frame, (self.newWidth, self.newHeight))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Reuse the canvas item (good)
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self._exportImgRef = img  # prevent GC

        if self.videoFrameId:
            self.canvas.itemconfig(self.videoFrameId, image=img)
            self.canvas.coords(self.videoFrameId, video_x, video_y)
        else:
            self.videoFrameId = self.canvas.create_image(
                video_x, video_y, image=img, anchor="nw", tags="layer_video"
            )
            self.canvas.tag_lower(self.videoFrameId)

        # IMPORTANT: give Tk a chance to layout + draw everything you just changed
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

    def _writerThreadLoop(self, frameQueue, tempVideoPath, fps, width, height, stopEvent):
        out = self._openVideoWriter(tempVideoPath, fps, width, height)
        try:
            while not stopEvent.is_set():
                item = frameQueue.get()
                if item is None:  # sentinel = done
                    break

                # item is RGB uint8 numpy array (H,W,3)
                frame = item
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            out.release()
            
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
        
    def processVideoAndSave(self, songName, originalVideoPath=None, originalAudioPath=None, fpsCap=0):
        """
        Export a line-distribution video by rendering the UI (video + labels/lyrics/markers) onto the canvas
        and capturing frames from the Tk window.

        Uses ONE timebase: exportTimeMs.

        - If self.isMusicVideo is True: sample the MV at exportTimeMs.
        If exportTimeMs exceeds MV duration, loops MV frames (safe fallback).
        - If self.isMusicVideo is False: loop the chosen short background video for the full audio duration.
        """
        import os, uuid, queue, threading
        import cv2
        from tqdm import tqdm

        os.makedirs("./finished_videos", exist_ok=True)
        finalOutputPath = os.path.join("./finished_videos", f"{songName}_line_distribution.mp4")

        # Make sure previous abort doesn't instantly kill this export
        try:
            self.parent.exportStopEvent.clear()
        except Exception:
            pass

        # Build cut ranges once
        self.cutRanges, self.cutStarts = self.buildCutRanges()

        cap = cv2.VideoCapture(originalVideoPath)
        if not cap.isOpened():
            raise FileNotFoundError(f"Video not found / failed to open: {originalVideoPath}")

        # FPS from source video (we still write at effectiveFps)
        srcFps = cap.get(cv2.CAP_PROP_FPS)
        if not srcFps or srcFps <= 0:
            srcFps = 30.0

        effectiveFps = min(srcFps, fpsCap) if fpsCap and fpsCap > 0 else srcFps
        if not effectiveFps or effectiveFps <= 0:
            effectiveFps = 30.0

        frameDuration = 1000.0 / float(effectiveFps)

        # Canvas dimensions
        self.canvas.update_idletasks()
        width = int(self.canvas.winfo_width())
        height = int(self.canvas.winfo_height())

        totalFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        isMusicVideo = bool(getattr(self, "isMusicVideo", True))

        # Total export duration:
        # - If you have audio, always prefer audio duration (keeps output length correct).
        # - If no audio path, fall back to video duration.
        totalDurationMs = 0
        if originalAudioPath:
            try:
                totalDurationMs = int(self._getWavDurationMs(originalAudioPath))
            except Exception:
                totalDurationMs = 0

        if totalDurationMs <= 0:
            # Fallback: video duration
            if totalFrames > 0:
                totalDurationMs = int(totalFrames * (1000.0 / float(srcFps)))
            else:
                totalDurationMs = 0

        if totalDurationMs <= 0:
            cap.release()
            raise RuntimeError("Could not determine export duration (audio duration and video duration both unavailable).")

        cappedTotalFrames = int(totalDurationMs / frameDuration)

        tempVideoPath = f"temp_video_{uuid.uuid4().hex}.mp4"
        frameQueue = queue.Queue(maxsize=16)
        writerStop = threading.Event()
        writerThread = threading.Thread(
            target=self._writerThreadLoop,
            args=(frameQueue, tempVideoPath, effectiveFps, width, height, writerStop),
            daemon=True
        )
        writerThread.start()

        # State
        currentChunkIndex = None
        frameIndex = 0
        goodFrames = 0
        maxChunkSeen = 0

        # Prepare UI baseline: start at first keep chunk
        self.parent.isPaused = False
        firstKeepChunk = self.getFirstKeepChunk(0)
        startTimeMs = int(firstKeepChunk * self.parent.chunk_duration)
        
        self._resetExportUiToChunk(firstKeepChunk)

        self.parent.activeLyricIds.clear()
        self.parent.updateCanvasForCurrentPosition(firstKeepChunk)
        self.canvas.update_idletasks()
        self.canvas.update()

        # Single timebase for everything
        exportTimeMs = float(startTimeMs)

        progressBar = tqdm(
            total=cappedTotalFrames,
            desc="Processing Video",
            unit="frame",
            leave=True,
            position=0,
            dynamic_ncols=True
        )

        try:
            while cap.isOpened() and frameIndex < cappedTotalFrames:
                if self.parent.exportStopEvent.is_set():
                    break

                # Chunk index derived from the ONE timeline
                chunkIndex = int(exportTimeMs / self.parent.chunk_duration)
                if chunkIndex > maxChunkSeen:
                    maxChunkSeen = chunkIndex

                # Skip cut chunks (but still advance time so we don't freeze)
                if self.isChunkInCutRanges(chunkIndex):
                    exportTimeMs += frameDuration
                    progressBar.update(1)
                    frameIndex += 1
                    continue

                # Select a video frame for this exportTimeMs
                frame = None
                if totalFrames > 0:
                    # Convert export time -> source frame number
                    desiredFrame = int((exportTimeMs / 1000.0) * float(srcFps))

                    if isMusicVideo:
                        # If MV is shorter than audio, loop MV frames safely
                        desiredFrame = desiredFrame % totalFrames
                    else:
                        # Looping background always loops
                        desiredFrame = desiredFrame % totalFrames

                    cap.set(cv2.CAP_PROP_POS_FRAMES, desiredFrame)
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        # Hard reset once
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            # If video decode fails, advance time and keep going (don’t deadlock export)
                            exportTimeMs += frameDuration
                            progressBar.update(1)
                            frameIndex += 1
                            continue
                else:
                    # No frame count available; attempt sequential read and loop on EOF
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            exportTimeMs += frameDuration
                            progressBar.update(1)
                            frameIndex += 1
                            continue

                # Render + capture
                try:
                    rgbFrame, currentChunkIndex = self.processFrame(frame, int(exportTimeMs), currentChunkIndex)
                    if self._isValidRgbFrame(rgbFrame, width, height):
                        frameQueue.put(rgbFrame)
                        goodFrames += 1
                except Exception as e:
                    print(f"⚠️ Frame {frameIndex} @ {int(exportTimeMs)}ms failed: {e}")

                # Advance the ONE timeline
                exportTimeMs += frameDuration
                progressBar.update(1)
                frameIndex += 1

                # Keep Tk responsive if this function is called on the main thread
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
            writerThread.join(timeout=30)
            writerStop.set()

            # Restore UI state
            try:
                self.parent._finishExportVideo()
            except Exception:
                pass

        # Abort / no frames guard
        if self.parent.exportStopEvent.is_set() or goodFrames <= 0:
            print("⛔ Export aborted or no frames produced. Skipping audio mux.")
            try:
                if os.path.exists(tempVideoPath):
                    os.remove(tempVideoPath)
            except Exception:
                pass
            return None

        # Mux audio
        self.addAudioToVideo(
            tempVideoPath,
            originalAudioPath,
            goodFrames,
            effectiveFps,
            finalOutputPath,
            maxChunkInclusive=maxChunkSeen,
        )

        try:
            os.remove(tempVideoPath)
        except Exception:
            pass

        print(f"✅ Saved: {finalOutputPath}")
        return finalOutputPath
    
    # WORKS!!!!! ADD MULTITHREADING!!
    def compileFramesToMP4(self, tempVideoPath, framesList, fps, width, height):
        """Compile all stored frames into an MP4 video"""
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(tempVideoPath, fourcc, fps, (width, height))

        print("🛠️ Compiling frames into MP4...")
        totalFrames = len(framesList)
        progressBar = tqdm(total=totalFrames, desc="Processing MP4", unit="frame", leave=True, position=0, dynamic_ncols=True)
        for i, frame in enumerate(framesList):
            if frame is not None:
                # Ensure correct size
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                # Write to video
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                progressBar.update(1)

        out.release()
        progressBar.close()
        print(f"✅ Video frames compiled into {tempVideoPath}")
    
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
       
    def addAudioToVideo(self, videoPath, audioPath, totalFrames, fps, outputPath, maxChunkInclusive=None):
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
        if not getattr(self, "cutRanges", None):
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", videoPath,
                "-i", audioPath,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-shortest",
                outputPath
            ], check=True)
            return

        keepRanges = self.buildKeepRangesSeconds(maxChunkInclusive)

        # Edge case: everything is cut (no keep ranges)
        if not keepRanges:
            # mux video with no audio (or silence). Here we output no audio:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", videoPath,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-an",
                outputPath
            ], check=True)
            return

        # Build filter_complex: [1:a]atrim=start=...:end=...,asetpts=PTS-STARTPTS[a0]; ...
        # then concat them: [a0][a1]...concat=n=N:v=0:a=1[aout]
        parts = []
        concatInputs = []
        for i, (s, e) in enumerate(keepRanges):
            parts.append(f"[1:a]atrim=start={s}:end={e},asetpts=PTS-STARTPTS[a{i}]")
            concatInputs.append(f"[a{i}]")

        filterComplex = ";".join(parts) + ";" + "".join(concatInputs) + f"concat=n={len(keepRanges)}:v=0:a=1[aout]"

        subprocess.run([
            "ffmpeg", "-y",
            "-i", videoPath,     # 0:v
            "-i", audioPath,     # 1:a
            "-filter_complex", filterComplex,
            "-map", "0:v:0",
            "-map", "[aout]",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-shortest",
            outputPath
        ], check=True)