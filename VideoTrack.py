import tkinter as tk
from tkinter import ttk
import cv2
import time
from PIL import Image, ImageTk, ImageGrab
import threading
from TrackItem import TrackItem
import numpy as np
import os
import subprocess
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class VideoTrackItem(TrackItem):
    def __init__(self, canvas, parent, videoPath, scale=100, scaleX=1.0, position=(0,0), baseHeight=720, isMusicVideo=True):
        super().__init__(scale, position, sourceImages={}, animations=[], type="video")
        self.canvas = canvas
        self.videoPath = videoPath
        self.parent = parent
        self.cap = cv2.VideoCapture(videoPath)
        self.scale = scale
        self.scaleX = scaleX
        self.videoFrameId = None
        self.isPlaying = False
        self.isPaused = False
        self.thread = None
        self.baseHeight = baseHeight
        self.currentFrame = None
        
        # Get video dimensions
        if self.cap.isOpened():
            self.frameWidth = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frameHeight = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        else:
            raise FileNotFoundError(f"Video file not found: {videoPath}")
        
        self.adjustScale(baseHeight)
        self.setPosition()
        self.isMusicVideo = isMusicVideo
        
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
        if not self.thread or not self.thread.is_alive():  # Check if the thread is not already running
            self.thread = threading.Thread(target=self._playVideo, daemon=True)
            self.thread.start()
        else:
            self.isPaused = False
        
    def pause(self):
        self.isPaused = True
        
    def stop(self):
        self.isPlaying = False
        self.isPaused = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)  # Wait briefly for the thread to terminate
    
    def resize(self, currentHeight):
        """Resize video dimensions dynamically."""
        self.adjustScale(currentHeight)
        if self.videoFrameId:
            self.canvas.delete(self.videoFrameId)
            self.videoFrameId = None

    def _playVideo(self):
        # Get video frame rate (frames per second)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError("Invalid FPS detected in video file.")

        frameDuration = 1000 / fps  # Duration of each frame in ms
        lastFrameTime = time.time()
        
        while self.isPlaying and self.cap.isOpened():
            if self.isPaused:
                time.sleep(0.02)  # Wait briefly while paused
                lastFrameTime = time.time() 
                continue
            
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (self.newWidth, self.newHeight))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            
            if self.videoFrameId:
                self.canvas.itemconfig(self.videoFrameId, image=img)
            else:
                self.videoFrameId = self.canvas.create_image(self.position[0], self.position[1], image=img, anchor="nw")
                  # Keep a reference to avoid garbage collection
                # Push the video to the back layer
                self.canvas.tag_lower(self.videoFrameId)
            self.canvas.image = img
            self.canvas.coords(self.videoFrameId, self.position[0], self.position[1])
            self.canvas.update()
            
            # Maintain FPS
            elapsedTime = time.time() - lastFrameTime
            sleepTime = max(0, (frameDuration / 1000) - elapsedTime)
            time.sleep(sleepTime)
            lastFrameTime = time.time()
        self.isPlaying = False

    def setPosition(self):
        x = 300 / 1920 * 1920 * self.scaleX - (self.newWidth / 2)
        self.position = (x, 0)
            
    def seek(self, timeMs):
        """Calculate the frame index based on the time in milliseconds"""
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps > 0:
            frameIndex = int((timeMs / 1000.0) * fps)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frameIndex)
    
    def captureCanvas(self, canvas):
        canvas.update()

        x = canvas.winfo_rootx() + 2
        y = canvas.winfo_rooty() + 2
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        #print(f"X: {x}, y: {y}, width: {width}, height: {height}")

        img = ImageGrab.grab(bbox=(x, y, x + width, y + height)).convert("RGBA")

        return img
        
    def processFrame(self, frame, currentTimeMs, currentChunkIndex):
        """Process a single frame: overlay Tkinter canvas and update chunkIndex if needed"""
        newChunkIndex = int(currentTimeMs / self.parent.chunk_duration)

        # Update the canvas **only if chunkIndex changes**
        if newChunkIndex != currentChunkIndex:
            self.parent.updateCanvasForCurrentPosition(newChunkIndex)
            currentChunkIndex = newChunkIndex        
        
        self.setPosition()
        video_x, video_y = int(self.position[0]), int(self.position[1])
        
        if hasattr(self, "videoFrameId") and self.videoFrameId:
            self.parent.canvas.delete(self.videoFrameId)
        
        frame = cv2.resize(frame, (self.newWidth, self.newHeight))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.videoFrameId = self.canvas.create_image(video_x, video_y, image=img, anchor="nw")
        self.canvas.tag_lower(self.videoFrameId)
            
        self.canvas.coords(self.videoFrameId, video_x, video_y)
        videoFrame = self.captureCanvas(self.canvas)

        finalFrame = np.array(videoFrame.convert("RGB"), dtype=np.uint8)
        if finalFrame.any():
            return finalFrame, currentChunkIndex
        else:
            print("⚠️ Warning: Empty frame detected. Skipping.")
            return None, currentChunkIndex

    def processVideoAndSave(self, outputPath="LineDistribution.mp4", fpsCap=0):
        """Extract each frame from the video, update the canvas, and save to MP4"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        originalFps = self.cap.get(cv2.CAP_PROP_FPS)
        effectiveFps = min(originalFps, fpsCap) if fpsCap > 0 else originalFps
        frameSkipRate = max(1, int(originalFps / effectiveFps)) if fpsCap > 0 else 1
        frameDuration = 1000 / effectiveFps  # Each frame's duration in milliseconds
        
        width = self.canvas.winfo_width()
        height = self.canvas.winfo_height()
        totalFrames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        totalDurationMs = totalFrames * (1000 / originalFps)  # Calculate total duration in ms
        cappedTotalFrames = int(totalDurationMs / frameDuration)  # Adjusted frame count based on FPS cap
        #print(f"Total Frames in Video: {totalFrames}")
    
        currentChunkIndex = -1
        frameIndex = 0
        audioPath = self.parent.testSongPath
        
        # Initialize progress bar
        progressBar = tqdm(total=cappedTotalFrames, desc="Processing Video", unit="frame", leave=True, position=0, dynamic_ncols=True)
        self.parent.isPaused = False

        framesList = [None] * cappedTotalFrames
        try:
            while self.cap.isOpened() and frameIndex < cappedTotalFrames:
                ret, frame = self.cap.read()

                # Skip frames based on FPS cap
                for _ in range(frameSkipRate - 1):
                    self.cap.read()

                if not ret:
                    break  # Stop when video ends

                currentTimeMs = int(frameIndex * frameDuration)

                try:
                    # Process frame and update canvas accordingly
                    finalFrame, currentChunkIndex = self.processFrame(frame, currentTimeMs, currentChunkIndex)
                    framesList[frameIndex] = finalFrame

                except Exception as e:
                    print(f"⚠️ Error processing frame at {currentTimeMs}ms: {e}")

                progressBar.update(1)  # Update progress bar
                frameIndex += 1
            
        except Exception as e:
            print(f"\n⚠️ Error during video processing: {e}")
            print("Saving current progress and adding audio...")

        finally:
            # Release resources even if an error occurs
            self.cap.release()
            progressBar.close()
            print("Video processing complete. Adding audio...")
    
        tempVideoPath = "temp_video.mp4"
        self.compileFramesToMP4(tempVideoPath, framesList, effectiveFps, width, height)
        self.addAudioToVideo(tempVideoPath, audioPath, cappedTotalFrames, effectiveFps, outputPath)
        os.remove(tempVideoPath)

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
        
    def addAudioToVideo(self, videoPath, audioPath, totalFrames, fps, outputPath):
        """Merge final Mp4 with audio"""
        chunkDurationSec = self.parent.chunk_duration / 1000
        subprocess.run([
                    "ffmpeg",
                    "-i", videoPath,  # Input video (without audio)
                    "-i", audioPath,  # Input audio
                    "-c:v", "libx264",  # Encode video properly
                    "-preset", "fast",  # Speed up encoding
                    "-crf", "23",  # Maintain quality
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",  # Encode audio with AAC
                    "-strict", "experimental",  # Ensure compatibility
                    "-shortest",  # Trim to shortest stream (avoid extra silence)
                    "-t", str(totalFrames / fps - chunkDurationSec),  # Trim audio 1 chunk shorter
                    outputPath  # Output file
                ], check=True)