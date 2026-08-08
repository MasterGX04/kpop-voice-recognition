import cv2
import os
import pygame
import numpy as np
from tqdm import tqdm

class LoopingBackground:
    def __init__(self, videoPath, audioPath):
        """Initialize the looping background video class."""
        print(f"Video path: {videoPath}")
        self.videoPath = videoPath
        self.audioPath = audioPath

        # Load video properties
        self.cap = cv2.VideoCapture(videoPath)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frameDuration = 1000 / self.fps  # Frame duration in ms
        self.videoLengthMs = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) * self.frameDuration)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Load audio using pygame and get duration in milliseconds
        pygame.mixer.init()
        self.audioDurationMs = int(pygame.mixer.Sound(audioPath).get_length() * 1000)

    def processLoopingVideo(self, outputPath="looping_background.mp4"):
        """Creates a looping video that matches the length of the audio."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset video to start
        
        totalFrames = int(self.audioDurationMs / self.frameDuration)
        frameIndex = 0
        progressBar = tqdm(total=totalFrames, desc="Processing Looping Background", unit="frame", leave=True, position=0, dynamic_ncols=True)

        framesList = []

        try:
            while frameIndex < totalFrames:
                # Reset video when it reaches the end
                if self.cap.get(cv2.CAP_PROP_POS_MSEC) >= self.videoLengthMs:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue  # Restart the video if it ends

                framesList.append(frame)
                progressBar.update(1)
                frameIndex += 1

        except Exception as e:
            print(f"\n⚠️ Error during looping video processing: {e}")

        finally:
            self.cap.release()
            progressBar.close()
            print("Looping background processing complete.")

        # Save looping video
        self.compileFramesToMP4(outputPath, framesList, self.fps, self.width, self.height)

    def compileFramesToMP4(self, outputPath, frames, fps, width, height):
        """Compiles the processed frames into an MP4 file."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(outputPath, fourcc, fps, (width, height))

        totalFrames = len(frames)
        progressBar = tqdm(total=totalFrames, desc="Processing Looping Background MP4", unit="frame", leave=True, position=0, dynamic_ncols=True)
        for frame in frames:
            if frame is not None:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                progressBar.update(1)

        out.release()
        print(f"Looping video saved as {outputPath}")
        
if __name__ == "__main__":
    loopingBackground = LoopingBackground("./training_data/IVE/CelestialBackground.mp4", "./training_data/IVE/Flu.mp3")
    loopingBackground.processLoopingVideo()