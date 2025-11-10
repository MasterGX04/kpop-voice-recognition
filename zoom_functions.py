import tkinter as tk

class ZoomManager:
    def __init__(self, canvas, parent, progressBar, songDuration, chunkDuration, pygame):
        self.canvas = canvas
        self.parent = parent
        self.progressBar = progressBar
        self.songDuration = songDuration
        self.chunkDuration = chunkDuration
        self.zoomLevel = 1.0 # Default zoom level
        self.totalWidth = 800 * self.zoomLevel
        self.pygame = pygame
        self.minChunksInView = 10
        self.maxChunksInView = int(songDuration / chunkDuration) # Max chunks in view
        self.currentChunksInView = self.maxChunksInView
        
        # Zoom Slider UI
        self.zoomFrame = tk.Frame(self.canvas, bg="black")
        self.zoomFrame.place(relx=1, rely=1, anchor="se")
        
        self.zoomOutLabel = tk.Label(self.zoomFrame, text="-", fg="white", bg="black", font=("Arial", 14))
        self.zoomOutLabel.pack(side="left", padx=5)
        
        # Create zoom slider
        self.zoomVar = tk.DoubleVar(value=1.0)
        self.zoomSlider = tk.Scale(
            self.zoomFrame,
            from_=1.0,
            to=25.0,
            resolution=0.1,
            orient=tk.HORIZONTAL,
            variable=self.zoomVar,
            command=self.onZoomChange,
            showvalue=0,
            length=150,
            bg="black",
            fg="white",
            highlightthickness=0
        )
        self.zoomSlider.pack(side="left")
        
        self.zoomInLabel = tk.Label(self.zoomFrame, text="+", fg="white", bg="black", font=("Arial", 14))
        self.zoomInLabel.pack(side="left", padx=5)
        
        self.zoomHidden = False

    def updateZoomLevel(self, newZoom):
        """Update the zoom level and adjust the visible range of the progress bar."""
        self.zoomLevel = float(newZoom)
        # Calculate new number of chunks to display based on zoom value
        self.currentChunksInView = int(self.maxChunksInView // self.zoomLevel)
        
        visibleDuration = self.currentChunksInView * self.parent.chunk_duration

        # Update the current section index based on playback offset and new visible duration
        if hasattr(self.parent, "playbackOffset"):
            playbackTime = self.parent.playbackOffset + self.pygame.mixer.music.get_pos()
            self.parent.currentSectionIndex = int(playbackTime // visibleDuration)
            self.parent.progressBarHandle.currentSectionIndex = self.parent.currentSectionIndex
            
        self.parent.updateProgressBar()
        # Update progress bar range
        
        self.parent.drawTimeMarkers()
        
    def toggleZoomUI(self):
        """Toggle visibility of the zoom UI elements."""
        self.zoomHidden = not self.zoomHidden  # Toggle state

        if self.zoomHidden:
            self.zoomFrame.place_forget()  # Hide frame and all elements
        else:
            self.zoomFrame.place(relx=1, rely=1, anchor="se")  # Restore position
        
    
    def onZoomChange(self, newZoom):
        if not self.parent.isPlaying:
            self.zoomVar.set(1.0)  # Reset zoom level to default
            return
        """Handle zoom changes and update visi ble range"""
        self.updateZoomLevel(newZoom)
        # self.parent.updateProgressBar()
        
    def getVisibleChunks(self):
        """Return the current range of visible chunks based on the zoom level."""
        return self.currentChunksInView
    
class ProgressBarHandle:
    def __init__(self, canvas, parent, progressBarWidth, chunkDuration):
        self.canvas = canvas
        self.parent = parent
        self.progressBarWidth = progressBarWidth
        self.chunkDuration = chunkDuration
        self.handleWidth = 10
        self.handle = self.canvas.create_rectangle(0, 0, self.handleWidth, 20, fill="green", outline="white")
        self.currentSectionIndex = 0
    
    def move(self, x, rootSectionIndex):
        """Move the progress bar handle to the specified x position."""
        newState = "hidden" if self.parent.uiHidden else "normal"
        # print(f"Root section index: {rootSectionIndex}, current SI: {self.currentSectionIndex}")
        if rootSectionIndex == self.currentSectionIndex:
            self.canvas.itemconfig(self.handle, state=newState)
            x = max(0, min(x, self.progressBarWidth))  # Ensure x is within bounds
            self.canvas.coords(self.handle, x - self.handleWidth / 2, 0, x + self.handleWidth / 2, 20)
        else:
            self.canvas.itemconfig(self.handle, state="hidden")
        
    def jump(self, x, rootSectionIndex):
        if rootSectionIndex == self.currentSectionIndex:
            self.move(x, rootSectionIndex)
        else:
            self.parent.currentSectionIndex = self.currentSectionIndex
            self.parent.drawMarkers(self.currentSectionIndex)
            self.move(x, self.currentSectionIndex)
            
    def updatePosition(self, currentTimeMs, totalDurationMs, rootSectionIndex, visibleDuration):
        if self.currentSectionIndex != rootSectionIndex:
            # Hide the handle if the section index does not match
            self.canvas.itemconfig(self.handle, state="hidden")
        else:
            # Show the handle and update its position if the section index matches
            self.canvas.itemconfig(self.handle, state="normal")
            progressRatio = (currentTimeMs % (self.chunkDuration * self.progressBarWidth)) / totalDurationMs
            x = progressRatio * self.progressBarWidth
            self.move(x)
        
class ProgressBarNavigator:
    def __init__(self, canvas, totalDurationMs, chunkDuration, onNavigateCallBack):
        self.canvas = canvas
        self.totalDurationMs = totalDurationMs
        self.chunkDuration = chunkDuration
        self.onNavigateCallback = onNavigateCallBack
        self.navigatorWidth = 10
        self.navigator = self.canvas.create_rectangle(0, 25, self.navigatorWidth, 45, fill="green", outline="black")
        
        self.canvas.tag_bind(self.navigator, "<B1-Motion>", self.onDrag)
        
    def onDrag(self, event):
        X = max(0, min(event.x, self.canvas.winfo_width()))
        self.canvas.coords(self.navigator, X, 25, X + self.navigatorWidth, 45)
        
        progressRatio = X / self.canvas.winfo_width()
        newStartTime = int(progressRatio * self.totalDurationMs)
        self.onNavigateCallback(newStartTime)