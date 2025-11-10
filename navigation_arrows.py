class NavigationArrows:
    def __init__(self, canvas, parent, progressBarCanvas):
        """
        Initialize the navigation arrows.
        :param canvas: The canvas where the arrows are drawn.
        :param parent: The parent class that manages the progress bar and time markers.
        """
        self.canvas = canvas
        self.parent = parent
        self.progressBarCanvas = progressBarCanvas
        self.arrowSize = 20  # Size of the arrow
        self.arrowPadding = 10  # Padding around the progress bar
        self.arrows = {}
        
        self.canvas.bind_all('<KeyPress-k>', lambda e: self.navigateLeft())
        self.canvas.bind_all('<KeyPress-l>', lambda e: self.navigateRight())
    
    def createArrows(self):
        self.progressBarCanvas.update_idletasks()
        
        """Creates the left and right navigation arrows"""
        progressBarX = self.progressBarCanvas.winfo_x() if self.progressBarCanvas.winfo_x() >= 0 else 159
        progressBarY = self.progressBarCanvas.winfo_y() if self.progressBarCanvas.winfo_y() >= 0 else 640
        progressBarWidth = self.progressBarCanvas.winfo_width() 
        progressBarHeight = self.progressBarCanvas.winfo_height()
        print(f"Progress Bar X: {progressBarX}, Y: {progressBarY}, width: {progressBarWidth}, height: {progressBarHeight}")
        centerY = progressBarY + progressBarHeight // 2
        
        self.canvas.delete("nav_arrow")
         
        # Left arrow positions
        leftArrowBaseX = progressBarX - self.arrowPadding
        leftArrowTipX = leftArrowBaseX + self.arrowSize
        leftArrowYTop = centerY - self.arrowSize // 2
        leftArrowYBottom = centerY + self.arrowSize // 2

        self.arrows["left"] = self.canvas.create_polygon(
            leftArrowBaseX, centerY,
            leftArrowTipX, leftArrowYTop,
            leftArrowTipX, leftArrowYBottom,
            fill="darkgray", outline="black", tags="nav_arrow"
        )
        self.canvas.tag_bind(self.arrows["left"], "<Button-1>", self.navigateLeft)
        self.canvas.tag_bind(self.arrows["left"], "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind(self.arrows["left"], "<Leave>", lambda e: self.canvas.config(cursor=""))

        # Right arrow - based on progressBarCanvas's actual right edge
        progressBarRightX = progressBarX + progressBarWidth
        rightArrowBaseX = progressBarRightX + self.arrowPadding
        rightArrowTipX = rightArrowBaseX - self.arrowSize
        rightArrowYTop = centerY - self.arrowSize // 2
        rightArrowYBottom = centerY + self.arrowSize // 2

        self.arrows["right"] = self.canvas.create_polygon(
            rightArrowBaseX, centerY,
            rightArrowTipX, rightArrowYTop,
            rightArrowTipX, rightArrowYBottom,
            fill="darkgray", outline="black", tags="nav_arrow"
        )
        self.canvas.tag_bind(self.arrows["right"], "<Button-1>", self.navigateRight)
        self.canvas.tag_bind(self.arrows["right"], "<Enter>", lambda e: self.canvas.config(cursor="hand2"))
        self.canvas.tag_bind(self.arrows["right"], "<Leave>", lambda e: self.canvas.config(cursor=""))

        self.canvas.tag_raise("nav_arrow")

    def navigateLeft(self, event):
        """Handle clicking left arrow"""
        if self.parent.progressBarHandle.currentSectionIndex > 0:
            self.parent.progressBarHandle.currentSectionIndex -= 1
            self.parent.drawTimeMarkers()
            self.parent.drawMarkers(self.parent.progressBarHandle.currentSectionIndex)
            
    def navigateRight(self, event):
        """Handle clicking right arrow"""
        totalSections = len(self.parent.chunks) // self.parent.zoomManager.currentChunksInView
        if self.parent.currentSectionIndex < totalSections - 1:
            self.parent.progressBarHandle.currentSectionIndex += 1
            self.parent.drawTimeMarkers()
            self.parent.drawMarkers(self.parent.progressBarHandle.currentSectionIndex)
        
    def updateArrows(self, progressBarCanvas):
        """Update the position of the arrows if the progress bar canvas is resized."""
        self.progressBarCanvas = progressBarCanvas
        self.createArrows()