from PIL import ImageTk, Image, ImageDraw
import tkinter as tk 
import tkinter.font as tkFont

class TrackItem:
    def __init__(self, scale=40, position=(0, 0), sourceImages=None, animations=None, parent=None, trackMember=None,  type="image"):
        """
        Initialize a TrackItem instance.

        :param scale: Integer (0-1000) representing the scaled height of the image.
        :param position: Tuple (x, y) for the image's position, scaled relative to base dimensions.
        :param animations: Placeholder for animations, currently empty.
        """
        self.trackMember = trackMember
        self.scale = max(0, min(scale, 1000))
        self.position = self._scalePosition(position)
        self.originalImages = sourceImages if sourceImages is not None else {}  # Store original PIL.Image objects
        self.sourceImages = {
            key: ImageTk.PhotoImage(img) for key, img in self.originalImages.items()
        } 
        self.imageId = None
        self.currentImageKey = "dark"
        self.animations = animations if animations is not None else []
        self.timerValue = 0.0  # Timer starts at 0.0 seconds
        self.parent = parent
        
        parentScaleX = getattr(self.parent, "scaleX", 1.0) if self.parent else 1.0
        parentScaleY = getattr(self.parent, "scaleY", 1.0) if self.parent else 1.0

        self.fontSize = 25
        self.font = tkFont.Font(family="Digital-7", size=self.fontSize, weight="bold")
        self.progressBarXStart = None
        
        self.timerX = 0
        self.timerY = 0
        self.lastUpdateChunk = 0
        
        if type == "image":
            numChunks = len(self.parent.chunks)
            self.timeline = [0.0] * numChunks
            self.positionTimeline = [0.0] * numChunks
            if self.parent.labels != []:
                self.initializeTimeline()
            self.memberColor = self.parent.getMemberColor(self.trackMember)
            clearImage = self.chromaKeyImage(self.originalImages["dark"], self.memberColor)
            self.originalImages["clear"] = clearImage  
            self.sourceImages["clear"] = ImageTk.PhotoImage(clearImage)
            self.setTimerPosition(parentScaleX, parentScaleY)
            
            # Set bar color
            lightImage = self.originalImages["light"]
            self.progressBarColor = self._getBarColor(lightImage)
            self.currentSlotIndex = None
        
        
    @staticmethod
    def _scalePosition(position):
        """
        Scale the position values relative to a base width and height.

        :param position: Tuple (x, y) representing raw x and y coordinates.
        :return: Tuple (scaled_x, scaled_y) with scaled values.
        """
        baseWidth = 1920
        baseHeight = 1080
        x, y = position
        return x / baseWidth, y / baseHeight
    
    @staticmethod
    def _getBarColor(image):
        image = image.convert("RGBA")
        rgb = image.getpixel((0, 150))
        
        if len(rgb) == 4:
            rgb = rgb[:3]
        
        hexColor = "#{:02x}{:02x}{:02x}".format(*rgb)
        return hexColor
    
    def setMaxTime(self, maxTime):
        self.maxTime = maxTime
    
    def setTimerPosition(self, parentScaleX=None, parentScaleY=None):
        if parentScaleX is None:
            parentScaleX = getattr(self.parent, "scaleX", 1.0)
        if parentScaleY is None:
            parentScaleY = getattr(self.parent, "scaleY", 1.0)
        
        self.fontSize = int(40 * parentScaleX)
        self.font.configure(size=self.fontSize)
        self.xOffset = int(700 * parentScaleX)
        self.yOffset = int(5 * parentScaleY)
        
    def rescalePositionTimeline(self, scaleY):
        """
        Rescale Y positions from base timeline using the current vertical scale.
        Called when the window is resized.
        """
        if not hasattr(self, "basePositionTimeline"):
            return

        self.positionTimeline = [
            y * scaleY if y != 0.0 else 0.0
            for y in self.basePositionTimeline
        ]
        
    def chromaKeyImage(self, image, keyColor):
        """Apply chroma keying to an image"""
        if keyColor == "#aa9f00": keyColor = "ffff00"
        keyColorRGB = tuple(int(keyColor.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        
        image = image.convert("RGBA")
        data = image.getdata()
        newData = []
        for item in data:
            if item[:3] == keyColorRGB:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)
        
        image.putdata(newData)
        return image
    
    def animatePosition(self, startY, endY, startChunk, endChunk):
        self.animations.append({
            "startY": startY,
            "endY": endY,
            "startChunk": startChunk,
            "endChunk": endChunk
        })
        
    def updateAnimations(self, currentChunk):
        """
        Update the member's vertical animation for the current chunk.
        Animations operate entirely in BASE Y-COORDS (not scaled).
        Scaling is applied later in updateElementPositions().
        """
        for anim in self.animations[:]:
            if currentChunk < anim["startChunk"]:
                continue  # Animation hasn't started yet
            
            totalChunks = anim["endChunk"] - anim["startChunk"]
            if totalChunks <= 0:
                self.animations.remove(anim)
                continue
            
            currentProgressChunk = currentChunk - anim["startChunk"]
            progress = min(currentProgressChunk / totalChunks, 1.0)
            # Linear interpolation for smooth transition
            startY = anim["startY"] # baseY, unscaled
            endY = anim["endY"] # baseY, unscaled
            interpolatedY = startY + (endY - startY) * progress
            
            # x, _ = self.parent.canvas.coords(self.imageId)
           # self.parent.canvas.coords(self.imageId, x, interpolatedY)
            self.positionTimeline[currentChunk] = interpolatedY
            
            # Cleanup finished animation
            if progress >= 1.0:
                self.positionTimeline[currentChunk] = endY
                #self.parent.canvas.coords(self.imageId, x, interpolatedY)
                self.animations.remove(anim)
    
    def getSlotOffsetForSlot(self, slotIndex):
        """
        Computes the vertical offset of this member based on slot index and scale.
        Replaces heightOffset[0] and heightOffset[1].
        """
        imgHeight = self.sourceImages[self.currentImageKey].height()
        baseOffset = int(10 * self.parent.scaleY)  # or 0 if you keep baseY independent
        return imgHeight * slotIndex + baseOffset
    
    def getHeightOffset(self):
        """
        Compute vertical offset dynamically from scaling and slot index.
        This guarantees it always exists and matches the current UI scale.
        """
        imgHeight = self.sourceImages[self.currentImageKey].height() # scaled image
        slot = self.currentSlotIndex
        spacing = int(15 * self.parent.scaleY)
        
        return imgHeight * slot + spacing
        
    def getMostRecentY(self, currentChunk):
        for c in range(currentChunk, -1, -1):
            if self.positionTimeline[c] != 0.0:
                return self.positionTimeline[c]
        return self.parent.canvas.coords(self.imageId)[1]  # fallback
                
    def checkAndSwap(self, currentChunk):
        """
        Checks timeline conditions and triggers swap animations.
        """
        currentSlotIndex = self.parent.slotMap[self.trackMember]
        currentValue = self.timeline[currentChunk]

        baseChunkLength = 12
        membersToPass = []
            
         # Step 1: Determine which members need to be passed
        for slot in range(currentSlotIndex - 1, -1, -1):
            otherKey = next(name for name, idx in self.parent.slotMap.items() if idx == slot)
            otherTrackItem = self.parent.memberImages[otherKey]
            
            # Look slightly ahead
            futureIndex = min(currentChunk + 4, len(self.timeline) - 1)
            otherValue = otherTrackItem.timeline[futureIndex]

            if currentValue > otherValue:
                membersToPass.append(otherKey)
            else:
                break
        
        # If nothing to pass just lock to current BASE Y
        # DOUBLE CHECK THIS IN CASE
        if not membersToPass:
            baseY = self.getMostRecentY(currentChunk)
            self.positionTimeline[currentChunk] = baseY
            return
        
        # Step 2: Animate the current member upward to new Y
        newSlot = currentSlotIndex - len(membersToPass)
        newY = self.getSlotOffsetForSlot(newSlot)
        currentY = self.getMostRecentY(currentChunk)

        fullLength = baseChunkLength + 2 * (len(membersToPass) - 1)
        self.animatePosition(
            startY=currentY,
            endY=newY,
            startChunk=currentChunk,
            endChunk=currentChunk + fullLength
        )
        self.positionTimeline[currentChunk + baseChunkLength] = newY
        self.currentSlotIndex = newSlot
        self.parent.slotMap[self.trackMember] = newSlot
        
        # Step 3: Animate each passed member down with staggered start
        for idx, passedKey in enumerate(membersToPass):
            passedTrackItem = self.parent.memberImages[passedKey]
            passedStartChunk = currentChunk + (idx * 2)
            originalY = passedTrackItem.getMostRecentY(passedStartChunk)
            passedEndChunk = passedStartChunk + baseChunkLength
            newSlot = passedTrackItem.currentSlotIndex + 1
            targetY = passedTrackItem.getSlotOffsetForSlot(newSlot)

            passedTrackItem.animatePosition(
                startY=originalY,
                endY=targetY,
                startChunk=passedStartChunk,
                endChunk=passedEndChunk
            )
            passedTrackItem.positionTimeline[passedEndChunk] = targetY
            passedTrackItem.currentSlotIndex = newSlot
            self.parent.slotMap[passedKey] = newSlot
    
        # Step 5: Lock this chunk’s Y
        baseY = self.getMostRecentY(currentChunk)
        self.positionTimeline[currentChunk] = baseY
    
    def initializeTimeline(self):
        numChunks = len(self.parent.chunks) # Double check this
        self.timeline = [0.0] * numChunks
        
        rawRanges = []
        for label in self.parent.labels:
            member, start, end = label[:3]
            if member == self.trackMember:
                rawRanges.append((start, end))
        
        if not rawRanges:
            return
        
        # Sort and merge overlapping/adjacent ranges
        rawRanges.sort(key=lambda r: r[0]) # Sort by start
        mergedRanges = [rawRanges[0]]
        for start, end in rawRanges[1:]:
            lastStart, lastEnd = mergedRanges[-1]
            if start <= lastEnd + 1:
                # Overlapping or directly adjacent: merge
                mergedRanges[-1] = (lastStart, max(lastEnd, end))
            else:
                mergedRanges.append((start, end))
        
        # 👉 last chunk where this member sings
        lastEnd = mergedRanges[-1][1]
        self.lastUpdateChunk = min(lastEnd, numChunks - 1)
    
        activeChunks = 0
        lastTime = 0.0
        rangeIdx = 0
        currentRange = mergedRanges[rangeIdx]
        
        # print("All labels:", self.parent.labels)
        for chunkIndex in range(numChunks):
            # still within some active range?
            while rangeIdx < len(mergedRanges) and chunkIndex > currentRange[1]:
                rangeIdx += 1
                if rangeIdx < len(mergedRanges):
                    currentRange = mergedRanges[rangeIdx]
                else:
                    currentRange = None
                    break
                
            if currentRange and currentRange[0] <= chunkIndex <= currentRange[1]:
                # inside active range: incremeent time
                activeChunks += 1
                lastTime = activeChunks * (self.parent.chunk_duration / 1000.0)
                self.timeline[chunkIndex] = lastTime
            else:
                self.timeline[chunkIndex] = lastTime
              
    def setImageId(self, imageId):
        self.imageId = imageId
        
    def getTimerX(self):
        if self.timerX:
            return self.timerX

    def setScale(self, scale):
        """
        Update the scale of the TrackItem.

        :param scale: Integer (0-1000) for the new scale value.
        """
        self.scale = max(0, min(scale, 1000))
        
    def setSourceImage(self, sourceImage):
        """
        Update the source image of the TrackItem.

        :param sourceImage: The new image this TrackItem should represent.
        """
        self.sourceImages = sourceImage
        
    def switchImage(self, imageKey):
        """
        Switch the current image being displayed by the TrackItem.

        :param imageKey: Key to select the new image from sourceImages.
        """
        if imageKey in self.sourceImages:
            self.currentImageKey = imageKey
        else:
            raise ValueError(f"Image key '{imageKey}' not found in sourceImages.")

    def getCurrentImage(self):
        return self.sourceImages[self.currentImageKey]
    
    def __repr__(self):
        """
        Return a string representation of the TrackItem instance.
        """
        return f"TrackItem(scale={self.scale}, position={self.position}, animations={self.animations})"
    
    def resizeImages(self, scale):
        """
        Resize all images ('dark' and 'light') to the new scale.
        :param scale: Scale factor (0-1000, where 100 is the normal size).
        """
        for key in self.originalImages:
            originalImage = self.originalImages[key]
            baseWidth, baseHeight = originalImage.size
            newWidth = int(baseWidth * (scale / 100))
            newHeight = int(baseHeight * (scale / 100))
            resizedImage = originalImage.resize((newWidth, newHeight))
            self.sourceImages[key] = ImageTk.PhotoImage(resizedImage)

        self.updateProgressBarGeometry()
        self.setTimerPosition()
    
    def updateTime(self):
        """
        Update the timer by one chunk, based on the parent's chunk duration (40ms by default).
        """
        if self.parent and hasattr(self.parent, "chunk_duration"):
            self.timerValue += self.parent.chunk_duration / 1000.0
         
    def saveLastTime(self, chunkIndex):
        self.timeline[chunkIndex] = self.timeline[chunkIndex - 1]
                 
    def drawTextForCurrentChunk(self, chunkIndex):
        """
        Draw the timer text at the appropriate position.
        :param draw: ImageDraw instance for drawing text
        """
        if chunkIndex == len(self.parent.chunks): return
        self.setPositionFromTimeline(chunkIndex)
        
        safeIndex = min(chunkIndex, len(self.timeline) - 1)
        value = self.timeline[safeIndex]
        
        timerText = f"{value:.1f}" if value > 0.0 else ''
        # print(f"Timer text: {timerText}")
        
        x, y = self.parent.canvas.coords(self.imageId)
        # Update timer position to align the top-right corner
        self.timerX = x + self.xOffset
        self.timerY = y + self.yOffset
        
        # print(f"Timer x: {self.timerX}, Timer y: {self.timerY}")
        
        if hasattr(self, "timerTextId") and self.timerTextId:
            self.parent.canvas.delete(self.timerTextId)
        
        self.timerTextId = self.parent.canvas.create_text(
            round(self.timerX),
            round(self.timerY),
            text=timerText,
            font=self.font,
            fill="white",
            anchor="ne"  # Anchor the text to the right (east)
        )
    
    def updateProgressBarGeometry(self):
        """
        Update progress bar height based on the current (scaled) image height.
        This makes the bar thickness scale with the member image.
        """
        imgHeight = self.sourceImages[self.currentImageKey].height()
        # e.g. 8% of image height, tweak factor as you like
        self.progressBarHeight = max(2, int(0.15 * imgHeight))
            
    def setPositionFromTimeline(self, currentChunk):
        """
        Sets the position of the TrackItem based on its positionTimeline for the given chunk.
        """
        if 0 <= currentChunk < len(self.positionTimeline):
            x, _ = self.parent.canvas.coords(self.imageId)
            self.parent.canvas.coords(self.imageId, x, self.positionTimeline[currentChunk])
            self.updateProgressBar(currentChunk, self.maxTime)
            
    def updateAndDrawTimer(self, chunkIndex):
        """
        Update the timer and draw it if the app is not paused.
        """
    
        if not self.parent.isPaused:
            self.drawTextForCurrentChunk(chunkIndex)
    
    def getProgressBarY(self):
        """
        Y coordinate where the progress bar should be drawn, based on the image’s
        position and scaled height.
        """
        _, imgY = self.parent.canvas.coords(self.imageId)
        imgHeight = self.sourceImages[self.currentImageKey].height()
        return imgY + int(0.7 * imgHeight)
    
    def findStartX(self):
        if self.progressBarXStart is None:
            darkImage = self.originalImages["dark"]
            memberColorRGBA = tuple(int(self.memberColor.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (255,)
            pixels = darkImage.load()
            width, _ = darkImage.size
            
            for x in range(width - 1, -1, -1):  # Iterate from the rightmost to the leftmost pixel
                pixel = pixels[x, 0]  # Access the first row only
                if pixel[:3] == memberColorRGBA[:3] and pixel[3] != 0:  # Check for member color and non-transparent alpha
                    self.progressBarXStart = x * (self.scale / 100 / 2)
                    return self.progressBarXStart  # Return the x-coordinate of the last matching pixel
            return 0  # Default to 0 if no match is found
        else:
            return self.progressBarXStart
    
    def createRoundedRectangleImage(self, width, height, color, radius):
        """Create a rounded rectangle image with Pillow."""
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Draw the rounded rectangle
        draw.rounded_rectangle(
            (0, 0, width, height), radius=radius, fill=color
        )

        # Convert the Pillow image to a Tkinter PhotoImage
        return ImageTk.PhotoImage(image)

    def initializeProgressBar(self):
        """Initialize the progress bar."""
        self.updateProgressBarGeometry()
        
        self.progressBarImage = self.createRoundedRectangleImage(
            0, self.progressBarHeight, self.progressBarColor, radius=self.progressBarHeight // 2
        )
        
        self.progressBarCanvasImage = self.parent.canvas.create_image(
            0, self.getProgressBarY(), anchor="nw", image=self.progressBarImage
        )
        
        self.parent.canvas.tag_lower(self.progressBarCanvasImage, self.imageId)
        
    def updateProgressBar(self, currentChunk, maxTime):
        currentTime = self.timeline[currentChunk]
        
        if maxTime == 0 or currentTime == 0.0:
            return
         
        progress = currentTime / maxTime
        xStart = 1920 * self.parent.scaleX * 1 / 16 - self.progressBarHeight // 2
        xEnd = min(xStart + progress * (self.timerX - xStart), self.timerX) # Update later
        # print(f"X start: {xStart}, xEnd: {xEnd}")
        
        barWidth = int(xEnd - xStart) if xEnd != 0 else 0
        y = self.getProgressBarY()
        
        color = self.progressBarColor if self.currentImageKey == 'light' else "#ffffff"
        # Update rectangle for the main bar
        self.progressBarImage = self.createRoundedRectangleImage(
            int(barWidth), self.progressBarHeight, color, radius=self.progressBarHeight // 2
        )
        
        self.parent.canvas.itemconfig(self.progressBarCanvasImage, image=self.progressBarImage)
        
        self.parent.canvas.coords(self.progressBarCanvasImage, xStart, y)