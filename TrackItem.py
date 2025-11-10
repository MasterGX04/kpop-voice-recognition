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
        
        self.fontSize = 25
        self.font = tkFont.Font(family="Digital-7", size=self.fontSize, weight="bold")
        self.progressBarXStart = None
        
        self.timerX = 0
        self.timerY = 0
        self.heightOffset = None
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
            self.setTimerPosition()
            
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
    
    def setTimerPosition(self):
        self.fontSize = int(40 * self.parent.scaleY)
        self.font.configure(size=self.fontSize)
        self.xOffset = int(700 * self.parent.scaleX)
        self.yOffset = int(15 * self.parent.scaleY)
        
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
            interpolatedY = anim["startY"] + (anim["endY"] - anim["startY"]) * progress
            
            # x, _ = self.parent.canvas.coords(self.imageId)
           # self.parent.canvas.coords(self.imageId, x, interpolatedY)
            self.positionTimeline[currentChunk] = interpolatedY
            
            if progress == 1.0:
                interpolatedY = anim["endY"]  # Lock at final position
                #self.parent.canvas.coords(self.imageId, x, interpolatedY)
                self.positionTimeline[currentChunk] = interpolatedY
                
    def initializeHeightOffset(self, keys):
        firstKey, secondKey = keys[0], keys[1]
        firstY = self.parent.canvas.coords(self.parent.memberImages[firstKey].imageId)[1]
        secondY = self.parent.canvas.coords(self.parent.memberImages[secondKey].imageId)[1]
        self.heightOffset = (secondY - firstY, firstY)  # (scaledHeight, yOffset)
        self.initializeProgressBar()
        
    def getMostRecentY(self, currentChunk):
        for c in range(currentChunk, -1, -1):
            if self.positionTimeline[c] != 0.0:
                return self.positionTimeline[c]
        return self.parent.canvas.coords(self.imageId)[1]  # fallback
                
    def checkAndSwap(self, currentChunk):
        """
        Checks timeline conditions and triggers swap animations.
        """
        keys = list(self.parent.memberImages.keys())
        if not hasattr(self, 'heightOffset') or not self.heightOffset:
            self.initializeHeightOffset(keys)
        
        currentSlotIndex = self.parent.slotMap[self.trackMember]
        currentValue = self.timeline[currentChunk]
        scaledHeight, yOffset = self.heightOffset

        baseChunkLength = 12
        membersToPass = []
            
         # Step 1: Determine which members need to be passed
        for slot in range(currentSlotIndex - 1, -1, -1):
            otherKey = next(name for name, idx in self.parent.slotMap.items() if idx == slot)
            otherTrackItem = self.parent.memberImages[otherKey]
            otherValue = otherTrackItem.timeline[min(currentChunk + 4, len(self.timeline) - 1)]

            if currentValue > otherValue:
                membersToPass.append(otherKey)
            else:
                break
        
        if not membersToPass:
            _, y = self.parent.canvas.coords(self.imageId)
            self.positionTimeline[currentChunk] = y
            return
        
        # Step 2: Animate the current member upward to new Y
        newSlot = currentSlotIndex - len(membersToPass)
        newY = newSlot * scaledHeight + yOffset
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
            targetY = newSlot * scaledHeight + yOffset

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
        _, y = self.parent.canvas.coords(self.imageId)
        self.positionTimeline[currentChunk] = y
    
    def initializeTimeline(self):
        self.timeline = [0.0] * len(self.parent.chunks)
        activeChunks = 0
        labelRanges = [] 
        # print("All labels:", self.parent.labels)
        for label in self.parent.labels:
            member, start, end = label[:3]
            if member == self.trackMember:
                labelRanges.append((start, end))
                for chunkIndex in range(start, end + 1):
                    activeChunks += 1
                    self.timeline[chunkIndex] = activeChunks * (self.parent.chunk_duration / 1000)
            
        totalChunks = len(self.timeline)
        currentRangeIndex = 0
        lastTime = 0.0
        
        for chunkIndex in range(totalChunks):
        # If within the current label range, skip (already calculated)
            if (currentRangeIndex < len(labelRanges) and labelRanges[currentRangeIndex][0] <= chunkIndex <= labelRanges[currentRangeIndex][1]):
                if chunkIndex == labelRanges[currentRangeIndex][1]:
                    lastTime = self.timeline[chunkIndex]
                    currentRangeIndex += 1  # Move to the next range
                    self.lastUpdateChunk = chunkIndex
                continue

            # Fill in time for chunks outside the active ranges
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
        self.timerY = y - self.yOffset
        
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
    
    def getProgressY(self):
        _, y = self.parent.canvas.coords(self.imageId)
        return  y + 0.7 * self.heightOffset[0]
    
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
        self.progressBarHeight = int(15 * self.parent.scaleY)
        y = self.getProgressY()
        self.progressBarImage = self.createRoundedRectangleImage(
            0, self.progressBarHeight, self.progressBarColor, radius=self.progressBarHeight // 2
        )
        
        self.progressBarCanvasImage = self.parent.canvas.create_image(
            0, self.getProgressY(), anchor="nw", image=self.progressBarImage
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
        y = self.getProgressY()
        
        color = self.progressBarColor if self.currentImageKey == 'light' else "#ffffff"
        # Update rectangle for the main bar
        self.progressBarImage = self.createRoundedRectangleImage(
            int(barWidth), self.progressBarHeight, color, radius=self.progressBarHeight // 2
        )
        
        self.parent.canvas.itemconfig(self.progressBarCanvasImage, image=self.progressBarImage)
        
        self.parent.canvas.coords(self.progressBarCanvasImage, xStart, y)