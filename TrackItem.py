from PIL import ImageTk, Image, ImageDraw
import tkinter.font as tkFont

class TrackItem:
    def __init__(self, scale=40, sourceImages=None, animations=None, parent=None, trackMember=None,  type="image"):
        """
        Initialize a TrackItem instance.

        :param scale: Integer (0-1000) representing the scaled height of the image.
        :param position: Tuple (x, y) for the image's position, scaled relative to base dimensions.
        :param animations: Placeholder for animations, currently empty.
        """
        self.trackMember = trackMember
        self.scale = max(0, min(scale, 1000))
        self.originalImages = sourceImages if sourceImages is not None else {}  # Store original PIL.Image objects
        self.sourceImages = {
            key: ImageTk.PhotoImage(img) for key, img in self.originalImages.items()
        } 
        self.imageId = None
        self.currentImageKey = "dark"
        self.animations = animations if animations is not None else []
        self.timerValue = 0.0  # Timer starts at 0.0 seconds
        self.parent = parent
        self.currentRole = "none" # "main" | "harmony" | "adlib" | "none"
        
        parentScaleX = getattr(self.parent, "scaleX", 1.0) if self.parent else 1.0
        parentScaleY = getattr(self.parent, "scaleY", 1.0) if self.parent else 1.0

        self.fontSize = 25
        self.font = tkFont.Font(family="Digital-7", size=self.fontSize, weight="bold")
        self.progressBarXStart = None
        
        self.timerX = 0
        self.timerY = 0
        self.maxTime = 0.0
        self.lastUpdateChunk = 0
        
        if type == "image":
            numChunks = len(self.parent.chunks)
            self.timeline = [0.0] * numChunks
            self.positionTimeline = [None] * numChunks
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
        self.xOffset = int(720 * parentScaleX)
        self.yOffset = int(5 * parentScaleY)
        
    def rescalePositionTimeline(self, scaleY):
        if not hasattr(self, "basePositionTimeline"):
            return

        baseH = self.parent.slotHeightBase
        pixH  = self.parent.slotHeightPx  # computed once in parent on resize

        if baseH <= 0 or pixH <= 0:
            return
    
        self.positionTimeline = [
            (int(round((y / baseH) * pixH)) if y is not None else None)
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
        Scaling is applied later in updateElementResizing().
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
        if hasattr(self.parent, "slotBaseYs") and 0 <= slotIndex < len(self.parent.slotBaseYs):
            return self.parent.slotBaseYs[slotIndex]
        return self.parent.slotHeightBase * slotIndex
        
    def getMostRecentY(self, currentChunk):
        for c in range(currentChunk, -1, -1):
            y = self.positionTimeline[c]
            if y is not None:
                return y
        return self.parent.canvas.coords(self.imageId)[1]
                
    def checkAndSwap(self, currentChunk):
        """
        Checks timeline conditions and triggers swap animations.
        """
        currentSlotIndex = self.parent.slotMap[self.trackMember]
        currentValue = self.timeline[currentChunk]
        
        baseChunkLength = 20
        membersToPass = []
        
        # Shanpshot the order at start of this chunk
        originalSlotMap = dict(self.parent.slotMap)
        originalTrackSlots = {
            name: track.currentSlotIndex
            for name, track in self.parent.memberImages.items()
        }
            
         # Step 1: Determine which members need to be passed
        for slot in range(currentSlotIndex - 1, -1, -1):
            otherKey = next(name for name, idx in originalSlotMap.items() if idx == slot)
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
        
        # Step 2: compute all target slots FIRST
        targetSlots = {}

        newSelfSlot = originalTrackSlots[self.trackMember] - len(membersToPass)
        targetSlots[self.trackMember] = newSelfSlot

        for passedKey in membersToPass:
            targetSlots[passedKey] = originalTrackSlots[passedKey] + 1

        # Step 3: animate self
        currentY = self.getMostRecentY(currentChunk)
        newY = self.getSlotOffsetForSlot(targetSlots[self.trackMember])

        fullLength = baseChunkLength + 2 * (len(membersToPass) - 1)
        endChunk = min(currentChunk + fullLength, len(self.positionTimeline) - 1)
        lockChunk = min(currentChunk + baseChunkLength, len(self.positionTimeline) - 1)

        self.animatePosition(
            startY=currentY,
            endY=newY,
            startChunk=currentChunk,
            endChunk=endChunk
        )
        self.positionTimeline[lockChunk] = newY

        # Step 4: animate passed members
        for idx, passedKey in enumerate(membersToPass):
            passedTrackItem = self.parent.memberImages[passedKey]
            passedStartChunk = min(currentChunk + (idx * 2), len(passedTrackItem.positionTimeline) - 1)
            passedEndChunk = min(passedStartChunk + baseChunkLength, len(passedTrackItem.positionTimeline) - 1)

            originalY = passedTrackItem.getMostRecentY(passedStartChunk)
            targetY = passedTrackItem.getSlotOffsetForSlot(targetSlots[passedKey])

            passedTrackItem.animatePosition(
                startY=originalY,
                endY=targetY,
                startChunk=passedStartChunk,
                endChunk=passedEndChunk
            )
            passedTrackItem.positionTimeline[passedEndChunk] = targetY

        # Step 5: NOW commit slot changes
        for memberName, newSlot in targetSlots.items():
            track = self.parent.memberImages[memberName]
            track.currentSlotIndex = newSlot
            self.parent.slotMap[memberName] = newSlot

        self.positionTimeline[currentChunk] = self.getMostRecentY(currentChunk)
    
    def initializeTimeline(self, includeBacking: bool = True):
        numChunks = len(self.parent.chunks)
        self.timeline = [0.0] * numChunks
        
        rawRanges = []
        for label in self.parent.labels:
            member, start, end, isBacking, isAdlib = label
            
            if member != self.trackMember:
                continue
        
            if (not includeBacking) and (isBacking and not isAdlib):
                continue
            
            rawRanges.append((start, end))
        
        if not rawRanges:
            self.lastUpdateChunk = 0
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
            newHeight = int(round(baseHeight * (scale / 100.0)))
            newWidth = int(round(baseWidth * (scale / 100.0)))
            resizedImage = originalImage.resize((newWidth, newHeight))
            self.sourceImages[key] = ImageTk.PhotoImage(resizedImage)

        if self.imageId is not None:
            self.initializeProgressBar()
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
    
    def _computeBaseBarStartX(self):
        """
        Find the rightmost x in the TOP row that matches member color (non-transparent),
        in ORIGINAL image pixel coordinates.
        Cached as self._barStartXBasePx.
        """
        if hasattr(self, "_barStartXBasePx") and self._barStartXBasePx is not None:
            return self._barStartXBasePx

        darkImage = self.originalImages["dark"].convert("RGBA")
        pixels = darkImage.load()
        width, _ = darkImage.size

        keyRgb = tuple(int(self.memberColor.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

        # Scan right -> left on row y=0 (your current behavior)
        for x in range(width - 1, -1, -1):
            r, g, b, a = pixels[x, 0]
            if a != 0 and (r, g, b) == keyRgb:
                self._barStartXBasePx = x
                return x

        self._barStartXBasePx = 0
        return 0
    
    def getProgressBarXStartCanvas(self):
        """
        Returns the xStart in CANVAS coordinates, based on current displayed image size and position.
        """
        baseX = self._computeBaseBarStartX()

        # Canvas position of the image (top-left because you use anchor="nw" in your create_image)
        imgX, _ = self.parent.canvas.coords(self.imageId)

        # Displayed image width (Tk PhotoImage width)
        displayedW = self.sourceImages[self.currentImageKey].width()

        baseW = self.originalImages["dark"].size[0]
        if baseW <= 0:
            return imgX

        # Map proportionally
        xStart = imgX + (baseX / baseW) * displayedW

        # If you want a small inset so rounded cap isn't clipped:
        xStart -= self.progressBarHeight // 4

        return xStart
    
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
    
    def _shadeHex(self, hexColor: str, amount: float) -> str:
        """
        amount in [-1.0 .. 1.0]
        < 0 → darken
        > 0 → lighten
        """
        r = int(hexColor[1:3], 16)
        g = int(hexColor[3:5], 16)
        b = int(hexColor[5:7], 16)

        if amount > 0:
            r += (255 - r) * amount
            g += (255 - g) * amount
            b += (255 - b) * amount
        else:
            r *= (1 + amount)
            g *= (1 + amount)
            b *= (1 + amount)

        return "#{:02x}{:02x}{:02x}".format(
            int(max(0, min(255, r))),
            int(max(0, min(255, g))),
            int(max(0, min(255, b))),
        )
    
    def getColor(self) -> str:
        role = getattr(self, "currentRole", "none")
        base = self.progressBarColor  # always hex

        if role == "main":
            return base
        if role == "harmony":
            return self._shadeHex(base, -0.35)  # darker
        if role == "adlib":
            return self._shadeHex(base, +0.35)  # lighter
        return "#ffffff"
       
    def updateProgressBar(self, currentChunk, maxTime):
        currentTime = self.timeline[currentChunk]
        
        if maxTime == 0:
            return
        
        if currentTime == 0.0:
            self.parent.canvas.itemconfig(self.progressBarCanvasImage, state="hidden")
            return
         
        self.parent.canvas.itemconfig(self.progressBarCanvasImage, state="normal")
        progress = currentTime / maxTime
        xStart = self.getProgressBarXStartCanvas()
        xEnd = min(xStart + progress * (self.timerX - xStart), self.timerX) # Update later
        # print(f"X start: {xStart}, xEnd: {xEnd}")
        
        barWidth = int(xEnd - xStart) if xEnd != 0 else 0
        y = self.getProgressBarY()
        
        color = self.getColor()
        # Update rectangle for the main bar
        self.progressBarImage = self.createRoundedRectangleImage(
            int(barWidth), 
            self.progressBarHeight, 
            color, radius=self.progressBarHeight // 2
        )
        
        self.parent.canvas.itemconfig(self.progressBarCanvasImage, image=self.progressBarImage)
        
        self.parent.canvas.coords(self.progressBarCanvasImage, xStart, y)