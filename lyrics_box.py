from PIL import Image, ImageTk
import tkinter as tk
import os
from pathlib import Path
from matplotlib import font_manager
from audio_processing import getSongsFromSameAlbum

class LyricBox:
    def __init__(self, canvas, parent, memberNames, circleImages, koreanLyric, romanization, englishTrans, startChunk, language, isAdLib=False, adLibDuration=0):
        self.canvas = canvas
        self.parent = parent
        self.memberNames = memberNames
        self.koreanLyric = koreanLyric
        self.romanization = romanization
        self.englishTrans = englishTrans
        self.startChunk = startChunk
        self.isAdLib = isAdLib  # New flag for ad-libs
        self.adLibDuration = adLibDuration 
        
        self.photoY = 0
        
        self.BASE_W = 1920
        self.BASE_H = 1080
        
        self.circleImages = circleImages
        self.memberPhotos = self.resizeMemberImages(circleImages)
        self._animCursor = 0
        self._heldBaseY = None
        self.textItems = []  # Store canvas item IDs
        self.totalHeight = 0  # To calculate total height of the lyric box
        self.isVisible = False
        self.language = language
        
        self.photoItemIds = []
        self._photoRefByItemId = {}
        
        # Assign colors for each member if multiple
        if isinstance(memberNames, list):
            self.memberColors = [parent.getMemberColor(name, forLyrics=True) for name in memberNames]
        else:
            self.memberColors = [parent.getMemberColor(memberNames, forLyrics=True)]
        
        self.startY = -self.totalHeight
        self.endY = self._pxY(5)
        
        # Set fonts [SCALE IT LATER]
        self.fontSize = 18
        fontFamily = "Pretendard Variable"
        self.font = (fontFamily, self.fontSize, "normal")
        self.boldFont = (fontFamily, self.fontSize, "bold")
        self.englishFont = (fontFamily, self.fontSize + 2)

        self.animations = []
        
        self.lyricsPadding = self._pxY(5)
        self.addLyricDuration = 9
        
        if isAdLib:
            self.createAdLibDisplay()
        else:
            self.createLyricDisplay()
            self.initializeLyricPosition()
    
    def resizeMemberImages(self, memberPhotos):
        """
        Resize already-loaded member photos to match the current canvas scale.

        Assumes memberPhotos is a list of PIL.Image objects (typically the 'circle' images),
        already selected/saved in the correct order (so indices stay consistent).

        Base design size: 100px at 1920x1080, i.e.
        - width ratio  = 100/1920 of the design width
        - height ratio = 100/1080 of the design height
        """
        if not memberPhotos:
            return []

        # Ensure list for consistency
        if not isinstance(memberPhotos, (list, tuple)):
            memberPhotos = [memberPhotos]

        # Current scale from parent (computed in onCanvasResize)
        scaleY = getattr(self.parent, "scaleY", 1.0)

        # Base (design) size derived from ratios
        # (Equivalent to "100px in a 1920x1080 design")
        baseImgH = int((150 / 1080) * 1080)  # = 100, kept explicit for sanity

        # Scale into actual canvas pixels
        imgH = max(1, int(baseImgH * scaleY))
        
        resizedTkPhotos = []
        for img in memberPhotos:
            try:
                # Support either PIL.Image or ImageTk.PhotoImage being passed in
                pilImg = img
                if isinstance(img, ImageTk.PhotoImage):
                    # If you ever pass PhotoImage by accident, you can't reliably get PIL back.
                    # Treat it as unsupported to avoid silent wrong behavior.
                    raise TypeError("resizeMemberImages expects PIL.Image objects, not ImageTk.PhotoImage.")

                resizedPil = pilImg.resize((imgH, imgH), Image.LANCZOS)
                resizedTkPhotos.append(ImageTk.PhotoImage(resizedPil))
            except Exception as e:
                print(f"Warning: Failed to resize member image: {e}")

        return resizedTkPhotos
    
    def _baseAnchor(self):
        # you asked specifically for the 760/1920 ratio
        baseX = self.parent.targetLyricsX
        baseY = int((10  / 1080) * self.BASE_H)   # keep your old top padding but ratio-based
        return baseX, baseY
    
    def _anchorXY(self):
        baseX, baseY = self._baseAnchor()

        scaleX = getattr(self.parent, "scaleX", 1.0)
        scaleY = getattr(self.parent, "scaleY", 1.0)

        offX = getattr(self.parent, "viewportOffsetX", 0)
        offY = getattr(self.parent, "viewportOffsetY", 0)

        x = offX + int(baseX * scaleX)
        y = offY + int(baseY * scaleY)
        return x, y
    
    def _storeOffset(self, itemId, itemX, itemY):
        # store RELATIVE offsets (dx, dy)
        self.textItemOffsets.append((itemId, itemX - self.originX, itemY - self.originY))
    
    def initializeLyricPosition(self):
        """
        Rule:
        - This lyric ALWAYS gets an "enter" animation from above -> endY (base units) over addLyricDuration chunks.
        - After that, it HOLDS at endY until another lyric adds a push-down animation later.
        - When this lyric appears, it pushes DOWN any lyrics that are currently stacked/active at this chunk.
        """
        startChunk = self.startChunk
        endChunk = self.startChunk + self.addLyricDuration
        
        # Keep this as BASE units (design pixels). Canvas conversion happens in render: canvasY = offY + baseY*scaleY.
        endYBase = 5

        scaleY = getattr(self.parent, "scaleY", 1.0)
        if scaleY <= 0:
            scaleY = 1.0

        # totalHeight is in CANVAS pixels; convert to BASE units for animation math
        totalHeightBase = self.totalHeight / scaleY
        
        # 1) Enter animation for THIS lyric (from above to endYBase)
        self.animatePosition(
            startY=-totalHeightBase,
            endY=endYBase,
            startChunk=startChunk,
            endChunk=endChunk
        )
        
        # 2) Compute how much vertical space THIS lyric occupies (in CANVAS px), then convert to BASE delta.
        # This delta is what we apply to existing lyrics as a push-down animation.
        numMembers = 1 if isinstance(self.memberNames, str) else len(self.memberNames)
        
        photoColumnHeightPx = 0
        if self.memberPhotos:
            photoHeight = self.memberPhotos[0].height()
            photoOverlapY = self._pxY(10)
            photoColumnHeightPx = (photoHeight * numMembers) - (photoOverlapY * max(0, numMembers - 1))
            
        extraPadY = self._pxY(10)
        
        additionalCanvasHeightPx = max(
            photoColumnHeightPx + self.lyricsPadding + extraPadY,
            self.totalHeight + self.lyricsPadding
        )
        
        pushDownBaseY = additionalCanvasHeightPx / scaleY
        # print(f"Push down base Y: {pushDownBaseY}")
        if pushDownBaseY <= 0:
            return
        
        # 3) Push down existing lyrics that are active at this chunk.
        # Preferred: use a parent-maintained active set (fast).
        existingLyricBoxes = []
        if hasattr(self.parent, "getActiveLyricBoxesAtChunk"):
            existingLyricBoxes = self.parent.getActiveLyricBoxesAtChunk(startChunk)
        elif hasattr(self.parent, "activeLyricIds"):
            # activeLyricIds is a set of lyric IDs (you can use startChunk IDs)
            for lid in list(self.parent.activeLyricIds):
                if lid == self.startChunk:
                    continue
                lb = self.parent.lyrics.get(lid)
                if lb:
                    existingLyricBoxes.append(lb)
                    
        # For each existing lyric, add a NEW animation segment that shifts it down by pushDownBaseY.
        for lb in existingLyricBoxes:
            if lb is self:
                continue

            # IMPORTANT: we want the baseY at THIS chunk, not whatever was cached globally.
            # So each lyric box needs a method like getBaseYAt(chunk) that returns its current held/animated baseY.
            if hasattr(lb, "getBaseYAt"):
                oldBaseY = lb.getBaseYAt(startChunk)
                if oldBaseY is None:
                    continue
            else:
                # Worst-case fallback: assume it is at the resting endYBase.
                oldBaseY = endYBase

            lb.animatePosition(
                startY=oldBaseY,
                endY=oldBaseY + pushDownBaseY,
                startChunk=startChunk,          # push starts now
                endChunk=endChunk              # and completes over the same addLyricDuration
            )
        
        self.animations.sort(key=lambda a: a["startChunk"])
    
    def resetAnimCursor(self):
        self._animCursor = 0
        self._heldBaseY = None     
        self._lastChunkIndex = None
     
    def createAdLibDisplay(self):
        """Creates a visual representation of an ad-lib lyric (top-right aligned)."""
        self.textItems = []
        self.textItemOffsets = []
        self.totalHeight = 0

        # Scaled spacing (no static pixels)
        padY = self._pxY(5)
        marginX = self._pxX(10)
        marginY = self._pxY(10)

        # --- Anchor (top-right) in design space ---
        # Use the fitted viewport mapping like everything else:
        # designX = BASE_W - margin, designY = some margin
        # then map to canvas using scale + viewportOffset
        scaleX = getattr(self.parent, "scaleX", 1.0)
        scaleY = getattr(self.parent, "scaleY", 1.0)
        offX = getattr(self.parent, "viewportOffsetX", 0)
        offY = getattr(self.parent, "viewportOffsetY", 0)

        baseW = getattr(self.parent, "baseWidth", 1920)
        baseH = getattr(self.parent, "baseHeight", 1080)

        # top-right anchor in BASE coords
        baseAnchorX = baseW - 10  # "10px" in design space; marginX handles scaling in canvas space
        baseAnchorY = 10

        # Convert base anchor to canvas coords, then apply additional scaled margins
        anchorX = offX + int(baseAnchorX * scaleX) - marginX
        anchorY = offY + int(baseAnchorY * scaleY) + marginY

        # Store origin for relative offsets (ad-lib box origin is its anchor point)
        self.originX = anchorX
        self.originY = anchorY

        # --- Create text (right-aligned) ---
        # Pick the displayed string
        if self.language == "Korean":
            displayText = self.koreanLyric
        else:
            displayText = self.englishTrans

        textId = self.canvas.create_text(
            anchorX, anchorY,
            text=displayText,
            font=self.englishFont,
            fill=self.memberColors[0],
            anchor="ne",
            state="normal"
        )
        self.textItems.append(textId)
        self._storeOffset(textId, anchorX, anchorY)

        textHeight = self._getItemHeight(textId)
        self.totalHeight = textHeight + padY

        # --- Animate in BASE Y units for resize safety ---
        # Convert canvas measurements back to base units for the animation system.
        totalHeightBase = self.totalHeight / scaleY

        # Start from just below the viewport bottom (base coords)
        # viewport bottom in canvas is: offY + newHeight. Converting to base gives ~baseH.
        startYBase = baseH + 20  # small cushion in base units
        midYBase = baseH / 2
        endYBase = (baseAnchorY + totalHeightBase)  # settle near the top margin area

        self.animateAdLibPosition(
            startY=startYBase,
            midY=midYBase,
            endY=endYBase,
            startChunk=self.startChunk,
            duration=self.adLibDuration
        )
    
    def animateAdLibPosition(self, startY, midY, endY, startChunk, duration):
        """
        Animate ad-lib in BASE Y coords: startY -> midY -> endY.
        Stores frames in BASE units for resize safety.
        """
        fadeDuration = int(duration) if duration and duration > 0 else 10
        half = max(1, fadeDuration // 2)  # avoid divide-by-zero

        anim = {
            "startChunk": startChunk,
            "endChunk": startChunk + fadeDuration,
            "frames": {}
        }
        self.animations.append(anim)

        # Phase 1: start -> mid
        for chunk in range(startChunk, startChunk + half):
            progress = (chunk - startChunk) / half
            baseY = startY + progress * (midY - startY)
            anim["frames"][chunk] = baseY

        # Phase 2: mid -> end
        for chunk in range(startChunk + half, startChunk + fadeDuration + 1):
            progress = (chunk - (startChunk + half)) / (fadeDuration - half if (fadeDuration - half) > 0 else 1)
            baseY = midY + progress * (endY - midY)
            anim["frames"][chunk] = baseY

        # Store in lyricPositions as BASE Y (consistent with the refactor)
        for chunk, baseY in anim["frames"].items():
            if chunk not in self.parent.lyricPositions:
                self.parent.lyricPositions[chunk] = []

            # Deduplicate by THIS ad-lib's identity (use startChunk arg, not self.startChunk)
            if any(entry[0] == startChunk for entry in self.parent.lyricPositions[chunk]):
                continue

            self.parent.lyricPositions[chunk].append((startChunk, baseY))
        
    def _pxX(self, basePx):
        return max(1, int(basePx * getattr(self.parent, "scaleX", 1.0)))

    def _pxY(self, basePx):
        return max(1, int(basePx * getattr(self.parent, "scaleY", 1.0)))

    def _px(self, basePx):
        s = min(getattr(self.parent, "scaleX", 1.0), getattr(self.parent, "scaleY", 1.0))
        return max(1, int(basePx * s))
    
    def createLyricDisplay(self):
        """Create a visual representation of the lyric box on the canvas."""
        self.originX, self.originY = self._anchorXY()
        x = self.originX
        y = self.originY 
        
        padding = self._pxY(5)
        textPhotoGapX = self._pxX(10)
        photoOverlapY = self._pxY(10)
        self.totalHeight = 0
        self.textItems = []
        self.textItemOffsets = []
        
        self.photoY = y + padding
        if self.memberPhotos:
            photoHeight = self.memberPhotos[0].height()
            photoWidth = self.memberPhotos[0].width()
            
            textX = x + photoWidth + textPhotoGapX
            
            for i, photo in enumerate(self.memberPhotos):
                photoId = self.canvas.create_image(
                    x , self.photoY, image=photo, anchor="nw", state="normal"
                    )
                self.photoItemIds.append(photoId)
                self._photoRefByItemId[photoId] = photo 
                self.textItems.append(photoId)
                self._storeOffset(photoId, x, self.photoY)

                if i < len(self.memberPhotos) - 1 or len(self.memberPhotos) == 1:
                    self.photoY += photoHeight - photoOverlapY
                # Offset text to the right of the photo
        else:
            textX = x
        
        # --- Member name ---
        if isinstance(self.memberNames, list) and len(self.memberNames) > 1:
            nameIds = []
            nameX = textX
            for i, name in enumerate(self.memberNames):
                partId = self.canvas.create_text(
                    nameX, y, text=name, font=self.font, fill=self.memberColors[i], anchor="nw", state="normal"
                )
                nameIds.append(partId)
                self.textItems.append(partId)
                self._storeOffset(partId, nameX, y)
                
                # Add spacing between names
                nameX += self._getItemWidth(partId) + self._pxX(5) 
            nameHeight = self._getItemHeight(nameIds[0]) if nameIds else 0
        else:
            nameIds = self.canvas.create_text(
                textX, y, text=self.memberNames,
                font=self.boldFont, fill=self.memberColors[0], 
                anchor="nw", state="normal"
            )  
            self.textItems.append(nameIds)
            self._storeOffset(nameIds, textX, y)
            nameHeight = self._getItemHeight(nameIds)    
        
        
        y += nameHeight + padding
        self.totalHeight += nameHeight + padding

        # --- Korean + Romanization ---
        if self.language == 'Korean':
            # Display Korean lyric (multi-line)
            for line in self.koreanLyric.split("\n"):
                self._createColorCodedText(textX, y, line, self.boldFont, self.memberColors)
                
                lastId = self.textItems[-1] if self.textItems else None
                lineHeight = self._getItemHeight(lastId) if lastId else 0
                y += lineHeight + padding
                self.totalHeight += lineHeight + padding

            # Display Romanization (multi-line, grey and not bold)
            for line in self.romanization.split("\n"):
                lineId = self.canvas.create_text(
                    textX, y, text=line, font=self.font, 
                    fill="grey", anchor="nw", state="normal"
                )
                self.textItems.append(lineId)
                self._storeOffset(lineId, textX, y)
                
                lineHeight = self._getItemHeight(lineId)
                y += lineHeight + padding
                self.totalHeight += lineHeight + padding

        # Display English translation (multi-line)
        for line in self.englishTrans.split("\n"):
            font = self.boldFont if self.language == 'Korean' else self.englishFont
            self._createColorCodedText(textX, y, line, font, self.memberColors)
            
            lastId = self.textItems[-1] if self.textItems else None
            lineHeight = self._getItemHeight(lastId) if lastId else 0

            y += lineHeight + padding
            self.totalHeight += lineHeight + padding
            
        self.hide()    
            
    def _createColorCodedText(self, x, y, text, font, colors):
        """
        Create multi-colored text where color changes at each '|'.
        All positions are stored as RELATIVE offsets from lyric box origin.
        """
        parts = text.split("|")  # Split text at '|'
        textX = x
        colorIndex = 0  # Start with the first member's color

        for part in parts:
            if not part:
                # Still advance color even if empty segment
                print(f"Color bug with {part}")
                if colorIndex < len(colors) - 1:
                    colorIndex += 1
                continue
            
            textId = self.canvas.create_text(
                textX, y, text=part, font=font, 
                fill=colors[colorIndex], anchor="nw", state="normal"
            )
            self.textItems.append(textId)
            self._storeOffset(textId, textX, y)
            
            textX += self._getItemWidth(textId)  # Move x position for next part
            # Cycle to next color if available
            if colorIndex < len(colors) - 1:
                colorIndex += 1
    
    def _getItemHeight(self, itemId):
        """Calculate the height of a canvas item."""
        bbox = self.canvas.bbox(itemId)
        if bbox:
            return bbox[3] - bbox[1]  # Height = bottom - top
        return 0
    
    def _getItemWidth(self, itemId):
        """Calculate the width of a canvas item."""
        bbox = self.canvas.bbox(itemId)
        return (bbox[2] - bbox[0]) if bbox else 0
    
    def setPosition(self, baseY):
        scaleY = getattr(self.parent, "scaleY", 1.0)
        offY = getattr(self.parent, "viewportOffsetY", 0)

        anchorX, _ = self._anchorXY() # Recompute in case resized
        self.originX = anchorX
        self.originY = offY + (baseY * scaleY)
        """Move the LyricBox to a fixed Y position on the canvas."""
        for itemId, dx, dy in self.textItemOffsets:
            #if (itemId == 53): print(f"Relative y: {relativeY} Actual y: {yPos + relativeY} for", itemId)
            self.canvas.coords(itemId, self.originX + dx, self.originY + dy)
            
    def show(self):
        """Make the lyric box visible on the canvas."""
        if not self.isVisible:
            for item in self.textItems:
                self.canvas.itemconfig(item, state="normal")
                self.canvas.tag_raise(item)
            self.isVisible = True
            
    def hide(self):
        """Hide the lyric box from the canvas."""
        for item in self.textItems:
            self.canvas.itemconfig(item, state="hidden")
        self.isVisible = False
    
    def getBaseYAt(self, chunkIndex):
        """
        Returns baseY if the lyric should be drawn at chunkIndex, else None.
        Contract: animate for a segment, then hold last endY until next segment.
        """
        anims = self.animations
        if not anims:
            return None

        # advance cursor while next animation has started
        while self._animCursor + 1 < len(anims) and anims[self._animCursor + 1]["startChunk"] <= chunkIndex:
            self._animCursor += 1

        anim = anims[self._animCursor]
        frames = anim["frames"]

        if chunkIndex in frames:
            self._heldBaseY = frames[chunkIndex]
            return self._heldBaseY

        # hold at last known value for this anim (usually endChunk frame)
        endChunk = anim["endChunk"]
        if endChunk in frames and chunkIndex > endChunk:
            self._heldBaseY = frames[endChunk]
            return self._heldBaseY

        return self._heldBaseY 
                    
    def animatePosition(self, startY, endY, startChunk, endChunk):
        """Precompute animation frames and store them per chunkIndex for smoother playback."""
        duration = endChunk - startChunk
        # Should never happen but just so it doesn't crash
        if duration <= 0:
            print(f"Duration for lyric from {startChunk} to {endChunk} is negative: {duration} ms")
            duration = 1
        
        anim = {
            "startChunk": startChunk,
            "endChunk": endChunk,
            "frames": {}
        }
        # Precompute baseY per chunk (inclusive)
        for chunk in range(startChunk, endChunk + 1):
            progress = (chunk - startChunk) / duration
            baseY = startY + progress * (endY - startY)
            anim["frames"][chunk] = baseY
            
        self.animations.append(anim)

        # Keep animations in chronological order so getBaseYAt's cursor is valid
        self.animations.sort(key=lambda a: a["startChunk"])
              
    
    def rebuildForResize(self):
        # preserve visibility
        wasVisible = self.isVisible

        # delete old canvas items
        for item in getattr(self, "textItems", []):
            try:
                self.canvas.delete(item)
            except Exception:
                pass

        self.textItems = []
        self.textItemOffsets = []
        self.photoItemIds = []
        self._photoRefByItemId = {}
        self.totalHeight = 0
        self.photoY = 0

        # reload photos at new scale (loadMemberPhotos uses parent.scaleX/scaleY)
        self.memberPhotos = self.resizeMemberImages(self.circleImages)

        # recreate UI items
        if self.isAdLib:
            self.createAdLibDisplay()
        else:
            self.createLyricDisplay()  
        
        # restore visibility
        # print(f"WAS {self.memberNames} visibile? {wasVisible}")
        if wasVisible:
            self.show()
        else:
            self.hide()        
    