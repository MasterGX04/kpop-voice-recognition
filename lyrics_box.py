from PIL import Image, ImageTk
import tkinter as tk

class LyricBox:
    def __init__(self, canvas, parent, memberNames, circleImages, 
                 koreanLyric, romanization, englishTrans, 
                 startChunk, language, isAdLib=False, adLibDuration=25):
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
        self.baseFontSizePx = 30
        
        self.fontSize = 20
        self.recalculateFontSize(baseFontPx=self.baseFontSizePx)
        self.animations = []
        
        self.lyricsPadding = self._pxY(5)
        self.addLyricDuration = 9
        
        if isAdLib:
            self.createAdLibDisplay()
        else:
            self.createLyricDisplay()
            self.initializeLyricPosition()
            
        self.lastChunkSeen = -1
    
    def recalculateFontSize(self, baseFontPx=15, minPx=12, maxPx=72, fontFamily="Pretendard Variable"):
        """
        Recompute font tuples based on current app height (scaleY).
        Design baseline: baseFontPx at 1080p (i.e., BASE_H).
        Call this whenever parent.scaleY changes (e.g., onCanvasResize).
        """
        scaleY = float(getattr(self.parent, "scaleY", 1.0) or 1.0)

        # Main scaling rule: font tracks fitted height scale
        size = int(round(baseFontPx * scaleY))

        # Clamp so it stays readable across extremes
        size = max(minPx, min(maxPx, size))
        self.fontSize = size
        
        self.font = (fontFamily, self.fontSize, "normal")
        self.boldFont = (fontFamily, self.fontSize, "bold")

        # If you want English slightly bigger than the base, scale the delta too
        # (keeps the +2 feeling consistent at different resolutions)
        englishDelta = max(1, int(round(2 * scaleY)))
        self.englishFont = (fontFamily, self.fontSize + englishDelta)
    
    def resizeMemberImages(self, memberPhotos):
        """
        Resize already-loaded member photos to match the current canvas scale.

        Assumes memberPhotos is a list of PIL.Image objects (typically the 'circle' images),
        already selected/saved in the correct order (so indices stay consistent).

        Base design size: 100px at 1920x1080, i.e.
        - width ratio  = 100/1920 of the design width
        - height ratio = 100/1080 of the design height
        """
        # Current scale from parent (computed in onCanvasResize)
        scaleY = getattr(self.parent, "scaleY", 1.0)

        # Base (design) size derived from ratios
        # (Equivalent to "100px in a 1920x1080 design")
        baseImgH = int((120 / 1080) * 1080)  # = 100, kept explicit for sanity
        imgH = max(1, int(baseImgH * scaleY))
        # Ensure list for consistency
        if not memberPhotos:
            transparent = Image.new("RGBA", (imgH, imgH), (0, 0, 0, 0))
            return [ImageTk.PhotoImage(transparent)]
        
        if not isinstance(memberPhotos, (list, tuple)):
            memberPhotos = [memberPhotos]
        
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
        startChunk = self.startChunk
        endChunk = self.startChunk + self.addLyricDuration

        endYBase = 5

        scaleY = getattr(self.parent, "scaleY", 1.0)
        if scaleY <= 0:
            scaleY = 1.0

        totalHeightBase = self.totalHeight / scaleY

        # 1) Enter animation for THIS lyric
        self.animatePosition(
            startY=-totalHeightBase,
            endY=endYBase,
            startChunk=startChunk,
            endChunk=endChunk
        )

        # 2) Compute push-down delta
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
        if pushDownBaseY <= 0:
            return

        # 3) Push down existing NON-ADLIB lyrics active at this chunk
        existingLyricBoxes = []

        if hasattr(self.parent, "getActiveLyricBoxesAtChunk"):
            existingLyricBoxes = self.parent.getActiveLyricBoxesAtChunk(startChunk) or []
        elif hasattr(self.parent, "activeLyricIds"):
            for lid in list(self.parent.activeLyricIds):
                lb = self.parent.lyrics.get(lid)
                if lb:
                    existingLyricBoxes.append(lb)

        for lb in existingLyricBoxes:
            # Always skip self
            if lb is self:
                continue

            # Skip ad-libs so they don't get stacked/pushed
            if getattr(lb, "isAdLib", False):
                continue

            # Also skip if THIS lyric is an ad-lib (extra safety; normally you don't call initializeLyricPosition for adlibs)
            if getattr(self, "isAdLib", False):
                continue

            oldBaseY = lb.getBaseYAt(startChunk) if hasattr(lb, "getBaseYAt") else endYBase
            if oldBaseY is None:
                continue

            lb.animatePosition(
                startY=oldBaseY,
                endY=oldBaseY + pushDownBaseY,
                startChunk=startChunk,
                endChunk=endChunk
            )

        self.animations.sort(key=lambda a: a["startChunk"])
    
    def resetAnimCursor(self):
        self._animCursor = 0
        self._heldBaseY = None     
        self._lastChunkIndex = None
     
    def createAdLibDisplay(self):
        """Creates a visual representation of an ad-lib lyric (top-right aligned) with member name + lyric text."""
        self.textItems = []
        self.textItemOffsets = []
        self.totalHeight = 0

        padY = self._pxY(5)
        lineGapY = self._pxY(2)
        marginX = self._pxX(10)
        marginY = self._pxY(10)

        scaleX = getattr(self.parent, "scaleX", 1.0)
        scaleY = getattr(self.parent, "scaleY", 1.0)
        offX = getattr(self.parent, "viewportOffsetX", 0)
        offY = getattr(self.parent, "viewportOffsetY", 0)

        baseW = getattr(self.parent, "baseWidth", 1920)
        baseH = getattr(self.parent, "baseHeight", 1080)

        # Store base anchor for resize-safe positioning
        self.baseAnchorX = baseW - 10
        self.baseAnchorY = 10

        # Convert to canvas coords
        anchorX = offX + int(self.baseAnchorX * scaleX) - marginX
        anchorY = offY + int(self.baseAnchorY * scaleY) + marginY

        # Store origin (canvas space)
        self.originX = anchorX
        self.originY = anchorY

        # --- Build member display (allow multiple) ---
        members = getattr(self, "memberNames", []) or []
        if isinstance(members, str):
            members = [members]
        memberDisplay = "/".join(members) if members else ""

        # --- Pick the displayed lyric string ---
        lines = []

        k = (getattr(self, "koreanLyric", "") or "").strip()
        r = (getattr(self, "romanization", "") or "").strip()
        e = (getattr(self, "englishTrans", "") or "").strip()

        if k:
            lines.append(("korean", k))
        if r:
            lines.append(("roman", r))
        if e:
            lines.append(("english", e))

        currentY = anchorY

        # 1) Member name line (bold)
        if memberDisplay:
            nameId = self.canvas.create_text(
                anchorX, currentY,
                text=memberDisplay,
                font=self.boldFont,
                fill=self.memberColors[0],
                anchor="ne",
                state="normal"
            )
            self.canvas.addtag_withtag("lyrics", nameId)
            self.textItems.append(nameId)
            self._storeOffset(nameId, anchorX, currentY)

            nameHeight = self._getItemHeight(nameId)
            currentY += nameHeight + lineGapY

        # 2) Add each lyric line (stacked)
        lineHeights = []
        for kind, text in lines:
            if not text:
                continue

            for line in str(text).split("\n"):
                if not line.strip():
                    continue

                # --- KOREAN ---
                if kind == "korean":
                    self._createColorCodedText(
                        anchorX,
                        currentY,
                        line,
                        self.boldFont,          # Korean is always bold
                        self.memberColors,
                        anchor="ne"
                    )
                    lastId = self.textItems[-1]
                    h = self._getItemHeight(lastId)

                # --- ROMANIZATION ---
                elif kind == "roman":
                    tid = self.canvas.create_text(
                        anchorX,
                        currentY,
                        text=line,
                        font=self.font,         # normal font
                        fill="grey",
                        anchor="ne",
                        state="normal"
                    )
                    self.canvas.addtag_withtag("lyrics", tid)
                    self.textItems.append(tid)
                    self._storeOffset(tid, anchorX, currentY)
                    h = self._getItemHeight(tid)

                # --- ENGLISH ---
                elif kind == "english":
                    # THIS is the key rule:
                    # If language is Korean → English is bold
                    # If language is English → English uses englishFont
                    fontToUse = self.boldFont if self.language == "Korean" else self.englishFont

                    tid = self.canvas.create_text(
                        anchorX,
                        currentY,
                        text=line,
                        font=fontToUse,
                        fill=self.memberColors[0],
                        anchor="ne",
                        state="normal"
                    )
                    self.canvas.addtag_withtag("lyrics", tid)
                    self.textItems.append(tid)
                    self._storeOffset(tid, anchorX, currentY)
                    h = self._getItemHeight(tid)

                else:
                    continue

                lineHeights.append(h)
                currentY += h + lineGapY
                
        # Measure lyric height in canvas units
        textHeightCanvas = sum(lineHeights) + lineGapY * max(0, len(lineHeights) - 1)

        # Total height (canvas units) used by normal on-screen checks
        # Include member name line if present
        totalCanvas = textHeightCanvas + padY
        if memberDisplay:
            totalCanvas += nameHeight + lineGapY
        self.totalHeight = totalCanvas

        # Store BASE height for “offscreen” checks (base units)
        self.adLibTextHeightBase = (totalCanvas / scaleY) if scaleY else totalCanvas

        # ---- IMPORTANT: make offsets bottom-anchored instead of top-anchored ----
        # Right now textItemOffsets were stored with dy measured from the block's TOP.
        # Convert them so dy is measured from the block's BOTTOM (i.e., extend upward).
        self.textItemOffsets = [
            (itemId, dx, dy - totalCanvas) for (itemId, dx, dy) in self.textItemOffsets
        ]

        # Animate: bottom -> beyond top
        cushionBase = 20
        startYBase = baseH + cushionBase          # bottom starts below screen
        endYBase   = -cushionBase                 # bottom goes above top

        self.animateAdLibPosition(
            startY=startYBase,
            endY=endYBase,
            startChunk=self.startChunk,
            durationChunks=self.adLibDuration
        )
        
        # print(f"Ad-lib animation for {memberDisplay}: {self.animations}")
        
    def setAdLibPosition(self, baseY):
        scaleX = getattr(self.parent, "scaleX", 1.0)
        scaleY = getattr(self.parent, "scaleY", 1.0)
        offX = getattr(self.parent, "viewportOffsetX", 0)
        offY = getattr(self.parent, "viewportOffsetY", 0)

        marginX = self._pxX(10)
        marginY = self._pxY(10)

        baseW = getattr(self.parent, "baseWidth", 1920)
        baseAnchorX = getattr(self, "baseAnchorX", baseW - 10)

        originX = offX + int(baseAnchorX * scaleX) - marginX
        originY = offY + int(baseY * scaleY) + marginY   # baseY is now the BOTTOM

        for itemId, dx, dy in self.textItemOffsets:
            self.canvas.coords(itemId, originX + dx, originY + dy)
            
    def animateAdLibPosition(self, startY, endY, startChunk, durationChunks):
        """
        Animate ad-lib in BASE Y coords: startY -> endY over durationChunks (chunk indices).
        Stores frames in BASE units for resize safety.
        """
        startChunk = int(startChunk)
        durationChunks = max(1, int(durationChunks))

        anim = {
            "startChunk": startChunk,
            "endChunk": startChunk + durationChunks,
            "frames": {}
        }

        # Precompute frames (inclusive)
        for i in range(durationChunks + 1):
            chunk = startChunk + i
            t = i / durationChunks
            baseY = startY + t * (endY - startY)
            anim["frames"][chunk] = baseY

        self.animations.append(anim)

        # Keep animations ordered so getBaseYAt cursor works
        self.animations.sort(key=lambda a: a["startChunk"])
        self._animCursor = 0
        self._heldBaseY = None
    
    def rebuildAdLibAnimation(self):
        # Clear any existing animations
        self.animations = []
        if hasattr(self, "resetAnimCursor"):
            self.resetAnimCursor()

        baseH = getattr(self.parent, "baseHeight", 1080)
        cushionBase = 20

        # Make sure adLibTextHeightBase is available (base units)
        if getattr(self, "adLibTextHeightBase", None) is None:
            scaleY = getattr(self.parent, "scaleY", 1.0) or 1.0
            heightCanvas = getattr(self, "totalHeight", 0)
            self.adLibTextHeightBase = heightCanvas / scaleY

        # Bottom-anchored animation (bottom starts below screen; ends above top)
        startYBase = baseH + cushionBase
        endYBase = -cushionBase

        self.adLibStartYBase = startYBase
        self.adLibEndYBase = endYBase

        self.animateAdLibPosition(
            startY=startYBase,
            endY=endYBase,
            startChunk=self.startChunk,
            durationChunks=self.adLibDuration
        )
    
    def updateAdLibForChunk(self, chunkIndex):
        baseY = self.getBaseYAt(chunkIndex)
        if baseY is None:
            self.setVisible(False)
            return

        textHeightBase = getattr(self, "adLibTextHeightBase", 0)  # base-units height
        baseH = getattr(self.parent, "baseHeight", 1080)          # base-units screen height
        cushionBase = 20

        # baseY is the BOTTOM of the adlib block (base units)
        topY = baseY - textHeightBase

        # Offscreen checks (base units)
        if baseY < -cushionBase:                  # bottom passed above top
            self.setVisible(False)
            return
        if topY > baseH + cushionBase:            # top still below bottom (hasn't entered yet)
            self.setVisible(False)
            return

        self.setVisible(True)

        scaleX = getattr(self.parent, "scaleX", 1.0)
        scaleY = getattr(self.parent, "scaleY", 1.0)
        offX = getattr(self.parent, "viewportOffsetX", 0)
        offY = getattr(self.parent, "viewportOffsetY", 0)

        marginX = self._pxX(10)
        marginY = self._pxY(10)

        baseW = getattr(self.parent, "baseWidth", 1920)
        baseAnchorX = getattr(self, "baseAnchorX", baseW - 10)

        originX = offX + int(baseAnchorX * scaleX) - marginX
        originY = offY + int(baseY * scaleY) + marginY  # baseY is bottom

        for itemId, dx, dy in self.textItemOffsets:
            self.canvas.coords(itemId, originX + dx, originY + dy)
    
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
                    x , self.photoY, image=photo, anchor="nw", state="normal", tags="lyrics"
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
                self.canvas.addtag_withtag("lyrics", partId)
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
            self.canvas.addtag_withtag("lyrics", nameIds)
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
                self.canvas.addtag_withtag("lyrics", lineId)
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
            
    def _createColorCodedText(self, x, y, text, font, colors, anchor="nw"):
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
                fill=colors[colorIndex], anchor=anchor, state="normal"
            )
            self.canvas.addtag_withtag("lyrics", textId)
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
        """Make the lyric box visible on the canvas (idempotent)."""
        # Actually show
        for item in self.textItems:
            self.canvas.itemconfig(item, state="normal")

        self.isVisible = True

        # Defer layering: mark dirty, let parent do it once
        if hasattr(self.parent, "lyricsLayerDirty"):
            self.parent.lyricsLayerDirty = True
            
    def hide(self):
        """Hide the lyric box from the canvas (idempotent)."""
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
        
        self.recalculateFontSize(baseFontPx=self.baseFontSizePx)

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
            
    def destroy(self):
        """Permanently remove this lyric box from the canvas (idempotent)."""
        # Delete any canvas items we created
        for item in getattr(self, "textItems", []):
            try:
                self.canvas.delete(item)
            except Exception:
                pass

        # Clear refs so nothing holds old images/items
        self.textItems = []
        self.textItemOffsets = []
        self.photoItemIds = []
        self._photoRefByItemId = {}
        self.isVisible = False
    