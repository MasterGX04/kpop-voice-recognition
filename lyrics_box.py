from PIL import Image, ImageTk
import tkinter as tk
import os
from matplotlib import font_manager
from audio_processing import getSongsFromSameAlbum

class LyricBox:
    def __init__(self, canvas, parent, memberName, koreanLyric, romanization, englishTrans, startChunk, language, isAdLib=False, adLibDuration=0):
        self.canvas = canvas
        self.parent = parent
        self.memberName = memberName
        self.koreanLyric = koreanLyric
        self.romanization = romanization
        self.englishTrans = englishTrans
        self.startChunk = startChunk
        self.isAdLib = isAdLib  # New flag for ad-libs
        self.adLibDuration = adLibDuration 
        
        self.photoY = 0
        
        self.memberPhotos = self.loadMemberPhotos(self.memberName)
        self.textItems = []  # Store canvas item IDs
        self.totalHeight = 0  # To calculate total height of the lyric box
        self.isVisible = False
        self.language = language
        
        # Assign colors for each member if multiple
        if isinstance(memberName, list):
            self.memberColors = [parent.getMemberColor(name) for name in memberName]
        else:
            self.memberColors = [parent.getMemberColor(memberName)]
        
        self.startY = -self.totalHeight
        self.endY = 5
        
        # Set fonts
        self.fontSize = 18
        self.fontPath = './fonts/CENTURY.TTF'
        fontProperties = font_manager.FontProperties(fname=self.fontPath)
        self.font = (fontProperties.get_name(), self.fontSize, "bold")
        self.englishFont = (fontProperties.get_name(), self.fontSize + 2)
        # self.adLibFont = (fontProperties.get_name(), self.fontSize - 2, "bold")

        self.animations = []
        
        self.lyricsPadding = 5
        self.addLyricDuration = 7
        
        if isAdLib:
            self.createAdLibDisplay()
        else:
            self.createLyricDisplay()
    
    def loadMemberPhotos(self, memberNames, ):
        """Load profile images for each member in a multi-member lyric."""
        if isinstance(memberNames, str):  # Convert single name to list for consistency
            memberNames = [memberNames]
        
        photos = []
        songsFromSameAlbum = getSongsFromSameAlbum()
        songName = os.path.splitext(os.path.basename(self.parent.testSongPath))[0]
        albumName = None
    
        for album, songs in songsFromSameAlbum[self.parent.selectedGroup].items():
            if songName in songs:
                albumName = album
                break  # Exit once we find the album
            
        for name in memberNames:
            imagePath = f"./group_icons/{self.parent.selectedGroup}/{albumName}/{name} Circle.png"
            if os.path.exists(imagePath):
                img = Image.open(imagePath)
                img = img.resize((80, 80))
                photos.append(ImageTk.PhotoImage(img))
            else:
                print(f"Warning: {imagePath} does not exist")
        return photos
    
    def initializeLyricPosition(self):
        chunkIndex = self.startChunk
        baseY = self.endY # Base Position
        
        # Find most recent chunk index with lyriccs
        recentChunkIndex = max(
            (key for key in self.parent.lyrics if (key is not None and key < chunkIndex)),
            default=None
        )
        
        if chunkIndex not in self.parent.lyricPositions:
            if recentChunkIndex is not None and (recentChunkIndex + self.addLyricDuration) in self.parent.lyricPositions:
                self.parent.lyricPositions[chunkIndex] = self.parent.lyricPositions[recentChunkIndex + self.addLyricDuration][:]
            else:
                self.parent.lyricPositions[chunkIndex] = []
        
        # Check if there exists a lyric at this chunk index
        if len(self.parent.lyricPositions[chunkIndex]) > 0:
            numMembers = 1 if isinstance(self.memberName, str) else len(self.memberName)
            additionalHeight = max(
                (self.photoY) * numMembers + self.lyricsPadding + 10, self.totalHeight + self.lyricsPadding
            )
            newY = additionalHeight
        else:
            newY = baseY

        # Animate current text
        self.animatePosition(
            startY=-self.totalHeight,
            endY=0,
            startChunk=self.startChunk,
            endChunk=self.startChunk + self.addLyricDuration
        )
        
        self.animateExistingLyrics(chunkIndex, newY)
        
    def createAdLibDisplay(self):
        """Creates a visual representation of an ad-lib lyric."""
        canvasWidth = self.canvas.winfo_width()
        canvasHeight = self.canvas.winfo_height()
        
        x = canvasWidth - 10
        y = canvasHeight
        padding = 5
        self.totalHeight = 0
        self.textItemOffsets = []
        
        # Display Ad-Lib Text
        if self.language == "Korean":
            textId = self.canvas.create_text(
                x, y, text=self.koreanLyric if self.language == 'Korean' else self.englishTrans,
                font=self.adLibFont, fill=self.memberColors[0], anchor="ne", state="normal"  # Align top-right
            )
            self.textItems.append(textId)
            self.textItemOffsets.append((textId, y))
            
            textHeight = self._getItemHeight(textId)
            self.totalHeight += textHeight + padding
        
        self.animateAdLibPosition(
            startY=canvasHeight,
            midY=canvasHeight / 2,
            endY=self.totalHeight,
            startChunk=self.startChunk,
            duration=self.adLibDuration
        )
    
    def animateAdLibPosition(self, startY, midY, endY, startChunk, duration):
        """Animates ad-lib from bottom → mid-screen → disappear."""
        fadeDuration = duration if duration > 0 else 10
        
        self.animations.append({
            "startChunk": startChunk,
            "endChunk": startChunk + fadeDuration,
            "frames":{}
        })
        
        canvasHeight = self.canvas.winfo_height()
        
        # Second phase: Fade out above screen
        for chunk in range(startChunk + fadeDuration // 2, startChunk + fadeDuration):
            progress = (chunk - (startChunk + fadeDuration // 2)) / (fadeDuration // 2)
            interpolatedY = round(midY - progress * (midY - endY), 1)
            self.animations[-1]["frames"][chunk] = max(0, min(interpolatedY, canvasHeight))

        # Store animation data for ad-lib movement
        for chunk, yPos in self.animations[-1]["frames"].items():
            if chunk not in self.parent.lyricPositions:
                self.parent.lyricPositions[chunk] = []
            self.parent.lyricPositions[chunk].append((self.startChunk, yPos))
        
    
    def createLyricDisplay(self):
        """Create a visual representation of the lyric box on the canvas."""
        x, y = 760 * self.parent.scaleX, 10 * self.parent.scaleY  # Adjust positioning (Needs to be updated)
        padding = 5
        self.totalHeight = 0
        self.textItemOffsets = []
        
        self.photoY = y
        if self.memberPhotos:
            photoHeight = self.memberPhotos[0].height()
            photoWidth = self.memberPhotos[0].width()
            
            textX = x + photoWidth + 10
            for i, photo in enumerate(self.memberPhotos):
                photoId = self.canvas.create_image(x, y, image=photo, anchor="nw", state="normal")
                self.textItems.append(photoId)
                self.textItemOffsets.append((photoId, self.photoY))
                if i < len(self.memberPhotos) - 1 or len(self.memberPhotos) == 1:
                    self.photoY += photoHeight - 10
                # Offset text to the right of the photo
        else:
            textX = x
        
        if isinstance(self.memberName, list) and len(self.memberName) > 1:
            nameId = []
            nameX = textX
            for i, name in enumerate(self.memberName):
                partId = self.canvas.create_text(
                    nameX, y, text=name, font=self.font, fill=self.memberColors[i], anchor="nw", state="normal"
                )
                nameId.append(partId)
                nameWidth = self._getItemWidth(partId)
                nameX += nameWidth + 5  # Add spacing between names
                self.textItemOffsets.append((partId, y))
            self.textItems.extend(nameId)
            nameHeight = self._getItemHeight(nameId[0])
        else:
            nameId = self.canvas.create_text(
                textX, y, text=self.memberName,
                font=self.font, fill=self.memberColors[0], anchor="nw", state="normal"
            )  
            self.textItems.append(nameId)
            self.textItemOffsets.append((nameId, y))
            nameHeight = self._getItemHeight(nameId)    
        
        
        y += nameHeight + padding
        self.totalHeight += nameHeight + padding

        if self.language == 'Korean':
            # Display Korean lyric (multi-line)
            for line in self.koreanLyric.split("\n"):
                self._createColorCodedText(textX, y, line, self.font, self.memberColors)
                y += self._getItemHeight(self.textItems[-1]) + padding
                self.totalHeight += self._getItemHeight(self.textItems[-1]) + padding

            # Display Romanization (multi-line, grey and not bold)
            for line in self.romanization.split("\n"):
                lineId = self.canvas.create_text(
                    textX, y, text=line,
                    font=self.font, fill="grey", anchor="nw",
                    state="normal"
                )
                self.textItems.append(lineId)
                lineHeight = self._getItemHeight(lineId)
                self.textItemOffsets.append((lineId, y))
                y += lineHeight + padding
                self.totalHeight += lineHeight + padding

        # Display English translation (multi-line)
        for line in self.englishTrans.split("\n"):
            font = self.font if self.language == 'Korean' else self.englishFont
            self._createColorCodedText(textX, y, line, font, self.memberColors)
            y += self._getItemHeight(self.textItems[-1]) + padding
            self.totalHeight += self._getItemHeight(self.textItems[-1]) + padding
            
        self.hide()    
            
    def _createColorCodedText(self, x, y, text, font, colors):
        """Creates multi-colored text where colors change after encountering '|' character."""
        parts = text.split("|")  # Split text at '|'
        textX = x
        colorIndex = 0  # Start with the first member's color

        for part in parts:
            textId = self.canvas.create_text(
                textX, y, text=part, font=font, fill=colors[colorIndex], anchor="nw", state="normal"
            )
            self.textItems.append(textId)
            textWidth = self._getItemWidth(textId)
            textX += textWidth  # Move x position for next part
            self.textItemOffsets.append((textId, y))
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
    
    def setPosition(self, yPos):
        """Move the LyricBox to a fixed Y position on the canvas."""
        for itemId, relativeY in self.textItemOffsets:
            #if (itemId == 53): print(f"Relative y: {relativeY} Actual y: {yPos + relativeY} for", itemId)
            self.canvas.coords(itemId, self.canvas.coords(itemId)[0], yPos + relativeY)
            
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
                 
    def animatePosition(self, startY, endY, startChunk, endChunk):
        """Precompute animation frames and store them per chunkIndex for smoother playback."""
        duration = endChunk - startChunk
        
        self.animations.append({
            "startChunk": startChunk,
            "endChunk": endChunk,
            "frames": {}
        })
        
        canvasHeight = self.canvas.winfo_height()
        removeFromFrame = False
        
        for chunk in range(startChunk, endChunk + 1):
            if removeFromFrame:
                # Stop storing Y positions after an overflow is detected
                break
                
            progress = (chunk - startChunk) / duration           
            
            interpolatedY = round(startY + progress * (endY - startY), 1)
            
            if interpolatedY > canvasHeight:
                removeFromFrame = True  # Flag to remove it from future frames
            
            # Store animation frame
            self.animations[-1]["frames"][chunk] = interpolatedY
            
            if chunk not in self.parent.lyricPositions:
                self.parent.lyricPositions[chunk] = []
                
            if any(entry[0] == self.startChunk for entry in self.parent.lyricPositions[chunk]):
                    continue  # Skip adding duplicate entry
            
            self.parent.lyricPositions[chunk].append((self.startChunk, interpolatedY))
              
    def animateExistingLyrics(self, chunkIndex, newLyricY):
        """Update animations for all existing lyrics when a new lyric is added."""
        # Called infinitely
        existingLyrics = list(self.parent.lyricPositions[chunkIndex])
        for existingStartChunk, oldY in existingLyrics:
            if existingStartChunk != self.startChunk:
                lyricBox = self.parent.lyrics[existingStartChunk]
                # print(f"Lyric box to change for {lyricBox.memberName}:", lyricBox.englishTrans)
                lyricBox.animatePosition(
                    startY=oldY,
                    endY=oldY+newLyricY,
                    startChunk=self.startChunk,
                    endChunk = self.startChunk + self.addLyricDuration
                )