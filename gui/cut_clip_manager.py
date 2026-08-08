from __future__ import annotations

class CutClipManager():
    def __init__(self, chunkDurationMs: int, jumpCallback):
        """
        jumpCallback(targetMs: int) -> None
        Should perform a safe jump: update playbackOffset, restart audio, seek video, etc.
        """
        self.chunkDurationMs = int(chunkDurationMs)
        self.jumpCallback = jumpCallback
        self.enabled = False

        self.cutRanges = []
        self.isCutChunk = []
        self.nextPlayableChunk = []
        
    def setIsEnabled(self, enabled: bool):
        self.enabled = bool(enabled)
    
    def toggleEnabled(self) -> bool:
        """
        Toggle clip-cut mode on/off.
        Returns the new enabled state.
        """
        newState = not self.enabled
        self.setIsEnabled(newState)
        # return newState
    
    def rebuild(self, labels, numChunks: int):
        """
        Build internal playback lookup tables for 'Cut' labels.

        This function DOES NOT:
        - modify or delete any labels
        - remove 'Cut' labels from labels.json
        - perform any playback skipping by itself

        What this function DOES:
        - Reads the existing labels list and extracts only labels whose member == "Cut"
        - Merges overlapping or adjacent Cut ranges into consolidated intervals
        - Builds:
            1) self.isCutChunk[chunkIndex] -> True if that chunk is inside a Cut range
            2) self.nextPlayableChunk[chunkIndex] -> the next chunk index >= chunkIndex
            that is NOT inside a Cut range

        These lookup tables are later USED by playback code (e.g. maybeSkip / maybeSkipNext)
        to decide WHEN to jump audio/video during playback.

        Important:
        - 'Cut' labels SHOULD still be saved to labels.json.
        They are part of the annotation data and must persist.
        - rebuild() is a PRECOMPUTATION step only.
        Skipping will not happen unless playback code explicitly calls maybeSkip(...)
        or maybeSkipNext(...) during playback.
        """
        n = int(numChunks)
        cutRanges = []

        for lab in labels:
            if len(lab) < 3:
                continue
            member, start, end = lab[:3]
            if str(member).strip().lower() != "cut":
                continue
            s = max(0, int(start))
            e = min(n - 1, int(end))
            if s <= e:
                cutRanges.append((s, e))

        cutRanges.sort()
        merged = []
        for s, e in cutRanges:
            if not merged or s > merged[-1][1] + 1:
                merged.append([s, e])
            else:
                merged[-1][1] = max(merged[-1][1], e)
        self.cutRanges = [(s, e) for s, e in merged]

        isCut = [False] * n
        for s, e in self.cutRanges:
            for i in range(s, e + 1):
                isCut[i] = True
        self.isCutChunk = isCut

        nextPlayable = [n] * (n + 1)
        nextPlayable[n] = n
        for i in range(n - 1, -1, -1):
            nextPlayable[i] = nextPlayable[i + 1] if isCut[i] else i
        self.nextPlayableChunk = nextPlayable
        
    def chunkFromMs(self, ms: int) -> int:
        if self.chunkDurationMs <= 0:
            return 0
        return int(ms // self.chunkDurationMs)

    def msFromChunk(self, chunk: int) -> int:
        return int(chunk) * self.chunkDurationMs

    def inCut(self, chunk: int) -> bool:
        if chunk < 0 or chunk >= len(self.isCutChunk):
            return False
        return self.isCutChunk[chunk]
    
    def nextPlayable(self, chunk: int) -> int:
        if chunk < 0:
            return 0
        if chunk >= len(self.nextPlayableChunk):
            return len(self.isCutChunk)  # n
        return self.nextPlayableChunk[chunk]
    
    def maybeSkip(self, currentChunk: int) -> bool:
        """
        Returns True if a jump happened (caller should stop normal tick processing that frame).
        """
        if not self.enabled:
            return False

        if not self.inCut(currentChunk):
            return False
        
        next = self.nextPlayable(currentChunk)
        if next >= len(self.isCutChunk):
            self.jumpCallback(None)
            return True  # nothing left; let caller handle end-of-song
        self.jumpCallback(self.msFromChunk(next))
        return True
    
    def maybeSkipNext(self, currentChunk: int) -> bool:
        """
        Optional variant: jump when the NEXT chunk is cut (your 99->251 example),
        so you don't spend a tick inside the cut region.
        """
        if not self.enabled:
            return False

        nxtChunk = currentChunk + 1
        if nxtChunk < len(self.isCutChunk) and self.inCut(nxtChunk):
            nxt = self.nextPlayable(nxtChunk)
            if nxt >= len(self.isCutChunk):
                self.jumpCallback(None) 
                return True
            self.jumpCallback(self.msFromChunk(nxt))
            return True
        return False    