import tkinter as tk

class LabelLaneRenderer:
    """
    Draws label bars in up to N fixed lanes above the progress bar (DAW style).
    Uses greedy interval lane assignment by start time.

    labels: list like [memberName, startChunk, endChunk, isBacking, isAdlib]
    """
    def __init__(
        self,
        canvas: tk.Canvas,
        zoomManager,
        progressBarCanvas,
        getLabelsFn,
        getMemberColorFn,
        *,
        chunksPerSecond=25.0,   # 40ms chunks => 25 chunks/sec
        maxLanes=4,
        laneHeight=12,
        laneGap=4,
        topPadding=10,
        barOutline="#000000",
        tag="label_bar",
    ):
        self.canvas = canvas
        self.zoomManager = zoomManager
        self.progressBarCanvas = progressBarCanvas
        self.getLabels = getLabelsFn
        self.getMemberColor = getMemberColorFn

        self.chunksPerSecond = chunksPerSecond
        self.maxLanes = maxLanes
        self.laneHeight = laneHeight
        self.laneGap = laneGap
        self.topPadding = topPadding
        self.barOutline = barOutline
        self.tag = tag
        
    def clear(self):
        self.canvas.delete(self.tag)
        
    def _durationSecs(self, start, end):
        # end is treated as exclusive in your tooltip math; keep consistent
        return round((end - start) / self.chunksPerSecond, 2)

    def _assignLanes(self, intervals):
        """
        intervals: list of dicts: { "member": str, "start": int, "end": int, ... }
        Sort by start, then greedy place into first lane where lastEnd < start,
        else spill into next lane. If still no lane (more than maxLanes overlap),
        clamp to last lane (lane maxLanes-1).
        """
        intervals.sort(key=lambda x: (x["start"], x["end"]))

        lastEnd = [-10**9] * self.maxLanes  # last end chunk in each lane
        for it in intervals:
            placed = False
            for lane in range(self.maxLanes):
                if lastEnd[lane] < it["start"]:
                    it["lane"] = lane
                    lastEnd[lane] = it["end"]
                    placed = True
                    break

            if not placed:
                # Too many overlaps. Robust fallback: clamp to last lane.
                it["lane"] = self.maxLanes - 1
                # Update lastEnd so later items still behave sensibly
                lastEnd[self.maxLanes - 1] = max(lastEnd[self.maxLanes - 1], it["end"])

        return intervals

    def drawSection(self, sectionIndex, progressBarWidth):
        """
        Draw label bars for labels that intersect the visible chunk window for sectionIndex.
        """
        self.clear()

        chunksInView = self.zoomManager.currentChunksInView
        windowStart = sectionIndex * chunksInView
        windowEnd = windowStart + chunksInView

        # pull labels and filter those that intersect the window
        raw = self.getLabels() or []
        intervals = []
        for lab in raw:
            if len(lab) < 3:
                continue
            member, start, end = lab[0], lab[1], lab[2]
            isBacking = lab[3] if len(lab) > 3 else False
            isAdlib = lab[4] if len(lab) > 4 else False

            # intersect check: [start,end) intersects [windowStart, windowEnd)
            if end <= windowStart or start >= windowEnd:
                continue
            
            interval = {
                "member": member,
                "start": start,
                "end": end,
                "isBacking": isBacking,
                "isAdlib": isAdlib,
            }   
            intervals.append(interval)

        if not intervals:
            return

        self._assignLanes(intervals)

        # Geometry
        baseX = self.progressBarCanvas.winfo_x()
        barY = self.progressBarCanvas.winfo_y()

        # Lanes sit above the progress bar; lane 0 is closest to bar, lane 3 highest
        # You can invert if you prefer.
        for it in intervals:
            lane = it["lane"]

            wrapsLeft = it["start"] < windowStart
            wrapsRight = it["end"] > windowEnd

            # Clamp interval to window for drawing
            s = max(it["start"], windowStart)
            e = min(it["end"], windowEnd)

            # Convert chunk -> x in pixels within the section
            sRel = (s - windowStart) / chunksInView
            eRel = (e - windowStart) / chunksInView

            x1 = self.canvas.canvasx(baseX + sRel * progressBarWidth)
            x2 = self.canvas.canvasx(baseX + eRel * progressBarWidth)

            # Minimum visible width
            if x2 - x1 < 2:
                x2 = x1 + 2

            y2 = barY - self.topPadding - lane * (self.laneHeight + self.laneGap)
            y1 = y2 - self.laneHeight
            
            yMid = (y1 + y2) / 2
            fill = self.getMemberColor(it["member"]) or "#888888"
            self._drawEdgeGlyphs(x1, x2, yMid, fill, left=wrapsLeft, right=wrapsRight)


            rectId = self.canvas.create_rectangle(
                x1, y1, x2, y2,
                fill=fill,
                outline=self.barOutline,
                width=1,
                tags=(self.tag,)
            )

            # Bonus: show duration in secs on the bar (keep short so it doesn’t clutter)
            dur = self._durationSecs(it["start"], it["end"])
            roleSuffix = ""
            if it["isAdlib"]:
                roleSuffix = " (A)"
            elif it["isBacking"]:
                roleSuffix = " (B)"

            text = f"{it['member']} {dur:.2f}s{roleSuffix}"

            # Only draw text if there's enough room
            if (x2 - x1) >= 60:
                self.canvas.create_text(
                    x1 + 4,
                    (y1 + y2) / 2,
                    text=text,
                    anchor="w",
                    fill="white",
                    font=("Arial", 8),
                    tags=(self.tag,)
                )
    # end drawSection
    
    def _drawEdgeGlyphs(self, xLeft, xRight, yMid, fill, *, left=False, right=False):
        """
        Draw small DAW-style continuation triangles at the left/right edges.
        xLeft/xRight are the bar edges; yMid is vertical center of the bar lane.
        """
        triW = 7
        triH = 7

        if left:
            # left-pointing triangle near xLeft
            pts = [
                xLeft + 2, yMid,                 # tip (points left)
                xLeft + 2 + triW, yMid - triH/2, # top-right
                xLeft + 2 + triW, yMid + triH/2, # bottom-right
            ]
            self.canvas.create_polygon(
                pts, fill=fill, outline="", tags=(self.tag,)
            )

        if right:
            # right-pointing triangle near xRight
            pts = [
                xRight - 2, yMid,                 # tip (points right)
                xRight - 2 - triW, yMid - triH/2, # top-left
                xRight - 2 - triW, yMid + triH/2, # bottom-left
            ]
            self.canvas.create_polygon(
                pts, fill=fill, outline="", tags=(self.tag,)
            )            
    