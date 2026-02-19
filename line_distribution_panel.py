import tkinter as tk
import statistics
import os, sys
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def resourcePath(*parts: str) -> str:
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, *parts)

class LineDistributionPanel:
    """
    Tkinter-embedded matplotlib donut pie + compact stats line.

    Expected app interface:
      - app.root : Tk root (or parent window)
      - app.chunks : list (used for timeline index safety)
      - app.memberImages : dict[str, TrackItem] where TrackItem.timeline is list of seconds over time
    """

    def __init__(self, app, *, parent=None, bg="black", dpi=100):
        self.app = app
        self.bg = bg
        self.dpi = dpi

        self.parent = parent if parent is not None else app.root
        self.frame = tk.Frame(self.parent, bg=self.bg, highlightthickness=0)
        self.frame.tkraise()

        self.statsLabel = tk.Label(
            self.frame,
            text="",
            fg="#dddddd",
            bg=self.bg,
            font=("Arial", 13)
        )
        self.statsLabel.pack(side="top", anchor="w", padx=12, pady=(10, 0))

        # fairness grade label (added in Step 3)
        self.gradeLabel = tk.Label(
            self.frame, text="", fg="white", bg=self.bg, font=("Arial", 14, "bold")
        )
        self.gradeLabel.pack(side="top", anchor="w", padx=12, pady=(4, 8))

        self.fig = Figure(figsize=(6, 4), dpi=self.dpi)
        self.ax = self.fig.add_subplot(111)

        self.canvasAgg = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvasWidget = self.canvasAgg.get_tk_widget()
        self.canvasWidget.configure(bg=self.bg, highlightthickness=0)

        # CHANGED: let it expand to fill
        self.canvasWidget.pack(side="top", fill="both", expand=True)

        self._placed = False
        self._spinAfterId = None
        self._startAngle = 90.0
        self.placeOverlay()  # Start visible; toggle later as needed

    def placeOverlay(self, *, relw=1.0, relh=1.0, padx=10, pady=10):
        """
        Place as an overlay panel on top of the canvas, without covering everything.
        Uses relative sizing so it scales a bit with window size.
        """
        # Place in a corner with a bounded footprint
        self.frame.place(
            relx=0.0,
            rely=0.0,
            anchor="nw",     # TOP-LEFT
            relwidth=relw,
            relheight=relh,
            x=0,
            y=0
        )   
        self._placed = True
    
    def _buildMemberColorMap(self) -> dict:
        """
        Returns {memberName: "#RRGGBB"} from self.app.members
        """
        m = {}
        members = getattr(self.app, "members", None) or []
        for entry in members:
            name = (entry.get("name") or "").strip()
            color = (entry.get("color") or "").strip()
            if name and color:
                # normalize: ensure it starts with '#'
                if not color.startswith("#"):
                    color = "#" + color
                m[name] = color
        return m

    def hide(self):
        self.stopSpin()
        try:
            self.frame.place_forget()
        except Exception:
            pass
        self._placed = False

    def show(self):
        if not self._placed:
            self.placeOverlay()
        self.startSpin()
            
    def startSpin(self, *, degPerTick=1.2, tickMs=50):
        """Slow spin using Tk after()."""
        self.stopSpin()

        def _tick():
            # Only keep spinning if still visible
            if not self._placed:
                self._spinAfterId = None
                return

            self._startAngle = (self._startAngle + degPerTick) % 360.0
            self.update(redrawOnly=True)  # don’t recompute stats if you don’t want to
            self._spinAfterId = self.frame.after(tickMs, _tick)

        _tick()

    def stopSpin(self):
        if self._spinAfterId is not None:
            try:
                self.frame.after_cancel(self._spinAfterId)
            except Exception:
                pass
            self._spinAfterId = None
            
    def update(self, redrawOnly=False):
        stats = self.computeStats() if not redrawOnly else getattr(self, "_lastStats", None)
        if stats is None:
            stats = self.computeStats()
        self._lastStats = stats

        items = stats["sortedItems"]
        meanS = stats["meanSeconds"]
        stdS = stats["stdSeconds"]
        fairness = stats["fairness"]

        self.statsLabel.config(
            text=f"Mean: {meanS:.2f}s   Std: {stdS:.2f}s   Fairness: {fairness:.3f}"
        )

        # grade label set in Step 3 (we’ll plug it in later)
        nMembers = len(items)  # or len(app.members) if you prefer
        grade, target = self._fairnessGrade(fairness, nMembers)
        self.gradeLabel.config(text=f"Grade: {grade}   Target({nMembers}): {target:.2f}")

        self.ax.clear()
        self.ax.set_facecolor(self.bg)

        if not items:
            self.ax.text(0.5, 0.5, "No data", ha="center", va="center", color="white")
            self.ax.axis("off")
            self.canvasAgg.draw()
            return

        colorMap = self._buildMemberColorMap()

        labels = [name for name, _ in items]
        values = [sec for _, sec in items]

        # ✅ per-slice color with fallback
        sliceColors = [colorMap.get(name, "#777777") for name in labels]

        wedges, _ = self.ax.pie(
            values,
            labels=None,
            startangle=self._startAngle,      
            counterclock=False,
            wedgeprops={"width": 0.45},
            colors=sliceColors
        )
        
        totalSeconds = sum(values) if values else 1.0
        legendLabels = [
            f"{n}: {s:.2f}s ({(100 * s / totalSeconds):.1f}%)"
            for (n, s) in items
        ]
        leg = self.ax.legend(
            wedges,
            legendLabels,
            loc="center left",
            bbox_to_anchor=(0.88, 0.5),   # 👈 pull it INSIDE the axes
            frameon=False,
            fontsize=12,                  # 👈 slightly bigger, readable
            handlelength=1.2,
            handletextpad=0.6,
            borderaxespad=0.0
        )

        # ✅ Make legend text readable while still matching each member theme.
        # We choose white/black based on the slice color brightness.
        legendTextColors = ["#000000" for _ in sliceColors]
        for textObj, txtColor in zip(leg.get_texts(), legendTextColors):
            textObj.set_color(txtColor)

        self.ax.set_title("Line Distribution (seconds)", color="white", fontsize=14)
        self.ax.axis("equal")
        self.fig.tight_layout(rect=[0.0, 0.0, 0.95, 1.0])
        self.canvasAgg.draw()

    def toggle(self, *, x=10, y=-10):
        if self._placed:
            self.hide()
        else:
            self.show(x=x, y=y)

    def _getMemberSecondsSung(self) -> dict:
        result = {}
        memberImages = getattr(self.app, "memberImages", None) or {}
        chunks = getattr(self.app, "chunks", None) or []

        for name, trackItem in memberImages.items():
            tl = getattr(trackItem, "timeline", None)
            if not tl:
                result[name] = 0.0
                continue

            # use the last available element safely
            lastIdx = len(tl) - 1
            if chunks:
                lastIdx = min(lastIdx, len(chunks) - 1)

            try:
                seconds = float(tl[lastIdx])
            except Exception:
                seconds = 0.0

            result[name] = max(0.0, seconds)

        return result

    def computeStats(self) -> dict:
        memberSeconds = self._getMemberSecondsSung()
        items = sorted(memberSeconds.items(), key=lambda kv: kv[1], reverse=True)
        values = [v for _, v in items]

        total = sum(values)
        if len(values) == 0:
            meanS, stdS = 0.0, 0.0
        elif len(values) == 1:
            meanS, stdS = values[0], 0.0
        else:
            meanS = statistics.mean(values)
            stdS = statistics.pstdev(values)

        fairness = 0.0
        if meanS > 1e-9:
            fairness = 1.0 - (stdS / meanS)

        return {
            "sortedItems": items,
            "meanSeconds": meanS,
            "stdSeconds": stdS,
            "fairness": fairness,
            "totalSeconds": total,
        }

    def _targetFairness(self, n: int) -> float:
        if n <= 4:
            return 0.85
        if n >= 7:
            return 0.70
        # n = 5 or 6: linearly slide 0.85 -> 0.70 across 3 steps (4->7)
        t = (n - 4) / 3.0
        return 0.85 + t * (0.70 - 0.85)
    
    def _fairnessGrade(self, fairness: float, n: int) -> tuple[str, float]:
        """
        Returns (letter, target) where letter is curved by group size.
        Bands are relative to target fairness.
        """
        target = self._targetFairness(n)
        delta = fairness - target

        w = 0.10  # band width in fairness points (tune this)

        # Midpoints for base grades
        # A centered at 0, B at -0.10, C at -0.20, D at -0.30
        midpoints = [
            ("A", 0.00),
            ("B", -1 * w),
            ("C", -2 * w),
            ("D", -3 * w),
        ]

        # S is special: anything sufficiently above target becomes S / S+
        # You can tune these thresholds to taste.
        if delta >= w:   # >= +0.15
            # Optional: S+ if very high
            return ("S+" if delta >= 2.5 * w else "S"), target 

        # F is special: far below target
        if delta < -3.5 * w:   # < -0.35
            # Optional: F- if extremely low
            return ("F-" if delta < -4.5 * w else "F"), target

        # Choose nearest midpoint among A/B/C/D
        base, mid = min(midpoints, key=lambda bm: abs(delta - bm[1]))

        # Define the band for this base grade as [mid - w/2, mid + w/2)
        lo = mid - 0.5 * w
        hi = mid + 0.5 * w

        # Clamp delta into the band when computing modifier
        # (edge cases near boundaries won’t blow up)
        x = min(max(delta, lo), hi - 1e-9)

        # progress 0..1 through band (bottom->top)
        progress = (x - lo) / (hi - lo)

        # Modifier rule:
        # bottom 40% => "-"
        # middle 20% => ""
        # top 40% => "+"
        if progress < 0.40:
            mod = "-"
        elif progress > 0.60:
            mod = "+"
        else:
            mod = ""

        return f"{base}{mod}", target
    
    def onResize(self, widthPx: int, heightPx: int):
            # Leave space for your top labels; adjust if you add more UI
            topUiPx = 90
            w = max(200, widthPx)
            h = max(200, heightPx - topUiPx)

            self.fig.set_size_inches(w / self.dpi, h / self.dpi, forward=True)
            self.canvasAgg.draw()
            
    def renderPanelRgba(self) -> Image.Image:
        """
        Returns a PIL RGBA image of the panel including:
        - donut + legend (matplotlib fig)
        - statsLabel + gradeLabel text (Tk labels) rendered onto the image

        Assumes self.update() has been called recently so fig + labels are current.
        """
        # 1) Render the matplotlib figure to a PNG buffer
        buf = BytesIO()
        self.fig.savefig(buf, format="png", dpi=self.dpi, transparent=True)
        buf.seek(0)
        im = Image.open(buf).convert("RGBA")

        # 2) Grab the current text from Tk labels
        statsText = ""
        gradeText = ""
        try:
            statsText = self.statsLabel.cget("text") or ""
        except Exception:
            pass
        try:
            gradeText = self.gradeLabel.cget("text") or ""
        except Exception:
            pass

        if not statsText and not gradeText:
            return im  # nothing to draw

        # 3) Draw text onto the image (top-left), matching your panel style
        draw = ImageDraw.Draw(im)

        # Try to match your label fonts; fallback to default if unavailable
        # (PIL needs a font file path to match Arial exactly)
        try:
            # If you have a bundled font path, use it here.
            fontPath = resourcePath("fonts/PretendardVariable.ttf")  # example
            fontStats = ImageFont.truetype(fontPath, 28)
            fontGrade = ImageFont.truetype(fontPath, 30)
            fontStats = ImageFont.truetype("arial.ttf", 28)
            fontGrade = ImageFont.truetype("arial.ttf", 30)
        except Exception:
            fontStats = ImageFont.load_default()
            fontGrade = ImageFont.load_default()

        # Background strip so text stays readable regardless of chart colors
        padX = 18
        padTop = 14
        lineGap = 8

        # Measure text sizes
        def _text_bbox(txt, font):
            if not txt:
                return (0, 0, 0, 0)
            return draw.textbbox((0, 0), txt, font=font)

        b1 = _text_bbox(statsText, fontStats)
        b2 = _text_bbox(gradeText, fontGrade)
        h1 = (b1[3] - b1[1]) if statsText else 0
        h2 = (b2[3] - b2[1]) if gradeText else 0

        # Total header height
        headerH = padTop + (h1 if h1 else 0) + (lineGap if (h1 and h2) else 0) + (h2 if h2 else 0) + padTop

        # Draw semi-opaque header background (works even if fig bg isn't pure black)
        header = Image.new("RGBA", (im.size[0], headerH), (0, 0, 0, 180))
        im.alpha_composite(header, dest=(0, 0))

        # Draw text
        y = padTop
        if statsText:
            draw.text((padX, y), statsText, font=fontStats, fill=(220, 220, 220, 255))
            y += h1 + (lineGap if gradeText else 0)
        if gradeText:
            draw.text((padX, y), gradeText, font=fontGrade, fill=(255, 255, 255, 255))

        return im