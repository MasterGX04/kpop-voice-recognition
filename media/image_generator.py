from PIL import Image, ImageDraw, ImageFont
import os

def _safeLoadFont(fontPath: str, size: int) -> ImageFont.ImageFont:
    """
    Never crashes. Tries:
      1) Provided TTF/OTF path (if exists)
      2) DejaVuSans.ttf (often shipped with Pillow on many installs)
      3) PIL built-in default bitmap font
    """
    size = max(1, int(size or 1))

    # 1) user/bundled font path
    if isinstance(fontPath, str):
        fp = fontPath.strip()
        if fp and os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass

    # 2) common fallback
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass

    # 3) last resort (bitmap font, size not respected)
    return ImageFont.load_default()

def _drawPerfectRing(canvas: Image.Image, ringColor, *, circleSize: int, ringThickness: int):
    """
    Draws a ring that *guarantees* it reaches the image edges with no 1px gaps.
    Does not rely on outline stroke centering.
    """
    ringThickness = int(ringThickness)
    circleSize = int(circleSize)

    # Outer ellipse hits the boundary exactly
    outerBBox = (0, 0, circleSize - 1, circleSize - 1)

    # Inner ellipse inset by thickness (this punches out the center)
    inset = ringThickness
    innerBBox = (inset, inset, (circleSize - 1) - inset, (circleSize - 1) - inset)

    # Make a mask: outer white, inner black
    mask = Image.new("L", (circleSize, circleSize), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.ellipse(outerBBox, fill=255)
    if innerBBox[2] > innerBBox[0] and innerBBox[3] > innerBBox[1]:
        mdraw.ellipse(innerBBox, fill=0)

    # Paint the ring color through the mask
    ringLayer = Image.new("RGBA", (circleSize, circleSize), (0, 0, 0, 0))
    rdraw = ImageDraw.Draw(ringLayer)
    rdraw.rectangle((0, 0, circleSize - 1, circleSize - 1), fill=ringColor)

    # The alpha_composite above ignores mask, so we paste with mask:
    canvas.paste(ringLayer, (0, 0), mask)

def make_member_card(
    member_name: str,
    color_x,  # ring color (e.g. "#FF00AA" or (255,0,170))
    color_y,  # background color (e.g. "#111111" or (17,17,17))
    circles_dir: str,
    output_dir: str,
    output_path: str,
    font_path: str,
    canvas_size=(1000, 300),
    circle_size=300,
    ring_thickness=8,
    text_pos=(330, -5),
    start_font_size=140,
    shrink_step=5,
    right_padding=10
):
    os.makedirs(output_dir, exist_ok=True)

    # 1) Create 1000x300 canvas with background color y
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    
    draw.rectangle(
        [(0, 0), (circle_size - 1, circle_size - 1)],
        fill=color_y
    )

    # 2) Load and place the circular image at [0,0]
    circle_filename = f"{member_name} Circle.png"
    circle_path = os.path.join(circles_dir, circle_filename)
    circle_img = Image.open(circle_path).convert("RGBA")

    # Ensure it's exactly 300x300 (you can change to LANCZOS for best quality)
    if circle_img.size != (circle_size, circle_size):
        circle_img = circle_img.resize((circle_size, circle_size), Image.LANCZOS)

    canvas.alpha_composite(circle_img, dest=(0, 0))

    # 4) Draw the 7px ring circle (300x300) at [0,0] with color x
    _drawPerfectRing(canvas, color_x, circleSize=circle_size, ringThickness=ring_thickness)

    # 5) Fit text: start at 140, shrink by -5 until it fits remaining width
    max_text_width = canvas_size[0] - text_pos[0] - right_padding
    font_size = start_font_size

    font = _safeLoadFont(font_path, font_size)

    def text_width(fnt: ImageFont.FreeTypeFont, txt: str) -> int:
        bbox = draw.textbbox((0, 0), txt, font=fnt)
        return bbox[2] - bbox[0]

    while text_width(font, member_name) > max_text_width and font_size > 5:
        font_size -= shrink_step
        font = _safeLoadFont(font_path, font_size)

    # 6) Draw the name at [330,25] in color y
    draw.text(text_pos, member_name, font=font, fill=color_y)

    # 7) Save as {memberName}.png in output_dir (keeps transparency)
    out_path = os.path.join(output_dir, output_path)
    canvas.save(out_path, format="PNG")
    return out_path

def make_dark_member_card(
    member_name: str,
    color_y,  # square color (variable)
    circles_dir: str,
    output_dir: str,
    output_path: str,  # e.g. f"Dark {member_name}.png"
    font_path: str,
    canvas_size=(1000, 300),
    circle_size=300,
    ring_thickness=8,
    text_pos=(330, -5),
    start_font_size=140,
    shrink_step=5,
    right_padding=10
):
    os.makedirs(output_dir, exist_ok=True)

    RING_COLOR = "#c9c9c9"
    TEXT_COLOR = "#ffffff"

    # 1) Create 1000x300 fully transparent canvas
    canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # 2) Draw the 300x300 square of color_y at [0,0] (behind everything)
    draw.rectangle([(0, 0), (circle_size - 1, circle_size - 1)], fill=color_y)

    # 3) Load member circle image
    circle_filename = f"{member_name} Circle.png"
    circle_path = os.path.join(circles_dir, circle_filename)
    circle_img = Image.open(circle_path).convert("RGBA")

    if circle_img.size != (circle_size, circle_size):
        circle_img = circle_img.resize((circle_size, circle_size), Image.LANCZOS)

    # 4) Greyscale the image (preserve alpha)
    r, g, b, a = circle_img.split()
    gray_rgb = Image.merge("RGB", (r, g, b)).convert("L")      # luminance
    gray_rgb = Image.merge("RGB", (gray_rgb, gray_rgb, gray_rgb))
    circle_img_gray = Image.merge("RGBA", (*gray_rgb.split(), a))

    # Paste the greyed member image
    canvas.alpha_composite(circle_img_gray, dest=(0, 0))

    # 5) Draw ring: ALWAYS #c9c9c9
    _drawPerfectRing(canvas, RING_COLOR, circleSize=circle_size, ringThickness=ring_thickness)

    # 6) Fit text and draw: ALWAYS white
    max_text_width = canvas_size[0] - text_pos[0] - right_padding
    font_size = start_font_size

    font = _safeLoadFont(font_path, font_size)

    def text_width(fnt: ImageFont.FreeTypeFont, txt: str) -> int:
        bbox = draw.textbbox((0, 0), txt, font=fnt)
        return bbox[2] - bbox[0]

    while text_width(font, member_name) > max_text_width and font_size > 5:
        font_size -= shrink_step
        font = _safeLoadFont(font_path, font_size)

    draw.text(text_pos, member_name, font=font, fill=TEXT_COLOR)

    # 7) Save output
    out_path = os.path.join(output_dir, output_path)
    canvas.save(out_path, format="PNG")
    return out_path