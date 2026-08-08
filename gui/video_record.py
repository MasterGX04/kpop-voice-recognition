import ctypes
from ctypes import wintypes
from PIL import Image

user32 = ctypes.WinDLL("user32", use_last_error=True)
gdi32  = ctypes.WinDLL("gdi32", use_last_error=True)

PW_RENDERFULLCONTENT = 0x00000002
SRCCOPY = 0x00CC0020
BI_RGB = 0

class RECT(ctypes.Structure):
    _fields_ = [("left", ctypes.c_long),
                ("top", ctypes.c_long),
                ("right", ctypes.c_long),
                ("bottom", ctypes.c_long)]

class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", wintypes.DWORD),
        ("biWidth", wintypes.LONG),
        ("biHeight", wintypes.LONG),
        ("biPlanes", wintypes.WORD),
        ("biBitCount", wintypes.WORD),
        ("biCompression", wintypes.DWORD),
        ("biSizeImage", wintypes.DWORD),
        ("biXPelsPerMeter", wintypes.LONG),
        ("biYPelsPerMeter", wintypes.LONG),
        ("biClrUsed", wintypes.DWORD),
        ("biClrImportant", wintypes.DWORD),
    ]

class BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", BITMAPINFOHEADER),
                ("bmiColors", wintypes.DWORD * 3)]

# ---- prototypes (important) ----
user32.GetClientRect.argtypes = [wintypes.HWND, ctypes.POINTER(RECT)]
user32.GetClientRect.restype = wintypes.BOOL

user32.GetDC.argtypes = [wintypes.HWND]
user32.GetDC.restype = wintypes.HDC

user32.ReleaseDC.argtypes = [wintypes.HWND, wintypes.HDC]
user32.ReleaseDC.restype = ctypes.c_int

user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
user32.PrintWindow.restype = wintypes.BOOL

gdi32.CreateCompatibleDC.argtypes = [wintypes.HDC]
gdi32.CreateCompatibleDC.restype = wintypes.HDC

gdi32.CreateCompatibleBitmap.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int]
gdi32.CreateCompatibleBitmap.restype = wintypes.HBITMAP

gdi32.SelectObject.argtypes = [wintypes.HDC, wintypes.HGDIOBJ]
gdi32.SelectObject.restype = wintypes.HGDIOBJ

gdi32.DeleteObject.argtypes = [wintypes.HGDIOBJ]
gdi32.DeleteObject.restype = wintypes.BOOL

gdi32.DeleteDC.argtypes = [wintypes.HDC]
gdi32.DeleteDC.restype = wintypes.BOOL

gdi32.BitBlt.argtypes = [wintypes.HDC, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                         wintypes.HDC, ctypes.c_int, ctypes.c_int, wintypes.DWORD]
gdi32.BitBlt.restype = wintypes.BOOL

gdi32.GetDIBits.argtypes = [wintypes.HDC, wintypes.HBITMAP, wintypes.UINT, wintypes.UINT,
                           wintypes.LPVOID, ctypes.POINTER(BITMAPINFO), wintypes.UINT]
gdi32.GetDIBits.restype = ctypes.c_int

def captureWindowClientRGBA(hwnd: int) -> Image.Image:
    hwnd = wintypes.HWND(hwnd)

    rect = RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        raise OSError(f"GetClientRect failed (err={ctypes.get_last_error()})")

    w = rect.right - rect.left
    h = rect.bottom - rect.top
    if w <= 0 or h <= 0:
        raise ValueError("Invalid window size")

    hdc_window = None
    hdc_mem = None
    hbmp = None
    old_obj = None

    try:
        hdc_window = user32.GetDC(hwnd)
        if not hdc_window:
            raise OSError(f"GetDC failed (err={ctypes.get_last_error()})")

        hdc_mem = gdi32.CreateCompatibleDC(hdc_window)
        if not hdc_mem:
            raise OSError(f"CreateCompatibleDC failed (err={ctypes.get_last_error()})")

        hbmp = gdi32.CreateCompatibleBitmap(hdc_window, w, h)
        if not hbmp:
            raise OSError(f"CreateCompatibleBitmap failed (err={ctypes.get_last_error()})")

        old_obj = gdi32.SelectObject(hdc_mem, hbmp)

        # Try PrintWindow first
        ok = user32.PrintWindow(hwnd, hdc_mem, PW_RENDERFULLCONTENT)

        # Fallback to BitBlt if PrintWindow fails
        if not ok:
            ok2 = gdi32.BitBlt(hdc_mem, 0, 0, w, h, hdc_window, 0, 0, SRCCOPY)
            if not ok2:
                raise OSError(f"PrintWindow+BitBlt failed (err={ctypes.get_last_error()})")

        bmi = BITMAPINFO()
        bmi.bmiHeader.biSize = ctypes.sizeof(BITMAPINFOHEADER)
        bmi.bmiHeader.biWidth = w
        bmi.bmiHeader.biHeight = -h  # top-down
        bmi.bmiHeader.biPlanes = 1
        bmi.bmiHeader.biBitCount = 32
        bmi.bmiHeader.biCompression = BI_RGB

        buf = (ctypes.c_byte * (w * h * 4))()
        res = gdi32.GetDIBits(hdc_mem, hbmp, 0, h, ctypes.byref(buf), ctypes.byref(bmi), 0)
        if res == 0:
            raise OSError(f"GetDIBits failed (err={ctypes.get_last_error()})")

        return Image.frombuffer("RGBA", (w, h), bytes(buf), "raw", "BGRA", 0, 1)

    finally:
        # cleanup (ALWAYS)
        try:
            if old_obj and hdc_mem:
                gdi32.SelectObject(hdc_mem, old_obj)
        except Exception:
            pass
        if hbmp:
            gdi32.DeleteObject(hbmp)
        if hdc_mem:
            gdi32.DeleteDC(hdc_mem)
        if hdc_window:
            user32.ReleaseDC(hwnd, hdc_window)