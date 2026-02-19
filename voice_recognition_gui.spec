# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Collect non-Python package data files that some ML libs expect at runtime
extra_datas = []
extra_datas += collect_data_files("x_clip")  # fixes missing x_clip/data/bpe_simple_vocab_16e6.txt
extra_datas += [
    ("images\\logo.ico", "images"),
    ("placeholder.png", "."),
    ("looping_background.mp4", "."),
    ("fonts\\PretendardVariable.ttf", "fonts"),
    ("fonts\\Hiragino Sans GB W3.ttf", "fonts")
]

binaries = []
binaries += [("ffmpeg.exe", "."), ("ffprobe.exe", ".")]
binaries += collect_dynamic_libs("cv2") 

a = Analysis(
    ['voice_recognition_gui.py'],
    pathex=[],
    binaries=binaries,
    datas=extra_datas,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Line Distribution Creator',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    windowed=True,
    icon='images\\logo.ico',
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)