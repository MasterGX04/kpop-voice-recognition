@echo off
cd /d "%~dp0"
start "" ".\venv\Scripts\pythonw.exe" -m gui.voice_recognition_gui
exit