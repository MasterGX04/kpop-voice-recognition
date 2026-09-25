@echo off
cd /d "%~dp0"
start "" ".\pyannote-env\Scripts\pythonw.exe" -m gui.voice_recognition_gui
exit