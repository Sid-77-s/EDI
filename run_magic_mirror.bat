@echo off
title Magic Mirror — Multimodal Spatial Visualization System
cd /d "%~dp0"
echo ============================================================
echo   Magic Mirror - Real-Time Multimodal Visualization
echo ============================================================
echo Starting camera, microphone, YOLO/MediaPipe, and NLP...
echo.
set PYTHONPATH=.;MagicMirror_Visualisation;magic_mirror_nlp/nlp/sandbox;magic_mirror_nlp/nlp
.venv\Scripts\python.exe integration_live_demo.py --show-hud %*
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo System exited with code %ERRORLEVEL%.
)
pause
