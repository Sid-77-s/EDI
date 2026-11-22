# Magic Mirror PowerShell Launcher
Set-Location -Path $PSScriptRoot
$env:PYTHONPATH = ".;MagicMirror_Visualisation;magic_mirror_nlp/nlp/sandbox;magic_mirror_nlp/nlp"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Magic Mirror - Real-Time Multimodal Visualization" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "Starting camera, microphone, YOLO/MediaPipe, and NLP..." -ForegroundColor Yellow
& "$PSScriptRoot\.venv\Scripts\python.exe" integration_live_demo.py --show-hud $args
