@echo off
title Push Predictions to Render
cd /d "%~dp0"

echo ========================================
echo   Pre-compute + Upload Predictions
echo ========================================
echo.

echo [1/2] Running predictions locally...
python -u -m horse.precompute
if %ERRORLEVEL% NEQ 0 (
    echo FAILED - precompute error
    pause
    exit /b 1
)

echo.
echo [2/2] Uploading cache to Render...
scp -r horse/data/predictions_cache srv-d6nh2p5m5p6s73ct5920@ssh.oregon.render.com:/var/data/horse/
if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS - Predictions live on Render!
) else (
    echo.
    echo UPLOAD FAILED - try running from Git Bash:
    echo   scp -r horse/data/predictions_cache srv-d6nh2p5m5p6s73ct5920@ssh.oregon.render.com:/var/data/horse/
)

echo.
pause
