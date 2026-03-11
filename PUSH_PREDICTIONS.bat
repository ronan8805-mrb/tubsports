@echo off
title Pre-compute Predictions
cd /d "%~dp0"

echo ========================================
echo   Pre-compute Predictions (Local)
echo ========================================
echo.

echo [1/1] Running predictions locally...
python -u -m horse.precompute
if %ERRORLEVEL% EQU 0 (
    echo.
    echo SUCCESS - Predictions cached locally. Served via Cloudflare Tunnel.
) else (
    echo.
    echo FAILED - precompute error
)

echo.
pause
