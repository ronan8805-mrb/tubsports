@echo off
title Sectionals Backfill (77 Weeks)
cd /d "%~dp0"

echo ============================================
echo   Sectionals Backfill - 77 Weeks
echo   RacingTV GPS split times
echo ============================================
echo.
echo   Range: 2024-07-05 to 2026-02-20
echo   Skips races already in DB
echo   Auto-restarts on crash
echo.
echo ============================================
echo.

:retry
python -m horse.scrapers.sectionals --start 2024-07-05 --end 2026-02-20
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Crashed with code %ERRORLEVEL%. Restarting in 10 seconds...
    timeout /t 10 /nobreak
    goto retry
)

echo.
echo ============================================
echo   Backfill complete.
echo ============================================
pause
