@echo off
title Horse Racing - Morning Odds Refresh
color 0B

cd /d "%~dp0"

echo.
echo ======================================================================
echo   MORNING ODDS REFRESH
echo   %DATE% %TIME%
echo   Run this at ~9am before racing starts
echo ======================================================================
echo.

:: ---- Step 1: Fetch fresh opening line odds from Racing API ----
echo [%TIME%] Step 1: Fetching morning odds from Racing API...
python -u -c "from horse.scrapers.racing_api import RacingAPIClient, fetch_odds_for_today; c = RacingAPIClient(); n = fetch_odds_for_today(c); print(f'  Odds fetched for {n} races')"
if errorlevel 1 (
    echo [%TIME%] ERROR: Odds fetch failed. Check Racing API credentials.
    goto :fail
)

:: ---- Step 2: Patch predictions cache with harmony scores ----
echo.
echo [%TIME%] Step 2: Computing harmony scores and patching cache...
python -u -m horse.refresh_harmony
if errorlevel 1 (
    echo [%TIME%] ERROR: Harmony refresh failed.
    goto :fail
)

echo.
color 0A
echo ======================================================================
echo   MORNING ODDS COMPLETE - %DATE% %TIME%
echo   Best Bets page will now show bookmaker odds and harmony scores.
echo ======================================================================
goto :end

:fail
color 0C
echo.
echo ======================================================================
echo   MORNING ODDS FAILED - %DATE% %TIME%
echo ======================================================================

:end
echo.
pause
