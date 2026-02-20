@echo off
title Horse Racing Daily Cycle
color 0E
echo.
echo ======================================================================
echo   HORSE RACING - DAILY CYCLE
echo   Scrape results  ^>  Reconcile predictions  ^>  Fetch tomorrow's card
echo ======================================================================
echo.
echo   This script runs the daily feedback loop:
echo     1. Scrape today's race RESULTS (so model can learn)
echo     2. Reconcile predictions vs actual outcomes
echo     3. Fetch tomorrow's RACECARDS (so model can predict)
echo.
echo   Run this every evening after racing finishes.
echo.
echo   Press any key to start...
pause >nul

cd /d "%~dp0"

echo.
echo [%TIME%] ===== STEP 1: Scraping latest results =====
echo [%TIME%] Fetching results from The Racing API...
python -u -m horse.scrapers.backfill --phases 2
if errorlevel 1 (
    echo [%TIME%] WARNING: Results scrape had errors, continuing...
)

echo.
echo [%TIME%] ===== STEP 2: Reconciling predictions =====
echo [%TIME%] Checking model predictions against actual results...
python -u -m horse.reconcile
if errorlevel 1 (
    echo [%TIME%] WARNING: Reconcile had errors, continuing...
)

echo.
echo [%TIME%] ===== STEP 3: Fetching tomorrow's racecards =====
echo [%TIME%] Getting upcoming race entries...
python -u -m horse.scrapers.backfill --phases 8
if errorlevel 1 (
    echo [%TIME%] WARNING: Racecard scrape had errors, continuing...
)

echo.
echo ======================================================================
echo   DAILY CYCLE COMPLETE
echo.
echo   Results scraped     - model can learn from today's outcomes
echo   Predictions checked - see Performance tab in the UI
echo   Racecards fetched   - tomorrow's races ready for prediction
echo.
echo   Next: Open the UI (START_HORSE.bat) to view predictions
echo ======================================================================
echo.
pause
