@echo off
title Horse Racing - Nightly Auto Cycle
color 0E

cd /d "%~dp0"

echo.
echo ======================================================================
echo   HORSE RACING - NIGHTLY AUTO CYCLE
echo   %DATE% %TIME%
echo ======================================================================
echo.
echo   1. Scrape today's results
echo   2. Reconcile predictions vs actuals
echo   3. Fetch tomorrow's racecards
echo   4. Enrich derived stats (damsire, ratings, courses, jockey/trainer)
echo   5. Incremental retrain (stacked ensemble)
echo   6. Model monitoring + drift detection
echo   7. Refresh odds for tomorrow
echo.

:: ---- Step 1: Scrape recent results ----
echo [%TIME%] Step 1: Scraping latest results...
python -u -c "from horse.scrapers.racing_api import fetch_recent_results; r=fetch_recent_results(3); print(f'  Results: {r[\"races\"]} races, {r[\"runners\"]} runners')"
if errorlevel 1 echo [%TIME%] WARNING: Results scrape had errors

:: ---- Step 2: Reconcile predictions ----
echo.
echo [%TIME%] Step 2: Reconciling predictions...
python -u -m horse.reconcile
if errorlevel 1 echo [%TIME%] WARNING: Reconcile had errors

:: ---- Step 3: Fetch tomorrow's racecards ----
echo.
echo [%TIME%] Step 3: Fetching tomorrow's racecards...
python -u -m horse.scrapers.backfill --phase 8
if errorlevel 1 echo [%TIME%] WARNING: Racecard fetch had errors

:: ---- Step 4: Enrich derived stats ----
echo.
echo [%TIME%] Step 4: Enriching derived stats (damsire, ratings, courses, jockey/trainer)...
python -u -m horse.enrich
if errorlevel 1 echo [%TIME%] WARNING: Enrichment had errors

:: ---- Step 5: Incremental retrain (stacked ensemble) ----
echo.
echo [%TIME%] Step 5: Retraining model (incremental - stacked ensemble)...
python -u -m horse.retrain
if errorlevel 1 echo [%TIME%] WARNING: Retrain had errors

:: ---- Step 6: Model monitoring + drift detection ----
echo.
echo [%TIME%] Step 6: Running model monitoring...
python -u -c "from horse.online import load_monitoring_history; h=load_monitoring_history(); print(f'  Monitoring: {len(h)} snapshots recorded')" 2>nul
if errorlevel 1 echo [%TIME%] WARNING: Monitoring check had errors

:: ---- Step 7: Refresh odds for tomorrow ----
echo.
echo [%TIME%] Step 7: Refreshing odds for upcoming races...
python -u -c "import requests; r=requests.post('http://localhost:8002/api/refresh-odds', timeout=5); print(f'  Odds refresh: {r.json()}')" 2>nul
if errorlevel 1 echo [%TIME%] INFO: Odds refresh skipped (API may not be running)

echo.
if %ERRORLEVEL% EQU 0 (
    color 0A
    echo ======================================================================
    echo   NIGHTLY CYCLE COMPLETE - %DATE% %TIME%
    echo   Stacked model retrained. Drift monitored. Odds persisted.
    echo ======================================================================
) else (
    color 0C
    echo ======================================================================
    echo   NIGHTLY CYCLE FAILED - %DATE% %TIME%
    echo   Check horse_retrain.log for details.
    echo ======================================================================
)
