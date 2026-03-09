@echo off
title Horse Racing - Data Enrichment Engine
color 0E

echo ============================================================
echo   DATA ENRICHMENT ENGINE
echo ============================================================
echo.
echo   Computes derived statistics from 2.28M raw results:
echo.
echo     1. Damsire stats      (grandoffspring win rates)
echo     2. Ratings history    (official rating snapshots)
echo     3. Course profiles    (draw bias, pace bias, direction)
echo     4. Jockey stats       (win rates, place rates)
echo     5. Trainer stats      (win rates, place rates)
echo.
echo   This unlocks features that are currently producing NaN.
echo   Run before retraining for maximum model power.
echo.
echo ============================================================
echo.

cd /d "%~dp0"
python -m horse.enrich

echo.
echo ============================================================
echo   ENRICHMENT COMPLETE
echo ============================================================
echo.
echo   Now run RETRAIN_HORSE.bat to retrain with full data.
echo.
pause
