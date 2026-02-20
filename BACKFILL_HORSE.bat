@echo off
title Horse Racing - Full Data Backfill (The Racing API)
echo ============================================================
echo   HORSE RACING - FULL DATA BACKFILL
echo   The Racing API (PRO) -^> horse.duckdb
echo   Last 12 months results + full career form per horse
echo   All countries ^| All surfaces ^| All race types
echo ============================================================
echo.
echo   Phase 1: Sync courses (~2 sec)
echo   Phase 2: Backfill results - last 12 months (~30-60 min)
echo   Phase 3: Validation report
echo   Phase 4: Horse profiles + career form history (~3-6 hours)
echo   Phase 5: Jockey + Trainer data (~1-2 hours)
echo   Phase 6: Sire/Dam/Damsire data (~2-4 hours)
echo   Phase 7: Cleanup + stats
echo   Phase 8: Upcoming racecards
echo.
echo   Total estimated time: ~7-13 hours (fully resumable)
echo   Press Ctrl+C at any time. Re-run to resume where you left off.
echo.
echo ============================================================
echo.

cd /d "%~dp0"

REM Run all phases (or pass --phase N for specific phase)
python -m horse.scrapers.backfill %*

echo.
echo ============================================================
echo   BACKFILL COMPLETE
echo ============================================================
pause
