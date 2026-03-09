@echo off
REM ============================================================
REM  DOWNLOAD_SECTIONALS.bat
REM  Scrapes YESTERDAY'S RacingTV GPS sectional data into horse.duckdb
REM
REM  IMPORTANT: RacingTV only keeps sectional data for ~24-48 hours.
REM             Run this EVERY DAY (ideally each morning for yesterday).
REM
REM  DAILY USE (run each morning):
REM    DOWNLOAD_SECTIONALS.bat
REM
REM  SPECIFIC DATE:
REM    DOWNLOAD_SECTIONALS.bat 2026-03-05 2026-03-05
REM
REM  DRY RUN:
REM    DOWNLOAD_SECTIONALS.bat --dry-run
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo  RacingTV Sectionals Downloader -- YESTERDAY ONLY
echo  %DATE% %TIME%
echo ============================================================

IF "%~1"=="--dry-run" (
    echo [DRY RUN] No data will be written.
    python -u -m horse.scrapers.sectionals --days 1 --dry-run
    GOTO :END
)

IF NOT "%~1"=="" IF NOT "%~2"=="" (
    echo Specific date range: %~1  to  %~2
    python -u -m horse.scrapers.sectionals --start %~1 --end %~2
    GOTO :END
)

REM Default: YESTERDAY only (data expires after ~48 hours)
python -u -m horse.scrapers.sectionals --days 1

:END
echo.
echo Done. %DATE% %TIME%
echo Run RETRAIN_HORSE.bat weekly to update the model with new sectional data.
pause