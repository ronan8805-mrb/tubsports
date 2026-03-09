@echo off
title Horse Racing - 15 Year Historical Download
color 0E

cd /d "%~dp0"

echo.
echo ======================================================================
echo   HORSE RACING - HISTORICAL DATA DOWNLOAD
echo   15 Years (2011 to %DATE:~6,4%)
echo ======================================================================
echo.
echo   This will download race results year-by-year from The Racing API
echo   and store them locally:
echo.
echo     Raw JSON:  horse\data\raw\YYYY_raw.json
echo     Parquet:   horse\data\results\YYYY.parquet
echo     Database:  horse\data\horse.duckdb
echo.
echo   Estimated time: 2-4 hours (API rate limited)
echo   Estimated disk: 5-8 GB total
echo.
echo   Press any key to start...
pause >nul

echo.
echo [%TIME%] Starting historical download...
echo.
python -u -m horse.scrapers.historical_download

echo.
if %ERRORLEVEL% EQU 0 (
    color 0A
    echo ======================================================================
    echo   DOWNLOAD COMPLETE - %DATE% %TIME%
    echo   Check historical_download.log for details.
    echo ======================================================================
) else (
    color 0C
    echo ======================================================================
    echo   DOWNLOAD HAD ERRORS - %DATE% %TIME%
    echo   Check historical_download.log for details.
    echo ======================================================================
)

echo.
echo   Press any key to close...
pause >nul
