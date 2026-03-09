@echo off
REM ============================================================
REM  IMPORT TIMEFORM SECTIONAL ARCHIVE
REM  Imports historical sectional data from Timeform Excel file
REM  into the sectionals table for ML training.
REM
REM  Usage:
REM    IMPORT_SECTIONALS.bat                   <- import default file
REM    IMPORT_SECTIONALS.bat --dry-run         <- preview without writing
REM    IMPORT_SECTIONALS.bat --file myfile.xlsx
REM ============================================================

cd /d "%~dp0"

echo ============================================================
echo  TIMEFORM SECTIONAL IMPORTER
echo ============================================================
echo.

REM Stop API server to release DB lock
echo Stopping API server to release database lock...
taskkill /F /IM python.exe /T >nul 2>&1
timeout /t 3 /nobreak >nul

REM Default file
set XLSX=SectionalArchive10240325.xlsx

REM Check for --file argument
:parse
if "%~1"=="--file" (
    set XLSX=%~2
    shift
    shift
    goto parse
)
if "%~1"=="--dry-run" (
    set DRYRUN=--dry-run
    shift
    goto parse
)

echo Importing from: %XLSX%
if defined DRYRUN echo Mode: DRY-RUN (no changes written)
echo.

python -u -m horse.scrapers.import_timeform --file "%XLSX%" %DRYRUN%

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ============================================================
    echo  IMPORT COMPLETE
    echo  Next step: run RETRAIN_HORSE.bat to use sectional features
    echo ============================================================
) else (
    echo.
    echo ERROR: Import failed. Check output above.
)

REM Restart API server
echo.
echo Restarting API server...
start "" /MIN python -u -m horse.api.main

pause
