@echo off
title Setup Windows Task Scheduler
cd /d "%~dp0"

echo.
echo ======================================================================
echo   SETTING UP AUTOMATIC SCHEDULE (run once as Administrator)
echo.
echo   23:00  Nightly cycle  - retrain, precompute, odds pass 1
echo   09:00  Morning odds   - re-fetch prices, harmony pass 2
echo   12:00  Midday odds    - re-fetch prices, harmony pass 3
echo ======================================================================
echo.

set NIGHTLY_TASK=HorseRacing_NightlyCycle
set MORNING_TASK=HorseRacing_MorningOdds
set MIDDAY_TASK=HorseRacing_MiddayOdds

set NIGHTLY_BAT=%~dp0NIGHTLY_CYCLE.bat
set ODDS_BAT=%~dp0MORNING_ODDS.bat

:: ---- Remove old tasks (safe to re-run) ----
schtasks /delete /tn "%NIGHTLY_TASK%" /f >nul 2>&1
schtasks /delete /tn "%MORNING_TASK%" /f >nul 2>&1
schtasks /delete /tn "%MIDDAY_TASK%"  /f >nul 2>&1

set ERRORS=0

:: ---- 1. Nightly cycle at 23:00 ----
echo Registering nightly cycle at 23:00...
schtasks /create ^
  /tn "%NIGHTLY_TASK%" ^
  /tr "\"%NIGHTLY_BAT%\"" ^
  /sc daily ^
  /st 23:00 ^
  /ru "%USERNAME%" ^
  /rl highest ^
  /f
if errorlevel 1 ( echo   ERROR: nightly task failed & set ERRORS=1 ) else ( echo   OK )

:: ---- 2. Morning odds at 09:00 ----
echo Registering morning odds refresh at 09:00...
schtasks /create ^
  /tn "%MORNING_TASK%" ^
  /tr "\"%ODDS_BAT%\"" ^
  /sc daily ^
  /st 09:00 ^
  /ru "%USERNAME%" ^
  /rl highest ^
  /f
if errorlevel 1 ( echo   ERROR: morning task failed & set ERRORS=1 ) else ( echo   OK )

:: ---- 3. Midday odds at 12:00 ----
echo Registering midday odds refresh at 12:00...
schtasks /create ^
  /tn "%MIDDAY_TASK%" ^
  /tr "\"%ODDS_BAT%\"" ^
  /sc daily ^
  /st 12:00 ^
  /ru "%USERNAME%" ^
  /rl highest ^
  /f
if errorlevel 1 ( echo   ERROR: midday task failed & set ERRORS=1 ) else ( echo   OK )

echo.
if "%ERRORS%"=="1" (
    color 0C
    echo ======================================================================
    echo   ONE OR MORE TASKS FAILED
    echo   Right-click this file and choose "Run as administrator" and retry.
    echo ======================================================================
) else (
    color 0A
    echo ======================================================================
    echo   ALL 3 TASKS SCHEDULED SUCCESSFULLY
    echo.
    echo   23:00  NIGHTLY_CYCLE.bat  retrain + precompute + odds pass 1
    echo   09:00  MORNING_ODDS.bat   fresh morning prices  + harmony pass 2
    echo   12:00  MORNING_ODDS.bat   pre-race prices       + harmony pass 3
    echo.
    echo   Nothing else to do. Runs automatically every day.
    echo ======================================================================
)

echo.
pause
