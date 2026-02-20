@echo off
title Horse Racing Prediction System
color 0A

echo.
echo ======================================================================
echo   HORSE RACING PREDICTIVE INTELLIGENCE SYSTEM
echo   RTX 5090 Accelerated  -  14D Engine
echo ======================================================================
echo.
echo   Horse 14D Engine   Frontend :5174   API :8002
echo.
echo   Press any key to start...
pause >nul

cd /d "%~dp0"

:: ---- Kill stale processes ----
echo.
echo [%TIME%] [1/3] Killing old processes...
taskkill /FI "WINDOWTITLE eq Horse API*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Horse Frontend*" /F >nul 2>&1
timeout /t 2 /nobreak >nul
echo          Done.

:: ---- Horse Backend ----
echo.
echo [%TIME%] [2/3] Starting Horse Racing API (port 8002)...
start "Horse API" /MIN cmd /c "cd /d "%~dp0" && python -u -m uvicorn horse.api.main:app --host 0.0.0.0 --port 8002"
timeout /t 3 /nobreak >nul

:: ---- Horse Frontend ----
echo [%TIME%] [3/3] Starting Horse Frontend (port 5174)...
start "Horse Frontend" /MIN cmd /c "cd /d "%~dp0\horse\frontend" && npm run dev"

timeout /t 6 /nobreak >nul

:: ---- Open browser ----
echo.
echo [%TIME%] Opening browser...
start http://localhost:5174

echo.
echo ======================================================================
echo   HORSE RACING SYSTEM RUNNING!
echo.
echo   HORSE RACING (14D Engine)
echo     Frontend:  http://localhost:5174
echo     API:       http://localhost:8002
echo     Health:    http://localhost:8002/api/health
echo.
echo   All predictions are live. No fake data.
echo ======================================================================
echo.
echo   Press any key to STOP all services...
pause >nul

echo.
echo [%TIME%] Stopping all services...
taskkill /FI "WINDOWTITLE eq Horse API*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Horse Frontend*" /F >nul 2>&1
echo Done.
