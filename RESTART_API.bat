@echo off
title Horse API (port 8002)
cd /d "%~dp0"

echo Killing old API on port 8002...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8002 ^| findstr LISTENING') do (
    taskkill /PID %%a /F >nul 2>&1
)
taskkill /FI "WINDOWTITLE eq Horse API*" /F >nul 2>&1
timeout /t 2 /nobreak >nul

echo Starting Horse Racing API on port 8002...
python -u -m uvicorn horse.api.main:app --host 0.0.0.0 --port 8002
pause
