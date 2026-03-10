@echo off
title Push to GitHub
cd /d "%~dp0"

echo ========================================
echo   Pushing to GitHub (tubsports)
echo ========================================
echo.

git remote set-url origin https://github.com/ronan8805-mrb/tubsports.git
echo Remote: https://github.com/ronan8805-mrb/tubsports.git
echo.

echo Pushing to origin main...
git push -u origin main
echo.

if %ERRORLEVEL% EQU 0 (
    echo SUCCESS - Code pushed to GitHub!
) else (
    echo FAILED - see error above
)

echo.
pause
