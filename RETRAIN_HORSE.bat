@echo off
title RETRAIN - Horse Racing AI (LightGBM + XGBoost GPU)
color 0B
echo.
echo ======================================================================
echo   UK/IRE HORSE RACING AI - RETRAIN
echo   RTX 5090 GPU Accelerated (XGBoost CUDA)
echo ======================================================================
echo.
echo   LightGBM (CPU) + XGBoost (GPU) ensemble
echo   ~120 features (form, jockey, trainer, breeding, draw, context)
echo   Leakage audit + time-based CV + calibration
echo.
echo   ESTIMATED TIME: ~5-15 minutes
echo.
echo   Press any key to start...
pause >nul

cd /d "%~dp0"

echo.
echo [%TIME%] Starting horse model retrain...
echo.
python -u -m horse.retrain

echo.
if %ERRORLEVEL% EQU 0 (
    color 0A
    echo ======================================================================
    echo   HORSE RETRAIN COMPLETE - SUCCESS
    echo ======================================================================
    echo.
    echo   Models saved to horse\data\models\
    echo   Now restart horse API to use new models.
    echo.
) else (
    color 0C
    echo ======================================================================
    echo   HORSE RETRAIN FAILED - Check errors above
    echo ======================================================================
    echo   Also check horse_retrain.log for details
)
echo Press any key to close...
pause >nul
