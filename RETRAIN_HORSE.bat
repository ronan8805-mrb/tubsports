@echo off
title RETRAIN - Horse Racing AI (4-Model Ensemble)
color 0B
echo.
echo ======================================================================
echo   UK/IRE HORSE RACING AI - RETRAIN
echo   RTX 5090 GPU Accelerated (XGBoost CUDA)
echo ======================================================================
echo.
echo   CatBoost + LightGBM + XGBoost + RandomForest ensemble
echo   130+ features (breeding, NLP, pace, market, sectionals)
echo   Purged walk-forward CV + isotonic calibration
echo.
echo   --full-rebuild : Rebuild ALL features from scratch
echo   --tune         : Run Optuna hyperparameter tuning
echo   (default)      : Incremental - only new races
echo.
echo   Press any key to start...
pause >nul

cd /d "%~dp0"

echo.
echo [%TIME%] Step 1: Data enrichment (damsire, ratings, courses, jockey/trainer stats)...
echo.
python -u -m horse.enrich

echo.
echo [%TIME%] Step 2: Model retrain...
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
