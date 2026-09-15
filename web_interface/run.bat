@echo off
title PPD Risk Assessment Server
color 0A
echo.
echo  ============================================
echo   PPD Risk Assessment  --  FastAPI Server
echo  ============================================
echo.
echo  [1/2] Installing / verifying dependencies...
pip install -r requirements.txt --upgrade --quiet
echo.
echo  [2/2] Starting server at http://localhost:8080
echo         Press CTRL+C to stop.
echo.
cd /d "%~dp0"
python -m uvicorn app:app --host 0.0.0.0 --port 8080 --reload
pause
