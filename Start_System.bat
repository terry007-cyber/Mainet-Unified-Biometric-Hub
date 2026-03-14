@echo off
title Mainet Biometric Launcher
color 0B
echo ========================================================
echo      MAINET UNIFIED BIOMETRIC SYSTEM - STARTUP
echo ========================================================
echo.

echo 1. Launching FACE RECOGNITION (Port 5000)...
start "Phase 1: Face" cmd /k "cd ma_bi_system && venv\Scripts\activate && python app.py"

echo 2. Launching FINGERPRINT SCANNER (Port 5001)...
start "Phase 2: Fingerprint" cmd /k "cd ma_fingerprint_system && venv\Scripts\activate && python app.py"

echo 3. Launching IRIS SCANNER (Port 5002)...
start "Phase 3: Iris" cmd /k "cd ma_iris_system && venv\Scripts\activate && python app.py"

echo 4. Launching CENTRAL HUB (Port 8080)...
start "Phase 4: Central Hub" cmd /k "cd ma_central_hub && venv\Scripts\activate && python app.py"

echo.
echo ========================================================
echo      SYSTEMS LAUNCHING... OPENING DASHBOARD IN 5s
echo ========================================================
timeout /t 5 >nul
start http://127.0.0.1:8080