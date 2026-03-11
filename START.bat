@echo off
title Smart Expense Tracker
color 0A

echo.
echo ============================================
echo   Smart Expense Tracker - ML Powered
echo ============================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.9+ from python.org
    pause
    exit /b 1
)

:: Change to script directory
cd /d "%~dp0"

:: Install dependencies if needed
echo [1/3] Checking dependencies...
pip install -r requirements.txt -q --disable-pip-version-check
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Check API key
if "%ANTHROPIC_API_KEY%"=="" (
    echo.
    echo [WARN] ANTHROPIC_API_KEY not set.
    echo        AI Advisor and Receipt Scanning need it.
    echo        Set it by running: set ANTHROPIC_API_KEY=your_key_here
    echo        Or add it to your system environment variables.
    echo.
    echo        App will still run - ML features work without the API key.
    echo.
)

echo [2/3] Starting ML engine...
echo [3/3] Launching app...
echo.
echo  Open your browser: http://127.0.0.1:5000
echo  Press Ctrl+C to stop
echo.
echo ============================================

python app.py

pause
