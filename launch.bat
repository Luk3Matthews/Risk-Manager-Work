@echo off
title Risk Manager Dashboard
cd /d "%~dp0"

echo ============================================
echo  Risk Manager - Emerging Risks Dashboard
echo ============================================
echo.

:: Create virtual environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo [1/3] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo ERROR: Could not create virtual environment.
        echo Make sure Python 3.12 is installed: https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Install/update dependencies
echo [2/3] Checking dependencies...
pip install -r requirements.txt --quiet --disable-pip-version-check
if errorlevel 1 (
    echo ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

:: Launch the dashboard
echo [3/3] Launching dashboard...
echo.
echo Dashboard will open at: http://localhost:8501
echo Press Ctrl+C to stop.
echo.
python -m streamlit run theme_engine/web_dashboard.py --server.headless false

pause
