@echo off
setlocal enabledelayedexpansion

REM ============================================================================
REM V7 Audio Authenticity Analyzer - Windows Application Launcher
REM ============================================================================

title V7 Audio Authenticity Analyzer

REM 1. Navigate reliably to application directory
cd /d "%~dp0"

echo ============================================================================
echo   V7 Audio Authenticity Analyzer - Local Launch Environment
echo ============================================================================
echo Application Directory : %CD%

REM 2. Determine Python interpreter (prefer local venv if present)
set "PY_CMD=python"
set "ENV_NAME=System Python"
if exist "%~dp0.venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0.venv\Scripts\python.exe"
    set "ENV_NAME=Local Virtual Environment (.venv)"
)
if exist "%~dp0venv\Scripts\python.exe" (
    set "PY_CMD=%~dp0venv\Scripts\python.exe"
    set "ENV_NAME=Local Virtual Environment (venv)"
)
echo Runtime Environment  : !ENV_NAME!

REM 3. Verify Python availability
"%PY_CMD%" --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Python interpreter not found!
    echo Please ensure Python 3.10+ is installed and available on PATH,
    echo or create a virtual environment in .venv.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('"%PY_CMD%" --version 2^>^&1') do set "PY_VER=%%v"
echo Python Version       : %PY_VER%

REM 4. Verify required core runtime dependencies
echo.
echo [1/3] Verifying runtime dependencies...
"%PY_CMD%" -c "import streamlit, torch, numpy, scipy, soundfile, matplotlib, reportlab" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] One or more required packages are missing!
    echo Please install dependencies using:
    echo     pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)
echo       Dependencies verified: OK

REM 5. Verify model checkpoint integrity
echo [2/3] Verifying model checkpoint integrity...
"%PY_CMD%" scripts\verify_checkpoint.py >nul 2>&1
if errorlevel 1 (
    echo.
    echo [ERROR] Checkpoint verification failed!
    echo Running detailed verification diagnostic:
    "%PY_CMD%" scripts\verify_checkpoint.py
    echo.
    pause
    exit /b 1
)
echo       Checkpoint integrity: OK (SHA-256 verified)

REM 6. Launch Streamlit Application
echo [3/3] Launching Streamlit web interface...
echo.
echo The application will open in your default browser at http://localhost:8501
echo Press Ctrl+C in this terminal window to stop the server.
echo ============================================================================
echo.

"%PY_CMD%" -m streamlit run app.py

if errorlevel 1 (
    echo.
    echo [NOTICE] Application terminated with exit code %errorlevel%.
    pause
)

endlocal
