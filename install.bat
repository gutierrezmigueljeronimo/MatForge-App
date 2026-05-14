@echo off
setlocal enabledelayedexpansion

echo ============================================================
echo  MatForge App -- Environment Setup
echo ============================================================
echo.

:: Verify Python 3.11 is available
py -3.11 --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.11 is not installed or not found.
    echo.
    echo Download Python 3.11 from:
    echo   https://www.python.org/downloads/release/python-3119/
    echo.
    echo During installation, check:
    echo   - Add Python to PATH
    echo   - Install for all users
    echo.
    pause
    exit /b 1
)

echo [1/3] Creating virtual environment with Python 3.11...
py -3.11 -m venv .venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)
echo       Done.
echo.

echo [2/3] Installing PyTorch with CUDA 11.8...
.venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo [ERROR] Failed to install PyTorch.
    echo.
    echo Check your internet connection and try again.
    pause
    exit /b 1
)
echo       Done.
echo.

echo [3/3] Installing remaining dependencies...
.venv\Scripts\pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies from requirements.txt.
    pause
    exit /b 1
)
echo       Done.
echo.

echo ============================================================
echo  Installation complete.
echo  Run launch_matforge.bat to start the application.
echo ============================================================
echo.
pause
