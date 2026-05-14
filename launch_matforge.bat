@echo off
setlocal enabledelayedexpansion

:: Verify the virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo.
    echo Run install.bat first to set up the environment.
    echo.
    pause
    exit /b 1
)

:: Verify model weights are present
if not exist "checkpoints\matforge\best_gan.pt" (
    echo [ERROR] MatForge model weights not found.
    echo.
    echo Expected: checkpoints\matforge\best_gan.pt
    echo.
    echo Download the weights from the latest release:
    echo   https://github.com/migueljeronimogutierrez/MatForge-App/releases/latest
    echo.
    pause
    exit /b 1
)

if not exist "checkpoints\sr\sr_ft_phase1_best_lpips.pt" (
    echo [ERROR] SR model weights not found.
    echo.
    echo Expected: checkpoints\sr\sr_ft_phase1_best_lpips.pt
    echo.
    echo Download the weights from the latest release:
    echo   https://github.com/migueljeronimogutierrez/MatForge-App/releases/latest
    echo.
    pause
    exit /b 1
)

echo ============================================================
echo  MatForge App
echo ============================================================
echo.
echo  Starting application at http://localhost:8501
echo  Press Ctrl+C in this window to stop the server.
echo.

call .venv\Scripts\activate.bat
streamlit run app.py
