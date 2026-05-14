# MatForge App — PowerShell launcher

# Verify the virtual environment exists
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[ERROR] Virtual environment not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Run install.bat first to set up the environment."
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

# Verify model weights are present
if (-not (Test-Path "checkpoints\matforge\best_gan.pt")) {
    Write-Host "[ERROR] MatForge model weights not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected: checkpoints\matforge\best_gan.pt"
    Write-Host ""
    Write-Host "Download the weights from the latest release:"
    Write-Host "  https://github.com/migueljeronimogutierrez/MatForge-App/releases/latest"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not (Test-Path "checkpoints\sr\sr_ft_phase1_best_lpips.pt")) {
    Write-Host "[ERROR] SR model weights not found." -ForegroundColor Red
    Write-Host ""
    Write-Host "Expected: checkpoints\sr\sr_ft_phase1_best_lpips.pt"
    Write-Host ""
    Write-Host "Download the weights from the latest release:"
    Write-Host "  https://github.com/migueljeronimogutierrez/MatForge-App/releases/latest"
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "============================================================"
Write-Host " MatForge App"
Write-Host "============================================================"
Write-Host ""
Write-Host " Starting application at http://localhost:8501"
Write-Host " Press Ctrl+C in this window to stop the server."
Write-Host ""

# Activate virtual environment and launch
& .venv\Scripts\Activate.ps1
streamlit run app.py
