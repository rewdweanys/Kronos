param(
    [string]$InstallPath = "$HOME\Documents\Stock-Picker\Kronos",
    [switch]$SkipClone
)

$ErrorActionPreference = "Stop"
$Repository = "https://github.com/rewdweanys/Kronos.git"
$Branch = "stock-picker-v1"

function Write-Step([string]$Message) {
    Write-Host "`n==> $Message" -ForegroundColor Cyan
}

Write-Step "Checking Git"
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required but was not found. Install Git for Windows and rerun this script."
}

if (-not $SkipClone) {
    if (-not (Test-Path $InstallPath)) {
        Write-Step "Cloning $Repository"
        New-Item -ItemType Directory -Force -Path (Split-Path $InstallPath) | Out-Null
        git clone $Repository $InstallPath
    }
}

if (-not (Test-Path (Join-Path $InstallPath ".git"))) {
    throw "$InstallPath is not a Git repository. Clone the fork there or rerun without -SkipClone."
}

Set-Location $InstallPath
Write-Step "Checking out $Branch"
git fetch origin
git checkout $Branch

Write-Step "Checking Python 3.11"
$Python311 = $null
try {
    $Python311 = (& py -3.11 -c "import sys; print(sys.executable)" 2>$null).Trim()
} catch {
    $Python311 = $null
}

if (-not $Python311) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        throw "Python 3.11 is missing and winget is unavailable. Install Python 3.11, then rerun."
    }
    Write-Step "Installing Python 3.11 alongside Python 3.14"
    winget install --exact --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
}

Write-Step "Creating isolated virtual environment"
& py -3.11 -m venv .venv
$Python = Join-Path $InstallPath ".venv\Scripts\python.exe"

Write-Step "Installing Kronos and stock-picker dependencies"
& $Python -m pip install --upgrade pip setuptools wheel
& $Python -m pip install -r requirements.txt
& $Python -m pip install -r requirements-stock-picker.txt

Write-Step "Running local diagnostics"
& $Python -m stock_picker doctor

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "First baseline scan: python -m stock_picker scan --model baseline"
Write-Host "Download Kronos-mini: python -m stock_picker warmup --model kronos-mini"
Write-Host "Your NVIDIA GPU currently reports a driver error. CPU mode works for initial testing."
