param(
    [string]$InstallRoot = "$env:LOCALAPPDATA\AudioToNotes",
    [string]$PythonExe = "python",
    [switch]$SkipContextMenu
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..')
Write-Host "Installing Audio-to-Notes to '$InstallRoot'" -ForegroundColor Cyan

if (-not (Test-Path $InstallRoot)) {
    New-Item -Path $InstallRoot -ItemType Directory -Force | Out-Null
}

$itemsToCopy = @(
    'app.py',
    'windows_entry.py',
    'requirements.txt'
)

foreach ($item in $itemsToCopy) {
    $source = Join-Path $repoRoot $item
    if (-not (Test-Path $source)) {
        throw "Required file '$item' not found at $source."
    }
    Copy-Item -Path $source -Destination $InstallRoot -Force
}

$processingSource = Join-Path $repoRoot 'processing'
$processingDest = Join-Path $InstallRoot 'processing'
if (Test-Path $processingSource) {
    Copy-Item -Path $processingSource -Destination $processingDest -Recurse -Force
}

Copy-Item -Path (Join-Path $PSScriptRoot 'install_context_menu.ps1') -Destination (Join-Path $InstallRoot 'install_context_menu.ps1') -Force
Copy-Item -Path (Join-Path $PSScriptRoot 'uninstall_context_menu.ps1') -Destination (Join-Path $InstallRoot 'uninstall_context_menu.ps1') -Force

$venvPath = Join-Path $InstallRoot 'venv'
if (-not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    & $PythonExe -m venv $venvPath
} else {
    Write-Host "Using existing virtual environment at $venvPath" -ForegroundColor Yellow
}

$pipPath = Join-Path $venvPath 'Scripts\pip.exe'
if (-not (Test-Path $pipPath)) {
    throw "pip executable not found at $pipPath"
}

Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
& $pipPath install --upgrade pip
& $pipPath install -r (Join-Path $InstallRoot 'requirements.txt')

$pythonwPath = Join-Path $venvPath 'Scripts\pythonw.exe'
if (-not (Test-Path $pythonwPath)) {
    $pythonwPath = Join-Path $venvPath 'Scripts\python.exe'
}

if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Warning "ffmpeg not detected in PATH. Install it separately for audio conversion support."
}

if (-not $SkipContextMenu) {
    $ctxScript = Join-Path $InstallRoot 'install_context_menu.ps1'
    & $ctxScript -PythonPath $pythonwPath -ScriptPath (Join-Path $InstallRoot 'windows_entry.py')
}

Write-Host "Installation complete." -ForegroundColor Green
Write-Host "Ensure the OPENAI_API_KEY environment variable is set before running the context menu action." -ForegroundColor Yellow
