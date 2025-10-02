param(
    [string]$InstallRoot = "$env:LOCALAPPDATA\AudioToNotes",
    [switch]$LeaveContextMenu
)

$ErrorActionPreference = "Stop"

if (-not $LeaveContextMenu) {
    $ctxScript = Join-Path $InstallRoot 'uninstall_context_menu.ps1'
    if (Test-Path $ctxScript) {
        & $ctxScript
    } else {
        Write-Warning "Context menu uninstall script not found. Removing registry keys directly."
        $extensions = @('.aac','.flac','.m4a','.mp3','.mp4','.mp4a','.ogg','.opus','.wav','.wma')
        foreach ($ext in $extensions) {
            $baseKey = "HKCU:\Software\Classes\SystemFileAssociations\$ext\shell\AudioToNotes_CreateNotes"
            if (Test-Path $baseKey) {
                Remove-Item -Path $baseKey -Recurse -Force
            }
        }
    }
}

if (Test-Path $InstallRoot) {
    Write-Host "Removing installation directory '$InstallRoot'" -ForegroundColor Yellow
    Remove-Item -Path $InstallRoot -Recurse -Force
}

Write-Host "Audio-to-Notes uninstalled." -ForegroundColor Green
