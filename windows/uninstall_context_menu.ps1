param(
    [string]$Verb = "AudioToNotes_CreateNotes"
)

$ErrorActionPreference = "Stop"

$extensions = @(
    '.aac',
    '.flac',
    '.m4a',
    '.mp3',
    '.mp4',
    '.mp4a',
    '.ogg',
    '.opus',
    '.wav',
    '.wma'
)

foreach ($ext in $extensions) {
    $baseKey = "HKCU:\Software\Classes\SystemFileAssociations\$ext\shell\$Verb"
    if (Test-Path $baseKey) {
        Remove-Item -Path $baseKey -Recurse -Force
    }
}

Write-Host "Context menu entry removed for verb '$Verb'." -ForegroundColor Yellow
