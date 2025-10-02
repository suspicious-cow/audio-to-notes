param(
    [Parameter(Mandatory = $true)]
    [string]$PythonPath,

    [Parameter(Mandatory = $true)]
    [string]$ScriptPath,

    [string]$MenuText = "Create Notes",

    [string]$Verb = "AudioToNotes_CreateNotes",

    [string]$IconPath = ""
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $PythonPath)) {
    throw "Python executable not found at '$PythonPath'."
}

if (-not (Test-Path $ScriptPath)) {
    throw "Windows entry script not found at '$ScriptPath'."
}

$command = '"{0}" "{1}" "%1"' -f (Resolve-Path $PythonPath), (Resolve-Path $ScriptPath)

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
    if (-not (Test-Path $baseKey)) {
        New-Item -Path $baseKey -Force | Out-Null
    }
    Set-ItemProperty -Path $baseKey -Name "" -Value $MenuText
    if ($IconPath) {
        Set-ItemProperty -Path $baseKey -Name "Icon" -Value $IconPath
    }

    $commandKey = "$baseKey\command"
    if (-not (Test-Path $commandKey)) {
        New-Item -Path $commandKey -Force | Out-Null
    }
    Set-ItemProperty -Path $commandKey -Name "" -Value $command
}

Write-Host "Context menu entry '$MenuText' installed for supported audio file types." -ForegroundColor Green
