$pythonPath = "C:\Users\Zain_\anaconda3\envs\audio-notes-gpu\python.exe"
$scriptPath = "C:\Users\Zain_\Dropbox\Personal\Data Science Projects\audio-to-notes\windows_entry.py"

& "$PSScriptRoot\windows\install_context_menu.ps1" -PythonPath $pythonPath -ScriptPath $scriptPath
