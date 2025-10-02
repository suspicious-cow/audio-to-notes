# Audio-to-Notes

Audio-to-Notes is a Python-based workflow that transcribes spoken audio and generates meeting notes using OpenAI Whisper and GPT models. The core CLI can process individual files or whole folders, and the Windows tooling adds an Explorer context-menu action so you can right-click an audio file and produce the transcript + notes alongside it.

## Features
- Wide audio format support (aac, flac, m4a, mp3, mp4, ogg, opus, wav, wma, and more).
- Automatic Whisper-powered transcription with smart chunking for long recordings.
- GPT-generated notes with prompt tweaks (e.g., rename "Zane" to "Zain").
- Timestamped output files saved next to the source audio.
- Optional Windows installer that sets up a virtual environment and Explorer "Create Notes" context-menu entry.

## Requirements
- Python 3.10 or newer (3.11+ recommended for Whisper).
- `ffmpeg` available on the system `PATH`.
- Python dependencies from `requirements.txt` (`openai`, `openai-whisper`, and their transitive packages such as PyTorch).
- An OpenAI API key with access to the requested GPT model (`OPENAI_API_KEY`).

---

## Python CLI Usage (Cross-Platform)

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Configure your API key**
   ```bash
   export OPENAI_API_KEY=sk-...  # PowerShell: $env:OPENAI_API_KEY = 'sk-...'
   ```

3. **Process a single file**
   ```bash
   python app.py path/to/audio.mp3
   ```
   Creates `audio_YYYYMMDD-HHMMSS-transcription.txt` and `audio_YYYYMMDD-HHMMSS-notes.txt` alongside the source file.

4. **Process a folder (skip files that already have outputs)**
   ```bash
   python app.py path/to/folder
   ```

5. **Legacy watch-loop behaviour**
   ```bash
   python app.py --loop
   ```
   Continuously scans the bundled `processing/` directory until no new files are found.

Helpful flags:
- `--chunk-length <seconds>` adjusts the Whisper chunk size (default 600 seconds).
- `--no-skip-existing` forces regeneration when scanning a folder.

---

## Windows Explorer Integration

The `windows/install.ps1` script packages the app for the current user, installs dependencies in an isolated virtual environment, and registers a **Create Notes** right-click action for supported audio/video extensions.

1. **Set your OpenAI API key once per machine**
   ```powershell
   [System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-...', 'User')
   ```

2. **Run the installer**
   ```powershell
   # From the repository root
   powershell -ExecutionPolicy Bypass -File .\windows\install.ps1
   ```
   Optional parameters:
   - `-InstallRoot "C:\\Tools\\AudioToNotes"` to choose a destination (defaults to `%LOCALAPPDATA%\AudioToNotes`).
   - `-PythonExe "C:\\Python311\\python.exe"` to use a specific interpreter.
   - `-SkipContextMenu` to skip registering the right-click action (keeping just the virtual environment).

3. **Use the context menu**
   - Right-click a supported audio file in Explorer.
   - Choose **Create Notes**.
   - After processing, a dialog lists the generated filenames; they are written next to the original file. A `CreateNotes.log` file in the install directory captures recent activity.

4. **Uninstall**
   ```powershell
   powershell -ExecutionPolicy Bypass -File .\windows\uninstall.ps1
   ```
   Add `-LeaveContextMenu` to remove the files but keep the Explorer entry.

---

## Troubleshooting
- **`ffmpeg` not found**: install it from https://ffmpeg.org/ and ensure the binaries are on `PATH`.
- **Missing or invalid OpenAI credentials**: confirm `OPENAI_API_KEY` is defined for the account running the command.
- **Whisper install issues on Windows**: ensure the Python version supports PyTorch wheels; see https://pytorch.org/get-started/locally/ if needed.
- **Context menu missing**: rerun the installer in a non-elevated PowerShell session, or confirm the extension is listed in `windows/install_context_menu.ps1`.

---

## Contributing
Pull requests and issues are welcome—open an issue first for significant changes.
