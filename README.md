# Audio-to-Notes

Audio-to-Notes is a Python application that streamlines the process of transcribing audio files and generating concise notes using state-of-the-art AI models. It features a simple graphical user interface (GUI) for easy file selection and title entry, and leverages NVIDIA NeMo's Parakeet for transcription and OpenAI's GPT-4.1 for note generation.

---

## Features
- **Audio File Support:** Accepts a wide range of audio formats (e.g., mp4a, wav, mp3, flac, m4a, ogg, etc.).
- **User-Friendly GUI:** Simple interface for selecting audio files and entering a custom title.
- **Automatic Transcription:** Uses NVIDIA Parakeet (via NeMo Toolkit) for accurate speech-to-text transcription.
- **AI-Powered Notes:** Summarizes transcriptions into clear, concise notes using OpenAI's GPT-4.1 API.
- **Organized Output:** Saves both transcription and notes with filenames containing the user title, current date/time, and a descriptive suffix.
- **Robust File Handling:** Handles audio conversion (via ffmpeg) to ensure compatibility with the transcription model.

---

## Requirements
- Python 3.8+
- NVIDIA GPU with CUDA drivers (for NeMo/Parakeet ASR)
- [nemo_toolkit[asr]](https://github.com/NVIDIA/NeMo) (for Parakeet ASR)
- [openai](https://pypi.org/project/openai/) Python package
- [tkinter](https://docs.python.org/3/library/tkinter.html) (for GUI)
- [ffmpeg](https://ffmpeg.org/) (system package, for audio conversion)
- [ffmpeg-python](https://github.com/kkroening/ffmpeg-python) (Python wrapper for ffmpeg)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd audio-to-notes
   ```
2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Install system dependencies:**
   - Ensure you have an NVIDIA GPU and CUDA drivers installed.
   - Install ffmpeg:
     ```bash
     sudo apt-get update && sudo apt-get install ffmpeg
     ```
4. **Set your OpenAI API key:**
   - Obtain an API key from [OpenAI](https://platform.openai.com/).
   - Set it as an environment variable:
     ```bash
     export OPENAI_API_KEY=your-key-here
     ```

---

## Usage

1. **Run the application:**
   ```bash
   python app.py
   ```
2. **Using the GUI:**
   - Click the button to select an audio file (any supported format).
   - Enter a descriptive title in the textbox.
   - Click the button to start processing.
   - The app will:
     - Convert the audio to a compatible format if needed.
     - Transcribe the audio using NeMo Parakeet.
     - Save the transcription as `<title>-<datetime>-transcription.txt`.
     - Generate notes from the transcription using OpenAI GPT-4.1.
     - Save the notes as `<title>-<datetime>-notes.txt`.
   - All output files are saved in the current working directory.

---

## Example

Suppose you have an audio file `meeting.m4a` and want to generate notes titled "Team Meeting". The app will produce files like:
- `Team Meeting-2025-05-18-15-30-12-transcription.txt`
- `Team Meeting-2025-05-18-15-30-12-notes.txt`

---

## Troubleshooting
- **CUDA/Driver Issues:** Ensure your NVIDIA drivers and CUDA are properly installed and compatible with your GPU.
- **ffmpeg Not Found:** Install ffmpeg using your system package manager (see above).
- **OpenAI API Errors:** Make sure your API key is valid and has sufficient quota.
- **tkinter Not Installed:** On some Linux systems, install with `sudo apt-get install python3-tk`.

---

## License
- Parakeet model: CC-BY-4.0
- This app: MIT License (or specify your own)

---

## Acknowledgments
- [NVIDIA NeMo Toolkit](https://github.com/NVIDIA/NeMo)
- [OpenAI GPT-4.1](https://platform.openai.com/docs/models/gpt-4)
- [ffmpeg](https://ffmpeg.org/)

---

## Contributing
Pull requests and issues are welcome! Please open an issue to discuss major changes first.
