import os
import sys
import subprocess
from datetime import datetime
import nemo.collections.asr as nemo_asr
import openai
import glob

# Helper: Convert audio to wav if needed
def convert_to_wav(input_path):
    if input_path.lower().endswith('.wav') or input_path.lower().endswith('.flac'):
        return input_path
    output_path = input_path + '.converted.wav'
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000', '-ac', '1', output_path
    ]
    subprocess.run(cmd, check=True)
    return output_path

# Helper: Transcribe audio using Parakeet
def transcribe_audio(audio_path):
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v2")
    output = asr_model.transcribe([audio_path])
    return output[0].text

# Helper: Generate notes using OpenAI
def generate_notes(transcription, api_key):
    openai.api_key = api_key
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are an expert in taking notes from audio transcriptions. I need you to create notes from the following transcription. Do not use any markdown, just stick to plain text."},
            {"role": "user", "content": transcription}
        ]
    )
    return response.choices[0].message.content.strip()

PROCESSING_FOLDER = os.path.join(os.path.dirname(__file__), "processing")

def process_all_in_folder(input_folder, api_key):
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith((
        '.wav', '.flac', '.mp4a', '.mp3', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.mp4'))]
    for filename in audio_files:
        title, ext = os.path.splitext(filename)
        # Count how many files contain the title in their name
        title_count = sum(1 for f in os.listdir(input_folder) if title in f)
        if title_count > 1:
            print(f"Skipping (multiple files with title present): {filename}")
            continue
        # Also skip .converted.wav files (never process them directly)
        if filename.endswith('.converted.wav'):
            continue
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{title}_{dt}"
        try:
            print(f"Processing: {filename}")
            wav_path = convert_to_wav(os.path.join(input_folder, filename))
            transcription = transcribe_audio(wav_path)
            trans_file = os.path.join(input_folder, f"{base}-transcription.txt")
            with open(trans_file, "w", encoding="utf-8") as f:
                f.write(transcription)
            notes = generate_notes(transcription, api_key)
            notes_file = os.path.join(input_folder, f"{base}-notes.txt")
            with open(notes_file, "w", encoding="utf-8") as f:
                f.write(notes)
            print(f"Done: {filename}\n  -> {trans_file}\n  -> {notes_file}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY.")
        sys.exit(1)
    if not os.path.exists(PROCESSING_FOLDER):
        os.makedirs(PROCESSING_FOLDER)
    process_all_in_folder(PROCESSING_FOLDER, api_key)
