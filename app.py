import os
import sys
import subprocess
from datetime import datetime
import whisper
import openai
import glob
import math

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

def split_wav(input_wav, chunk_length_sec=600):
    """Split a wav file into N-second chunks. Returns list of chunk file paths."""
    import wave
    import contextlib
    chunk_paths = []
    with contextlib.closing(wave.open(input_wav, 'rb')) as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        duration = n_frames / float(framerate)
        n_chunks = math.ceil(duration / chunk_length_sec)
    for i in range(n_chunks):
        start = i * chunk_length_sec
        out_path = f"{input_wav}.chunk{i}.wav"
        cmd = [
            'ffmpeg', '-y', '-i', input_wav,
            '-ss', str(start), '-t', str(chunk_length_sec),
            '-acodec', 'copy', out_path
        ]
        subprocess.run(cmd, check=True)
        chunk_paths.append(out_path)
    return chunk_paths

# Helper: Transcribe audio using OpenAI Whisper
def transcribe_audio(audio_path):
    # Load Whisper model (using 'base' for good balance of speed/accuracy)
    print("Loading Whisper model...")
    model = whisper.load_model("base")
    
    # If file is long, split into chunks and transcribe each
    import wave
    import contextlib
    chunk_length_sec = 600  # 10 minutes
    
    try:
        with contextlib.closing(wave.open(audio_path, 'rb')) as wf:
            n_frames = wf.getnframes()
            framerate = wf.getframerate()
            duration = n_frames / float(framerate)
    except Exception:
        # If wave can't read it, just try transcribing directly
        print("Transcribing audio file...")
        result = model.transcribe(audio_path)
        return result["text"]
    
    if duration <= chunk_length_sec:
        print("Transcribing audio file...")
        result = model.transcribe(audio_path)
        return result["text"]
    
    # Split and transcribe each chunk
    print(f"File is long ({duration:.1f}s), splitting into chunks...")
    chunk_paths = split_wav(audio_path, chunk_length_sec)
    texts = []
    for i, chunk in enumerate(chunk_paths):
        print(f"Transcribing chunk {i+1}/{len(chunk_paths)}...")
        result = model.transcribe(chunk)
        texts.append(result["text"])
        os.remove(chunk)
    return '\n'.join(texts)

# Helper: Generate notes using OpenAI
def generate_notes(transcription, api_key):
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4o",  # Updated to use gpt-4o
        messages=[
            {"role": "system", "content": "You are an expert in taking notes from audio transcriptions. I need you to create notes from the following transcription. Do not use any markdown, just stick to plain text. Make sure to capture key points and action items from the meeting transcription. Change all instances of Zane to Zain."},
            {"role": "user", "content": transcription}
        ]
    )
    content = response.choices[0].message.content
    return content if content else "No notes generated."

PROCESSING_FOLDER = os.path.join(os.path.dirname(__file__), "processing")

def process_all_in_folder(input_folder, api_key):
    audio_files = [f for f in os.listdir(input_folder) if f.lower().endswith((
        '.wav', '.flac', '.mp4a', '.mp3', '.ogg', '.m4a', '.aac', '.wma', '.opus', '.mp4'))]
    processed_any = False
    for filename in audio_files:
        # Never process .converted.wav directly
        if filename.endswith('.converted.wav'):
            continue
        title, ext = os.path.splitext(filename)
        # Find all output files for this title
        trans_files = [f for f in os.listdir(input_folder) if f.startswith(title + '_') and '-transcription.txt' in f]
        notes_files = [f for f in os.listdir(input_folder) if f.startswith(title + '_') and '-notes.txt' in f]
        need_trans = not trans_files
        need_notes = not notes_files
        if not (need_trans or need_notes):
            # Both outputs exist, skip
            continue
        dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = f"{title}_{dt}"
        try:
            print(f"Processing: {filename}")
            wav_path = convert_to_wav(os.path.join(input_folder, filename))
            transcription = None
            if need_trans or need_notes:
                transcription = transcribe_audio(wav_path)
            if need_trans:
                trans_file = os.path.join(input_folder, f"{base}-transcription.txt")
                with open(trans_file, "w", encoding="utf-8") as f:
                    f.write(str(transcription))
                print(f"  -> {trans_file}")
            if need_notes:
                if transcription is None:
                    # Should not happen, but just in case
                    with open(trans_files[0], "r", encoding="utf-8") as f:
                        transcription = f.read()
                notes = generate_notes(transcription, api_key)
                notes_file = os.path.join(input_folder, f"{base}-notes.txt")
                with open(notes_file, "w", encoding="utf-8") as f:
                    f.write(notes)
                print(f"  -> {notes_file}")
            processed_any = True
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    return processed_any

if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY.")
        sys.exit(1)
    if not os.path.exists(PROCESSING_FOLDER):
        os.makedirs(PROCESSING_FOLDER)
    # Loop until no more files are processed
    while True:
        processed = process_all_in_folder(PROCESSING_FOLDER, api_key)
        if not processed:
            print("No more new files to process. Exiting.")
            break
