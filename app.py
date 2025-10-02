import argparse
import contextlib
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import openai
import whisper

SUPPORTED_EXTENSIONS: tuple[str, ...] = (
    ".wav",
    ".flac",
    ".mp4a",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".mp4",
)

PROCESSING_FOLDER = Path(__file__).resolve().parent / "processing"
DEFAULT_CHUNK_LENGTH_SEC = 600  # 10 minutes


def convert_to_wav(input_path: Path) -> tuple[Path, bool]:
    """Return a wav version of the input audio path and whether it is temporary."""
    input_path = input_path.resolve()
    if input_path.suffix.lower() == ".wav":
        return input_path, False
    output_path = input_path.with_suffix(input_path.suffix + ".converted.wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path, True


def get_wav_duration_seconds(wav_path: Path) -> float:
    """Compute the duration of a wav file in seconds."""
    import wave

    with contextlib.closing(wave.open(str(wav_path), "rb")) as wf:
        n_frames = wf.getnframes()
        framerate = wf.getframerate()
        if framerate == 0:
            raise ValueError("Invalid WAV file: zero framerate")
        return n_frames / float(framerate)


def split_wav(input_wav: Path, chunk_length_sec: int = DEFAULT_CHUNK_LENGTH_SEC) -> tuple[list[Path], Path]:
    """Split a wav file into chunks, returning chunk paths and the temp directory."""
    duration = get_wav_duration_seconds(input_wav)
    chunks_needed = max(1, math.ceil(duration / chunk_length_sec))
    temp_dir = Path(tempfile.mkdtemp(prefix="audio_notes_chunks_"))
    chunk_paths: list[Path] = []
    try:
        for i in range(chunks_needed):
            start = i * chunk_length_sec
            out_path = temp_dir / f"chunk_{i:03d}.wav"
            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                str(input_wav),
                "-ss",
                str(start),
                "-t",
                str(chunk_length_sec),
                "-acodec",
                "copy",
                str(out_path),
            ]
            subprocess.run(cmd, check=True)
            chunk_paths.append(out_path)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return chunk_paths, temp_dir


def transcribe_audio(
    audio_path: Path,
    *,
    model: whisper.Whisper,
    chunk_length_sec: int = DEFAULT_CHUNK_LENGTH_SEC,
) -> str:
    """Transcribe audio using Whisper, chunking longer files automatically."""
    try:
        duration = get_wav_duration_seconds(audio_path)
    except Exception:
        print("Transcribing audio file...")
        result = model.transcribe(str(audio_path))
        return result["text"]

    if duration <= chunk_length_sec:
        print("Transcribing audio file...")
        result = model.transcribe(str(audio_path))
        return result["text"]

    print(f"File is long ({duration:.1f}s), splitting into chunks...")
    chunk_paths, temp_dir = split_wav(audio_path, chunk_length_sec)
    texts: list[str] = []
    try:
        for idx, chunk in enumerate(chunk_paths, start=1):
            print(f"Transcribing chunk {idx}/{len(chunk_paths)}...")
            result = model.transcribe(str(chunk))
            texts.append(result["text"])
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return "\n".join(texts)


def generate_notes(transcription: str, api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in taking notes from audio transcriptions. "
                    "I need you to create notes from the following transcription. "
                    "Do not use any markdown, just stick to plain text. Make sure to "
                    "capture key points and action items from the meeting transcription. "
                    "Change all instances of Zane to Zain."
                ),
            },
            {"role": "user", "content": transcription},
        ],
    )
    content = response.choices[0].message.content
    return content if content else "No notes generated."


def ensure_api_key(api_key: Optional[str]) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not key:
        print("Error: OpenAI API key required. Set OPENAI_API_KEY or pass --api-key.")
        sys.exit(1)
    return key


def find_existing_outputs(output_dir: Path, base_title: str) -> tuple[list[Path], list[Path]]:
    trans_files = sorted(output_dir.glob(f"{base_title}_*-transcription.txt"))
    notes_files = sorted(output_dir.glob(f"{base_title}_*-notes.txt"))
    return trans_files, notes_files


def process_file(
    input_path: Path,
    api_key: str,
    *,
    chunk_length_sec: int = DEFAULT_CHUNK_LENGTH_SEC,
    skip_existing: bool = False,
    model: Optional[whisper.Whisper] = None,
) -> Optional[dict[str, Path]]:
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.exists() or not input_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {input_path}")

    if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    output_dir = input_path.parent
    base_title = input_path.stem
    trans_files, notes_files = find_existing_outputs(output_dir, base_title)
    if skip_existing and trans_files and notes_files:
        print(f"Skipping {input_path.name}: existing transcription and notes detected.")
        return None

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    transcription_path = output_dir / f"{base_title}_{timestamp}-transcription.txt"
    notes_path = output_dir / f"{base_title}_{timestamp}-notes.txt"

    print(f"Processing: {input_path}")
    wav_path, should_cleanup_wav = convert_to_wav(input_path)

    close_model = False
    if model is None:
        print("Loading Whisper model...")
        model = whisper.load_model("base")
        close_model = True

    try:
        transcription = transcribe_audio(
            wav_path,
            model=model,
            chunk_length_sec=chunk_length_sec,
        )
    finally:
        if should_cleanup_wav and wav_path.exists():
            wav_path.unlink()
        if close_model:
            # Ensure resources are freed when we created the model.
            del model

    with open(transcription_path, "w", encoding="utf-8") as f:
        f.write(transcription)
    print(f"  -> {transcription_path}")

    notes = generate_notes(transcription, api_key)
    with open(notes_path, "w", encoding="utf-8") as f:
        f.write(notes)
    print(f"  -> {notes_path}")

    return {"transcription_path": transcription_path, "notes_path": notes_path}


def process_all_in_folder(
    input_folder: Path,
    api_key: str,
    *,
    chunk_length_sec: int = DEFAULT_CHUNK_LENGTH_SEC,
    skip_existing: bool = True,
) -> bool:
    input_folder = Path(input_folder).expanduser().resolve()
    if not input_folder.exists() or not input_folder.is_dir():
        raise NotADirectoryError(f"Folder not found: {input_folder}")

    audio_files = [
        f
        for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.endswith(".converted.wav")
    ]
    if not audio_files:
        return False

    print(f"Found {len(audio_files)} file(s) to check in {input_folder}")
    model = whisper.load_model("base")
    processed_any = False
    for audio_file in audio_files:
        try:
            result = process_file(
                audio_file,
                api_key,
                chunk_length_sec=chunk_length_sec,
                skip_existing=skip_existing,
                model=model,
            )
            if result is not None:
                processed_any = True
        except Exception as exc:
            print(f"Error processing {audio_file.name}: {exc}")
    return processed_any


def run_processing_loop(
    folder: Path,
    api_key: str,
    *,
    chunk_length_sec: int = DEFAULT_CHUNK_LENGTH_SEC,
    skip_existing: bool = True,
) -> None:
    while True:
        processed = process_all_in_folder(
            folder,
            api_key,
            chunk_length_sec=chunk_length_sec,
            skip_existing=skip_existing,
        )
        if not processed:
            print("No more new files to process. Exiting.")
            break


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe audio files and generate notes using OpenAI Whisper and GPT.",
    )
    parser.add_argument(
        "target",
        nargs="?",
        help="Audio file or directory to process. If omitted, the processing folder is used in loop mode.",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="OpenAI API key (overrides OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=DEFAULT_CHUNK_LENGTH_SEC,
        help="Chunk length in seconds when splitting long audio (default: 600).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Process files even if matching transcripts and notes already exist when scanning a folder.",
    )
    parser.set_defaults(skip_existing=True)
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Continuously scan the processing folder until no new files are found (default action when no target is provided).",
    )
    return parser.parse_args(argv)


def cli_main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    api_key = ensure_api_key(args.api_key)

    if args.target:
        target = Path(args.target).expanduser()
        if target.is_file():
            process_file(
                target,
                api_key,
                chunk_length_sec=args.chunk_length,
                skip_existing=False,
            )
        elif target.is_dir():
            processed = process_all_in_folder(
                target,
                api_key,
                chunk_length_sec=args.chunk_length,
                skip_existing=args.skip_existing,
            )
            if not processed:
                print("No files processed.")
        else:
            print(f"Error: target not found -> {target}")
            sys.exit(1)
    else:
        loop = True if args.loop or not args.target else False
        if loop:
            run_processing_loop(
                PROCESSING_FOLDER,
                api_key,
                chunk_length_sec=args.chunk_length,
                skip_existing=args.skip_existing,
            )
        else:
            processed = process_all_in_folder(
                PROCESSING_FOLDER,
                api_key,
                chunk_length_sec=args.chunk_length,
                skip_existing=args.skip_existing,
            )
            if not processed:
                print("No files processed.")


if __name__ == "__main__":
    cli_main()
