import argparse
import ctypes
import sys
import traceback
from pathlib import Path
from typing import Optional

from app import (
    DEFAULT_CHUNK_LENGTH_SEC,
    ensure_api_key,
    process_file,
)

MB_ICONINFORMATION = 0x40
MB_ICONERROR = 0x10
MB_OK = 0x0


def show_message(title: str, message: str, *, error: bool = False) -> None:
    flags = MB_OK | (MB_ICONERROR if error else MB_ICONINFORMATION)
    ctypes.windll.user32.MessageBoxW(None, message, title, flags)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Windows entry point for the Audio-to-Notes application. Processes a single audio file "
            "and drops the transcript + notes alongside the source file."
        )
    )
    parser.add_argument("input_path", help="Audio file to transcribe and summarize.")
    parser.add_argument(
        "--api-key",
        dest="api_key",
        help="Optional OpenAI API key override. Falls back to OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--chunk-length",
        type=int,
        default=DEFAULT_CHUNK_LENGTH_SEC,
        help="Chunk length in seconds when splitting long audio (default: 600).",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress Windows message boxes (useful for automated scenarios).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    api_key = ensure_api_key(args.api_key)

    target_path = Path(args.input_path).expanduser()
    try:
        result = process_file(
            target_path,
            api_key,
            chunk_length_sec=args.chunk_length,
            skip_existing=False,
        )
    except Exception as exc:
        tb = traceback.format_exc()
        if not args.silent:
            show_message(
                "Create Notes",
                f"Failed to process '{target_path.name}'.\n{exc}",
                error=True,
            )
        raise SystemExit(1) from exc

    if result is None:
        if not args.silent:
            show_message(
                "Create Notes",
                f"No notes were generated for '{target_path.name}'.",
                error=True,
            )
        raise SystemExit(2)

    transcription_path = result["transcription_path"]
    notes_path = result["notes_path"]

    if not args.silent:
        show_message(
            "Create Notes",
            (
                f"Created transcription and notes for '{target_path.name}'.\n\n"
                f"Transcription: {transcription_path.name}\n"
                f"Notes: {notes_path.name}"
            ),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
