import argparse
import contextlib
import importlib
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
import torch

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
DEFAULT_CHUNK_LENGTH_SEC = 40  # Canary-Qwen performs best around 30–45s chunks


class CanaryTranscriber:
    """Wrapper around NVIDIA Canary-Qwen SALM for chunked transcription."""

    def __init__(
        self,
        *,
        model_name: str = "nvidia/canary-qwen-2.5b",
        max_new_tokens: int = 512,
        device: Optional[str] = None,
    ) -> None:
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        print(f"Loading Canary-Qwen model on {self.device}...")
        fsdp_module = importlib.import_module("torch.distributed.fsdp")
        if not hasattr(fsdp_module, "fully_shard"):

            def _passthrough_fully_shard(target=None, *_, **__):
                return target

            fsdp_module.fully_shard = _passthrough_fully_shard  # type: ignore[attr-defined]
        dtensor_module = importlib.import_module("torch.distributed.tensor")
        if not hasattr(dtensor_module, "Replicate"):

            class _Replicate:
                def __repr__(self) -> str:  # pragma: no cover - trivial
                    return "Replicate()"

            dtensor_module.Replicate = _Replicate  # type: ignore[attr-defined]
        if not hasattr(dtensor_module, "Shard"):

            class _Shard:
                def __init__(self, dim: int) -> None:
                    self.dim = dim

                def __repr__(self) -> str:  # pragma: no cover - trivial
                    return f"Shard(dim={self.dim})"

            dtensor_module.Shard = _Shard  # type: ignore[attr-defined]
        if hasattr(dtensor_module, "__all__"):
            for _symbol in ("Replicate", "Shard"):
                if _symbol not in dtensor_module.__all__:
                    dtensor_module.__all__.append(_symbol)
        # PyTorch 2.4 lacks the padding_side keyword that NeMo expects.
        rnn_module = importlib.import_module("torch.nn.utils.rnn")
        pad_sequence_fn = getattr(rnn_module, "pad_sequence")
        import inspect

        if "padding_side" not in inspect.signature(pad_sequence_fn).parameters:
            original_pad_sequence = pad_sequence_fn

            def _pad_sequence_with_side(
                sequences,
                batch_first: bool = False,
                padding_value: float = 0.0,
                padding_side: str = "right",
            ):
                if padding_side == "right":
                    return original_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)
                if padding_side != "left":
                    raise ValueError(f"Unsupported padding_side: {padding_side}")
                if not sequences:
                    return original_pad_sequence(sequences, batch_first=batch_first, padding_value=padding_value)

                max_len = max(seq.size(0) for seq in sequences)
                trailing_shape = sequences[0].shape[1:]
                output_shape = (len(sequences), max_len, *trailing_shape) if batch_first else (max_len, len(sequences), *trailing_shape)
                output = sequences[0].new_full(output_shape, padding_value)

                if batch_first:
                    for idx, seq in enumerate(sequences):
                        length = seq.size(0)
                        output[idx, max_len - length :] = seq
                else:
                    for idx, seq in enumerate(sequences):
                        length = seq.size(0)
                        output[max_len - length :, idx] = seq
                return output

            rnn_module.pad_sequence = _pad_sequence_with_side  # type: ignore[assignment]
        salm_module = importlib.import_module("nemo.collections.speechlm2.models.salm")
        salm_cls = getattr(salm_module, "SALM")
        hf_parts_module = importlib.import_module("nemo.collections.speechlm2.parts.hf_hub")
        omegaconf_module = importlib.import_module("omegaconf")
        transformers_utils = importlib.import_module("transformers.utils")

        cached_file_fn = getattr(transformers_utils, "cached_file")
        omega_conf = getattr(omegaconf_module, "OmegaConf")
        config_name = getattr(hf_parts_module, "CONFIG_NAME")

        resolved_config = cached_file_fn(
            model_name,
            config_name,
            cache_dir=None,
            force_download=False,
            proxies=None,
            resume_download=False,
            local_files_only=False,
            token=None,
            revision=None,
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        if resolved_config is None:
            raise RuntimeError(f"Missing {config_name} for {model_name}")

        cfg_dict = omega_conf.to_container(omega_conf.load(resolved_config))
        if not isinstance(cfg_dict, dict):
            raise TypeError("Expected model config to deserialize to a dict")
        cfg_dict["pretrained_weights"] = False

        base_from_pretrained = None
        for base in salm_cls.__mro__:
            if base.__name__ == "PyTorchModelHubMixin":
                base_from_pretrained = base._from_pretrained.__get__(salm_cls, salm_cls)
                break
        if base_from_pretrained is None:
            raise RuntimeError("PyTorchModelHubMixin not found in SALM inheritance chain")

        # Load weights via the vanilla PyTorchModelHubMixin to avoid extra kwargs that Nemo's mixin forwards.
        self.model = base_from_pretrained(
            model_id=model_name,
            revision=None,
            cache_dir=None,
            force_download=False,
            local_files_only=False,
            token=None,
            map_location=self.device,
            strict=False,
            cfg=cfg_dict,
        )
        self.model.to(self.device)
        self.model.eval()
        self.prompt_template = "Transcribe the following: {}"

    def transcribe(self, audio_path: Path) -> str:
        prompt = self.prompt_template.format(self.model.audio_locator_tag)
        prompts = [[{"role": "user", "content": prompt, "audio": [str(audio_path)]}]]
        with torch.inference_mode():
            answer_ids = self.model.generate(
                prompts=prompts,
                max_new_tokens=self.max_new_tokens,
            )
        transcript = self.model.tokenizer.ids_to_text(answer_ids[0].cpu())
        return transcript.strip()

    def close(self) -> None:
        if hasattr(self, "model"):
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


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
    transcriber: CanaryTranscriber,
    chunk_length_sec: int = DEFAULT_CHUNK_LENGTH_SEC,
) -> str:
    """Transcribe audio using Canary-Qwen, chunking longer files automatically."""
    try:
        duration = get_wav_duration_seconds(audio_path)
    except Exception:
        print("Transcribing audio file...")
        return transcriber.transcribe(audio_path)

    if duration <= chunk_length_sec:
        print("Transcribing audio file...")
        return transcriber.transcribe(audio_path)

    print(f"File is long ({duration:.1f}s), splitting into chunks...")
    chunk_paths, temp_dir = split_wav(audio_path, chunk_length_sec)
    texts: list[str] = []
    try:
        for idx, chunk in enumerate(chunk_paths, start=1):
            print(f"Transcribing chunk {idx}/{len(chunk_paths)}...")
            texts.append(transcriber.transcribe(chunk))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
    return "\n".join(texts)


def generate_notes(transcription: str, api_key: str) -> str:
    client = openai.OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-5",
        messages = [
    {
        "role": "system",
        "content": (
            "ROLE\n"
            "You are an expert meeting-notes synthesizer.\n\n"
            "OBJECTIVE\n"
            "Convert the provided raw meeting transcript into a single, clean set of notes that follow the exact plain-text template below, regardless of meeting type.\n\n"
            "GLOBAL RULES\n"
            "- Use plain text only. Do not use any markdown (no #, *, _, tables, or emojis).\n"
            "- Replace every instance of the name \"Zane\" with \"Zain\" everywhere (speakers, attendees, tasks, quotes).\n"
            "- Do not invent facts. If information is missing, write \"Not stated\".\n"
            "- Keep names as they appear (after the Zane→Zain replacement). Do not infer last names unless present.\n"
            "- Normalize dates to YYYY-MM-DD and times to 12-hour HH:MM AM/PM. Include timezone if present in the transcript.\n"
            "- Convert relative dates (e.g., \"next Friday\") to exact dates only if the transcript provides a reference date; otherwise keep the relative phrase.\n"
            "- Be concise and specific. Remove filler and repetition. One idea per bullet.\n"
            "- De-duplicate repeated points and action items; keep the most complete version.\n"
            "- If timestamps exist, attach the most relevant timestamp to each action item.\n\n"
            "OUTPUT TEMPLATE (copy these headings exactly; fill every field; use \"Not stated\" if unknown)\n\n"
            "MEETING NOTES\n"
            "TITLE:\n"
            "DATE:\n\n"
            "EXECUTIVE SUMMARY (3–6 BULLETS):\n"
            "- \n"
            "- \n"
            "- \n"
            "- \n\n"
            "AGENDA & COVERAGE:\n"
            "- Item 1: Covered / Partially Covered / Skipped — 1–2 sentence summary\n"
            "- Item 2: ...\n"
            "- Additional items: ...\n\n"
            "KEY DISCUSSION POINTS (GROUPED BY TOPIC):\n"
            "- Topic: 2–4 sentence summary\n"
            "  - Notable details:\n"
            "  - Metrics/dates mentioned:\n"
            "  - Dependencies:\n\n"
            "DECISIONS MADE:\n"
            "- Decision: Rationale — Owner — Effective date\n\n"
            "RISKS / BLOCKERS:\n"
            "- Risk: Impact — Mitigation — Owner\n\n"
            "OPEN QUESTIONS:\n"
            "- Question — Owner — Needed by (YYYY-MM-DD)\n\n"
            "FOLLOW-UPS / NEXT STEPS:\n"
            "- Next step — Owner — Target date\n\n"
            "PARKING LOT:\n"
            "- Item to revisit\n\n"
            "QUALITY CHECKS (perform silently; do not include this section in the output)\n"
            "1) All \"Zane\" → \"Zain\".\n"
            "2) No markdown characters used.\n"
            "3) All headings present and in the same order.\n"
            "4) \"Not stated\" used where information is missing.\n"
            "5) Duplicates merged; owners and due dates standardized.\n"
        ),
    },
    {"role": "user", "content": transcription},
]

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
    transcriber: Optional[CanaryTranscriber] = None,
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

    close_transcriber = False
    if transcriber is None:
        transcriber = CanaryTranscriber()
        close_transcriber = True

    try:
        transcription = transcribe_audio(
            wav_path,
            transcriber=transcriber,
            chunk_length_sec=chunk_length_sec,
        )
    finally:
        if should_cleanup_wav and wav_path.exists():
            wav_path.unlink()
        if close_transcriber:
            transcriber.close()

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
    transcriber = CanaryTranscriber()
    processed_any = False
    try:
        for audio_file in audio_files:
            try:
                result = process_file(
                    audio_file,
                    api_key,
                    chunk_length_sec=chunk_length_sec,
                    skip_existing=skip_existing,
                    transcriber=transcriber,
                )
                if result is not None:
                    processed_any = True
            except Exception as exc:
                print(f"Error processing {audio_file.name}: {exc}")
    finally:
        transcriber.close()
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
        description="Transcribe audio files with NVIDIA Canary-Qwen and summarize notes with GPT.",
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
        help=f"Chunk length in seconds when splitting long audio (default: {DEFAULT_CHUNK_LENGTH_SEC}).",
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
    return parser.parse_args(list(argv) if argv is not None else None)


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
