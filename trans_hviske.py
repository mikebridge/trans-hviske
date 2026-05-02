#!/usr/bin/env python3
"""
Generate a WebVTT file from an audio file using WhisperX.

Example:
  trans-hviske input.mp3 --language da --model large-v2

Requirements:
  pip install whisperx
  ffmpeg must be installed and available on PATH
"""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Transcribe audio to WebVTT with WhisperX")
    parser.add_argument("audio_file", help="Path to input audio file, e.g. input.mp3")
    parser.add_argument("--language", default="da", help="Source language code, default: da")
    parser.add_argument("--model", default="large-v2", help="Whisper model name, default: large-v2")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Inference device")
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="Compute type, e.g. int8 for CPU, float16 for CUDA"
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Reduce if memory is tight")
    parser.add_argument(
        "--output",
        help="Output VTT path. Default: <input_stem>.vtt"
    )
    return parser.parse_args()


def cleanup_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def vtt_escape(text: str) -> str:
    return html.escape(text, quote=False)


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    ms_total = int(round(seconds * 1000))
    hours = ms_total // 3_600_000
    ms_total %= 3_600_000
    minutes = ms_total // 60_000
    ms_total %= 60_000
    secs = ms_total // 1000
    millis = ms_total % 1000
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def normalize_segment(segment: dict[str, Any]) -> dict[str, Any]:
    start = float(segment.get("start", 0.0) or 0.0)
    end = float(segment.get("end", start) or start)
    text = cleanup_text(str(segment.get("text", "") or ""))
    return {
        "start": start,
        "end": max(end, start + 0.01),
        "text": text,
    }


def transcribe_and_align(
        whisperx: Any,
        audio: Any,
        model_name: str,
        language: str,
        device: str,
        compute_type: str,
        batch_size: int
) -> dict[str, Any]:
    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language
    )

    result = model.transcribe(
        audio,
        batch_size=batch_size,
        language=language,
        print_progress=True
    )

    align_language = result.get("language") or language
    model_a, metadata = whisperx.load_align_model(
        language_code=align_language,
        device=device
    )

    aligned = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
        print_progress=True
    )

    return aligned


def write_vtt(output_path: Path, segments: list[dict[str, Any]]) -> None:
    lines: list[str] = ["WEBVTT", ""]

    cue_number = 1
    for seg in segments:
        if not seg["text"]:
            continue

        start_ts = format_timestamp(seg["start"])
        end_ts = format_timestamp(seg["end"])
        text = vtt_escape(seg["text"])

        lines.append(str(cue_number))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(text)
        lines.append("")
        cue_number += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    import whisperx

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    output_path = Path(args.output) if args.output else audio_file.with_suffix(".vtt")

    audio = whisperx.load_audio(str(audio_file))

    duration = len(audio) / 16000
    print(f"Loaded {audio_file.name} ({duration:.1f}s)")
    print("Transcribing...")
    result = transcribe_and_align(
        whisperx=whisperx,
        audio=audio,
        model_name=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size
    )

    segments = [normalize_segment(seg) for seg in result.get("segments", [])]
    write_vtt(output_path, segments)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
