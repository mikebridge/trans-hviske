#!/usr/bin/env python3
"""
Generate a bilingual WebVTT file from an audio file using WhisperX.

Output format per cue:
  Danish line in yellow
  English line in white

Example:
  trans-hviske input.mp3 --language da --model large-v2

Requirements:
  pip install whisperx
  ffmpeg must be installed and available on PATH

Notes:
- WhisperX provides alignment and uses a faster-whisper backend. Its documented
  Python flow is: load_model() -> transcribe() -> load_align_model() -> align(). 
- This script runs two passes:
    1) original-language transcription
    2) English translation
- Then it maps each Danish cue to overlapping English text and writes one VTT file.
"""

from __future__ import annotations

import argparse
import html
import math
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create bilingual Danish+English VTT with WhisperX")
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
        help="Output VTT path. Default: <input_stem>.bilingual.vtt"
    )
    return parser.parse_args()


def cleanup_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def vtt_escape(text: str) -> str:
    # Escape HTML-sensitive characters but preserve WebVTT class tags that we add later.
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


def join_overlapping_text(
        base_start: float,
        base_end: float,
        other_segments: list[dict[str, Any]],
        overlap_threshold: float = 0.15
) -> str:
    """
    Return concatenated text from segments in other_segments that overlap the
    [base_start, base_end] interval by at least overlap_threshold seconds.
    """
    matches: list[str] = []

    for seg in other_segments:
        overlap = min(base_end, seg["end"]) - max(base_start, seg["start"])
        if overlap >= overlap_threshold:
            if seg["text"]:
                matches.append(seg["text"])

    if matches:
        text = " ".join(matches)
        return cleanup_text(text)

    # Fallback: pick the closest segment center if there was no direct overlap.
    base_center = (base_start + base_end) / 2
    closest = None
    closest_distance = math.inf

    for seg in other_segments:
        seg_center = (seg["start"] + seg["end"]) / 2
        distance = abs(seg_center - base_center)
        if distance < closest_distance and seg["text"]:
            closest_distance = distance
            closest = seg["text"]

    return cleanup_text(closest or "")


def transcribe_and_align(
        whisperx: Any,
        audio: Any,
        model_name: str,
        language: str,
        task: str,
        device: str,
        compute_type: str,
        batch_size: int
) -> dict[str, Any]:
    """
    Run WhisperX transcription and alignment.

    task:
      - "transcribe" for source-language transcript
      - "translate" for English translation

    Some WhisperX versions expose the underlying faster-whisper task through
    model.transcribe(..., task="translate"). If your installed version rejects
    the task argument, see the note below after the script.
    """
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
        task=task
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
        return_char_alignments=False
    )

    return aligned


def write_vtt(
        output_path: Path,
        danish_segments: list[dict[str, Any]],
        english_segments: list[dict[str, Any]]
) -> None:
    lines: list[str] = [
        "WEBVTT",
        "",
        "STYLE",
        "::cue(.da) { color: yellow; }",
        "::cue(.en) { color: white; }",
        "",
    ]

    cue_number = 1

    for da in danish_segments:
        if not da["text"]:
            continue

        en_text = join_overlapping_text(da["start"], da["end"], english_segments)
        if not en_text:
            en_text = ""

        start_ts = format_timestamp(da["start"])
        end_ts = format_timestamp(da["end"])

        da_text = vtt_escape(da["text"])
        en_text = vtt_escape(en_text)

        lines.append(str(cue_number))
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(f"<c.da>{da_text}</c.da>")
        lines.append(f"<c.en>{en_text}</c.en>")
        lines.append("")
        cue_number += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()

    import whisperx

    audio_file = Path(args.audio_file)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    output_path = Path(args.output) if args.output else audio_file.with_suffix(".bilingual.vtt")

    audio = whisperx.load_audio(str(audio_file))

    print("Transcribing original language...")
    original_result = transcribe_and_align(
        whisperx=whisperx,
        audio=audio,
        model_name=args.model,
        language=args.language,
        task="transcribe",
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size
    )

    print("Translating to English...")
    english_result = transcribe_and_align(
        whisperx=whisperx,
        audio=audio,
        model_name=args.model,
        language=args.language,
        task="translate",
        device=args.device,
        compute_type=args.compute_type,
        batch_size=args.batch_size
    )

    danish_segments = [normalize_segment(seg) for seg in original_result.get("segments", [])]
    english_segments = [normalize_segment(seg) for seg in english_result.get("segments", [])]

    write_vtt(output_path, danish_segments, english_segments)

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
