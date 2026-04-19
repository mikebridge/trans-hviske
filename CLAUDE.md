# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

CLI tool that takes an audio file and emits a single-language WebVTT subtitle file using WhisperX (transcription + word-level alignment). Defaults to Danish.

## Install & Run

```bash
pip install -e .
trans-hviske input.mp3 --language da --device cpu --compute-type int8
# or: --device cuda --compute-type float16
```

Output defaults to `<input_stem>.vtt` next to the input; override with `--output`.

## Layout

- `trans_hviske.py` — single-module source (flat layout, declared via `[tool.setuptools] py-modules` in `pyproject.toml`).
- `pyproject.toml` — setuptools build; `[project.scripts]` wires `trans-hviske` to `trans_hviske:main`.
- `ffmpeg` must be on `PATH` at runtime; `whisperx` is the sole Python dependency.
- No tests.

If the module ever grows beyond a single file, promote `trans_hviske.py` to a package directory (`trans_hviske/__init__.py` + submodules) and drop the `py-modules` line — the `[project.scripts]` entry point keeps working.

## Architecture

Single-pass WhisperX pipeline in `trans_hviske.py`:

1. `transcribe_and_align` — `whisperx.load_model` → `model.transcribe` → `whisperx.load_align_model` → `whisperx.align` for word-level timings.
2. `write_vtt` emits one cue per segment with the aligned timestamps.

Translation is out of scope — if bilingual output is needed again, restore the `task="translate"` pass and the cue-merging logic from git history.
