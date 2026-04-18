# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

CLI tool that takes a Danish audio file and emits a bilingual WebVTT subtitle file (Danish cue in yellow, English translation in white).

## Install & Run

```bash
pip install -e .
trans-hviske input.mp3 --language da --device cpu --compute-type int8
# or: --device cuda --compute-type float16
```

Output defaults to `<input_stem>.bilingual.vtt` next to the input; override with `--output`.

## Layout

- `trans_hviske.py` — single-module source (flat layout, declared via `[tool.setuptools] py-modules` in `pyproject.toml`).
- `pyproject.toml` — setuptools build; `[project.scripts]` wires `trans-hviske` to `trans_hviske:main`.
- `ffmpeg` must be on `PATH` at runtime; `whisperx` is the sole Python dependency.
- No tests.

If the module ever grows beyond a single file, promote `trans_hviske.py` to a package directory (`trans_hviske/__init__.py` + submodules) and drop the `py-modules` line — the `[project.scripts]` entry point keeps working.

## Architecture

Two-pass WhisperX pipeline in `trans_hviske.py`:

1. `transcribe_and_align(..., task="transcribe")` — source-language (Danish) transcript, then `whisperx.align` for word-level timings.
2. `transcribe_and_align(..., task="translate")` — English translation pass (faster-whisper's translate task), also aligned.
3. `write_vtt` iterates Danish segments as the timing spine. For each Danish cue, `join_overlapping_text` picks English segments that overlap by ≥ 0.15s; if none overlap, it falls back to the English segment whose center is closest. Each cue emits `<c.da>…</c.da>` and `<c.en>…</c.en>` under a `STYLE` block that colors the classes.

The Danish transcript drives cue boundaries — English is fitted to it, not the reverse. Changing the overlap threshold or the fallback in `join_overlapping_text` is how you'd tune alignment quality.

Both passes call `whisperx.load_model(..., language=args.language)` even for the translate pass; the task distinction is passed through `model.transcribe(..., task=...)`. Some WhisperX versions don't accept `task` on `transcribe` — the docstring at the top of the file flags this.
