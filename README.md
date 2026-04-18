# trans-hviske

Generate a bilingual (Danish + English) WebVTT subtitle file from an audio file, using WhisperX.

This is just a quick demo.

## Install

`ffmpeg` must be on `PATH`.

```bash
pip install -e .
```

## Usage

```bash
# CPU
trans-hviske input.mp3 --language da --device cpu --compute-type int8

# CUDA
trans-hviske input.mp3 --language da --device cuda --compute-type float16
```
