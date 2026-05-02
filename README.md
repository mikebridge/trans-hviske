# trans-hviske

Generate a single-language WebVTT subtitle file from an audio file, using WhisperX. Defaults to Danish.

## Install

`ffmpeg` must be on `PATH`.

```bash
# CPU only
pip install -e .

# CUDA (NVIDIA GPU). Also pulls cuBLAS + cuDNN 9 user-space libs.
# Requires an NVIDIA driver already installed on the host (verify with `nvidia-smi`).
pip install -e '.[cuda]'
```

CUDA is NVIDIA-only — it does not run on macOS (Apple Silicon or otherwise).

## Usage

```bash
# CPU
trans-hviske input.mp3 --language da --device cpu --compute-type int8

# CUDA
trans-hviske input.mp3 --language da --device cuda --compute-type float16
```
