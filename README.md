# whisper-ov

CLI for Whisper ASR powered by OpenVINO with CPU, Intel GPU, and NPU support, plus NVIDIA GPUs via CUDA
(faster-whisper).

## Features

- Transcribe audio files using Whisper models accelerated by OpenVINO
- Support for CPU, GPU (officially Intel only), and NPU inference devices
- Optional NVIDIA GPU support via CUDA ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)), used
  automatically when available
- Automatic model download from HuggingFace (pre-converted OpenVINO and CTranslate2 formats)
- Multiple output formats: TXT, SRT, VTT
- Command-line interface with rich output
- Automatic language detection

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python package manager)
- Python >= 3.10 (automatically managed by uv)
- FFmpeg (for audio decoding)

### Install from source

```bash
# Clone the repository
git clone https://github.com/r-da/WhisperOV
cd WhisperOV

# Create virtual environment and install dependencies
uv sync
```

### NVIDIA GPUs (CUDA)

OpenVINO's `GPU` device only works with Intel GPUs. Its OpenCL backend may list NVIDIA/AMD GPUs too:
`whisper-ov devices` marks them as unsupported and `-d GPU` refuses them. To run on an NVIDIA GPU, install the optional `cuda` extra, which
adds faster-whisper (CTranslate2) and the CUDA 12 cuBLAS/cuDNN libraries (about 2.5 GB on disk):

```bash
uv sync --extra cuda
```

Requirements: an NVIDIA driver supporting CUDA 12 or newer and a GPU with compute capability 5.3 or newer, in practice
Pascal (GTX 10xx) and later. The prebuilt CTranslate2 has no kernels for older GPUs: Maxwell cards such as the
GTX 9xx, 950M or 750 Ti (compute capability 5.0/5.2) are detected but rejected, and `AUTO` uses OpenVINO. The
CUDA toolkit is not needed. On WSL2 the Windows NVIDIA driver is enough. Check that the GPU is detected with
`uv run whisper-ov devices` (it shows up as `CUDA:0`).

The libraries installed by the extra are loaded automatically. If CTranslate2 still can't find them (e.g. errors about
`libcublas.so.12` or `libcudnn`), set `LD_LIBRARY_PATH` before running:

```bash
export LD_LIBRARY_PATH=$(uv run python -c 'import nvidia.cublas.lib as a, nvidia.cudnn.lib as b; print(*a.__path__, *b.__path__, sep=":")')
```

### Running commands

```bash
uv run whisper-ov --help
```

### Install in development mode

```bash
uv sync --all-extras
```

## Usage

Run commands with `uv run` to automatically use the project's virtual environment:

### Transcribe an audio file

```bash
uv run whisper-ov transcribe <audio_file>
```

### Available options

```bash
uv run whisper-ov transcribe audio.mp3 \
  --model openai/whisper-large-v3-turbo \
  --device AUTO \
  --language it \
  --format srt,vtt \
  --output-dir ./subtitles
```

### Devices

- `AUTO` (default) — NVIDIA GPU via CUDA when the `cuda` extra is installed and a GPU is detected, otherwise OpenVINO
  `AUTO`. If loading or running the model on CUDA fails, it prints a warning and falls back to OpenVINO.
- `CUDA`, `CUDA:<n>` — force the NVIDIA GPU (no fallback)
- `CPU`, `GPU`, `NPU` — OpenVINO devices (`GPU` is officially Intel GPUs only)

On CUDA, `--compute-type` selects the CTranslate2 precision. The default `auto` picks the fastest type supported by the
GPU (float16 on recent cards, int8/float32 on older ones). The `-int8` model variants default to `int8_float16`. INT4
variants are not available on CUDA and always run on OpenVINO.

### Improving punctuation and spelling

Whisper imitates the style of the text it sees as the "previous transcription". You can pass a short sample text
to steer the spelling of names and technical terms, and (to a lesser extent) punctuation and capitalization:

- `--hotwords TEXT` — applied to **every** 30s window (audio longer than 30s)
- `--initial-prompt TEXT` — applied to the **first** 30s window only

```bash
uv run whisper-ov transcribe meeting.mp3 -l it \
  --hotwords "Intelligenza artificiale: AI hype, AI winter, GPT, LISP, Dartmouth."
```

Both are empty by default, and blank values are ignored. Tips: write the prompt in the language of the audio and keep
it to one or two sentences.

> **Warning:** the prompt is a hint, not an instruction, and it is not guaranteed to help. In a quick test on a 60s
> Italian recording, models that already punctuate well (`whisper-large-v3`, `whisper-large-v3-turbo-int4`) gained
> nothing from it, and `--hotwords` (applied to every window) made things worse in the second 30s window:
> a skipped sentence, invented words ("il governo dell'AI"), and with `whisper-large-v3` part of the prompt
> ("Come funziona...") repeated at the end of the transcript. A list of domain terms fixed a misrecognized word
> ("AIP" → "AI hype") only with `whisper-small`. `--initial-prompt` was safer but had little effect. Always compare
> the result against a run without prompt on a sample of your audio.

### Commands

- `uv run whisper-ov transcribe <file>` — Transcribe an audio file
- `uv run whisper-ov devices` — Show available inference devices (OpenVINO and CUDA)
- `uv run whisper-ov models` — List available Whisper models

### Output formats

- `txt` — Plain text transcript
- `srt` — SubRip subtitle format with timestamps
- `vtt` — WebVTT subtitle format with timestamps
- `all` — Generate all formats (default)

Output can be sent to stdout with `-o -` (txt only).

## Supported Models

| Model ID | Size (GB) | CUDA |
|---|---|---|
| openai/whisper-tiny | 0.07 | yes |
| openai/whisper-base | 0.14 | yes |
| openai/whisper-small | 0.50 | yes |
| openai/whisper-medium | 1.42 | yes |
| openai/whisper-large-v3 | 2.91 | yes |
| openai/whisper-large-v3-int4 | 1.00 | no |
| openai/whisper-large-v3-int8 | 1.50 | yes |
| openai/whisper-large-v3-turbo | 1.52 | yes |
| openai/whisper-large-v3-turbo-int4 | 0.75 | no |
| openai/whisper-large-v3-turbo-int8 | 1.00 | yes |

Models are automatically downloaded on first use: from the OpenVINO HuggingFace collection into
`~/.cache/whisper-ov/models`, and for CUDA from the CTranslate2 conversions (Systran, mobiuslabsgmbh) into
`~/.cache/whisper-ov/models-ct2`.

## Project Structure

```
.
├── src/
│   └── whisper_ov/
│       ├── __init__.py      # Package init, version
│       ├── cli.py           # CLI entry point (Typer)
│       ├── whisper.py       # OpenVINO WhisperPipeline wrapper, model table, downloads
│       ├── whisper_ct2.py   # faster-whisper (CUDA) wrapper
│       ├── audio.py         # Audio loading with PyAV
│       └── formats.py       # SRT/VTT output format helpers
├── pyproject.toml
└── uv.lock
```

## License

MIT License. See [LICENSE](LICENSE) for details.