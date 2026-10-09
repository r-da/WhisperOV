# whisper-ov

CLI for Whisper ASR powered by OpenVINO with CPU, GPU, and NPU support.

## Features

- Transcribe audio files using Whisper models accelerated by OpenVINO
- Support for CPU, GPU, and NPU inference devices
- Automatic model download from HuggingFace (pre-converted OpenVINO formats)
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
- `uv run whisper-ov devices` — Show available OpenVINO devices
- `uv run whisper-ov models` — List available Whisper models

### Output formats

- `txt` — Plain text transcript
- `srt` — SubRip subtitle format with timestamps
- `vtt` — WebVTT subtitle format with timestamps
- `all` — Generate all formats (default)

Output can be sent to stdout with `-o -` (txt only).

## Supported Models

| Model ID | Size (GB) |
|---|---|
| openai/whisper-tiny | 0.07 |
| openai/whisper-base | 0.14 |
| openai/whisper-small | 0.50 |
| openai/whisper-medium | 1.42 |
| openai/whisper-large-v3 | 2.91 |
| openai/whisper-large-v3-int4 | 1.00 |
| openai/whisper-large-v3-int8 | 1.50 |
| openai/whisper-large-v3-turbo | 1.52 |
| openai/whisper-large-v3-turbo-int4 | 0.75 |
| openai/whisper-large-v3-turbo-int8 | 1.00 |

Models are automatically downloaded from the OpenVINO HuggingFace collection on first use and cached in `~/.cache/whisper-ov`.

## Project Structure

```
.
├── src/
│   └── whisper_ov/
│       ├── __init__.py      # Package init, version
│       ├── cli.py           # CLI entry point (Typer)
│       ├── whisper.py       # OpenVINO WhisperPipeline wrapper
│       ├── audio.py         # Audio loading with PyAV
│       └── formats.py       # SRT/VTT output format helpers
├── pyproject.toml
└── uv.lock
```

## License

MIT License. See [LICENSE](LICENSE) for details.