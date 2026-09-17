"""CLI for whisper-ov - Whisper ASR using OpenVINO."""

import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from whisper_ov.audio import load_audio
from whisper_ov.formats import generate_srt, generate_vtt
from whisper_ov.whisper import list_models
from whisper_ov.whisper import transcribe as _transcribe

app = typer.Typer(
    name="whisper-ov",
    help="Whisper ASR powered by OpenVINO — CPU, GPU, NPU",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()
stderr_console = Console(file=sys.stderr)


def _parse_format(format_str: str) -> list[str]:
    """Parse comma-separated format string into list of formats."""
    return [f.strip() for f in format_str.split(",") if f.strip()]


def _resolve_output_name(audio_path: Path, output: str | None, output_dir: str | None) -> tuple[Path, str]:
    """Resolve the output directory and base filename.

    Returns:
        Tuple of (output_dir, base_filename_without_extension).
    """
    # Default output dir is the directory of the input audio file
    out_dir = audio_path.parent

    # If --output-dir is explicitly set, use it
    if output_dir:
        out_dir = Path(output_dir)

    if output:
        if output == "-":
            return out_dir, "-"
        base = Path(output)
        if base.suffix:
            name = base.stem
        else:
            name = base.name
        # If output includes a directory AND --output-dir was not set, use it
        if base.parent != Path(".") and not output_dir:
            out_dir = base.parent
    else:
        name = audio_path.stem

    return out_dir, name


@app.callback()
def callback() -> None:
    """whisper-ov - Whisper ASR powered by OpenVINO."""


@app.command()
def transcribe(
    audio_file: str = typer.Argument(
        ...,
        help="Path to the audio file to transcribe",
        dir_okay=False,
        exists=True,
    ),
    model: str = typer.Option(
        "openai/whisper-large-v3-turbo",
        "--model",
        "-m",
        help="Whisper model ID or path",
    ),
    device: str = typer.Option(
        "AUTO",
        "--device",
        "-d",
        help="Inference device (CPU, GPU, NPU, AUTO)",
        case_sensitive=False,
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code (e.g. en, it, fr). Auto-detect if not specified.",
    ),
    cache_dir: str = typer.Option(
        "~/.cache/whisper-ov",
        "--cache-dir",
        help="Directory for cached models",
    ),
    format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Output format(s): txt, srt, vtt, or comma-separated list. Default: all",
    ),
    output: str | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Output filename base (without extension). Default: same as input file name. Use '-' for stdout (txt only).",
    ),
    output_dir: str | None = typer.Option(
        None,
        "--output-dir",
        help="Directory for output files. Default: same as input file directory",
    ),
) -> None:
    """Transcribe an audio file to text using Whisper + OpenVINO."""
    audio_path = Path(audio_file)

    # Resolve output paths
    out_dir, base_name = _resolve_output_name(audio_path, output, output_dir)

    # Parse formats
    if format == "all":
        formats = ["txt", "srt", "vtt"]
    else:
        formats = _parse_format(format)

    if output == "-" and formats != ["txt"]:
        stderr_console.print("[yellow]stdout (-o -) only supports txt format, ignoring other formats[/yellow]")
        formats = ["txt"]

    # Load audio
    stderr_console.print(f"Loading audio: {audio_path.name}...", style="yellow")
    try:
        audio = load_audio(str(audio_path))
    except Exception as e:
        stderr_console.print(f"[red]Error loading audio: {e}[/red]")
        raise typer.Exit(1)

    duration = len(audio) / 16000
    stderr_console.print(
        f"Audio duration: {duration:.1f}s, samples: {len(audio)}",
        style="dim",
    )

    # Transcribe with timing
    stderr_console.print(f"Transcribing on {device.upper()}...", style="green")
    try:
        t0 = time.perf_counter()
        result = _transcribe(audio, model_id=model, device=device.upper(), language=language, cache_dir=cache_dir)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        stderr_console.print(f"[red]Transcription error: {e}[/red]")
        raise typer.Exit(1)

    stderr_console.print(f"Transcription completed in {elapsed:.2f}s", style="dim")

    full_text = result["text"]
    segments = result["segments"]

    # Build content dict
    content = {}
    if "txt" in formats:
        content["txt"] = full_text
    if "srt" in formats and segments:
        content["srt"] = generate_srt(segments)
    if "vtt" in formats and segments:
        content["vtt"] = generate_vtt(segments)

    # Output
    if output == "-":
        console.print(full_text)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    extensions = {"txt": ".txt", "srt": ".srt", "vtt": ".vtt"}
    for fmt, text in content.items():
        ext = extensions[fmt]
        out_path = out_dir / f"{base_name}{ext}"
        out_path.write_text(text, encoding="utf-8")
        stderr_console.print(f"Output written to {out_path}", style="dim")


@app.command()
def devices() -> None:
    """Show available OpenVINO devices."""
    try:
        import openvino as ov
    except ImportError:
        console.print("[red]openvino not installed. Install with: pip install openvino openvino-genai[/red]")
        raise typer.Exit(1)

    core = ov.Core()

    table = Table(title="OpenVINO Devices")
    table.add_column("Device", style="cyan")
    table.add_column("Full Name", style="white")
    table.add_column("Architecture", style="dim")

    for device_name in core.available_devices:
        full_name = core.get_property(device_name, "FULL_DEVICE_NAME")
        arch = core.get_property(device_name, "DEVICE_ARCHITECTURE")
        table.add_row(device_name, full_name, arch)

    console.print(table)


@app.command()
def models() -> None:
    """List available Whisper models."""
    model_list = list_models()

    table = Table(title="Available Whisper Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Size (GB)", style="dim", justify="right")

    for entry in model_list:
        table.add_row(entry["model_id"], entry["name"], f"{entry['size_gb']:.2f}")

    console.print(table)


def main() -> None:
    """Entry point for the whisper-ov CLI."""
    app()


if __name__ == "__main__":
    main()
