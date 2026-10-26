"""CLI for whisper-ov - Whisper ASR using OpenVINO."""

import importlib.util
import os
import sys
import time
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from whisper_ov import whisper_ct2
from whisper_ov.audio import load_audio
from whisper_ov.formats import generate_srt, generate_vtt
from whisper_ov.whisper import list_models, unsupported_gpu_reason
from whisper_ov.whisper import transcribe as _transcribe

app = typer.Typer(
    name="whisper-ov",
    help="Whisper ASR powered by OpenVINO (CPU, Intel GPU, NPU) and faster-whisper (NVIDIA CUDA)",
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


def _warn_no_fp16(cuda: dict, model: str, compute_type: str) -> None:
    """Warn that fp16 models on a GPU without float16 (Pascal GTX 10xx) fall back to int8/float32."""
    if compute_type == "auto" and not model.endswith("-int8"):
        reason = whisper_ct2.no_fp16_reason(cuda)
        if reason:
            stderr_console.print(reason, style="yellow")


def _transcribe_with_backend(audio, device: str, model: str, compute_type: str, **kwargs) -> dict:
    """Run on CUDA (faster-whisper) or OpenVINO depending on the device; AUTO prefers CUDA when available."""
    device = device.upper()
    cuda_index = whisper_ct2.parse_device(device)

    if cuda_index is not None:
        if importlib.util.find_spec("faster_whisper") is None:
            raise RuntimeError("CUDA support is not installed. Install it with: uv sync --extra cuda")
        # Without this check a missing device only surfaces mid-inference as "invalid device ordinal"
        cuda_devices = whisper_ct2.cuda_devices()
        if cuda_index >= len(cuda_devices):
            raise RuntimeError(
                f"CUDA:{cuda_index} not found, CTranslate2 sees {len(cuda_devices)} CUDA device(s). "
                "Check the NVIDIA driver (nvidia-smi) and CUDA_VISIBLE_DEVICES"
            )
        if not cuda_devices[cuda_index]["supported"]:
            raise RuntimeError(whisper_ct2.unsupported_reason(cuda_devices[cuda_index]))
        _warn_no_fp16(cuda_devices[cuda_index], model, compute_type)
        stderr_console.print(f"Transcribing on CUDA:{cuda_index}...", style="green")
        return whisper_ct2.transcribe(
            audio, model_id=model, device_index=cuda_index, compute_type=compute_type, **kwargs
        )

    if device == "AUTO" and whisper_ct2.supports_model(model) and importlib.util.find_spec("faster_whisper"):
        cuda_devices = whisper_ct2.cuda_devices()
        usable = [d for d in cuda_devices if d["supported"]]
        if usable:
            cuda = usable[0]
            _warn_no_fp16(cuda, model, compute_type)
            stderr_console.print(f"Transcribing on CUDA:{cuda['index']} (AUTO)...", style="green")
            try:
                return whisper_ct2.transcribe(
                    audio, model_id=model, device_index=cuda["index"], compute_type=compute_type, **kwargs
                )
            except Exception as e:  # noqa: BLE001 - any CUDA failure (missing cuDNN, driver issues) falls back
                stderr_console.print(f"[yellow]CUDA transcription failed ({e}), falling back to OpenVINO[/yellow]")
        elif cuda_devices:
            stderr_console.print(f"{whisper_ct2.unsupported_reason(cuda_devices[0])}, using OpenVINO", style="yellow")
        else:
            stderr_console.print("No CUDA device detected, using OpenVINO", style="yellow")

    # Fail early instead of with an OpenVINO kernel compilation error
    reason = unsupported_gpu_reason(device)
    if reason:
        raise RuntimeError(reason)

    stderr_console.print(f"Transcribing on {device}...", style="green")
    return _transcribe(audio, model_id=model, device=device, **kwargs)


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
        help="Inference device: CPU, GPU (OpenVINO, officially Intel only), NPU, CUDA or CUDA:<n> (NVIDIA), "
        "AUTO (CUDA if available, otherwise OpenVINO)",
        case_sensitive=False,
    ),
    language: str | None = typer.Option(
        None,
        "--language",
        "-l",
        help="Language code (e.g. en, it, fr). Auto-detect if not specified.",
    ),
    compute_type: str = typer.Option(
        "auto",
        "--compute-type",
        help="CTranslate2 compute type, CUDA only (auto, float16, int8_float16, int8, float32, ...)",
    ),
    cache_dir: str = typer.Option(
        "~/.cache/whisper-ov",
        "--cache-dir",
        help="Directory for cached models",
    ),
    initial_prompt: str | None = typer.Option(
        None,
        "--initial-prompt",
        help="Text used as previous transcription for the first 30s window only (style, spelling).",
    ),
    hotwords: str | None = typer.Option(
        None,
        "--hotwords",
        help="Text used as previous transcription for every 30s window (style, punctuation, names, terms).",
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
    """Transcribe an audio file to text using Whisper + OpenVINO or faster-whisper (CUDA)."""
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
    try:
        t0 = time.perf_counter()
        result = _transcribe_with_backend(
            audio,
            device=device,
            model=model,
            compute_type=compute_type,
            language=language,
            cache_dir=cache_dir,
            initial_prompt=initial_prompt,
            hotwords=hotwords,
        )
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
        out_path.write_text(text, encoding="utf-8", newline="\n")
        stderr_console.print(f"Output written to {out_path}", style="dim")


@app.command()
def devices() -> None:
    """Show available inference devices (OpenVINO and CUDA)."""
    try:
        import openvino as ov
    except ImportError:
        console.print("[red]openvino not installed. Install dependencies with: uv sync[/red]")
        raise typer.Exit(1)

    core = ov.Core()

    table = Table(title="Inference Devices")
    table.add_column("Device", style="cyan")
    table.add_column("Full Name", style="white")
    table.add_column("Architecture", style="dim")

    unsupported = []
    for device_name in core.available_devices:
        full_name = core.get_property(device_name, "FULL_DEVICE_NAME")
        arch = core.get_property(device_name, "DEVICE_ARCHITECTURE")
        reason = unsupported_gpu_reason(device_name, core)
        if reason:
            arch = f"{arch} (unsupported)"
            unsupported.append(reason)
        table.add_row(device_name, full_name, arch)

    cuda_devices = whisper_ct2.cuda_devices()
    for cuda in cuda_devices:
        name = f"{cuda['name'] or 'NVIDIA GPU'} (faster-whisper)"
        arch = ", ".join(cuda["compute_types"])
        if cuda["compute_capability"] is not None:
            major, minor = cuda["compute_capability"]
            arch = f"sm_{major}{minor}: {arch}" if cuda["supported"] else f"sm_{major}{minor}: unsupported"
        table.add_row(f"CUDA:{cuda['index']}", name, arch)

    console.print(table)

    if importlib.util.find_spec("faster_whisper") is None:
        stderr_console.print("CUDA support not installed (NVIDIA GPUs): uv sync --extra cuda", style="dim")
    elif not cuda_devices:
        stderr_console.print(
            "CUDA support installed but no CUDA device detected: check the NVIDIA driver (nvidia-smi) "
            "and CUDA_VISIBLE_DEVICES",
            style="yellow",
        )
    for reason in unsupported:
        stderr_console.print(reason, style="yellow")
    for cuda in cuda_devices:
        if not cuda["supported"]:
            stderr_console.print(whisper_ct2.unsupported_reason(cuda), style="yellow")
        elif reason := whisper_ct2.no_fp16_reason(cuda):
            stderr_console.print(reason, style="yellow")


@app.command()
def models() -> None:
    """List available Whisper models."""
    model_list = list_models()

    table = Table(title="Available Whisper Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Size (GB)", style="dim", justify="right")
    table.add_column("CUDA", justify="center")

    for entry in model_list:
        table.add_row(entry["model_id"], entry["name"], f"{entry['size_gb']:.2f}", "yes" if entry["cuda"] else "no")

    console.print(table)


def _terminate_process(code: int) -> None:
    """End the current Windows process without notifying the loaded DLLs.

    os._exit is not enough: the CRT's _exit goes through ExitProcess, which still calls every DLL's
    DllMain(DLL_PROCESS_DETACH) after killing the other threads, and that is where the GPU driver hangs.
    """
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.TerminateProcess.argtypes = (wintypes.HANDLE, wintypes.UINT)
    kernel32.TerminateProcess.restype = wintypes.BOOL
    kernel32.TerminateProcess(kernel32.GetCurrentProcess(), code)
    # Only reached if TerminateProcess failed
    os._exit(code)


def main() -> None:
    """Entry point for the whisper-ov CLI."""
    if sys.platform != "win32":
        app()
        return
    # On Windows the OpenVINO GPU plugin / Intel OpenCL driver can deadlock while its DLLs unload at process exit,
    # after all output is written: flush and terminate without running the native teardown
    try:
        app()
        code = 0
    except SystemExit as e:
        if e.code is None or isinstance(e.code, int):
            code = e.code or 0
        else:
            print(e.code, file=sys.stderr)
            code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    _terminate_process(code)


if __name__ == "__main__":
    main()
