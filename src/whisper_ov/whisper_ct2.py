"""faster-whisper (CTranslate2) wrapper for speech-to-text on NVIDIA GPUs via CUDA."""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

from whisper_ov.whisper import HF_TO_CT2, _download_model, _model_dir

# Cache subdir for CTranslate2 models, kept apart from the OpenVINO ones in "models"
CT2_SUBDIR = "models-ct2"

# cuBLAS/cuDNN from the pip wheels of the `cuda` extra, in dependency order. CTranslate2 dlopens them by soname,
# so loading them globally first avoids having to put their directories on LD_LIBRARY_PATH
_CUDA_LIBS = (
    ("cublas", "libcublasLt.so.12"),
    ("cublas", "libcublas.so.12"),
    ("cudnn", "libcudnn.so.9"),
)

_cuda_libs_loaded = False


def _preload_cuda_libs() -> None:
    """Load the pip-installed CUDA libraries, if any; a system-wide CUDA install is used otherwise."""
    global _cuda_libs_loaded
    if _cuda_libs_loaded or not sys.platform.startswith("linux"):
        return
    _cuda_libs_loaded = True

    try:
        import nvidia
    except ImportError:
        return

    for package, lib in _CUDA_LIBS:
        for base in nvidia.__path__:
            path = Path(base) / package / "lib" / lib
            if path.is_file():
                try:
                    ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
                except OSError as e:
                    print(f"Warning: could not load {path}: {e}", file=sys.stderr)
                break


def parse_device(device: str) -> int | None:
    """Return the CUDA device index for 'CUDA' or 'CUDA:<n>', None for any other device."""
    name, _, index = device.upper().partition(":")
    if name != "CUDA":
        return None
    if not index:
        return 0
    if not index.isdigit():
        raise ValueError(f"Invalid CUDA device '{device}', expected CUDA or CUDA:<index>")
    return int(index)


def cuda_devices() -> list[dict]:
    """Return the CUDA devices visible to CTranslate2, empty if the `cuda` extra is missing or no GPU is found.

    Each entry has 'index' and 'compute_types' (sorted list of supported CTranslate2 compute types).
    """
    _preload_cuda_libs()
    try:
        import ctranslate2
    except ImportError:
        return []

    try:
        count = ctranslate2.get_cuda_device_count()
    except RuntimeError:
        return []

    devices = []
    for index in range(count):
        try:
            compute_types = sorted(ctranslate2.get_supported_compute_types("cuda", index))
        except RuntimeError:
            compute_types = []
        devices.append({"index": index, "compute_types": compute_types})
    return devices


def supports_model(hf_model_id: str) -> bool:
    """Whether the model has a CTranslate2 version (INT4 variants do not)."""
    return HF_TO_CT2.get(hf_model_id, "") is not None


def _get_ct2_repo_id(hf_model_id: str) -> str:
    """Map a HuggingFace Whisper model ID to its CTranslate2 pre-converted repo ID."""
    if hf_model_id in HF_TO_CT2:
        repo_id = HF_TO_CT2[hf_model_id]
        if repo_id is None:
            raise ValueError(
                f"'{hf_model_id}' is not available on CUDA (no INT4 support): use the INT8 or fp16 variant"
            )
        return repo_id
    # If not in our known list, try the Systran naming scheme (openai/whisper-x -> Systran/faster-whisper-x)
    name = hf_model_id.split("/")[-1]
    return f"Systran/faster-{name}"


def _resolve_model_path(hf_model_id: str, cache_dir: str) -> str:
    """Return the local CTranslate2 model directory, downloading it if needed."""
    repo_id = _get_ct2_repo_id(hf_model_id)
    # Named after the repo, so the INT8 variants share the download of their fp16 model
    model_dir = _download_model(repo_id, _model_dir(cache_dir, CT2_SUBDIR, repo_id))
    return str(model_dir)


def _resolve_compute_type(hf_model_id: str, compute_type: str) -> str:
    """Map 'auto' to INT8 weights for the -int8 model variants; any other value is passed through."""
    if compute_type == "auto" and hf_model_id.endswith("-int8"):
        # CTranslate2 falls back to the closest type the GPU supports (e.g. int8_float32 without fp16)
        return "int8_float16"
    return compute_type


def transcribe(
    audio: np.ndarray,
    model_id: str = "openai/whisper-large-v3-turbo",
    device: str = "cuda",
    device_index: int = 0,
    language: str | None = None,
    cache_dir: str = "~/.cache/whisper-ov",
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    compute_type: str = "auto",
) -> dict:
    """Transcribe audio using faster-whisper (CTranslate2).

    Args:
        audio: numpy array of 16kHz mono audio (float32).
        model_id: HuggingFace Whisper model ID (e.g. 'openai/whisper-large-v3-turbo').
        device: CTranslate2 device ('cuda', or 'cpu' for testing without a GPU).
        device_index: CUDA device index.
        language: Language code (e.g. 'en', 'it', 'fr'). None for auto-detect.
        cache_dir: Directory for cached models.
        initial_prompt: Text fed as the previous transcription to the first 30s window only.
        hotwords: Text fed as the previous transcription to every 30s window.
        compute_type: CTranslate2 compute type; 'auto' picks the fastest one supported by the device.

    Returns:
        Dict with keys 'text' (str) and 'segments' (list of dicts with 'text', 'start', 'end').
    """
    _preload_cuda_libs()
    from faster_whisper import WhisperModel

    cache_dir = os.path.expanduser(cache_dir)
    compute_type = _resolve_compute_type(model_id, compute_type)

    print(f"Loading Whisper model '{model_id}' on {device.upper()} ({compute_type})...", file=sys.stderr)
    model_path = _resolve_model_path(model_id, cache_dir)
    model = WhisperModel(model_path, device=device, device_index=device_index, compute_type=compute_type)

    # faster-whisper takes plain language codes, unlike the <|xx|> tokens used by openvino-genai
    gen_kwargs = {"task": "transcribe"}
    if language:
        gen_kwargs["language"] = language.removeprefix("<|").removesuffix("|>")
    # Blank prompts are ignored: whitespace would still be fed to the decoder as "previous text"
    for key, value in (("initial_prompt", initial_prompt), ("hotwords", hotwords)):
        if value and value.strip():
            gen_kwargs[key] = value.strip()

    print("Transcribing...", file=sys.stderr)
    segments_iter, _info = model.transcribe(audio, **gen_kwargs)

    # Segments are generated lazily: the actual decoding happens while iterating
    segments = []
    texts = []
    for segment in segments_iter:
        texts.append(segment.text)
        segments.append(
            {
                "text": segment.text.strip(),
                "start": segment.start,
                "end": segment.end,
            }
        )

    return {
        "text": "".join(texts).strip(),
        "segments": segments,
    }
