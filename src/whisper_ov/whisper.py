"""OpenVINO WhisperPipeline wrapper for speech-to-text."""

import os
import sys
from pathlib import Path

import numpy as np

# Map HuggingFace Whisper model IDs to OpenVINO pre-converted repos
# Format: (hf_model_id, openvino_repo_id, display_name, approximate_size_gb)
KNOWN_MODELS = [
    ("openai/whisper-tiny", "OpenVINO/whisper-tiny-fp16-ov", "tiny", 0.07),
    ("openai/whisper-base", "OpenVINO/whisper-base-fp16-ov", "base", 0.14),
    ("openai/whisper-small", "OpenVINO/whisper-small-fp16-ov", "small", 0.50),
    ("openai/whisper-medium", "OpenVINO/whisper-medium-fp16-ov", "medium", 1.42),
    ("openai/whisper-large-v3", "OpenVINO/whisper-large-v3-fp16-ov", "large-v3", 2.91),
    ("openai/whisper-large-v3-int4", "OpenVINO/whisper-large-v3-int4-ov", "large-v3 (INT4)", 1.0),
    ("openai/whisper-large-v3-int8", "OpenVINO/whisper-large-v3-int8-ov", "large-v3 (INT8)", 1.5),
    ("openai/whisper-large-v3-turbo", "OpenVINO/whisper-large-v3-turbo-fp16-ov", "large-v3-turbo", 1.52),
    ("openai/whisper-large-v3-turbo-int4", "OpenVINO/whisper-large-v3-turbo-int4-ov", "large-v3-turbo (INT4)", 0.75),
    ("openai/whisper-large-v3-turbo-int8", "OpenVINO/whisper-large-v3-turbo-int8-ov", "large-v3-turbo (INT8)", 1.0),
]

HF_TO_OV = {hf: ov for hf, ov, _, _ in KNOWN_MODELS}


def _get_ov_repo_id(hf_model_id: str) -> str:
    """Map a HuggingFace Whisper model ID to its OpenVINO pre-converted repo ID."""
    if hf_model_id in HF_TO_OV:
        return HF_TO_OV[hf_model_id]
    # If not in our known list, try to construct the OpenVINO repo name
    name = hf_model_id.split("/")[-1]
    return f"OpenVINO/{name}-fp16-ov"


def _download_model(hf_model_id: str, ov_repo_id: str, cache_dir: str) -> Path:
    """Download the OpenVINO pre-converted model from HuggingFace Hub.

    Args:
        hf_model_id: Original HuggingFace model ID (for display).
        ov_repo_id: OpenVINO pre-converted repo ID on HuggingFace.
        cache_dir: Directory to cache the model.

    Returns:
        Path to the downloaded model directory.
    """
    import huggingface_hub as hf_hub

    name = hf_model_id.split("/")[-1]
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    model_dir = Path(cache_dir) / "models" / name

    if model_dir.exists():
        has_model = any(f.suffix in (".xml", ".onnx", ".bin") for f in model_dir.rglob("*"))
        if has_model:
            print(f"Using cached model: {model_dir}", file=sys.stderr)
            return model_dir

    model_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {ov_repo_id} to {model_dir}...", file=sys.stderr)
    hf_hub.snapshot_download(ov_repo_id, local_dir=str(model_dir))
    print(f"Model downloaded to {model_dir}", file=sys.stderr)

    return model_dir


def _resolve_model_path(hf_model_id: str, cache_dir: str) -> str:
    """Check if model is cached locally, otherwise download from OpenVINO HF collection.

    Returns the path to the local model directory.
    """
    ov_repo_id = _get_ov_repo_id(hf_model_id)
    model_dir = _download_model(hf_model_id, ov_repo_id, cache_dir)
    return str(model_dir)


def transcribe(
    audio: np.ndarray,
    model_id: str = "openai/whisper-large-v3-turbo",
    device: str = "AUTO",
    language: str | None = None,
    cache_dir: str = "~/.cache/whisper-ov",
) -> dict:
    """Transcribe audio using OpenVINO WhisperPipeline.

    Args:
        audio: numpy array of 16kHz mono audio (float32).
        model_id: HuggingFace Whisper model ID (e.g. 'openai/whisper-large-v3-turbo').
        device: OpenVINO device (CPU, GPU, NPU, AUTO).
        language: Language code (e.g. 'en', 'it', 'fr'). None for auto-detect.
        cache_dir: Directory for cached models.

    Returns:
        Dict with keys 'text' (str) and 'segments' (list of dicts with 'text', 'start', 'end').
    """
    from openvino_genai import WhisperPipeline

    cache_dir = os.path.expanduser(cache_dir)

    # Resolve model path — downloads from OpenVINO HF collection if not cached
    print(f"Loading Whisper model '{model_id}' on {device}...", file=sys.stderr)
    model_path = _resolve_model_path(model_id, cache_dir)

    # Create pipeline from local model directory
    pipeline = WhisperPipeline(model_path, device)

    # Build generation config as kwargs
    # Note: passing WhisperGenerationConfig object directly causes "vector::reserve" error
    # in openvino-genai 2026.x, so we use kwargs instead
    gen_kwargs = {"task": "transcribe", "return_timestamps": True}
    if language:
        if not (language.startswith("<|") and language.endswith("|>")):
            language = f"<|{language}|>"
        gen_kwargs["language"] = language

    # Run inference — audio must be a list of floats
    print("Transcribing...", file=sys.stderr)
    audio_list = audio.tolist()
    result = pipeline.generate(audio_list, **gen_kwargs)

    texts = result.texts
    if not texts:
        raise RuntimeError("Transcription returned no text")

    # Build segments list from timestamps
    # result.chunks is a flat list of WhisperDecodedResultChunk objects
    # Each chunk has: start_ts, end_ts, text
    segments = []
    if hasattr(result, "chunks") and result.chunks:
        for chunk in result.chunks:
            segments.append(
                {
                    "text": chunk.text.strip(),
                    "start": chunk.start_ts,
                    "end": chunk.end_ts,
                }
            )

    return {
        "text": texts[0],
        "segments": segments,
    }


def list_models() -> list[dict]:
    """Return list of known Whisper models with metadata."""
    return [
        {
            "model_id": hf_id,
            "name": name,
            "size_gb": size_gb,
        }
        for hf_id, _, name, size_gb in KNOWN_MODELS
    ]
