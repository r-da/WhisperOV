"""Audio decoding using PyAV. Supports any audio format, outputs 16kHz mono numpy array."""

import av
import av.audio.resampler
import numpy as np


def load_audio(path: str) -> np.ndarray:
    """Decode any audio file to 16kHz mono numpy float32 array.

    Args:
        path: Path to the audio file (any format supported by FFmpeg).

    Returns:
        numpy.ndarray of shape (n_samples,) with dtype float32.

    Raises:
        FileNotFoundError: If the file does not exist.
        RuntimeError: If no audio stream is found or decoding fails.
    """
    container = av.open(path, mode="r")

    audio_stream = None
    for stream in container.streams:
        if stream.type == "audio":
            audio_stream = stream
            break

    if audio_stream is None:
        raise RuntimeError(f"No audio stream found in '{path}'")

    # Create resampler: convert to 16kHz mono float32
    resampler = av.audio.resampler.AudioResampler(
        format="flt",
        layout="mono",
        rate=16000,
    )

    audio_frames = []
    for frame in container.decode(audio_stream):
        resampled_frames = resampler.resample(frame)
        for resampled_frame in resampled_frames:
            buffer = resampled_frame.to_ndarray().squeeze(axis=0)
            audio_frames.append(buffer)

    container.close()

    if not audio_frames:
        raise RuntimeError(f"No audio data could be decoded from '{path}'")

    audio = np.concatenate(audio_frames, axis=0)
    return audio.astype(np.float32)
