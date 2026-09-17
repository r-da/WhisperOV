"""Output format helpers: SRT, VTT, and timestamp formatting."""


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as WebVTT timestamp: HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = round((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_srt(segments: list[dict]) -> str:
    """Generate SRT subtitle content from segments.

    Args:
        segments: List of dicts with keys 'text', 'start', 'end' (timestamps in seconds).

    Returns:
        SRT formatted string.
    """
    parts = []
    for i, seg in enumerate(segments, 1):
        text = seg["text"].strip()
        if not text:
            continue
        parts.append(f"{i}\n")
        parts.append(f"{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n")
        parts.append(f"{text}\n")
    return "\n".join(parts)


def generate_vtt(segments: list[dict]) -> str:
    """Generate WebVTT subtitle content from segments.

    Args:
        segments: List of dicts with keys 'text', 'start', 'end' (timestamps in seconds).

    Returns:
        WebVTT formatted string.
    """
    header = "WEBVTT\n\n"
    parts = []
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue
        parts.append(f"{format_timestamp_vtt(seg['start'])} --> {format_timestamp_vtt(seg['end'])}\n{text}")
    return header + "\n".join(parts)
