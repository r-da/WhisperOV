"""Output format helpers: SRT, VTT, and timestamp formatting."""


def _seconds_to_ms(seconds: float) -> int:
    """Convert seconds to integer milliseconds, rounding and carrying over."""
    ms = round(seconds * 1000)
    return max(ms, 0)


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    ms = _seconds_to_ms(seconds)
    hours, remainder = divmod(ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as WebVTT timestamp: HH:MM:SS.mmm"""
    ms = _seconds_to_ms(seconds)
    hours, remainder = divmod(ms, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def _normalize_text(text: str) -> str:
    """Normalize text: CRLF/CR to LF, strip each line, filter empty lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line)


def _escape_vtt_payload(text: str) -> str:
    """Escape & and < in VTT payload to prevent malformed markup."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    return text


def _sort_segments(segments: list[dict]) -> list[dict]:
    """Sort segments by start time, then end time, preserving original order for ties."""
    return sorted(segments, key=lambda s: (s.get("start", 0), s.get("end", 0)))


def _validate_segment(seg: dict) -> bool:
    """Return True if segment has valid start < end and non-empty text."""
    start = seg.get("start", 0)
    end = seg.get("end", 0)
    text = seg.get("text", "").strip()
    if not text:
        return False
    return not end <= start


def generate_srt(segments: list[dict]) -> str:
    """Generate SRT subtitle content from segments.

    Args:
        segments: List of dicts with keys 'text', 'start', 'end' (timestamps in seconds).

    Returns:
        SRT formatted string.
    """
    segments = _sort_segments(segments)
    blocks = []
    index = 0
    for seg in segments:
        if not _validate_segment(seg):
            continue
        index += 1
        text = _normalize_text(seg["text"])
        blocks.append(f"{index}\n{format_timestamp_srt(seg['start'])} --> {format_timestamp_srt(seg['end'])}\n{text}")
    if not blocks:
        return ""
    return "\n\n".join(blocks) + "\n"


def generate_vtt(segments: list[dict]) -> str:
    """Generate WebVTT subtitle content from segments.

    Args:
        segments: List of dicts with keys 'text', 'start', 'end' (timestamps in seconds).

    Returns:
        WebVTT formatted string.
    """
    header = "WEBVTT\n\n"
    segments = _sort_segments(segments)
    parts = []
    for seg in segments:
        if not _validate_segment(seg):
            continue
        text = _normalize_text(seg["text"])
        text = _escape_vtt_payload(text)
        parts.append(f"{format_timestamp_vtt(seg['start'])} --> {format_timestamp_vtt(seg['end'])}\n{text}")
    if not parts:
        return header
    return header + "\n\n".join(parts) + "\n"