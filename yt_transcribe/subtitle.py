"""Subtitle export utilities for SRT and WebVTT formats."""

from pathlib import Path
from typing import Any


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format.

    Args:
        seconds: Time in seconds

    Returns:
        Timestamp string in format: HH:MM:SS,mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds to WebVTT timestamp format.

    Args:
        seconds: Time in seconds

    Returns:
        Timestamp string in format: HH:MM:SS.mmm
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def export_srt(words: list[dict], output_path: Path) -> None:
    """Export word-level timestamps to SRT format.

    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        output_path: Path to output SRT file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    counter = 1

    for word_entry in words:
        word = word_entry.get("word", "").strip()
        start = word_entry.get("start", 0)
        end = word_entry.get("end", 0)

        if not word:
            continue

        lines.append(str(counter))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(word)
        lines.append("")
        counter += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_vtt(words: list[dict], output_path: Path) -> None:
    """Export word-level timestamps to WebVTT format.

    Args:
        words: List of word dicts with 'word', 'start', 'end' keys
        output_path: Path to output VTT file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["WEBVTT", ""]

    for word_entry in words:
        word = word_entry.get("word", "").strip()
        start = word_entry.get("start", 0)
        end = word_entry.get("end", 0)

        if not word:
            continue

        lines.append(f"{format_timestamp_vtt(start)} --> {format_timestamp_vtt(end)}")
        lines.append(word)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_srt_from_segments(segments: list[dict], output_path: Path) -> None:
    """Export segment-level timestamps to SRT format.

    Args:
        segments: List of segment dicts with 'text', 'start', 'end' keys
        output_path: Path to output SRT file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    counter = 1

    for segment in segments:
        text = segment.get("text", "").strip()
        start = segment.get("start", 0)
        end = segment.get("end", 0)

        if not text:
            continue

        lines.append(str(counter))
        lines.append(f"{format_timestamp(start)} --> {format_timestamp(end)}")
        lines.append(text)
        lines.append("")
        counter += 1

    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_vtt_from_segments(segments: list[dict], output_path: Path) -> None:
    """Export segment-level timestamps to WebVTT format.

    Args:
        segments: List of segment dicts with 'text', 'start', 'end' keys
        output_path: Path to output VTT file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["WEBVTT", ""]

    for segment in segments:
        text = segment.get("text", "").strip()
        start = segment.get("start", 0)
        end = segment.get("end", 0)

        if not text:
            continue

        lines.append(f"{format_timestamp_vtt(start)} --> {format_timestamp_vtt(end)}")
        lines.append(text)
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
