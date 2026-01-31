"""YouTube video downloader using yt-dlp."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yt_dlp


@dataclass
class VideoInfo:
    id: str
    title: str
    audio_path: Path
    duration: float


def _parse_rate_limit(rate_limit: str) -> int | None:
    """Parse rate limit string (e.g., '5M', '10M', '1G') to bytes."""
    if not rate_limit:
        return None

    rate_limit = rate_limit.strip().upper()
    multipliers = {
        'K': 1024,
        'M': 1024 * 1024,
        'G': 1024 * 1024 * 1024,
    }

    for suffix, multiplier in multipliers.items():
        if rate_limit.endswith(suffix):
            try:
                value = float(rate_limit[:-1])
                return int(value * multiplier)
            except ValueError:
                return None

    # Try plain number (assume bytes)
    try:
        return int(rate_limit)
    except ValueError:
        return None


def download_channel(
    channel_url: str,
    output_dir: Path,
    max_videos: int | None = None,
    rate_limit: str | None = None,
    cookies_file: str | None = None,
    download_archive: str | None = None,
    progress_callback: Callable[[str, str, str], None] | None = None,
) -> list[VideoInfo]:
    """Download all audio from a YouTube channel in a single yt-dlp call.

    Args:
        channel_url: URL of the YouTube channel
        output_dir: Directory to save audio files
        max_videos: Maximum number of videos to download
        rate_limit: Rate limit (e.g., "5M", "10M")
        cookies_file: Path to cookies.txt file
        download_archive: Path to yt-dlp download archive file
        progress_callback: Optional callback for progress updates (video_id, status, title)

    Returns:
        List of VideoInfo for downloaded files
    """

    output_template = str(output_dir / "%(id)s.%(ext)s")

    postprocessors = [
        {
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }
    ]

    ydl_opts: dict[str, Any] = {
        "format": "bestaudio/best",
        "postprocessors": postprocessors,
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
    }

    # Add rate limiting
    if rate_limit:
        rate_limit_bytes = _parse_rate_limit(rate_limit)
        if rate_limit_bytes:
            ydl_opts["ratelimit"] = rate_limit_bytes

    # Add cookies
    if cookies_file:
        ydl_opts["cookiefile"] = cookies_file

    # Add download archive for resume support
    if download_archive:
        ydl_opts["download_archive"] = download_archive

    # Limit number of videos
    if max_videos:
        ydl_opts["playlistend"] = max_videos

    # Store downloaded video info
    downloaded_videos: list[VideoInfo] = []

    def progress_hook(d: dict[str, Any]) -> None:
        """Progress hook for yt-dlp."""
        if d["status"] == "finished" and progress_callback:
            video_id = d.get("info_dict", {}).get("id", "unknown")
            title = d.get("info_dict", {}).get("title", "Untitled")
            progress_callback(video_id, "downloaded", title)

    ydl_opts["progress_hooks"] = [progress_hook]

    # Convert channel URL to videos URL
    if "/@" in channel_url:
        handle = channel_url.split("/@")[-1].split("/")[0]
        videos_url = f"https://www.youtube.com/@{handle}/videos"
    elif "/channel/" in channel_url:
        channel_id = channel_url.split("/channel/")[-1].split("/")[0]
        videos_url = f"https://www.youtube.com/channel/{channel_id}/videos"
    elif "/c/" in channel_url:
        custom_name = channel_url.split("/c/")[-1].split("/")[0]
        videos_url = f"https://www.youtube.com/c/{custom_name}/videos"
    else:
        videos_url = channel_url.rstrip("/") + "/videos"

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:  # pyrefly: ignore[bad-argument-type]
        info = ydl.extract_info(videos_url, download=True)

        if info is None:
            return []

        entries = info.get("entries") or []

        for entry in entries:
            if entry is None:
                continue

            video_id: str = entry.get("id", "unknown")
            title: str = entry.get("title") or "Untitled"
            duration: float = entry.get("duration", 0) or 0.0
            audio_path = output_dir / f"{video_id}.mp3"

            # Only add if the file actually exists (was downloaded)
            if audio_path.exists():
                downloaded_videos.append(
                    VideoInfo(
                        id=video_id,
                        title=title,
                        audio_path=audio_path,
                        duration=duration,
                    )
                )

    return downloaded_videos
