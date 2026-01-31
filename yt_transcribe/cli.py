"""CLI interface for yt-transcribe."""

import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .asr import ASRTranscriber, get_backend
from .cleaner import TranscriptCleaner
from .downloader import VideoInfo, download_channel

console = Console()


def transcribe_single_video(
    video_info: VideoInfo,
    api_endpoint: str,
    api_key: str,
    model: str,
    asr_model: str | None,
    backend: str,
    cpu_threads: int,
) -> dict:
    """Transcribe a single video (audio already downloaded).

    Args:
        video_info: VideoInfo with audio_path already set
        api_endpoint: LLM API endpoint for transcript cleaning
        api_key: LLM API key
        model: LLM model name for cleaning
        asr_model: ASR model name
        backend: ASR backend
        cpu_threads: Number of CPU threads for ASR

    Returns:
        Dict with transcription result or error
    """
    try:
        # Set CPU threads
        if cpu_threads > 0:
            os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
            os.environ["MKL_NUM_THREADS"] = str(cpu_threads)

        transcriber = ASRTranscriber(model_name=asr_model, backend=backend)
        transcriber.load()

        cleaner = TranscriptCleaner(
            api_endpoint=api_endpoint,
            api_key=api_key,
            model=model,
        )

        raw_transcript = transcriber.transcribe(video_info.audio_path)
        cleaned_transcript = cleaner.clean(raw_transcript)

        return {
            "title": video_info.title,
            "id": video_info.id,
            "transcript": cleaned_transcript,
        }
    except Exception as e:
        return {
            "title": video_info.title,
            "id": video_info.id,
            "error": str(e),
        }


def process_videos_sequential(
    videos: list[VideoInfo],
    transcriber,
    cleaner,
    keep_audio: bool,
    console,
) -> list:
    """Process videos sequentially (single process).

    Args:
        videos: List of VideoInfo (audio already downloaded)
        transcriber: ASRTranscriber instance
        cleaner: TranscriptCleaner instance
        keep_audio: Whether to keep audio files
        console: Rich console for output

    Returns:
        List of transcription results
    """
    results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for i, video_info in enumerate(videos, 1):
            task = progress.add_task(
                f"[{i}/{len(videos)}] {video_info.title[:50]}...",
                total=None,
            )

            try:
                progress.update(task, description=f"[{i}/{len(videos)}] Transcribing: {video_info.title[:40]}...")
                raw_transcript = transcriber.transcribe(video_info.audio_path)

                progress.update(task, description=f"[{i}/{len(videos)}] Cleaning: {video_info.title[:40]}...")
                cleaned_transcript = cleaner.clean(raw_transcript)

                results.append({
                    "index": i,
                    "title": video_info.title,
                    "transcript": cleaned_transcript,
                    "id": video_info.id,
                })

                if not keep_audio and video_info.audio_path.exists():
                    video_info.audio_path.unlink()

            except Exception as e:
                console.print(f"[red]Error processing {video_info.title}: {e}[/]")
                continue

            progress.remove_task(task)

    return results


def process_videos_parallel(
    videos: list[VideoInfo],
    api_endpoint: str,
    api_key: str,
    model: str,
    asr_model: str | None,
    backend: str,
    cpu_threads: int,
    max_workers: int,
    keep_audio: bool,
    console,
) -> list:
    """Process videos in parallel using multiple processes.

    Args:
        videos: List of VideoInfo (audio already downloaded)
        api_endpoint: LLM API endpoint for transcript cleaning
        api_key: LLM API key
        model: LLM model name for cleaning
        asr_model: ASR model name
        backend: ASR backend
        cpu_threads: Number of CPU threads for ASR
        max_workers: Maximum number of parallel workers
        keep_audio: Whether to keep audio files
        console: Rich console for output

    Returns:
        List of transcription results
    """
    results = []
    completed = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        overall_task = progress.add_task(
            f"Processing {len(videos)} videos with {max_workers} workers...",
            total=len(videos),
        )

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = {}
            for video_info in videos:
                future = executor.submit(
                    transcribe_single_video,
                    video_info,
                    api_endpoint,
                    api_key,
                    model,
                    asr_model,
                    backend,
                    cpu_threads,
                )
                futures[future] = video_info

            # Collect results as they complete
            for future in as_completed(futures):
                video_info = futures[future]
                try:
                    result = future.result(timeout=600)  # 10 min timeout per video
                    if result and "error" not in result:
                        results.append(result)
                        # Clean up audio if not keeping
                        if not keep_audio and video_info.audio_path.exists():
                            video_info.audio_path.unlink()
                    else:
                        console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/]")
                except Exception as e:
                    console.print(f"[red]Error processing {video_info.title}: {e}[/]")

                completed += 1
                progress.update(overall_task, completed=completed)

    # Sort by title for consistent ordering
    results.sort(key=lambda x: x.get("title", ""))
    return results


@click.command()
@click.argument("channel_url")
@click.option(
    "--api-endpoint",
    envvar="LLM_API_ENDPOINT",
    required=True,
    help="LLM API endpoint for transcript cleaning (or set LLM_API_ENDPOINT env var)",
)
@click.option(
    "--api-key",
    envvar="LLM_API_KEY",
    required=True,
    help="LLM API key (or set LLM_API_KEY env var)",
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    help="LLM model name for cleaning",
)
@click.option(
    "--max-videos",
    "-n",
    type=int,
    default=None,
    help="Maximum number of videos to process",
)
@click.option(
    "--asr-model",
    default=None,
    help="ASR model name (auto-selected based on backend)",
)
@click.option(
    "--backend",
    type=click.Choice(["auto", "faster-whisper", "mlx", "mlx-plus", "cuda", "rocm", "mps", "cpu"]),
    default="auto",
    help="ASR backend (default: auto-detect, prefers faster-whisper)",
)
@click.option(
    "--workers",
    "-j",
    type=int,
    default=1,
    help="Number of parallel workers for processing videos (default: 1)",
)
@click.option(
    "--cpu-threads",
    type=int,
    default=0,
    help="Number of CPU threads for ASR (0 = auto, default: 0)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default=None,
    help="Output file path (default: stdout)",
)
@click.option(
    "--keep-audio",
    is_flag=True,
    help="Keep downloaded audio files",
)
@click.option(
    "--audio-dir",
    type=click.Path(),
    default=None,
    help="Directory to store audio files (implies --keep-audio)",
)
@click.option(
    "--rate-limit",
    type=str,
    default=None,
    help="Download rate limit (e.g., '5M', '10M')",
)
@click.option(
    "--cookies-file",
    type=click.Path(),
    default=None,
    help="Path to cookies.txt file for authentication",
)
@click.option(
    "--download-archive",
    type=click.Path(),
    default=None,
    help="Path to yt-dlp download archive file (for resume support)",
)
def main(
    channel_url: str,
    api_endpoint: str,
    api_key: str,
    model: str,
    max_videos: int | None,
    asr_model: str | None,
    backend: str,
    workers: int,
    cpu_threads: int,
    output: str | None,
    keep_audio: bool,
    audio_dir: str | None,
    rate_limit: str | None,
    cookies_file: str | None,
    download_archive: str | None,
):
    """Transcribe all videos from a YouTube channel.

    CHANNEL_URL: YouTube channel URL (e.g., https://www.youtube.com/@channelname)

    This tool uses yt-dlp to download all audio from a channel, then transcribes
    it using local ASR and cleans the transcript using an LLM.

    Examples:
        # Basic usage
        yt-transcribe "https://www.youtube.com/@channel" --api-key "xxx"

        # Download 50 videos with rate limiting and archive support
        yt-transcribe "https://www.youtube.com/@channel" --api-key "xxx" \\
            --max-videos 50 --rate-limit 5M --download-archive archive.txt

        # Parallel processing with 2 workers
        yt-transcribe "https://www.youtube.com/@channel" --api-key "xxx" -j 2
    """
    if audio_dir:
        keep_audio = True
        work_dir = Path(audio_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = Path(tempfile.mkdtemp())

    # Phase 1: Bulk download all audio from channel
    console.print("[bold blue]Downloading audio from channel...[/]")

    download_archive_path = (
        Path(download_archive) if download_archive else work_dir / ".download_archive"
    )

    def download_progress(video_id: str, status: str, title: str) -> None:
        """Progress callback for downloads."""
        console.print(f"[dim]Downloaded: {title[:50]}...[/]")

    video_infos = download_channel(
        channel_url=channel_url,
        output_dir=work_dir,
        max_videos=max_videos,
        rate_limit=rate_limit,
        cookies_file=cookies_file,
        download_archive=str(download_archive_path),
        progress_callback=download_progress,
    )

    if not video_infos:
        console.print("[yellow]No new videos to download (check --download-archive)[/]")
        return

    console.print(f"[green]Downloaded {len(video_infos)} videos[/]")

    # Phase 2: Transcribe in parallel
    detected_backend = backend if backend != "auto" else get_backend()

    # Set CPU threads for ASR
    if cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        console.print(f"[dim]Using {cpu_threads} CPU threads for ASR[/]")

    if workers > 1:
        console.print(f"[bold blue]Using {workers} parallel workers for transcription[/]")
        results = process_videos_parallel(
            videos=video_infos,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model=model,
            asr_model=asr_model,
            backend=detected_backend,
            cpu_threads=cpu_threads,
            max_workers=workers,
            keep_audio=keep_audio,
            console=console,
        )
    else:
        console.print(f"[bold blue]Loading ASR model (backend: {detected_backend})...[/]")
        transcriber = ASRTranscriber(
            model_name=asr_model,
            backend=detected_backend,
        )
        transcriber.load()
        console.print(f"[green]ASR ready: {transcriber.backend_info}[/]")

        cleaner = TranscriptCleaner(
            api_endpoint=api_endpoint,
            api_key=api_key,
            model=model,
        )

        results = process_videos_sequential(
            videos=video_infos,
            transcriber=transcriber,
            cleaner=cleaner,
            keep_audio=keep_audio,
            console=console,
        )

    output_text = format_output(results)

    if output:
        Path(output).write_text(output_text, encoding="utf-8")
        console.print(f"[green]Output saved to {output}[/]")
    else:
        console.print("\n" + output_text)

    if not keep_audio and work_dir.exists():
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)


def format_output(results: list[dict]) -> str:
    """Format results into the requested output format."""
    lines = []
    for r in results:
        lines.append(f"===video {r['index']}: {r['title']}===")
        lines.append(r["transcript"])
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
