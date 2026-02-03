"""CLI interface for yt-transcribe."""

import os
import subprocess
import tempfile
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .asr import ASRTranscriber, TranscriptionResult, get_backend
from .cleaner import TranscriptCleaner
from .downloader import VideoInfo, download_channel
from .subtitle import export_srt, export_vtt

console = Console()


def convert_to_wav(audio_path: Path) -> Path:
    """Convert audio file to WAV format using FFmpeg.

    Args:
        audio_path: Path to input audio file

    Returns:
        Path to output WAV file
    """
    wav_path = audio_path.with_suffix(".wav")

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(audio_path),
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-y",  # Overwrite output file
                str(wav_path),
            ],
            check=True,
            capture_output=True,
        )
        return wav_path
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}") from e
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg.")


def stream_transcription(transcriber: ASRTranscriber, audio_path: Path) -> Iterator[str]:
    """Stream transcription output in real-time (MLX only).

    Args:
        transcriber: Loaded ASRTranscriber instance
        audio_path: Path to audio file

    Yields:
        Chunks of transcribed text as they become available
    """
    # For MLX backend with streaming support
    result = transcriber.transcribe(audio_path)

    # If the result has segments, yield them progressively
    if result.segments:
        for segment in result.segments:
            yield segment.get("text", "")
    else:
        # Fallback to yielding full text
        yield result.text


def transcribe_single_video(
    video_info: VideoInfo,
    api_endpoint: str,
    api_key: str,
    model: str,
    backend: str,
    asr_model: str | None,
    language: str | None,
    with_timestamps: bool,
) -> dict:
    """Transcribe a single video (audio already downloaded).

    Args:
        video_info: VideoInfo with audio_path already set
        api_endpoint: LLM API endpoint for transcript cleaning
        api_key: LLM API key
        model: LLM model name for cleaning
        backend: ASR backend
        asr_model: ASR model name
        language: Language code
        with_timestamps: Whether to enable word-level timestamps

    Returns:
        Dict with transcription result or error
    """
    try:
        # Convert to WAV for MLX backend
        audio_path = video_info.audio_path
        wav_path: Path | None = None

        if backend == "mlx":
            wav_path = convert_to_wav(audio_path)
            audio_path = wav_path

        transcriber = ASRTranscriber(
            backend=backend,
            model_name=asr_model,
            language=language,
        )
        transcriber.load()

        # Use aligner backend for timestamps if requested
        if with_timestamps and backend == "mlx":
            from .asr import MLXAlignerBackend
            aligner = MLXAlignerBackend()
            aligner.load()
            result = aligner.transcribe(audio_path, language=language)
        else:
            result = transcriber.transcribe(audio_path)

        cleaner = TranscriptCleaner(
            api_endpoint=api_endpoint,
            api_key=api_key,
            model=model,
        )
        cleaned_transcript = cleaner.clean(result.text)

        return {
            "title": video_info.title,
            "id": video_info.id,
            "transcript": cleaned_transcript,
            "words": result.words if with_timestamps else None,
            "language": result.language,
        }
    except Exception as e:
        return {
            "title": video_info.title,
            "id": video_info.id,
            "error": str(e),
        }
    finally:
        # Clean up temporary WAV file
        if wav_path and wav_path.exists():
            wav_path.unlink()


def process_videos_sequential(
    videos: list[VideoInfo],
    transcriber,
    cleaner,
    keep_audio: bool,
    console,
    language: str | None = None,
    with_timestamps: bool = False,
    subtitle_format: str = "none",
    stream: bool = False,
) -> list:
    """Process videos sequentially (single process).

    Args:
        videos: List of VideoInfo (audio already downloaded)
        transcriber: ASRTranscriber instance
        cleaner: TranscriptCleaner instance
        keep_audio: Whether to keep audio files
        console: Rich console for output
        language: Language code
        with_timestamps: Whether to generate word-level timestamps
        subtitle_format: Subtitle export format (none, srt, vtt, both)
        stream: Whether to enable streaming output

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
                # Convert to WAV for MLX backend
                audio_path = video_info.audio_path
                wav_path: Path | None = None

                if transcriber.backend_name == "mlx":
                    progress.update(task, description=f"[{i}/{len(videos)}] Converting: {video_info.title[:40]}...")
                    wav_path = convert_to_wav(audio_path)
                    audio_path = wav_path

                if with_timestamps and transcriber.backend_name == "mlx":
                    progress.update(task, description=f"[{i}/{len(videos)}] Transcribing with timestamps: {video_info.title[:40]}...")
                    from .asr import MLXAlignerBackend
                    aligner = MLXAlignerBackend()
                    aligner.load()
                    result = aligner.transcribe(audio_path, language=language)
                else:
                    if stream:
                        progress.update(task, description=f"[{i}/{len(videos)}] Streaming transcription: {video_info.title[:40]}...")
                        console.print(f"[dim]Transcribing: {video_info.title}[/]")
                        text_chunks = []
                        for chunk in stream_transcription(transcriber, audio_path):
                            text_chunks.append(chunk)
                            console.print(chunk, end="")
                        console.print()
                        result = TranscriptionResult(text="".join(text_chunks))
                    else:
                        progress.update(task, description=f"[{i}/{len(videos)}] Transcribing: {video_info.title[:40]}...")
                        result = transcriber.transcribe(audio_path)

                progress.update(task, description=f"[{i}/{len(videos)}] Cleaning: {video_info.title[:40]}...")
                cleaned_transcript = cleaner.clean(result.text)

                video_result = {
                    "index": i,
                    "title": video_info.title,
                    "transcript": cleaned_transcript,
                    "id": video_info.id,
                    "words": result.words if with_timestamps else None,
                    "language": result.language,
                }

                # Export subtitles if requested
                if with_timestamps and subtitle_format != "none" and result.words:
                    base_name = video_info.id
                    if subtitle_format in ("srt", "both"):
                        srt_path = Path.cwd() / f"{base_name}.srt"
                        export_srt(result.words, srt_path)
                        console.print(f"[dim]Exported: {srt_path}[/]")
                    if subtitle_format in ("vtt", "both"):
                        vtt_path = Path.cwd() / f"{base_name}.vtt"
                        export_vtt(result.words, vtt_path)
                        console.print(f"[dim]Exported: {vtt_path}[/]")

                results.append(video_result)

                # Clean up
                if not keep_audio and video_info.audio_path.exists():
                    video_info.audio_path.unlink()
                if wav_path and wav_path.exists():
                    wav_path.unlink()

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
    backend: str,
    asr_model: str | None,
    language: str | None,
    with_timestamps: bool,
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
        backend: ASR backend
        asr_model: ASR model name
        language: Language code
        with_timestamps: Whether to enable word-level timestamps
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
            futures = {}
            for video_info in videos:
                future = executor.submit(
                    transcribe_single_video,
                    video_info,
                    api_endpoint,
                    api_key,
                    model,
                    backend,
                    asr_model,
                    language,
                    with_timestamps,
                )
                futures[future] = video_info

            for future in as_completed(futures):
                video_info = futures[future]
                try:
                    result = future.result(timeout=600)
                    if result and "error" not in result:
                        results.append(result)
                        if not keep_audio and video_info.audio_path.exists():
                            video_info.audio_path.unlink()
                    else:
                        console.print(f"[red]Error: {result.get('error', 'Unknown error')}[/]")
                except Exception as e:
                    console.print(f"[red]Error processing {video_info.title}: {e}[/]")

                completed += 1
                progress.update(overall_task, completed=completed)

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
    "--backend",
    type=click.Choice(["auto", "mlx", "faster-whisper"]),
    default="auto",
    help="ASR backend (default: auto-detects MLX on Apple Silicon)",
)
@click.option(
    "--language",
    "-l",
    type=str,
    default="auto",
    help="Language for transcription (auto, chinese, english, japanese, korean, french, german, spanish, russian, portuguese, italian, dutch)",
)
@click.option(
    "--asr-model",
    default=None,
    help="ASR model size: for MLX: 0.6b, 1.7b; for faster-whisper: tiny, base, small, medium, large-v3",
)
@click.option(
    "--with-timestamps",
    is_flag=True,
    help="Enable word-level timestamps (MLX only, useful for subtitle generation)",
)
@click.option(
    "--subtitle-format",
    type=click.Choice(["none", "srt", "vtt", "both"]),
    default="none",
    help="Subtitle export format (requires --with-timestamps)",
)
@click.option(
    "--stream/--no-stream",
    default=True,
    help="Enable streaming transcription output (default: enabled for MLX)",
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
    backend: str,
    language: str,
    asr_model: str | None,
    with_timestamps: bool,
    subtitle_format: str,
    stream: bool,
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
    it using local ASR (MLX Qwen3-ASR on Apple Silicon, faster-whisper fallback) and
    cleans the transcript using an LLM.

    Examples:
        # Basic usage (auto-detects MLX on Apple Silicon)
        yt-transcribe "https://www.youtube.com/@channel" --api-key "key"

        # With streaming output
        yt-transcribe "https://www.youtube.com/@channel" --api-key "key" --stream

        # With timestamps and SRT export
        yt-transcribe "https://www.youtube.com/@channel" --api-key "key" \\
            --with-timestamps --subtitle-format srt

        # Explicit language selection
        yt-transcribe "https://www.youtube.com/@channel" --api-key "key" \\
            --language english

        # Use larger model
        yt-transcribe "https://www.youtube.com/@channel" --api-key "key" \\
            --asr-model 1.7b

        # Force faster-whisper on Apple Silicon
        yt-transcribe "https://www.youtube.com/@channel" --api-key "key" \\
            --backend faster-whisper
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

    # Phase 2: Transcribe
    detected_backend = backend if backend != "auto" else get_backend()
    console.print(f"[dim]Using backend: {detected_backend}[/]")

    # Set CPU threads for ASR
    if cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ["MKL_NUM_THREADS"] = str(cpu_threads)
        console.print(f"[dim]Using {cpu_threads} CPU threads for ASR[/]")

    if workers > 1:
        # Parallel processing doesn't support streaming or subtitles
        if stream:
            console.print("[yellow]Streaming disabled in parallel mode[/]")
        if subtitle_format != "none":
            console.print("[yellow]Subtitle export disabled in parallel mode[/]")

        console.print(f"[bold blue]Using {workers} parallel workers for transcription[/]")
        results = process_videos_parallel(
            videos=video_infos,
            api_endpoint=api_endpoint,
            api_key=api_key,
            model=model,
            backend=detected_backend,
            asr_model=asr_model,
            language=language if language != "auto" else None,
            with_timestamps=with_timestamps,
            max_workers=workers,
            keep_audio=keep_audio,
            console=console,
        )
    else:
        console.print(f"[bold blue]Loading ASR model (backend: {detected_backend})...[/]")
        transcriber = ASRTranscriber(
            backend=detected_backend,
            model_name=asr_model,
            language=language if language != "auto" else None,
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
            language=language if language != "auto" else None,
            with_timestamps=with_timestamps,
            subtitle_format=subtitle_format,
            stream=stream and detected_backend == "mlx",
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
