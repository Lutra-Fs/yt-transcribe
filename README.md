# yt-transcribe

YouTube channel batch transcription tool using local ASR models for speech recognition, with LLM API for cleaning transcriptions.

**Dual backend architecture:**
- **MLX Qwen3-ASR** (primary on Apple Silicon) - Fast, optimized local transcription
- **faster-whisper** (fallback) - Reliable cross-platform transcription

> Transcription cleaning prompt inspired by [RookieRicardoR](https://x.com/RookieRicardoR/status/2011959082509615288)

## Installation

**Note:** MLX support requires Python 3.11 or 3.12. The project uses Python 3.11 by default.

```bash
# Install dependencies (uses Python 3.11 by default)
uv sync

# If you have dependency conflicts with MLX, try:
uv sync --prerelease=allow
```

For Apple Silicon with MLX support:
```bash
# Install with MLX extras
uv sync --all-extras
# or
uv sync --extra mlx

# If that fails, use prerelease mode:
uv sync --prerelease=allow
```

## Usage

### Basic Usage

```bash
# Basic usage (auto-detects MLX on Apple Silicon)
uv run yt-transcribe "https://www.youtube.com/@channelname" \
    --api-endpoint "https://api.openai.com/v1" \
    --api-key "your-api-key"

# Using environment variables
export LLM_API_ENDPOINT="https://api.openai.com/v1"
export LLM_API_KEY="your-api-key"
uv run yt-transcribe "https://www.youtube.com/@channelname"
```

### Streaming Output

```bash
# Enable progress updates during transcription (default)
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" --stream

# Disable progress updates
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" --no-stream
```

**Note:** Streaming currently shows progress updates, not real-time token-by-token output.

### Word-Level Timestamps & Subtitles

```bash
# Generate word-level timestamps for subtitle export (MLX only)
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" \
    --with-timestamps --subtitle-format srt

# Export both SRT and VTT formats
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" \
    --with-timestamps --subtitle-format both
```

### Language Selection

```bash
# Auto-detect language (default)
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" --language auto

# Specify language explicitly
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" --language english
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" --language chinese
```

### Model Selection

```bash
# Use larger MLX model (slower but more accurate)
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" --asr-model 1.7b

# Force faster-whisper backend
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "key" \
    --backend faster-whisper --asr-model base
```

### Other Options

```bash
# Limit number of videos
uv run yt-transcribe "https://www.youtube.com/@channelname" -n 5

# Save to file
uv run yt-transcribe "https://www.youtube.com/@channelname" -o transcripts.txt

# Use other LLM model for cleaning
uv run yt-transcribe "https://www.youtube.com/@channelname" --model gpt-4o

# Parallel processing with 2 workers
uv run yt-transcribe "https://www.youtube.com/@channelname" -j 2

# Download with rate limiting and archive support
uv run yt-transcribe "https://www.youtube.com/@channel" --api-key "xxx" \
    --max-videos 50 --rate-limit 5M --download-archive archive.txt
```

## Options

| Option | Description |
|--------|-------------|
| `--api-endpoint` | LLM API endpoint (required, or set `LLM_API_ENDPOINT` env var) |
| `--api-key` | LLM API key (required, or set `LLM_API_KEY` env var) |
| `--model` | LLM model name (default: `gpt-4o-mini`) |
| `-n, --max-videos` | Maximum number of videos to process |
| `--backend` | ASR backend: `auto`, `mlx`, `faster-whisper` (default: `auto`) |
| `-l, --language` | Language: `auto`, `chinese`, `english`, `japanese`, `korean`, `french`, `german`, `spanish`, `russian`, `portuguese`, `italian`, `dutch` |
| `--asr-model` | Model size: MLX (`0.6b`, `1.7b`), faster-whisper (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `--with-timestamps` | Enable word-level timestamps (MLX only, for subtitles) |
| `--subtitle-format` | Subtitle export: `none`, `srt`, `vtt`, `both` (requires `--with-timestamps`) |
| `--stream/--no-stream` | Enable/disable streaming output (default: enabled for MLX) |
| `-j, --workers` | Number of parallel workers (default: 1) |
| `-o, --output` | Output file path |
| `--keep-audio` | Keep downloaded audio files |
| `--audio-dir` | Audio files storage directory |
| `--rate-limit` | Download rate limit (e.g., `5M`, `10M`) |
| `--download-archive` | Path to yt-dlp archive file (for resume support) |

## Language Support

| Language | Code | MLX Qwen3-ASR | faster-whisper |
|----------|------|---------------|----------------|
| Auto-detect | `auto` | Yes | Yes |
| Chinese | `chinese` / `zh` | Yes | Yes |
| English | `english` / `en` | Yes | Yes |
| Japanese | `japanese` / `ja` | Yes | Yes |
| Korean | `korean` / `ko` | Yes | Yes |
| Cantonese | `cantonese` / `yue` | Yes | No |
| French | `french` / `fr` | Yes | Yes |
| German | `german` / `de` | Yes | Yes |
| Spanish | `spanish` / `es` | Yes | Yes |
| Russian | `russian` / `ru` | Yes | Yes |
| Portuguese | `portuguese` / `pt` | Yes | Yes |
| Italian | `italian` / `it` | Yes | Yes |
| Dutch | `dutch` / `nl` | Yes | Yes |

## Backend Architecture

### MLX Qwen3-ASR (Primary on Apple Silicon)
- **Models:** Qwen3-ASR-0.6B-8bit (default), Qwen3-ASR-1.7B-8bit
- **Features:** Streaming output, word-level alignment, multi-language support
- **Performance:** Optimized for Apple Silicon (M1/M2/M3/M4)

### faster-whisper (Fallback)
- **Models:** tiny, base, small, medium, large-v3
- **Features:** Cross-platform, CPU/GPU support
- **Use case:** Non-Apple Silicon systems, or when MLX is unavailable

### Auto-Detection
The tool automatically selects MLX on Apple Silicon when available, falling back to faster-whisper otherwise. Use `--backend` to override.

## Output Format

```
===video 1: Video Title 1===
Transcription content 1

===video 2: Video Title 2===
Transcription content 2
```

### Subtitle Files

When using `--with-timestamps --subtitle-format srt`, subtitle files are generated:

```
video_id.srt  # SRT format
video_id.vtt  # WebVTT format (if --subtitle-format vtt or both)
```

## Requirements

- Python 3.11+
- FFmpeg (for audio extraction)
- Apple Silicon M1/M2/M3/M4 (for MLX backend, optional)

## Troubleshooting

### FFmpeg not found
If you get an error about FFmpeg not being found, install it:
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### MLX backend not available on Apple Silicon
If MLX is not being auto-detected on Apple Silicon, ensure MLX is installed:
```bash
uv sync --extra mlx
```

### Subtitle export not working
Subtitle export requires `--with-timestamps` which is only supported with the MLX backend. Make sure:
1. You're using Apple Silicon with MLX installed
2. You're using `--backend mlx` or let it auto-detect
3. You passed `--with-timestamps --subtitle-format srt`

### Large model download on first run
On first use, the ASR models will be downloaded automatically. Model sizes:
- MLX 0.6B: ~600MB
- MLX 1.7B: ~1.7GB
- faster-whisper tiny: ~40MB
- faster-whisper base: ~140MB

## FAQ

**Q: Can I use this on non-Apple Silicon?**
A: Yes, the tool will automatically use faster-whisper backend on non-Apple Silicon systems.

**Q: How accurate is the transcription?**
A: MLX Qwen3-ASR is highly accurate for Chinese and English content. For other languages, it still performs well. The LLM cleaning step significantly improves readability.

**Q: Can I use a different LLM for cleaning?**
A: Yes, use `--model` to specify any OpenAI-compatible model (e.g., `gpt-4o`, `claude-3-haiku`, etc.).

**Q: Why is streaming output only available with MLX?**
A: The MLX backend supports streaming natively. faster-whisper processes the entire audio at once.

**Q: Can I download videos without transcribing?**
A: Use `--keep-audio --audio-dir ./audio` to keep the downloaded audio files.

## License

MIT
