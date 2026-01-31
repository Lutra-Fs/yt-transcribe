# yt-transcribe

YouTube channel batch transcription tool using local faster-whisper model for speech recognition, with LLM API for cleaning transcriptions.

> Transcription cleaning prompt inspired by [RookieRicardoR](https://x.com/RookieRicardoR/status/2011959082509615288)

## Installation

```bash
uv sync
```

## Usage

```bash
# Basic usage
uv run yt-transcribe "https://www.youtube.com/@channelname" \
    --api-endpoint "https://api.openai.com/v1" \
    --api-key "your-api-key"

# Using environment variables
export LLM_API_ENDPOINT="https://api.openai.com/v1"
export LLM_API_KEY="your-api-key"
uv run yt-transcribe "https://www.youtube.com/@channelname"

# Limit number of videos
uv run yt-transcribe "https://www.youtube.com/@channelname" -n 5

# Save to file
uv run yt-transcribe "https://www.youtube.com/@channelname" -o transcripts.txt

# Specify whisper model size
uv run yt-transcribe "https://www.youtube.com/@channelname" --asr-model base

# Use other LLM model
uv run yt-transcribe "https://www.youtube.com/@channelname" --model gpt-4o
```

## Options

| Option | Description |
|--------|-------------|
| `--api-endpoint` | LLM API endpoint (required, or set `LLM_API_ENDPOINT` env var) |
| `--api-key` | LLM API key (required, or set `LLM_API_KEY` env var) |
| `--model` | LLM model name (default: `gpt-4o-mini`) |
| `-n, --max-videos` | Maximum number of videos to process |
| `--asr-model` | Whisper model size: `tiny`, `base`, `small`, `medium`, `large` (default: `tiny`) |
| `-o, --output` | Output file path |
| `--keep-audio` | Keep downloaded audio files |
| `--audio-dir` | Audio files storage directory |

## Backend

This project uses **faster-whisper** as the speech recognition backend, supporting:
- CPU
- CUDA (NVIDIA GPU)
- Other devices supported by faster-whisper

## Output Format

```
===video 1: Video Title 1===
Transcription content 1

===video 2: Video Title 2===
Transcription content 2
```

## Requirements

- Python 3.11+
- FFmpeg (for audio extraction)
