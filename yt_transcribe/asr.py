"""ASR transcription with MLX Qwen3-ASR and faster-whisper backends."""

from __future__ import annotations

import platform
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TranscriptionResult:
    """Result of ASR transcription with optional metadata."""
    text: str
    segments: list[dict] | None = None
    words: list[dict] | None = None
    language: str | None = None


class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    @abstractmethod
    def load(self) -> None:
        """Load the ASR model."""
        pass

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        """Transcribe audio file and return result with metadata."""
        pass


class FasterWhisperBackend(ASRBackend):
    """faster-whisper backend for reliable full audio transcription."""

    MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
    LANGUAGE_MAP = {
        "auto": None,
        "chinese": "zh",
        "english": "en",
        "japanese": "ja",
        "korean": "ko",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "russian": "ru",
        "portuguese": "pt",
        "italian": "it",
        "dutch": "nl",
    }

    def __init__(
        self,
        model_size: str = "tiny",
        device: str = "cpu",
        compute_type: str = "int8",
        num_workers: int = 1,
        cpu_threads: int = 0,  # 0 = auto
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.num_workers = num_workers
        self.cpu_threads = cpu_threads
        self.model: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
            num_workers=self.num_workers,
            cpu_threads=self.cpu_threads,
        )

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        self.load()
        assert self.model is not None

        lang_code = self._normalize_language(language)
        segments, info = self.model.transcribe(
            str(audio_path),
            language=lang_code,
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,
        )

        full_text = "".join([segment.text for segment in segments])

        return TranscriptionResult(
            text=full_text,
            language=info.language if lang_code is None else lang_code,
        )

    def _normalize_language(self, language: str | None) -> str | None:
        """Normalize language name to ISO code."""
        if language is None or language == "auto":
            return None
        return self.LANGUAGE_MAP.get(language.lower(), language)


class MLXBackend(ASRBackend):
    """MLX Qwen3-ASR backend for Apple Silicon.

    Uses mlx-audio library for high-performance transcription on Apple Silicon.
    """

    MODELS = {
        "0.6b": "mlx-community/Qwen3-ASR-0.6B-8bit",
        "1.7b": "mlx-community/Qwen3-ASR-1.7B-8bit",
    }

    # Map our language names to the model's expected language names
    # The Qwen3-ASR model uses full English names by default
    LANGUAGE_MAP = {
        "auto": "English",  # Default
        "chinese": "Chinese",
        "english": "English",
        "japanese": "Japanese",
        "korean": "Korean",
        "cantonese": "Cantonese",
        "french": "French",
        "german": "German",
        "spanish": "Spanish",
        "russian": "Russian",
        "portuguese": "Portuguese",
        "italian": "Italian",
        "dutch": "Dutch",
    }

    def __init__(self, model_size: str = "0.6b"):
        if model_size not in self.MODELS:
            raise ValueError(
                f"Invalid MLX model size: {model_size}. "
                f"Choose from: {', '.join(self.MODELS.keys())}"
            )
        self.model_size = model_size
        self.model: Any = None
        self.model_path: str | None = None

    def load(self) -> None:
        if self.model is not None:
            return
        from mlx_audio.stt.utils import load_model

        self.model_path = self.MODELS[self.model_size]
        self.model = load_model(self.model_path)

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        from mlx_audio.stt.generate import generate_transcription

        self.load()
        assert self.model is not None

        lang_name = self._normalize_language(language)
        segments = generate_transcription(
            model=self.model,
            audio=str(audio_path),
            language=lang_name,
            verbose=False,
        )

        # Extract text from segments
        text = ""
        for segment in segments:
            if isinstance(segment, dict):
                text += segment.get("text", "")
            else:
                text += str(segment)

        return TranscriptionResult(
            text=text,
            segments=segments if isinstance(segments[0] if segments else None, dict) else None,
            language=language or "auto",
        )

    def _normalize_language(self, language: str | None) -> str:
        """Normalize language name to the model's expected format."""
        if language is None or language == "auto":
            # Use default (English)
            return "English"
        return self.LANGUAGE_MAP.get(language.lower(), "English")

    @property
    def supported_languages(self) -> list[str]:
        """Return list of supported languages."""
        return list(self.LANGUAGE_MAP.keys())


class MLXAlignerBackend(ASRBackend):
    """MLX Qwen3-ForcedAligner backend for word-level timestamps.

    Provides word-level timing information for subtitle generation.
    """

    MODEL_PATH = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    # Map our language names to the model's expected language names
    LANGUAGE_MAP = {
        "chinese": "Chinese",
        "english": "English",
        "japanese": "Japanese",
        "korean": "Korean",
        "cantonese": "Cantonese",
        "french": "French",
        "german": "German",
        "spanish": "Spanish",
        "russian": "Russian",
        "portuguese": "Portuguese",
        "italian": "Italian",
        "dutch": "Dutch",
    }

    def __init__(self):
        self.model: Any = None

    def load(self) -> None:
        if self.model is not None:
            return
        from mlx_audio.stt.utils import load_model
        self.model = load_model(self.MODEL_PATH)

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        """Run full transcription with word-level timestamps.

        The ForcedAligner model provides both transcription and word-level timestamps.
        """
        from mlx_audio.stt.generate import generate_transcription

        self.load()
        assert self.model is not None

        lang_name = self._normalize_language(language)

        # Generate transcription with timestamps
        segments = generate_transcription(
            model=self.model,
            audio=str(audio_path),
            language=lang_name,
            verbose=False,
        )

        # Extract text and words from segments
        text = ""
        words: list[dict] = []
        for segment in segments:
            if isinstance(segment, dict):
                segment_text = segment.get("text", "")
                text += segment_text
                # Extract word-level timestamps if available
                if "words" in segment:
                    words.extend(segment.get("words", []))
            else:
                text += str(segment)

        return TranscriptionResult(
            text=text,
            words=words if words else None,
            language=language or "auto",
        )

    def align(self, audio_path: Path, transcript: str, language: str | None = None) -> list[dict]:
        """Align transcript to audio for word-level timestamps.

        Args:
            audio_path: Path to audio file
            transcript: Transcript text to align
            language: Language code

        Returns:
            List of word dicts with 'word', 'start', 'end' keys
        """
        from mlx_audio.stt.generate import generate_transcription

        self.load()
        assert self.model is not None

        lang_name = self._normalize_language(language)

        # Use forced alignment by providing the text
        segments = generate_transcription(
            model=self.model,
            audio=str(audio_path),
            text=transcript,
            language=lang_name,
            verbose=False,
        )

        # Extract word-level timestamps
        words: list[dict] = []
        for segment in segments:
            if isinstance(segment, dict) and "words" in segment:
                words.extend(segment.get("words", []))

        return words

    def _normalize_language(self, language: str | None) -> str:
        """Normalize language name to the model's expected format."""
        if language is None:
            return "Chinese"  # Default to Chinese for aligner
        return self.LANGUAGE_MAP.get(language.lower(), "Chinese")


def get_backend() -> str:
    """Auto-detect the best available ASR backend.

    Returns:
        'mlx' if on Apple Silicon with MLX available, else 'faster-whisper'
    """
    # Check if we're on Apple Silicon
    if platform.processor() == "arm" and platform.system() == "Darwin":
        try:
            import mlx_audio  # noqa: F401
            return "mlx"
        except ImportError:
            pass
    return "faster-whisper"


class ASRTranscriber:
    """ASR transcriber with auto-detecting backend selection."""

    def __init__(
        self,
        backend: str | None = None,
        model_name: str | None = None,
        language: str | None = None,
    ):
        """
        Args:
            backend: Backend name ('auto', 'mlx', 'faster-whisper')
            model_name: Model size identifier ('0.6b', '1.7b' for MLX;
                         'tiny', 'base', etc. for faster-whisper)
            language: Language code or name (e.g., 'auto', 'chinese', 'english')
        """
        self._backend: ASRBackend | None = None
        self._backend_name = backend or "auto"
        self._model_name = model_name
        self._language = language

    def load(self) -> None:
        if self._backend is not None:
            return

        backend_type = self._backend_name
        if backend_type == "auto":
            backend_type = get_backend()

        if backend_type == "mlx":
            model = self._model_name or "0.6b"
            self._backend = MLXBackend(model_size=model)
        elif backend_type == "faster-whisper":
            model = self._model_name or "tiny"
            self._backend = FasterWhisperBackend(model_size=model)
        else:
            raise ValueError(
                f"Unknown backend: {backend_type}. "
                f"Choose from: auto, mlx, faster-whisper"
            )

        self._backend.load()

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        self.load()
        assert self._backend is not None
        return self._backend.transcribe(audio_path, language=self._language)

    @property
    def backend_info(self) -> str:
        backend_type = self._backend_name if self._backend_name != "auto" else get_backend()
        model = self._model_name or ("0.6b" if backend_type == "mlx" else "tiny")
        return f"{backend_type} ({model})"

    @property
    def backend_name(self) -> str:
        return self._backend_name if self._backend_name != "auto" else get_backend()
