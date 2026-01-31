"""ASR transcription with faster-whisper backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ASRBackend(ABC):
    """Abstract base class for ASR backends."""

    @abstractmethod
    def load(self) -> None:
        pass

    @abstractmethod
    def transcribe(self, audio_path: Path) -> str:
        pass


class FasterWhisperBackend(ASRBackend):
    """faster-whisper backend for reliable full audio transcription."""

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

    def transcribe(self, audio_path: Path) -> str:
        self.load()
        assert self.model is not None
        segments, info = self.model.transcribe(
            str(audio_path),
            language="zh",
            beam_size=5,
            vad_filter=True,
            word_timestamps=False,  # Disable for speed
        )
        # Combine all segments into full text
        full_text = "".join([segment.text for segment in segments])
        return full_text


class ASRTranscriber:
    """ASR transcriber using faster-whisper backend."""

    def __init__(
        self,
        model_name: str | None = None,
        backend: str | None = None,
    ):
        self._backend: ASRBackend | None = None
        self._model_name = model_name or "tiny"

    def load(self) -> None:
        if self._backend is not None:
            return

        self._backend = FasterWhisperBackend(model_size=self._model_name)
        self._backend.load()

    def transcribe(self, audio_path: Path) -> str:
        self.load()
        assert self._backend is not None
        return self._backend.transcribe(audio_path)

    @property
    def backend_info(self) -> str:
        return f"faster-whisper ({self._model_name})"
