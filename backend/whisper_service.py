"""Utilities for running Whisper transcription workloads."""

from __future__ import annotations

import io
import os
import threading
from dataclasses import asdict, dataclass
from io import BytesIO
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

try:  # pragma: no cover - import errors handled gracefully
    import whisper  # type: ignore
except ImportError as exc:  # pragma: no cover
    whisper = None  # type: ignore
    _whisper_import_error = exc
else:
    _whisper_import_error = None


@dataclass
class TranscriptionSegment:
    """Represents a segment of transcribed audio."""

    id: int
    start: float
    end: float
    text: str

    def dict(self) -> Dict[str, float | str]:
        return asdict(self)


@dataclass
class TranscriptionResult:
    """Full transcription output."""

    text: str
    segments: List[TranscriptionSegment]
    language: Optional[str]
    detected_language: Optional[str]


@dataclass
class StreamingUpdate:
    """Represents a streaming update delivered to websocket clients."""

    text: str
    delta: str
    language: Optional[str]


@dataclass
class _StreamingSession:
    buffer: io.BytesIO
    last_text: str = ""
    language: Optional[str] = None


class WhisperService:
    """Abstraction around the OpenAI Whisper ASR models."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        if whisper is None:  # pragma: no cover - ensures helpful message in missing dependency scenarios
            raise ImportError(
                "The `whisper` package is required but not installed. Install with `pip install openai-whisper`."
            ) from _whisper_import_error

        if model_name is None:
            model_name = os.getenv("WHISPER_MODEL_NAME", "small")
        if device is None:
            device = os.getenv("WHISPER_DEVICE")

        self.model_name = model_name
        self.device = device
        self._model = None
        self._model_lock = threading.Lock()
        self._sessions: Dict[str, _StreamingSession] = {}
        self._session_lock = threading.Lock()
        self._min_buffer_bytes = 32_000
        self._target_sample_rate = 16000

    def _load_model(self):
        if self._model is None:
            with self._model_lock:
                if self._model is None:
                    self._model = whisper.load_model(self.model_name, device=self.device)
        return self._model

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> TranscriptionResult:
        model = self._load_model()
        audio_array = self._decode_audio_bytes(audio_bytes)
        result = model.transcribe(
            audio_array,
            language=language,
            task="transcribe",
            fp16=False,
        )

        return self._to_transcription_result(result)

    def start_stream(self, session_id: str, language: Optional[str] = None) -> None:
        with self._session_lock:
            self._sessions[session_id] = _StreamingSession(buffer=io.BytesIO(), language=language)

    def set_stream_language(self, session_id: str, language: Optional[str]) -> None:
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.language = language

    def process_stream_chunk(self, session_id: str, chunk: bytes) -> Optional[StreamingUpdate]:
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown streaming session: {session_id}")

            session.buffer.write(chunk)
            buffer_value = session.buffer.getvalue()
            baseline_text = session.last_text
            language = session.language

        if len(buffer_value) < self._min_buffer_bytes:
            return None

        result = self.transcribe_bytes(buffer_value, language)
        if result.text == baseline_text:
            return None

        delta = result.text[len(baseline_text) :].strip()
        if not delta and result.text != baseline_text:
            delta = result.text

        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_text = result.text

        return StreamingUpdate(text=result.text, delta=delta, language=result.detected_language or result.language)

    def finalize_stream(self, session_id: str) -> Dict[str, Optional[str]]:
        with self._session_lock:
            session = self._sessions.get(session_id)
            if session is None:
                raise ValueError(f"Unknown streaming session: {session_id}")
            buffer_value = session.buffer.getvalue()
            language = session.language

        result = self.transcribe_bytes(buffer_value, language)

        with self._session_lock:
            session = self._sessions.get(session_id)
            if session:
                session.last_text = result.text

        return {
            "text": result.text,
            "language": result.language,
            "detected_language": result.detected_language,
        }

    def end_stream(self, session_id: str) -> None:
        with self._session_lock:
            self._sessions.pop(session_id, None)

    def close(self) -> None:
        with self._session_lock:
            self._sessions.clear()

    @staticmethod
    def _to_transcription_result(raw_result: Dict[str, Any]) -> TranscriptionResult:
        segments = [
            TranscriptionSegment(
                id=segment.get("id", idx),
                start=float(segment.get("start", 0.0)),
                end=float(segment.get("end", 0.0)),
                text=segment.get("text", "").strip(),
            )
            for idx, segment in enumerate(raw_result.get("segments", []))
        ]

        return TranscriptionResult(
            text=raw_result.get("text", "").strip(),
            segments=segments,
            language=raw_result.get("language"),
            detected_language=raw_result.get("language"),
        )

    def _decode_audio_bytes(self, audio_bytes: bytes) -> np.ndarray:
        try:
            audio, sample_rate = sf.read(BytesIO(audio_bytes), dtype="float32")
        except RuntimeError as exc:
            raise ValueError("Unable to decode audio payload. Ensure the file is a valid WAV/MP3/FLAC clip.") from exc

        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sample_rate != self._target_sample_rate:
            audio = resample_poly(audio, self._target_sample_rate, sample_rate)

        return audio

