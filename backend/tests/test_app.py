"""Backend integration tests with service fakes."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

from backend.app import create_app
from backend.summarizer_service import SOAPNote, SummarizationResult
from backend.whisper_service import StreamingUpdate, TranscriptionResult, TranscriptionSegment


class FakeWhisperService:
    def __init__(self) -> None:
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def transcribe_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> TranscriptionResult:
        segment = TranscriptionSegment(id=0, start=0.0, end=1.0, text="Hello world")
        return TranscriptionResult(text="Hello world", segments=[segment], language=language or "en", detected_language="en")

    def start_stream(self, session_id: str, language: Optional[str] = None) -> None:
        self.sessions[session_id] = {"language": language, "chunks": []}

    def set_stream_language(self, session_id: str, language: Optional[str]) -> None:
        if session_id in self.sessions:
            self.sessions[session_id]["language"] = language

    def process_stream_chunk(self, session_id: str, chunk: bytes) -> Optional[StreamingUpdate]:
        session = self.sessions.setdefault(session_id, {"language": "en", "chunks": []})
        session["chunks"].append(chunk)
        return StreamingUpdate(text="Hello world", delta="world", language=session.get("language") or "en")

    def finalize_stream(self, session_id: str) -> Dict[str, Optional[str]]:
        session = self.sessions.get(session_id, {"language": "en"})
        return {"text": "Hello world", "language": session.get("language"), "detected_language": "en"}

    def end_stream(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def close(self) -> None:  # pragma: no cover - nothing to clean up for fake
        self.sessions.clear()


class FakeSummarizerService:
    def summarize_to_soap(self, transcript: str, patient_context: Optional[Dict[str, Any]] = None) -> SummarizationResult:
        soap = SOAPNote(subjective="S", objective="O", assessment="A", plan="P")
        return SummarizationResult(soap_note=soap, raw_summary="Subjective: S\nObjective: O\nAssessment: A\nPlan: P")


class FakeEvaluationLogger:
    def __init__(self) -> None:
        self.transcripts = []
        self.summaries = []

    def log_transcript(self, transcript: str, metadata: Optional[Dict[str, Any]] = None, *, session_id: Optional[str] = None) -> int:
        self.transcripts.append({"transcript": transcript, "metadata": metadata, "session_id": session_id})
        return len(self.transcripts)

    def log_summary(
        self,
        *,
        transcript: str,
        soap_note: SOAPNote,
        raw_summary: str,
        session_id: Optional[str] = None,
        reference_note: Optional[str] = None,
    ) -> Dict[str, float]:
        metrics = {"rouge1_fmeasure": 1.0}
        self.summaries.append(
            {
                "transcript": transcript,
                "soap_note": soap_note.dict(),
                "raw_summary": raw_summary,
                "session_id": session_id,
                "reference_note": reference_note,
                "metrics": metrics,
            }
        )
        return metrics

    def close(self) -> None:  # pragma: no cover - nothing to clean up for fake
        self.transcripts.clear()
        self.summaries.clear()


@pytest.fixture()
def client() -> TestClient:
    app = create_app()
    app.dependency_overrides = {}
    app.state.whisper_service = FakeWhisperService()
    app.state.summarizer_service = FakeSummarizerService()
    app.state.evaluation_logger = FakeEvaluationLogger()

    with TestClient(app) as test_client:
        yield test_client


def test_healthcheck(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_endpoint(client: TestClient) -> None:
    files = {"file": ("sample.wav", b"fake audio", "audio/wav")}
    response = client.post("/transcribe", files=files)
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "Hello world"
    assert body["segments"][0]["text"] == "Hello world"


def test_summarize_endpoint(client: TestClient) -> None:
    payload = {"transcript": "Conversation about symptoms."}
    response = client.post("/summarize", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["soap_note"]["assessment"] == "A"
    assert "rouge1_fmeasure" in body["metrics"]


def test_websocket_transcription_flow(client: TestClient) -> None:
    with client.websocket_connect("/ws/transcribe") as websocket:
        start_message = websocket.receive_json()
        assert start_message["event"] == "session_started"
        session_id = start_message["session_id"]

        websocket.send_bytes(b"chunk")
        partial = websocket.receive_json()
        assert partial["event"] == "partial_transcript"
        assert partial["text"] == "Hello world"

        websocket.send_text(json.dumps({"event": "finalize"}))
        final = websocket.receive_json()
        assert final["event"] == "final_transcript"
        assert final["session_id"] == session_id

