"""FastAPI application entrypoint for the clinical note pipeline."""

from __future__ import annotations

import json
import logging
from functools import partial
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.evaluation import EvaluationLogger
from backend.summarizer_service import SOAPNote, SummarizerService
from backend.whisper_service import StreamingUpdate, WhisperService


logger = logging.getLogger(__name__)


class Segment(BaseModel):
    """Represents a Whisper segment returned to the client."""

    id: int
    start: float
    end: float
    text: str


class TranscriptionResponse(BaseModel):
    """Response model for the `/transcribe` endpoint."""

    text: str
    segments: List[Segment]
    language: Optional[str] = None
    detected_language: Optional[str] = None


class SummarizationRequest(BaseModel):
    """Request payload for the `/summarize` endpoint."""

    transcript: str
    session_id: Optional[str] = None
    patient_context: Optional[Dict[str, Any]] = None
    reference_note: Optional[str] = None


class SummarizationResponse(BaseModel):
    """Response payload for summarized SOAP notes."""

    soap_note: SOAPNote
    raw_summary: str
    metrics: Optional[Dict[str, float]] = None
    session_id: Optional[str] = None


def create_app() -> FastAPI:
    """Factory to create the FastAPI application instance."""

    application = FastAPI(title="Clinical Note Pipeline", version="0.1.0")

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.on_event("startup")
    async def _startup() -> None:  # pragma: no cover - executed by framework
        logger.info("Bootstrapping services for clinical pipeline")
        if not getattr(application.state, "whisper_service", None):
            application.state.whisper_service = WhisperService()
        if not getattr(application.state, "summarizer_service", None):
            application.state.summarizer_service = SummarizerService()
        if not getattr(application.state, "evaluation_logger", None):
            application.state.evaluation_logger = EvaluationLogger()

    @application.on_event("shutdown")
    async def _shutdown() -> None:  # pragma: no cover - executed by framework
        logger.info("Shutting down services")
        whisper_service: WhisperService | None = getattr(application.state, "whisper_service", None)
        if whisper_service:
            whisper_service.close()
        evaluation_logger: EvaluationLogger | None = getattr(application.state, "evaluation_logger", None)
        if evaluation_logger:
            evaluation_logger.close()

    return application


app = create_app()


def get_whisper_service() -> WhisperService:
    return app.state.whisper_service


def get_summarizer_service() -> SummarizerService:
    return app.state.summarizer_service


def get_evaluation_logger() -> EvaluationLogger:
    return app.state.evaluation_logger


@app.get("/health")
async def healthcheck() -> Dict[str, str]:
    """Simple healthcheck endpoint."""

    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = None,
    whisper_service: WhisperService = Depends(get_whisper_service),
    evaluation_logger: EvaluationLogger = Depends(get_evaluation_logger),
) -> TranscriptionResponse:
    """Transcribe an uploaded audio file using Whisper."""

    if file.content_type not in {"audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "application/octet-stream"}:
        raise HTTPException(status_code=415, detail=f"Unsupported audio format: {file.content_type}")

    audio_bytes = await file.read()

    result = await run_in_threadpool(
        whisper_service.transcribe_bytes,
        audio_bytes,
        language,
        file.content_type,
    )

    await run_in_threadpool(
        partial(
            evaluation_logger.log_transcript,
            result.text,
            metadata={"language": result.language, "detected_language": result.detected_language},
            session_id=None,
        )
    )

    segments = [Segment(**segment.dict()) for segment in result.segments]
    return TranscriptionResponse(
        text=result.text,
        segments=segments,
        language=result.language,
        detected_language=result.detected_language,
    )


@app.post("/summarize", response_model=SummarizationResponse)
async def summarize_text(
    payload: SummarizationRequest,
    summarizer_service: SummarizerService = Depends(get_summarizer_service),
    evaluation_logger: EvaluationLogger = Depends(get_evaluation_logger),
) -> SummarizationResponse:
    """Summarize transcript text into a structured SOAP note."""

    if not payload.transcript.strip():
        raise HTTPException(status_code=400, detail="Transcript text is required for summarization")

    summary_result = await run_in_threadpool(
        summarizer_service.summarize_to_soap,
        payload.transcript,
        payload.patient_context,
    )

    metrics = await run_in_threadpool(
        partial(
            evaluation_logger.log_summary,
            transcript=payload.transcript,
            soap_note=summary_result.soap_note,
            raw_summary=summary_result.raw_summary,
            session_id=payload.session_id,
            reference_note=payload.reference_note,
        )
    )

    return SummarizationResponse(
        soap_note=summary_result.soap_note,
        raw_summary=summary_result.raw_summary,
        metrics=metrics,
        session_id=payload.session_id,
    )


@app.websocket("/ws/transcribe")
async def websocket_transcription(
    websocket: WebSocket,
    whisper_service: WhisperService = Depends(get_whisper_service),
    evaluation_logger: EvaluationLogger = Depends(get_evaluation_logger),
) -> None:
    """Handle bi-directional transcription streaming sessions."""

    await websocket.accept()
    session_id = str(uuid4())
    await websocket.send_json({"event": "session_started", "session_id": session_id})

    whisper_service.start_stream(session_id)

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message and message["bytes"] is not None:
                chunk = message["bytes"]
                update: Optional[StreamingUpdate] = await run_in_threadpool(
                    whisper_service.process_stream_chunk,
                    session_id,
                    chunk,
                )

                if update is not None:
                    await websocket.send_json(
                        {
                            "event": "partial_transcript",
                            "session_id": session_id,
                            "text": update.text,
                            "delta": update.delta,
                            "language": update.language,
                        }
                    )

            elif "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"event": "error", "detail": "Invalid JSON payload"})
                    continue

                if payload.get("event") == "config":
                    language = payload.get("language")
                    whisper_service.set_stream_language(session_id, language)
                    await websocket.send_json({"event": "config_applied", "session_id": session_id, "language": language})
                elif payload.get("event") == "finalize":
                    final_result = await run_in_threadpool(whisper_service.finalize_stream, session_id)
                    await websocket.send_json({"event": "final_transcript", "session_id": session_id, **final_result})
                    await run_in_threadpool(
                        partial(
                            evaluation_logger.log_transcript,
                            final_result.get("text", ""),
                            metadata={"language": final_result.get("language"), "session_id": session_id},
                            session_id=session_id,
                        )
                    )
                else:
                    await websocket.send_json({"event": "error", "detail": "Unknown event"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for session %s", session_id)
    finally:
        whisper_service.end_stream(session_id)

