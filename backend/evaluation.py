"""Evaluation utilities, metrics, and persistence for the clinical pipeline."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from backend.summarizer_service import SOAPNote


logger = logging.getLogger(__name__)

Base = declarative_base()


class TranscriptLog(Base):
    __tablename__ = "transcript_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(128), nullable=True, index=True)
    transcript = Column(Text, nullable=False)
    language = Column(String(32), nullable=True)
    detected_language = Column(String(32), nullable=True)
    extras = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class SummaryLog(Base):
    __tablename__ = "summary_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transcript_id = Column(Integer, ForeignKey("transcript_logs.id"), nullable=True)
    session_id = Column(String(128), nullable=True, index=True)
    raw_summary = Column(Text, nullable=False)
    subjective = Column(Text, nullable=False)
    objective = Column(Text, nullable=False)
    assessment = Column(Text, nullable=False)
    plan = Column(Text, nullable=False)
    reference_note = Column(Text, nullable=True)
    metrics = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    transcript = relationship("TranscriptLog", backref="summaries")


class MetricEvaluator:
    """Compute quality metrics for generated clinical summaries."""

    def __init__(self, rouge_types: Optional[list[str]] = None, bert_lang: str = "en") -> None:
        from rouge_score import rouge_scorer

        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeLsum"]
        self.rouge_scorer = rouge_scorer.RougeScorer(self.rouge_types, use_stemmer=True)
        self.bert_lang = bert_lang

    def compute(self, prediction: str, reference: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        reference = reference.strip()
        prediction = prediction.strip()

        if not reference or not prediction:
            return metrics

        rouge_scores = self.rouge_scorer.score(reference, prediction)
        for name, score in rouge_scores.items():
            metrics[f"{name}_precision"] = float(score.precision)
            metrics[f"{name}_recall"] = float(score.recall)
            metrics[f"{name}_fmeasure"] = float(score.fmeasure)

        try:
            from bert_score import score as bert_score

            precision, recall, fmeasure = bert_score(
                [prediction], [reference], lang=self.bert_lang, rescale_with_baseline=True
            )
            metrics["bertscore_precision"] = float(precision[0])
            metrics["bertscore_recall"] = float(recall[0])
            metrics["bertscore_f1"] = float(fmeasure[0])
        except Exception as exc:  # pragma: no cover - optional dependency runtime errors
            logger.warning("BERTScore computation failed: %s", exc)

        return metrics


class EvaluationLogger:
    """Persist transcripts, SOAP notes, and derived metrics to a database."""

    def __init__(self, database_url: Optional[str] = None) -> None:
        default_url = "postgresql://postgres:postgres@localhost:5432/clinical_notes"
        self.database_url = database_url or os.getenv("DATABASE_URL", default_url)

        try:
            self.engine = create_engine(self.database_url, future=True)
            Base.metadata.create_all(self.engine)
        except SQLAlchemyError as exc:
            if database_url or os.getenv("DATABASE_URL"):
                raise
            fallback_url = "sqlite:///./clinical_notes.db"
            logger.warning(
                "Failed to initialize database at %s (%s). Falling back to %s.",
                self.database_url,
                exc,
                fallback_url,
            )
            self.database_url = fallback_url
            self.engine = create_engine(self.database_url, future=True)
            Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False, future=True)
        self.metric_evaluator = MetricEvaluator()

    @contextmanager
    def _session_scope(self) -> Session:
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except SQLAlchemyError:
            session.rollback()
            raise
        finally:
            session.close()

    def log_transcript(
        self,
        transcript: str,
        metadata: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
    ) -> int:
        metadata = metadata or {}
        language = metadata.get("language")
        detected_language = metadata.get("detected_language")

        with self._session_scope() as session:
            record = TranscriptLog(
                session_id=session_id,
                transcript=transcript,
                language=language,
                detected_language=detected_language,
                extras=metadata,
            )
            session.add(record)
            session.flush()
            return record.id

    def log_summary(
        self,
        *,
        transcript: str,
        soap_note: SOAPNote,
        raw_summary: str,
        session_id: Optional[str] = None,
        reference_note: Optional[str] = None,
    ) -> Dict[str, float]:
        reference_text = reference_note or transcript
        metrics = self.metric_evaluator.compute(raw_summary, reference_text) if reference_text else {}

        with self._session_scope() as session:
            transcript_id = self._find_transcript_id(session, session_id)
            record = SummaryLog(
                transcript_id=transcript_id,
                session_id=session_id,
                raw_summary=raw_summary,
                subjective=soap_note.subjective,
                objective=soap_note.objective,
                assessment=soap_note.assessment,
                plan=soap_note.plan,
                reference_note=reference_note,
                metrics=metrics,
            )
            session.add(record)

        return metrics

    def close(self) -> None:
        self.engine.dispose()

    def _find_transcript_id(self, session: Session, session_id: Optional[str]) -> Optional[int]:
        if not session_id:
            return None
        transcript = (
            session.query(TranscriptLog)
            .filter(TranscriptLog.session_id == session_id)
            .order_by(TranscriptLog.created_at.desc())
            .first()
        )
        return transcript.id if transcript else None
