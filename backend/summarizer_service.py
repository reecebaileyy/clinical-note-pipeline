"""Summarization utilities for producing structured SOAP notes."""

from __future__ import annotations

import logging
import os
import re
import threading
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

try:  # pragma: no cover - heavy dependency imported lazily
    from transformers import pipeline
except ImportError as exc:  # pragma: no cover
    pipeline = None
    _transformers_import_error = exc
else:
    _transformers_import_error = None


logger = logging.getLogger(__name__)


class SOAPNote(BaseModel):
    """Structured clinical note in SOAP format."""

    subjective: str
    objective: str
    assessment: str
    plan: str


@dataclass
class SummarizationResult:
    """Container for summarization outputs."""

    soap_note: SOAPNote
    raw_summary: str

    def dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["soap_note"] = self.soap_note.dict()
        return payload


class SummarizerService:
    """Wraps a Hugging Face summarization model for SOAP note generation."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[int] = None,
        max_length: int = 512,
        min_length: int = 120,
    ) -> None:
        if pipeline is None:  # pragma: no cover
            raise ImportError(
                "The `transformers` package is required but not installed. Install with `pip install transformers`."
            ) from _transformers_import_error

        if model_name is None:
            model_name = os.getenv("SUMMARIZER_MODEL_NAME", "facebook/bart-large-cnn")
        if device is None:
            device_env = os.getenv("SUMMARIZER_DEVICE")
            device = int(device_env) if device_env else None

        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.min_length = min_length
        self._pipeline = None
        self._lock = threading.Lock()

    def summarize_to_soap(
        self,
        transcript: str,
        patient_context: Optional[Dict[str, Any]] = None,
    ) -> SummarizationResult:
        summarizer = self._get_pipeline()
        prompt = self._build_prompt(transcript, patient_context)

        summary_output = summarizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            min_length=min(self.min_length, self.max_length - 8),
            do_sample=False,
        )

        summary_text = summary_output[0]["summary_text"].strip()
        soap_note = self._coerce_to_soap(summary_text)
        return SummarizationResult(soap_note=soap_note, raw_summary=summary_text)

    def _get_pipeline(self):
        if self._pipeline is None:
            with self._lock:
                if self._pipeline is None:
                    logger.info("Loading summarization model %s", self.model_name)
                    kwargs = {"model": self.model_name, "task": "summarization"}
                    if self.device is not None:
                        kwargs["device"] = self.device
                    self._pipeline = pipeline(**kwargs)
        return self._pipeline

    @staticmethod
    def _build_prompt(transcript: str, patient_context: Optional[Dict[str, Any]]) -> str:
        header = [
            "You are a clinical documentation assistant summarizing a patient-clinician conversation.",
            "Produce a concise SOAP note (Subjective, Objective, Assessment, Plan) using complete sentences.",
            "Always include each SOAP section label followed by a colon.",
        ]

        if patient_context:
            context_block = "\n".join([f"- {key}: {value}" for key, value in patient_context.items()])
            header.append("Patient Context:\n" + context_block)

        header.append("Conversation Transcript:\n" + transcript.strip())
        header.append(
            "\nRespond only with the SOAP note, formatted as:\nSubjective: ...\nObjective: ...\nAssessment: ...\nPlan: ..."
        )
        return "\n\n".join(header)

    @staticmethod
    def _coerce_to_soap(summary_text: str) -> SOAPNote:
        sections = {section: "" for section in ["subjective", "objective", "assessment", "plan"]}
        current_key = "subjective"

        for line in summary_text.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                normalized = key.strip().lower()
                if normalized in sections:
                    current_key = normalized
                    sections[current_key] = value.strip()
                    continue
            sections[current_key] = (sections[current_key] + " " + line.strip()).strip()

        if any(value == "" for value in sections.values()):
            SummarizerService._fill_empty_sections(sections, summary_text)

        return SOAPNote(**sections)

    @staticmethod
    def _fill_empty_sections(sections: Dict[str, str], summary_text: str) -> None:
        sentences: List[str] = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", summary_text)
            if sentence.strip()
        ]

        if not sentences:
            sentences = [summary_text.strip()] if summary_text.strip() else ["Information unavailable."]

        keys = list(sections.keys())
        sentence_idx = 0
        for key in keys:
            if sections[key]:
                continue
            sections[key] = sentences[sentence_idx % len(sentences)]
            sentence_idx += 1

