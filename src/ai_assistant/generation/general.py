"""Standalone general-knowledge generation with explicit provenance.

This module deliberately has no retrieval or context imports.  It is the
isolated phase-one path used for questions that are not being answered from
CrediX documents.  The response contract makes that fact machine-readable so
callers cannot accidentally present it as a cited RAG answer.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

from ..llm.base import LLMProvider
from ..llm.factory import get_llm
from ..llm.models import LLMRequest
from .errors import GeneralGenerationError


GENERAL_RESULT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

GENERAL_DISCLAIMER_EN = "AI-generated answer; not retrieved from CrediX documents."
GENERAL_DISCLAIMER_AR = "إجابة مولّدة بالذكاء الاصطناعي وليست مستخرجة من مستندات CrediX."


@dataclass(frozen=True, slots=True)
class GeneralAnswer:
    """An answer whose provenance is always explicitly general/AI-generated."""

    original_query: str
    answer: str
    answer_mode: str
    provenance: str
    citations: list[str]
    disclaimer: str
    provider: str
    model: str
    generation_latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    validation_status: str = "validated"
    prompt_version: str = "general_v1"

    def __post_init__(self) -> None:
        if self.answer_mode != "general":
            raise ValueError("GeneralAnswer.answer_mode must be 'general'")
        if self.provenance != "ai_generated":
            raise ValueError("GeneralAnswer.provenance must be 'ai_generated'")
        if self.citations:
            raise ValueError("general answers cannot contain citations")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_arabic(query: str) -> bool:
    arabic = sum("\u0600" <= char <= "\u06ff" for char in query)
    latin = sum(char.isascii() and char.isalpha() for char in query)
    return arabic > latin


def general_system_prompt() -> str:
    return (
        "You are the CrediX general AI assistant. Answer the user's general question helpfully and accurately "
        "using your general model knowledge only. Do not claim that any fact came from CrediX documents, policies, "
        "or retrieved evidence. Do not invent access to customer records or internal data. If the user asks for a "
        "specific CrediX policy, case, application, or document fact, state that this general path has no document "
        "evidence and the question must be answered by the grounded document assistant. Answer in the user's language. "
        "Return only JSON with one string field named 'answer'."
    )


def general_user_prompt(query: str) -> str:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    language = "Arabic" if _is_arabic(query) else "English"
    return f"RESPONSE LANGUAGE: {language}\n\nQUESTION:\n{query.strip()}"


def _parse_general_result(raw: str) -> str:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise GeneralGenerationError("general model returned invalid structured output") from exc
    if not isinstance(payload, dict) or set(payload) != {"answer"} or not isinstance(payload["answer"], str):
        raise GeneralGenerationError("general model returned an invalid answer object")
    answer = payload["answer"].strip()
    if not answer:
        raise GeneralGenerationError("general model returned an empty answer")
    return answer


class GeneralGenerator:
    """Generate without retrieval and return an explicit AI-only provenance contract."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.last_trace: dict[str, object] = {}

    def generate(self, query: str) -> GeneralAnswer:
        user_prompt = general_user_prompt(query)
        response = self.provider.generate(
            LLMRequest(
                general_system_prompt(),
                user_prompt,
                temperature=0.2,
                response_schema=GENERAL_RESULT_SCHEMA,
                response_schema_name="general_answer",
            )
        )
        answer = _parse_general_result(response.text)
        self.last_trace = {
            "answer_mode": "general",
            "provenance": "ai_generated",
            "citations": [],
            "provider": response.provider,
            "model": response.model,
            "prompt_version": "general_v1",
        }
        return GeneralAnswer(
            original_query=query.strip(),
            answer=answer,
            answer_mode="general",
            provenance="ai_generated",
            citations=[],
            disclaimer=GENERAL_DISCLAIMER_AR if _is_arabic(query) else GENERAL_DISCLAIMER_EN,
            provider=response.provider,
            model=response.model,
            generation_latency_seconds=response.latency_seconds,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
        )


class GeneralAssistant:
    """Small provider-backed orchestrator for the phase-one general path."""

    def __init__(self, *, generator: GeneralGenerator | None = None, provider: LLMProvider | None = None) -> None:
        self._generator = generator or GeneralGenerator(provider or get_llm())

    def answer(self, query: str) -> GeneralAnswer:
        return self._generator.generate(query)

