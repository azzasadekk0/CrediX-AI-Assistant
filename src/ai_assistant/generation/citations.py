"""Deterministic structured-output and ContextPack citation validation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace

from ..models import ContextPack
from .errors import CitationValidationError, StructuredOutputError
from .response import ResolvedSource


@dataclass(frozen=True, slots=True)
class StructuredGroundedResult:
    answer: str
    citations: list[str]
    no_answer: bool


def canonicalize_citations(result: StructuredGroundedResult, context_pack: ContextPack) -> tuple[StructuredGroundedResult, int]:
    """Map only exact, uniquely-owned current ContextPack chunk IDs to E handles.

    This deliberately has no fuzzy matching or source-ID guessing. A no-answer
    result is left untouched so its existing empty-citation contract remains
    enforced by ``validate_citations``.
    """
    normalized, count, _ = canonicalize_citations_with_details(result, context_pack)
    return normalized, count


def canonicalize_citations_with_details(result: StructuredGroundedResult, context_pack: ContextPack, *, include_decorated: bool = True) -> tuple[StructuredGroundedResult, int, list[dict[str, object]]]:
    """Canonicalize exact identifiers and retain only safe local diagnostics."""
    if result.no_answer:
        return result, 0, []
    chunk_to_handles: dict[str, list[str]] = {}
    for handle, source in context_pack.citation_map.items():
        for chunk_id in source["chunk_ids"]:
            chunk_to_handles.setdefault(chunk_id, []).append(handle)
    normalized: list[str] = []
    normalization_count = 0
    details: list[dict[str, object]] = []
    for citation in result.citations:
        if citation in context_pack.citation_map:
            value = citation
            status = "direct"
        else:
            decorated = _decorated_e_handle(citation) if include_decorated else None
            if decorated and decorated in context_pack.citation_map:
                value, status = decorated, "decorated_normalized"
            else:
                matches = chunk_to_handles.get(citation, [])
                value = matches[0] if len(matches) == 1 else citation
                status = "normalized" if len(matches) == 1 else "ambiguous" if matches else "unresolved"
            normalization_count += value != citation
        details.append({"input": citation, "output": value, "status": status})
        if value not in normalized:
            normalized.append(value)
    return replace(result, citations=normalized), normalization_count, details


def _decorated_e_handle(value: str) -> str | None:
    """Return only explicitly allowed whole-identifier E-handle wrappers."""
    match = re.fullmatch(r"(?:EVIDENCE\s+(E\d+)|\[(E\d+)\]|\[EVIDENCE\s+(E\d+)\])", value)
    if not match:
        return None
    return next(group for group in match.groups() if group)


def raw_first_pass_payload(text: str) -> dict[str, object] | None:
    """Capture untouched structured fields for diagnostics before parser cleanup."""
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return {"answer": value.get("answer"), "citations": list(value["citations"]) if isinstance(value.get("citations"), list) else value.get("citations"),
            "no_answer": value.get("no_answer")}


def parse_grounded_result(text: str) -> StructuredGroundedResult:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise StructuredOutputError("provider did not return valid structured JSON") from exc
    if not isinstance(value, dict) or set(value) != {"answer", "citations", "no_answer"}:
        raise StructuredOutputError("grounded result must contain exactly answer, citations, and no_answer")
    answer, citations, no_answer = value["answer"], value["citations"], value["no_answer"]
    if not isinstance(answer, str) or (not answer.strip() and not no_answer):
        raise StructuredOutputError("grounded factual answer must be non-empty")
    if not isinstance(citations, list) or not all(isinstance(handle, str) for handle in citations):
        raise StructuredOutputError("grounded citations must be an array of strings")
    if not isinstance(no_answer, bool):
        raise StructuredOutputError("grounded no_answer must be boolean")
    return StructuredGroundedResult(answer.strip(), list(dict.fromkeys(citations)), no_answer)


def validate_citations(result: StructuredGroundedResult, context_pack: ContextPack) -> list[ResolvedSource]:
    invalid = [handle for handle in result.citations if handle not in context_pack.citation_map]
    if invalid:
        raise CitationValidationError(invalid, "grounded result cited unavailable evidence")
    if result.no_answer:
        if result.citations:
            raise CitationValidationError(result.citations, "no-answer result must not cite factual evidence")
        return []
    if not result.citations:
        raise CitationValidationError([], "a factual grounded answer requires at least one citation")
    return [ResolvedSource.from_citation_map(handle, context_pack.citation_map[handle]) for handle in result.citations]
