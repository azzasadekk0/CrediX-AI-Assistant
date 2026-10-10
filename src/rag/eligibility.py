"""Shared conservative gates for embedding and index eligibility."""
from __future__ import annotations

from typing import Any, Iterable


def release_blockers(chunk: dict[str, Any]) -> list[str]:
    reasons = []
    if chunk.get("verification_status") != "verified":
        reasons.append("source_not_verified")
    if chunk.get("text_quality_status") != "reviewed":
        reasons.append("text_quality_not_reviewed")
    if chunk.get("pdf_page_start") is None or chunk.get("pdf_page_end") is None:
        reasons.append("pdf_page_reference_missing")
    return reasons


def release_blocking_chunks(chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"chunk_id": row.get("chunk_id"), "reasons": release_blockers(row)}
            for row in chunks if release_blockers(row)]
