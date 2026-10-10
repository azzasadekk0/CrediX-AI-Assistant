"""Final grounded answer and source provenance models."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ResolvedSource:
    handle: str
    chunk_ids: list[str]
    parent_id: str
    document_id: str
    document_title: str | None
    page_start: int | None
    page_end: int | None
    article_number: str | None
    provision_number: str | None

    @classmethod
    def from_citation_map(cls, handle: str, source: dict[str, Any]) -> "ResolvedSource":
        return cls(handle, list(source["chunk_ids"]), source["parent_id"], source["document_id"],
                   source.get("document_title"), source.get("page_start"), source.get("page_end"),
                   source.get("article_number"), source.get("provision_number"))


@dataclass(frozen=True, slots=True)
class GroundedAnswer:
    original_query: str
    answer: str
    no_answer: bool
    citations: list[str]
    resolved_sources: list[ResolvedSource]
    provider: str
    model: str
    generation_latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    validation_status: str
    prompt_version: str
    context_metadata: dict[str, Any]
    citation_repair_attempted: bool = False
    citation_normalization_used: bool = False
    citation_normalization_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
