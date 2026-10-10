"""Internal result models for the AI Assistant retrieval pipeline."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


RetrieverName = Literal["dense", "bm25"]


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """One unmodified retrieval hit plus the metadata needed by later stages."""

    chunk_id: str
    document_id: str
    document_title: str | None
    text_original: str
    parent_id: str
    pdf_page_start: int | None
    pdf_page_end: int | None
    article_number: str | None
    provision_number: str | None
    score: float
    retriever: RetrieverName
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation without altering scores."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HybridRetrievalResult:
    """Independent dense and BM25 retrieval lists for one original query."""

    original_query: str
    dense_results: list[RetrievalCandidate]
    bm25_results: list[RetrievalCandidate]

    def to_dict(self) -> dict[str, Any]:
        """Return separate raw-score result lists suitable for inspection output."""
        return {
            "original_query": self.original_query,
            "dense_results": [candidate.to_dict() for candidate in self.dense_results],
            "bm25_results": [candidate.to_dict() for candidate in self.bm25_results],
        }


@dataclass(frozen=True, slots=True)
class FusedRetrievalCandidate:
    """One de-duplicated rank-fusion hit, retaining both source observations."""

    chunk_id: str
    document_id: str
    document_title: str | None
    text_original: str
    parent_id: str
    pdf_page_start: int | None
    pdf_page_end: int | None
    article_number: str | None
    provision_number: str | None
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_score: float | None
    bm25_score: float | None
    retrieved_by: list[RetrieverName]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return all original fields and rank-fusion diagnostics."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FusedRetrievalResult:
    """Raw ranked lists and their deterministic fused candidate pool."""

    original_query: str
    dense_results: list[RetrievalCandidate]
    bm25_results: list[RetrievalCandidate]
    fused_results: list[FusedRetrievalCandidate]
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "dense_results": [candidate.to_dict() for candidate in self.dense_results],
            "bm25_results": [candidate.to_dict() for candidate in self.bm25_results],
            "fused_results": [candidate.to_dict() for candidate in self.fused_results],
            "configuration": self.configuration,
        }


@dataclass(frozen=True, slots=True)
class RerankedCandidate:
    """A candidate ranked solely by a cross-encoder relevance score."""

    chunk_id: str
    document_id: str
    document_title: str | None
    text_original: str
    parent_id: str
    pdf_page_start: int | None
    pdf_page_end: int | None
    article_number: str | None
    provision_number: str | None
    reranker_score: float
    original_candidate_rank: int
    rrf_score: float | None
    dense_rank: int | None
    bm25_rank: int | None
    dense_score: float | None
    bm25_score: float | None
    retrieved_by: list[RetrieverName]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RerankedRetrievalResult:
    """Candidate provenance and final cross-encoder ranking for one query."""

    original_query: str
    dense_results: list[RetrievalCandidate]
    bm25_results: list[RetrievalCandidate]
    candidates: list[FusedRetrievalCandidate]
    reranked_results: list[RerankedCandidate]
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "dense_results": [candidate.to_dict() for candidate in self.dense_results],
            "bm25_results": [candidate.to_dict() for candidate in self.bm25_results],
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "reranked_results": [candidate.to_dict() for candidate in self.reranked_results],
            "configuration": self.configuration,
        }


@dataclass(frozen=True, slots=True)
class ExpandedEvidence:
    """One source-faithful evidence block ready for context selection."""

    citation_handle: str | None
    chunk_ids: list[str]
    primary_chunk_id: str
    parent_id: str
    document_id: str
    document_title: str | None
    text_original: str
    expanded_text: str
    reranker_rank: int
    reranker_score: float
    dense_rank: int | None
    bm25_rank: int | None
    rrf_score: float | None
    pdf_page_start: int | None
    pdf_page_end: int | None
    article_number: str | None
    provision_number: str | None
    expansion_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ContextPack:
    """A deterministic, cited collection of source-only evidence blocks."""

    original_query: str
    evidence_items: list[ExpandedEvidence]
    selected_chunk_ids: list[str]
    selected_parent_ids: list[str]
    document_ids: list[str]
    citation_map: dict[str, dict[str, Any]]
    evidence_count: int
    unique_parent_count: int
    unique_document_count: int
    estimated_context_tokens: int
    configuration: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "original_query": self.original_query,
            "evidence_items": [item.to_dict() for item in self.evidence_items],
            "selected_chunk_ids": self.selected_chunk_ids,
            "selected_parent_ids": self.selected_parent_ids,
            "document_ids": self.document_ids,
            "citation_map": self.citation_map,
            "evidence_count": self.evidence_count,
            "unique_parent_count": self.unique_parent_count,
            "unique_document_count": self.unique_document_count,
            "estimated_context_tokens": self.estimated_context_tokens,
            "configuration": self.configuration,
        }

    def render_structured(self) -> str:
        """Render complete internal source blocks for diagnostics and inspection."""
        blocks: list[str] = []
        for item in self.evidence_items:
            article = " / ".join(value for value in (item.article_number, item.provision_number) if value) or ""
            blocks.append(
                f"[EVIDENCE {item.citation_handle}]\n\n"
                f"Document: {item.document_title or item.document_id}\n"
                f"Page: {item.pdf_page_start or ''}{('-' + str(item.pdf_page_end)) if item.pdf_page_end and item.pdf_page_end != item.pdf_page_start else ''}\n"
                f"Article / Provision: {article}\n"
                f"Chunk ID(s): {', '.join(item.chunk_ids)}\n"
                f"Parent ID: {item.parent_id}\n\n"
                f"Text:\n{item.expanded_text}"
            )
        return "\n\n".join(blocks)

    def render_for_llm(self) -> str:
        """Render only citation-facing provenance and unchanged canonical evidence text.

        Internal chunk, parent, and document identifiers remain in this
        ContextPack and its citation map, but are deliberately omitted from
        the LLM's citation vocabulary.  This keeps E-handles as the sole
        identifiers the model should emit.
        """
        blocks: list[str] = []
        for item in self.evidence_items:
            article = " / ".join(value for value in (item.article_number, item.provision_number) if value) or ""
            page = f"{item.pdf_page_start or ''}{('-' + str(item.pdf_page_end)) if item.pdf_page_end and item.pdf_page_end != item.pdf_page_start else ''}"
            blocks.append(
                f"[EVIDENCE {item.citation_handle}]\n\n"
                f"Document: {item.document_title or 'Untitled source'}\n"
                f"Page: {page}\n"
                f"Article / Provision: {article}\n\n"
                f"Text:\n{item.expanded_text}"
            )
        return "\n\n".join(blocks)
