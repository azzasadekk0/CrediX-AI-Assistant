"""Independent BGE-M3/FAISS and BM25 retrieval over the certified RAG bundle."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

from ..config import RetrievalConfig
from ..models import HybridRetrievalResult, RetrievalCandidate, RetrieverName


class HybridRetriever:
    """Expose the existing CrediX retrievers without fusion or reranking."""

    def __init__(self, config: RetrievalConfig | None = None, *, adapter: ModuleType | None = None) -> None:
        self.config = config or RetrievalConfig()
        self._adapter = adapter or import_module("src.rag.retrieval_adapter")

    def retrieve_dense(self, query: str, top_k: int | None = None) -> list[RetrievalCandidate]:
        """Return BGE-M3 + FAISS results, retaining the raw FAISS scores."""
        self._validate_query(query)
        limit = self._resolve_top_k(top_k, self.config.dense_top_k, "top_k")
        embedding = self._adapter.embed_query(query)
        raw_results = self._adapter.dense_search(embedding, top_k=limit)
        return [self._to_candidate(result, "dense") for result in raw_results]

    def retrieve_bm25(self, query: str, top_k: int | None = None) -> list[RetrievalCandidate]:
        """Return BM25 results, retaining the raw BM25 scores."""
        self._validate_query(query)
        limit = self._resolve_top_k(top_k, self.config.bm25_top_k, "top_k")
        raw_results = self._adapter.bm25_search(query, top_k=limit)
        return [self._to_candidate(result, "bm25") for result in raw_results]

    def retrieve(
        self,
        query: str,
        dense_top_k: int | None = None,
        bm25_top_k: int | None = None,
    ) -> HybridRetrievalResult:
        """Run both retrievers and keep their ranked lists completely separate."""
        self._validate_query(query)
        return HybridRetrievalResult(
            original_query=query,
            dense_results=self.retrieve_dense(query, dense_top_k),
            bm25_results=self.retrieve_bm25(query, bm25_top_k),
        )

    @staticmethod
    def _validate_query(query: str) -> None:
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if not query.strip():
            raise ValueError("query must not be empty or whitespace-only")

    @staticmethod
    def _resolve_top_k(value: int | None, default: int, name: str) -> int:
        resolved = default if value is None else value
        if not isinstance(resolved, int) or isinstance(resolved, bool) or resolved < 1:
            raise ValueError(f"{name} must be a positive integer")
        return resolved

    def _to_candidate(self, raw_result: dict[str, Any], retriever: RetrieverName) -> RetrievalCandidate:
        """Adapt an adapter result and verify its chunk and parent references exist."""
        try:
            chunk_id = raw_result["chunk_id"]
            document_id = raw_result["document_id"]
            text_original = raw_result["text_original"]
            score = raw_result["score"]
            metadata = dict(raw_result["metadata"])
        except (KeyError, TypeError) as exc:
            raise ValueError("retrieval adapter returned an invalid result") from exc

        canonical_chunk = self._adapter.get_chunk(chunk_id)
        parent_id = metadata.get("parent_id")
        if not isinstance(parent_id, str) or not parent_id:
            raise ValueError(f"retrieval result {chunk_id} is missing parent_id")
        self._adapter.get_parent_context(parent_id)

        return RetrievalCandidate(
            chunk_id=chunk_id,
            document_id=document_id,
            document_title=metadata.get("document_title", canonical_chunk.get("document_title")),
            text_original=text_original,
            parent_id=parent_id,
            pdf_page_start=metadata.get("pdf_page_start"),
            pdf_page_end=metadata.get("pdf_page_end"),
            article_number=metadata.get("article_number"),
            provision_number=metadata.get("provision_number"),
            score=float(score),
            retriever=retriever,
            metadata=metadata,
        )
