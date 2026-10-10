"""Runtime construction of deterministic, grounded ContextPacks."""
from __future__ import annotations

from ..models import ContextPack, RerankedCandidate
from ..reranking.reranked_retriever import RerankedRetriever
from .config import ContextConfig, load_frozen_context_config
from .expander import ContextExpander
from .selector import EvidenceSelector


class ContextBuilder:
    """Use frozen reranking and frozen context settings unless explicitly injected."""

    def __init__(self, config: ContextConfig | None = None, *, reranked_retriever: RerankedRetriever | None = None,
                 expander: ContextExpander | None = None, selector: EvidenceSelector | None = None) -> None:
        self.config = config or load_frozen_context_config()
        # Keep one retriever/reranker for the lifetime of this context builder,
        # but construct it lazily.  Loading the cross-encoder may move several
        # GB to CUDA; rebuilding it for every query caused severe latency and
        # GPU contention, while eager loading would make unit tests and API
        # startup pay the cost even when retrieval is not used.
        self._reranked_retriever = reranked_retriever
        self._expander = expander or ContextExpander()
        self._selector = selector or EvidenceSelector()

    def build(self, query: str) -> ContextPack:
        if self._reranked_retriever is None:
            self._reranked_retriever = RerankedRetriever()
        result = self._reranked_retriever.retrieve(query)
        return self.build_from_candidates(result.original_query, result.reranked_results)

    def build_from_candidates(self, query: str, candidates: list[RerankedCandidate]) -> ContextPack:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        # ``neighbor_expansion`` was part of the frozen schema but previously
        # had no runtime effect.  Make it operational while preserving the
        # configured strategy for all existing releases.  Neighbor expansion
        # is source-only and keeps every adjacent chunk in the citation map.
        expansion_strategy = self.config.expansion_strategy
        if self.config.neighbor_expansion and expansion_strategy == "chunk_only":
            expansion_strategy = "neighbor_1"
        expanded = [self._expander.expand(candidate, rank, expansion_strategy)
                    for rank, candidate in enumerate(candidates[:self.config.candidate_top_k], start=1)]
        selected, total_tokens, duplicates_removed = self._selector.select(expanded, self.config)
        chunk_ids = list(dict.fromkeys(chunk_id for item in selected for chunk_id in item.chunk_ids))
        parent_ids = list(dict.fromkeys(
            parent_id
            for item in selected
            for parent_id in item.metadata.get("source_parent_ids", [item.parent_id])
        ))
        document_ids = list(dict.fromkeys(item.document_id for item in selected))
        citation_map = {
            item.citation_handle or "": {
                "chunk_ids": item.chunk_ids, "parent_id": item.parent_id, "document_id": item.document_id,
                "parent_ids": item.metadata.get("source_parent_ids", [item.parent_id]),
                "chunk_parent_ids": item.metadata.get("chunk_parent_ids", {}),
                "document_title": item.document_title, "page_start": item.pdf_page_start, "page_end": item.pdf_page_end,
                "article_number": item.article_number, "provision_number": item.provision_number,
            } for item in selected
        }
        configuration = {**self.config.to_dict(), "duplicate_blocks_removed": duplicates_removed}
        return ContextPack(query, selected, chunk_ids, parent_ids, document_ids, citation_map, len(selected),
                           len(parent_ids), len(document_ids), total_tokens, configuration)
