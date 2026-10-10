"""Candidate collection and cross-encoder reranking without changing retrieval."""
from __future__ import annotations

from dataclasses import replace
import logging
import re
from typing import Iterable

from ..models import (FusedRetrievalCandidate, HybridRetrievalResult, RerankedCandidate,
                      RerankedRetrievalResult, RetrievalCandidate)
from ..retrieval.fused_retriever import load_frozen_config
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.rrf import RRFConfig, fuse_ranked_candidates
from .config import RerankerConfig, load_frozen_reranker_config
from .cross_encoder import CrossEncoderReranker


logger = logging.getLogger(__name__)


# Mixed-language and numeric regulatory questions are disproportionately likely
# to miss the lexical/semantic candidate pool: the corpus is Arabic while user
# queries often contain terms such as ``credit registry`` or thresholds such as
# ``300``.  Keep the normal DEV-frozen pool for ordinary queries, but give these
# high-risk queries a wider first-stage pool.  This is deliberately a small,
# deterministic trigger rather than a free-form query rewrite.
_ADAPTIVE_DOMAIN_TERMS = frozenset({
    "credit", "registry", "check", "loan", "loans", "creditworthiness",
    "cbe", "msme", "limit", "limits", "pd", "lgd",
})
_LATIN_TOKEN_RE = re.compile(r"[a-z][a-z0-9_-]*", re.IGNORECASE)
# Include Arabic-Indic and Eastern Arabic-Indic (Persian) digits.  The BM25
# normalizer folds both ranges to ASCII, so the adaptive trigger must recognize
# both spellings as well; otherwise a query such as ``۳۰۰ ألف`` silently uses
# the narrow D10/B10 pool even though its ASCII equivalent widens correctly.
_NUMERIC_TOKEN_RE = re.compile(r"(?<!\w)[0-9٠-٩۰-۹]+(?!\w)")


def needs_adaptive_candidate_pool(query: str) -> bool:
    """Return whether a query should use the wider candidate pool.

    The trigger is intentionally conservative and explainable.  It catches
    domain-specific English/code-switching and explicit numeric constraints,
    while not changing behavior for ordinary Arabic questions.
    """
    if not isinstance(query, str):
        return False
    latin_terms = {token.lower() for token in _LATIN_TOKEN_RE.findall(query)}
    return bool(latin_terms & _ADAPTIVE_DOMAIN_TERMS) or bool(_NUMERIC_TOKEN_RE.search(query))


def build_candidate_strategy(dense_results: list[RetrievalCandidate], bm25_results: list[RetrievalCandidate],
                             config: RerankerConfig) -> list[FusedRetrievalCandidate]:
    """Construct one complete strategy in memory, retaining all source provenance."""
    dense = dense_results[:config.dense_candidate_k]
    bm25 = bm25_results[:config.bm25_candidate_k]
    frozen = load_frozen_config()
    if config.candidate_strategy == "dense_top_20":
        return fuse_ranked_candidates(dense, [], RRFConfig(len(dense) or 1, 1, config.rerank_candidate_limit,
                                                            frozen.rrf_k, frozen.dense_weight, frozen.bm25_weight))
    fusion_config = replace(frozen, dense_candidate_k=len(dense) or 1, bm25_candidate_k=len(bm25) or 1,
                            fusion_top_k=len({candidate.chunk_id for candidate in dense + bm25}) or 1)
    fused = fuse_ranked_candidates(dense, bm25, fusion_config)
    if config.candidate_strategy == "rrf_top_30":
        return fused[:config.rerank_candidate_limit]
    # Raw-union strategies deliberately retain source-list order rather than RRF order.
    by_id = {candidate.chunk_id: candidate for candidate in fused}
    ordered_ids = dict.fromkeys(candidate.chunk_id for candidate in dense + bm25)
    return [by_id[chunk_id] for chunk_id in ordered_ids][:config.rerank_candidate_limit]


class RerankedRetriever:
    """Retrieve candidates once and let the cross encoder determine final rank."""

    def __init__(self, config: RerankerConfig | None = None, *, retriever: HybridRetriever | None = None,
                 reranker: CrossEncoderReranker | None = None) -> None:
        self.config = config or load_frozen_reranker_config()
        self._retriever = retriever or HybridRetriever()
        self._reranker = reranker
        self._reranker_error: str | None = None
        self._retrieval_error: str | None = None
        if self._reranker is None:
            try:
                self._reranker = CrossEncoderReranker(self.config)
            except Exception as exc:
                # The cross-encoder is an optimization layer, not the source
                # of truth. CPU/serverless hosts may not have CUDA, torch,
                # transformers, or room for the pinned model. Keep hybrid
                # retrieval available and expose degraded mode in diagnostics.
                self._reranker_error = f"{type(exc).__name__}: {exc}"
                logger.warning("Cross-encoder unavailable; using fused retrieval fallback: %s", self._reranker_error)

    def retrieve(self, query: str, *, output_top_k: int | None = None) -> RerankedRetrievalResult:
        """Preserve the exact query, candidate provenance, and deterministic ranking."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        limit = self.config.output_top_k if output_top_k is None else output_top_k
        adaptive = needs_adaptive_candidate_pool(query)
        effective_limit = max(self.config.rerank_candidate_limit, 40) if adaptive else self.config.rerank_candidate_limit
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > effective_limit:
            raise ValueError("output_top_k must be positive and not exceed rerank_candidate_limit")
        # The frozen DEV configuration remains the default.  For known
        # code-switching/numeric failure modes, widen only the first-stage
        # candidate pool; output_top_k and the frozen model remain unchanged.
        effective_config = self.config
        if adaptive:
            effective_config = replace(
                self.config,
                candidate_strategy="union_d20_b20",
                dense_candidate_k=max(self.config.dense_candidate_k, 20),
                bm25_candidate_k=max(self.config.bm25_candidate_k, 20),
                rerank_candidate_limit=max(self.config.rerank_candidate_limit, 40),
                output_top_k=limit,
            )
        try:
            raw = self._retriever.retrieve(query, dense_top_k=effective_config.dense_candidate_k,
                                           bm25_top_k=effective_config.bm25_candidate_k)
        except Exception as exc:
            # Dense retrieval needs the optional encoder/FAISS stack and its
            # model cache. A CPU/serverless deployment can still answer from
            # the certified lexical index when that optional layer is absent.
            retrieve_bm25 = getattr(self._retriever, "retrieve_bm25", None)
            if not callable(retrieve_bm25):
                raise
            bm25_results = retrieve_bm25(query, top_k=effective_config.bm25_candidate_k)
            if not bm25_results:
                raise
            self._retrieval_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Dense retrieval failed; using BM25 fallback: %s", self._retrieval_error)
            raw = HybridRetrievalResult(query, [], bm25_results)
        candidates = build_candidate_strategy(raw.dense_results, raw.bm25_results, effective_config)
        if self._reranker is None:
            reranked = rerank_candidates(candidates, self._fallback_scores(candidates))
        else:
            try:
                scores = self._reranker.score(query, [candidate.text_original for candidate in candidates])
            except Exception as exc:
                # Loading may succeed while the first batch still fails (for
                # example because of a CUDA OOM). Degrade to the same audited
                # fused order instead of converting a valid query to a 500.
                self._reranker_error = f"{type(exc).__name__}: {exc}"
                self._reranker = None
                logger.warning("Cross-encoder scoring failed; using fused retrieval fallback: %s", self._reranker_error)
                scores = self._fallback_scores(candidates)
            reranked = rerank_candidates(candidates, scores)
        configuration = self.config.to_dict()
        configuration.update({
            "adaptive_candidate_pool": adaptive,
            "effective_dense_candidate_k": effective_config.dense_candidate_k,
            "effective_bm25_candidate_k": effective_config.bm25_candidate_k,
            "effective_rerank_candidate_limit": effective_config.rerank_candidate_limit,
            "reranker_available": self._reranker is not None,
            "reranker_fallback": self._reranker is None,
            "retrieval_degraded": self._retrieval_error is not None,
        })
        if self._reranker_error:
            configuration["reranker_error"] = self._reranker_error
        if self._retrieval_error:
            configuration["retrieval_error"] = self._retrieval_error
        return RerankedRetrievalResult(query, raw.dense_results, raw.bm25_results, candidates, reranked[:limit], configuration)

    @staticmethod
    def _fallback_scores(candidates: list[FusedRetrievalCandidate]) -> list[float]:
        """Return deterministic RRF scores when cross-encoder ranking is unavailable."""

        return [
            candidate.rrf_score if candidate.rrf_score is not None else -float(index)
            for index, candidate in enumerate(candidates)
        ]

    def runtime_info(self) -> dict[str, object]:
        """Expose reranker runtime diagnostics while retaining a single model instance."""
        if self._reranker is None:
            return {
                "model_id": self.config.model_id,
                "model_revision": self.config.model_revision,
                "requested_device": self.config.device,
                "resolved_device": "unavailable",
                "reranker_available": False,
                "reranker_error": self._reranker_error,
                "retrieval_degraded": self._retrieval_error is not None,
                "retrieval_error": self._retrieval_error,
            }
        return {**self._reranker.runtime_info().to_dict(), "reranker_available": True, "reranker_fallback": False}


def _to_reranked(candidate: FusedRetrievalCandidate, score: float, original_rank: int) -> RerankedCandidate:
    return RerankedCandidate(
        chunk_id=candidate.chunk_id, document_id=candidate.document_id, document_title=candidate.document_title,
        text_original=candidate.text_original, parent_id=candidate.parent_id,
        pdf_page_start=candidate.pdf_page_start, pdf_page_end=candidate.pdf_page_end,
        article_number=candidate.article_number, provision_number=candidate.provision_number,
        reranker_score=float(score), original_candidate_rank=original_rank, rrf_score=candidate.rrf_score,
        dense_rank=candidate.dense_rank, bm25_rank=candidate.bm25_rank,
        dense_score=candidate.dense_score, bm25_score=candidate.bm25_score,
        retrieved_by=list(candidate.retrieved_by), metadata=dict(candidate.metadata),
    )


def rerank_candidates(candidates: list[FusedRetrievalCandidate], scores: list[float]) -> list[RerankedCandidate]:
    """Map one raw score per candidate and apply the documented stable tie-break."""
    if len(scores) != len(candidates):
        raise ValueError("reranker returned a score count different from candidate count")
    reranked = [_to_reranked(candidate, score, rank) for rank, (candidate, score) in enumerate(zip(candidates, scores), start=1)]
    reranked.sort(key=lambda item: (-item.reranker_score, item.original_candidate_rank,
                                    -(item.rrf_score if item.rrf_score is not None else float("-inf")),
                                    item.dense_rank if item.dense_rank is not None else float("inf"), item.chunk_id))
    return reranked
