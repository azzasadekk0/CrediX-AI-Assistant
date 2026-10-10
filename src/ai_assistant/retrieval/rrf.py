"""Pure deterministic Reciprocal Rank Fusion over existing ranked lists."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from ..models import FusedRetrievalCandidate, RetrievalCandidate


@dataclass(frozen=True, slots=True)
class RRFConfig:
    """Validated rank-fusion settings; candidate pools are supplied by callers."""

    dense_candidate_k: int = 20
    bm25_candidate_k: int = 20
    fusion_top_k: int = 30
    rrf_k: int = 60
    dense_weight: float = 1.0
    bm25_weight: float = 1.0

    def __post_init__(self) -> None:
        for name in ("dense_candidate_k", "bm25_candidate_k", "fusion_top_k", "rrf_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("dense_weight", "bm25_weight"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be greater than zero")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def fuse_ranked_candidates(
    dense_results: Iterable[RetrievalCandidate],
    bm25_results: Iterable[RetrievalCandidate],
    config: RRFConfig,
) -> list[FusedRetrievalCandidate]:
    """Fuse ranked candidates using only their 1-based positions, never raw scores.

    Exact-score ties are resolved by best rank, then Dense rank, BM25 rank, and
    chunk ID. This gives stable results across runs and Python versions.
    """
    observations: dict[str, dict[str, object]] = {}
    _observe(observations, dense_results, "dense", config)
    _observe(observations, bm25_results, "bm25", config)

    fused = [_to_fused(value) for value in observations.values()]
    fused.sort(
        key=lambda item: (
            -item.rrf_score,
            min(rank for rank in (item.dense_rank, item.bm25_rank) if rank is not None),
            item.dense_rank if item.dense_rank is not None else float("inf"),
            item.bm25_rank if item.bm25_rank is not None else float("inf"),
            item.chunk_id,
        )
    )
    return fused[:config.fusion_top_k]


def _observe(
    observations: dict[str, dict[str, object]],
    results: Iterable[RetrievalCandidate],
    name: str,
    config: RRFConfig,
) -> None:
    weight = config.dense_weight if name == "dense" else config.bm25_weight
    for rank, candidate in enumerate(results, start=1):
        record = observations.setdefault(
            candidate.chunk_id,
            {"candidate": candidate, "dense_rank": None, "bm25_rank": None,
             "dense_score": None, "bm25_score": None, "rrf_score": 0.0},
        )
        rank_key = f"{name}_rank"
        # A malformed source list with a duplicate chunk retains its first rank.
        if record[rank_key] is not None:
            continue
        record[rank_key] = rank
        record[f"{name}_score"] = candidate.score
        record["rrf_score"] = float(record["rrf_score"]) + weight / (config.rrf_k + rank)
        if name == "dense":
            record["candidate"] = candidate


def _to_fused(record: dict[str, object]) -> FusedRetrievalCandidate:
    candidate = record["candidate"]
    assert isinstance(candidate, RetrievalCandidate)
    dense_rank = record["dense_rank"]
    bm25_rank = record["bm25_rank"]
    return FusedRetrievalCandidate(
        chunk_id=candidate.chunk_id,
        document_id=candidate.document_id,
        document_title=candidate.document_title,
        text_original=candidate.text_original,
        parent_id=candidate.parent_id,
        pdf_page_start=candidate.pdf_page_start,
        pdf_page_end=candidate.pdf_page_end,
        article_number=candidate.article_number,
        provision_number=candidate.provision_number,
        rrf_score=float(record["rrf_score"]),
        dense_rank=dense_rank if isinstance(dense_rank, int) else None,
        bm25_rank=bm25_rank if isinstance(bm25_rank, int) else None,
        dense_score=float(record["dense_score"]) if record["dense_score"] is not None else None,
        bm25_score=float(record["bm25_score"]) if record["bm25_score"] is not None else None,
        retrieved_by=[name for name, rank in (("dense", dense_rank), ("bm25", bm25_rank)) if rank is not None],
        metadata=dict(candidate.metadata),
    )
