"""Dependency-light chunk-level ranking metrics for retrieval evaluation."""
from __future__ import annotations

import math
from typing import Iterable


def relevant_ranks(retrieved_chunk_ids: Iterable[str], relevant_chunk_ids: Iterable[str], k: int = 10) -> list[int]:
    """Return 1-based ranks of unique relevant chunks within the first ``k`` results."""
    if k < 1:
        raise ValueError("k must be positive")
    relevant = set(relevant_chunk_ids)
    seen: set[str] = set()
    ranks: list[int] = []
    for rank, chunk_id in enumerate(list(retrieved_chunk_ids)[:k], start=1):
        if chunk_id in relevant and chunk_id not in seen:
            ranks.append(rank)
            seen.add(chunk_id)
    return ranks


def recall_at_k(retrieved_chunk_ids: Iterable[str], relevant_chunk_ids: Iterable[str], k: int) -> float:
    """Fraction of all relevant chunks retrieved in the first ``k`` ranks."""
    relevant = set(relevant_chunk_ids)
    return len(relevant_ranks(retrieved_chunk_ids, relevant, k)) / len(relevant) if relevant else 0.0


def hit_rate_at_k(retrieved_chunk_ids: Iterable[str], relevant_chunk_ids: Iterable[str], k: int) -> float:
    """Whether at least one relevant chunk appears in the first ``k`` ranks."""
    return float(bool(relevant_ranks(retrieved_chunk_ids, relevant_chunk_ids, k)))


def reciprocal_rank_at_k(retrieved_chunk_ids: Iterable[str], relevant_chunk_ids: Iterable[str], k: int = 10) -> float:
    """Reciprocal rank of the first relevant chunk, or zero when none is retrieved."""
    ranks = relevant_ranks(retrieved_chunk_ids, relevant_chunk_ids, k)
    return 1.0 / ranks[0] if ranks else 0.0


def ndcg_at_k(retrieved_chunk_ids: Iterable[str], relevant_chunk_ids: Iterable[str], k: int = 10) -> float:
    """Binary-relevance normalized discounted cumulative gain at ``k``."""
    relevant = set(relevant_chunk_ids)
    if not relevant:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1) for rank in relevant_ranks(retrieved_chunk_ids, relevant, k))
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def ranking_metrics(retrieved_chunk_ids: Iterable[str], relevant_chunk_ids: Iterable[str]) -> dict[str, float]:
    """Compute all standard baseline metrics for a single answerable query."""
    retrieved = list(retrieved_chunk_ids)
    relevant = list(relevant_chunk_ids)
    return {
        "recall_at_1": recall_at_k(retrieved, relevant, 1),
        "recall_at_3": recall_at_k(retrieved, relevant, 3),
        "recall_at_5": recall_at_k(retrieved, relevant, 5),
        "recall_at_10": recall_at_k(retrieved, relevant, 10),
        "mrr_at_10": reciprocal_rank_at_k(retrieved, relevant, 10),
        "ndcg_at_10": ndcg_at_k(retrieved, relevant, 10),
        "hit_rate_at_1": hit_rate_at_k(retrieved, relevant, 1),
        "hit_rate_at_5": hit_rate_at_k(retrieved, relevant, 5),
        "hit_rate_at_10": hit_rate_at_k(retrieved, relevant, 10),
    }


def mean_metrics(metric_rows: Iterable[dict[str, float]]) -> dict[str, float]:
    """Calculate a macro average across answerable queries."""
    rows = list(metric_rows)
    if not rows:
        return {}
    return {name: sum(row[name] for row in rows) / len(rows) for name in rows[0]}
