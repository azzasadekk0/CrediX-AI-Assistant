"""RRF orchestration over the existing independent CrediX retrievers."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from ..models import FusedRetrievalResult
from .hybrid_retriever import HybridRetriever
from .rrf import RRFConfig, fuse_ranked_candidates


FROZEN_CONFIG_PATH = Path(__file__).with_name("rrf_config.json")


def load_frozen_config(path: Path = FROZEN_CONFIG_PATH) -> RRFConfig:
    """Load the DEV-selected configuration without silently changing settings."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"frozen RRF configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"frozen RRF configuration is invalid JSON: {path}") from exc
    fields = RRFConfig.__dataclass_fields__
    missing = sorted(set(fields) - set(data))
    if missing:
        raise ValueError(f"frozen RRF configuration is missing: {', '.join(missing)}")
    return RRFConfig(**{name: data[name] for name in fields})


class FusedRetriever:
    """Run Dense and BM25 once, then deterministically fuse their ranked lists."""

    def __init__(self, config: RRFConfig | None = None, *, retriever: HybridRetriever | None = None) -> None:
        self.config = config or load_frozen_config()
        self._retriever = retriever or HybridRetriever()

    def retrieve(
        self,
        query: str,
        dense_candidate_k: int | None = None,
        bm25_candidate_k: int | None = None,
        fusion_top_k: int | None = None,
        rrf_k: int | None = None,
        dense_weight: float | None = None,
        bm25_weight: float | None = None,
    ) -> FusedRetrievalResult:
        """Return raw source lists and a de-duplicated RRF candidate list."""
        overrides = {name: value for name, value in {
            "dense_candidate_k": dense_candidate_k,
            "bm25_candidate_k": bm25_candidate_k,
            "fusion_top_k": fusion_top_k,
            "rrf_k": rrf_k,
            "dense_weight": dense_weight,
            "bm25_weight": bm25_weight,
        }.items() if value is not None}
        config = replace(self.config, **overrides)
        raw = self._retriever.retrieve(
            query, dense_top_k=config.dense_candidate_k, bm25_top_k=config.bm25_candidate_k,
        )
        return FusedRetrievalResult(
            original_query=query,
            dense_results=raw.dense_results,
            bm25_results=raw.bm25_results,
            fused_results=fuse_ranked_candidates(raw.dense_results, raw.bm25_results, config),
            configuration=config.to_dict(),
        )
