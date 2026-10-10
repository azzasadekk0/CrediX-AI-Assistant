"""Validated configuration for the standalone BGE cross-encoder reranker."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path


MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
CANDIDATE_STRATEGIES = {"dense_top_20", "rrf_top_30", "union_d10_b10", "union_d20_b10", "union_d20_b20"}
FROZEN_RERANKER_CONFIG_PATH = Path(__file__).with_name("reranker_config.json")


@dataclass(frozen=True, slots=True)
class RerankerConfig:
    """Explicit, reproducible cross-encoder and candidate-source settings."""

    model_id: str = MODEL_ID
    model_revision: str = MODEL_REVISION
    device: str = "auto"
    batch_size: int = 8
    max_length: int = 512
    candidate_strategy: str = "union_d20_b20"
    dense_candidate_k: int = 20
    bm25_candidate_k: int = 20
    rerank_candidate_limit: int = 40
    output_top_k: int = 30

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_revision:
            raise ValueError("model_id and model_revision must be non-empty")
        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
        if self.candidate_strategy not in CANDIDATE_STRATEGIES:
            raise ValueError(f"unsupported candidate_strategy: {self.candidate_strategy}")
        for name in ("batch_size", "max_length", "dense_candidate_k", "bm25_candidate_k", "rerank_candidate_limit", "output_top_k"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.output_top_k > self.rerank_candidate_limit:
            raise ValueError("output_top_k must not exceed rerank_candidate_limit")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_frozen_reranker_config(path: Path = FROZEN_RERANKER_CONFIG_PATH) -> RerankerConfig:
    """Load only known, validated settings from the frozen DEV-selected JSON file."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"frozen reranker configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"frozen reranker configuration is invalid JSON: {path}") from exc
    fields = RerankerConfig.__dataclass_fields__
    missing = sorted(set(fields).difference(data))
    if missing:
        raise ValueError(f"frozen reranker configuration is missing: {', '.join(missing)}")
    values = {name: data[name] for name in fields}
    requested = os.getenv("CREDIX_RERANKER_DEVICE")
    if requested is not None:
        values["device"] = requested.strip().lower()
    return RerankerConfig(**values)
