"""Validated and frozen configuration for deterministic context construction."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


EXPANSION_STRATEGIES = {"chunk_only", "parent", "chunk_plus_parent", "parent_deduplicated", "neighbor_1"}
TOKEN_ESTIMATION_METHODS = {"whitespace_approx_v1"}
FROZEN_CONTEXT_CONFIG_PATH = Path(__file__).with_name("context_config.json")


@dataclass(frozen=True, slots=True)
class ContextConfig:
    """Context settings selected only from the DEV split."""

    candidate_top_k: int = 5
    expansion_strategy: str = "chunk_only"
    max_context_tokens: int = 2000
    deduplicate_parents: bool = False
    neighbor_expansion: bool = False
    token_estimation_method: str = "whitespace_approx_v1"
    selection_split: str = "dev"
    golden_dataset_version: str = "golden_retrieval_v2"

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_top_k, int) or isinstance(self.candidate_top_k, bool) or self.candidate_top_k < 1:
            raise ValueError("candidate_top_k must be a positive integer")
        if self.expansion_strategy not in EXPANSION_STRATEGIES:
            raise ValueError(f"unsupported expansion_strategy: {self.expansion_strategy}")
        if not isinstance(self.max_context_tokens, int) or isinstance(self.max_context_tokens, bool) or self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be a positive integer")
        if not isinstance(self.deduplicate_parents, bool) or not isinstance(self.neighbor_expansion, bool):
            raise ValueError("deduplicate_parents and neighbor_expansion must be booleans")
        if self.token_estimation_method not in TOKEN_ESTIMATION_METHODS:
            raise ValueError(f"unsupported token_estimation_method: {self.token_estimation_method}")
        if self.selection_split != "dev":
            raise ValueError("selection_split must be 'dev' for a frozen context configuration")
        if not isinstance(self.golden_dataset_version, str) or not self.golden_dataset_version:
            raise ValueError("golden_dataset_version must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_frozen_context_config(path: Path = FROZEN_CONTEXT_CONFIG_PATH) -> ContextConfig:
    """Load the complete frozen configuration, failing clearly if it is unusable."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"frozen context configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"frozen context configuration is invalid JSON: {path}") from exc
    fields = ContextConfig.__dataclass_fields__
    missing = sorted(set(fields).difference(data))
    if missing:
        raise ValueError(f"frozen context configuration is missing: {', '.join(missing)}")
    return ContextConfig(**{name: data[name] for name in fields})
