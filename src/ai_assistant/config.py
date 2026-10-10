"""Configuration for the standalone retrieval baseline."""
from __future__ import annotations

import os
from dataclasses import dataclass


def _positive_int_from_env(name: str, default: int) -> int:
    """Read a positive integer setting while failing clearly on invalid input."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Top-k defaults for the independent dense and lexical retrievers."""

    dense_top_k: int = _positive_int_from_env("CREDIX_DENSE_TOP_K", 20)
    bm25_top_k: int = _positive_int_from_env("CREDIX_BM25_TOP_K", 20)

    def __post_init__(self) -> None:
        for name, value in (("dense_top_k", self.dense_top_k), ("bm25_top_k", self.bm25_top_k)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
