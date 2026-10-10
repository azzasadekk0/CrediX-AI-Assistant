"""Batched zero-shot scoring with the pinned BGE cross-encoder revision."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Iterable

from .config import RerankerConfig


@dataclass(frozen=True, slots=True)
class RerankerRuntimeInfo:
    """Machine-independent runtime diagnostics for the loaded reranker."""

    model_id: str
    model_revision: str
    requested_device: str
    resolved_device: str
    torch_version: str
    torch_cuda_version: str | None
    cuda_available: bool
    gpu_name: str | None
    batch_size: int
    max_length: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_device(requested_device: str, torch_module=None) -> str:
    """Resolve an explicit device request without silently weakening CUDA requests."""
    if requested_device not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be 'cpu', 'cuda', or 'auto'")
    if torch_module is None:
        import torch as torch_module
    available = torch_module.cuda.is_available()
    if requested_device == "cuda" and not available:
        raise ValueError("CUDA was requested for reranking but is not available")
    return "cuda" if requested_device in {"cuda", "auto"} and available else "cpu"


class CrossEncoderReranker:
    """Load a sequence-classification cross encoder and score query/passage pairs."""

    def __init__(self, config: RerankerConfig, *, tokenizer=None, model=None) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.model = model
        self.load_seconds = 0.0
        self._torch = None
        self.resolved_device: str | None = None
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be supplied together")
        if tokenizer is None:
            self._load()
        else:
            import torch
            self._torch = torch
            self.resolved_device = resolve_device(config.device, torch)

    def _load(self) -> None:
        started = time.perf_counter()
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.resolved_device = resolve_device(self.config.device, torch)
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id, revision=self.config.model_revision)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.config.model_id, revision=self.config.model_revision)
        self.model.to(self.resolved_device)
        self.model.eval()
        self.load_seconds = time.perf_counter() - started

    def score(self, query: str, passages: Iterable[str]) -> list[float]:
        """Return raw logits for query/text_original pairs in the supplied order."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        texts = list(passages)
        if not all(isinstance(text, str) and text for text in texts):
            raise ValueError("passages must contain non-empty strings")
        if not texts:
            return []
        torch = self._torch
        if torch is None:
            import torch as torch_module
            torch = torch_module
            self._torch = torch
        scores: list[float] = []
        for offset in range(0, len(texts), self.config.batch_size):
            batch = texts[offset:offset + self.config.batch_size]
            encoded = self.tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                                     max_length=self.config.max_length, return_tensors="pt")
            encoded = {name: value.to(self.resolved_device or "cpu") for name, value in encoded.items()}
            with torch.no_grad():
                logits = self.model(**encoded).logits
            scores.extend(float(value) for value in logits.reshape(-1).detach().cpu().tolist())
        return scores

    def warmup(self) -> float:
        """Initialize the selected execution path without changing production results."""
        started = time.perf_counter()
        self.score("warmup", ["warmup"])
        self._synchronize()
        return time.perf_counter() - started

    def _synchronize(self) -> None:
        if self.resolved_device == "cuda":
            assert self._torch is not None
            self._torch.cuda.synchronize()

    def runtime_info(self) -> RerankerRuntimeInfo:
        """Return device and version diagnostics without exposing local cache paths."""
        import torch

        resolved = self.resolved_device or resolve_device(self.config.device, torch)
        available = torch.cuda.is_available()
        return RerankerRuntimeInfo(
            model_id=self.config.model_id, model_revision=self.config.model_revision,
            requested_device=self.config.device, resolved_device=resolved,
            torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
            cuda_available=available, gpu_name=torch.cuda.get_device_name(0) if available else None,
            batch_size=self.config.batch_size, max_length=self.config.max_length,
        )

    def model_identity(self) -> dict[str, str | float]:
        """Return immutable model identity and runtime versions for experiment artifacts."""
        import torch
        import transformers

        resolved = getattr(getattr(self.model, "config", None), "_commit_hash", None) or self.config.model_revision
        return {"model_id": self.config.model_id, "resolved_revision": resolved,
                "transformers_version": transformers.__version__, "torch_version": torch.__version__,
                "device": self.resolved_device or self.config.device, "model_load_seconds": self.load_seconds}
