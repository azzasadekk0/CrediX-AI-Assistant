"""Benchmark the frozen reranker in one persistent CPU or CUDA process."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import replace
from pathlib import Path

from ..reranking.config import load_frozen_reranker_config
from ..reranking.reranked_retriever import RerankedRetriever, build_candidate_strategy


QUERY = "ما شروط منح التسهيلات الائتمانية؟"


def _summary(values: list[float]) -> dict[str, float]:
    return {"mean_seconds": statistics.mean(values), "median_seconds": statistics.median(values),
            "min_seconds": min(values), "max_seconds": max(values)}


def _timed(reranker: RerankedRetriever, operation):
    scorer = reranker._reranker
    scorer._synchronize()
    started = time.perf_counter()
    result = operation()
    scorer._synchronize()
    return time.perf_counter() - started, result


def benchmark(device: str) -> dict[str, object]:
    config = replace(load_frozen_reranker_config(), device=device)
    reranker = RerankedRetriever(config)
    warmup_seconds = reranker._reranker.warmup()
    first_seconds, first = _timed(reranker, lambda: reranker.retrieve(QUERY))
    warm_requests = [_timed(reranker, lambda: reranker.retrieve(QUERY))[0] for _ in range(5)]
    retrieval_seconds, raw = _timed(reranker, lambda: reranker._retriever.retrieve(QUERY, dense_top_k=10, bm25_top_k=10))
    candidates = build_candidate_strategy(raw.dense_results, raw.bm25_results, config)
    reranking_seconds, _ = _timed(reranker, lambda: reranker._reranker.score(QUERY, [candidate.text_original for candidate in candidates]))
    info = reranker.runtime_info()
    result: dict[str, object] = {
        "runtime": info, "model_load_seconds": reranker._reranker.load_seconds, "warmup_seconds": warmup_seconds,
        "first_request_seconds": first_seconds, "warm_requests": _summary(warm_requests),
        "retrieval_only_seconds": retrieval_seconds, "reranking_only_seconds": reranking_seconds,
        "total_seconds": retrieval_seconds + reranking_seconds, "candidates_reranked": len(candidates),
        "provenance_complete": all(hasattr(first.reranked_results[0], field) for field in (
            "chunk_id", "document_id", "text_original", "parent_id", "dense_rank", "bm25_rank",
            "dense_score", "bm25_score", "rrf_score", "retrieved_by", "reranker_score", "metadata")),
    }
    if info["resolved_device"] == "cuda":
        import torch
        result["peak_allocated_vram_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_reserved_vram_bytes"] = torch.cuda.max_memory_reserved()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ai_assistant/evaluation/reranker_runtime_benchmark_v1.json"))
    args = parser.parse_args()
    result = benchmark(args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
