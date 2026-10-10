"""One-shot held-out TEST evaluation for the frozen D10+B10/top-15 RAG branch.

No gold labels are supplied to the model.  They are used only after completed
generation to calculate evaluation metrics.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..context.builder import ContextBuilder
from ..context.config import ContextConfig, load_frozen_context_config
from ..generation.generator import GroundedGenerator
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from ..reranking.reranked_retriever import RerankedRetriever
from .evaluate_context import DEFAULT_CACHE as DEV_RERANK_CACHE, _candidate
from .evaluate_generation import DEFAULT_BUNDLE, DEFAULT_OUTPUT, DEFAULT_SPLITS, DEFAULT_V2_DATASET, DEFAULT_V2_META, _record, load_and_validate_splits, load_cases
from .evaluate_generation_citation_facing import _unknown_identifier_diagnosis
from .evaluate_generation_final_dev import _answerability, _multi_evidence, _repair_token_usage, _upstream_miss
from .evaluate_generation_provider import ROOT, _canonicalization_metrics, _safe_component, _write_checkpoint, _write_review
from .recover_generation import _metrics


TEST_RERANK_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "reranker_test_d10_b10_v1.json"
TEST_CONTEXT_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "generation_test_contexts_d10_b10_top15_v1.json"
DEV_REPORT = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_grounded_v1_citation_facing_d10_b10_top15_final_v1.json"


def _frozen_config() -> ContextConfig:
    config = load_frozen_context_config()
    expected = {"candidate_top_k": 15, "expansion_strategy": "chunk_only", "max_context_tokens": 2000}
    if any(getattr(config, key) != value for key, value in expected.items()):
        raise ValueError("TEST evaluator requires the approved frozen D10+B10/top-15 context configuration")
    return config


def _test_cases() -> tuple[list[Any], list[str], dict[str, Any]]:
    all_cases = load_cases(DEFAULT_V2_DATASET)
    meta = json.loads(DEFAULT_V2_META.read_text(encoding="utf-8"))
    splits = load_and_validate_splits(all_cases, DEFAULT_SPLITS, meta)
    ids = splits["test"] + splits["test_no_answer"]
    by_id = {case.query_id: case for case in all_cases}
    return [by_id[query_id] for query_id in ids], ids, meta


def _rerank_identity(meta: dict[str, Any], query_ids: list[str]) -> dict[str, Any]:
    dev_identity = json.loads(DEV_RERANK_CACHE.read_text(encoding="utf-8"))["identity"]
    return {"frozen_dev_reranker_identity": dev_identity, "split": "test", "query_ids": query_ids,
            "benchmark_version": meta["dataset_version"]}


def _load_or_create_test_reranks(cases: list[Any], query_ids: list[str], meta: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    identity = _rerank_identity(meta, query_ids)
    if TEST_RERANK_CACHE.exists():
        payload = json.loads(TEST_RERANK_CACHE.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("TEST reranker cache identity does not match frozen D10+B10")
    else:
        payload = {"schema_version": 1, "identity": identity, "split": "test", "results": [], "runtime": None}
    rows = {row["query_id"]: row for row in payload["results"]}
    retriever: RerankedRetriever | None = None
    TEST_RERANK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if case.query_id in rows:
            continue
        retriever = retriever or RerankedRetriever()
        result = retriever.retrieve(case.query)
        payload["runtime"] = retriever.runtime_info()
        rows[case.query_id] = {"query_id": case.query_id, "query": case.query,
                               "reranked_results": [item.to_dict() for item in result.reranked_results]}
        payload["results"] = [rows[query_id] for query_id in query_ids if query_id in rows]
        TEST_RERANK_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows, payload.get("runtime") or {}


def _context_identity(meta: dict[str, Any], config: ContextConfig, rerank_identity: dict[str, Any]) -> dict[str, Any]:
    return {"reranker_cache_identity": rerank_identity, "context_config": config.to_dict(),
            "split": "test", "benchmark_version": meta["dataset_version"]}


def _load_or_create_test_contexts(cases: list[Any], query_ids: list[str], reranks: dict[str, dict[str, Any]], meta: dict[str, Any], config: ContextConfig) -> dict[str, dict[str, Any]]:
    identity = _context_identity(meta, config, _rerank_identity(meta, query_ids))
    if TEST_CONTEXT_CACHE.exists():
        payload = json.loads(TEST_CONTEXT_CACHE.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("TEST ContextPack cache identity does not match frozen configuration")
    else:
        payload = {"schema_version": 1, "identity": identity, "split": "test", "results": []}
    rows = {row["query_id"]: row for row in payload["results"]}
    builder = ContextBuilder(config)
    TEST_CONTEXT_CACHE.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if case.query_id in rows:
            continue
        started = time.perf_counter()
        pack = builder.build_from_candidates(case.query, [_candidate(item) for item in reranks[case.query_id]["reranked_results"]])
        rows[case.query_id] = {"query_id": case.query_id, "query": case.query, "answerable": case.answerable,
                               "context_pack": pack.to_dict(), "context_build_seconds": time.perf_counter() - started,
                               "rerank_seconds_if_uncached": 0.0}
        payload["results"] = [rows[query_id] for query_id in query_ids if query_id in rows]
        TEST_CONTEXT_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rows


def _paths(settings: LLMSettings) -> tuple[Path, Path, Path]:
    stem = f"generation_test_{_safe_component(settings.provider)}_{_safe_component(settings.model)}_grounded_v1_citation_facing_d10_b10_top15_final_v1"
    return DEFAULT_OUTPUT / f"{stem}.jsonl", DEFAULT_OUTPUT / f"{stem}.json", ROOT / "artifacts" / "ai_assistant" / "cache" / f"{stem}.progress.json"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _dev_comparison(report: dict[str, Any]) -> dict[str, Any]:
    if not DEV_REPORT.exists():
        return {"available": False}
    dev = json.loads(DEV_REPORT.read_text(encoding="utf-8"))
    return {"available": True,
            "answerability": {"dev": dev.get("answerability", {}), "test": report.get("answerability", {})},
            "citation": {"dev": dev.get("citation", {}), "test": report.get("citation", {})}}


def run(*, pacing_seconds: float = 12.0) -> dict[str, Any]:
    config = _frozen_config()
    cases_list, query_ids, meta = _test_cases()
    cases = {case.query_id: case for case in cases_list}
    reranks, reranker_runtime = _load_or_create_test_reranks(cases_list, query_ids, meta)
    cache = _load_or_create_test_contexts(cases_list, query_ids, reranks, meta, config)
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    review_path, report_path, progress_path = _paths(settings)
    records = {row["query_id"]: {key: value for key, value in row.items() if key != "review_status"} for row in _load_rows(review_path)} if review_path.exists() else {}
    generator = GroundedGenerator(get_llm(settings))
    rate_limited = False
    attempted = 0
    for query_id in query_ids:
        prior = records.get(query_id)
        if prior and (prior.get("validation_success") or prior.get("error_type") != "LLMRateLimitError"):
            continue
        if attempted:
            time.sleep(pacing_seconds)
        record = _record(cases[query_id], cache[query_id], generator)
        record["trace"] = generator.last_trace
        records[query_id] = record
        _write_review(review_path, records, query_ids)
        _write_checkpoint(progress_path, records, query_ids, runtime)
        attempted += 1
        if record.get("error_type") == "LLMRateLimitError":
            rate_limited = True
            break
    ordered = [records[query_id] for query_id in query_ids if query_id in records]
    provider_failures = [row["query_id"] for row in ordered if (row.get("error_type") or "").startswith("LLM")]
    base_metrics = _metrics(ordered, cases) if ordered else {}
    token_usage = {**base_metrics.get("token_usage", {}), "repair_calls": _repair_token_usage(ordered)}
    if "dev_total_tokens" in token_usage:
        token_usage["test_total_tokens"] = token_usage.pop("dev_total_tokens")
    report = {
        "schema_version": 1, "split": "test", "runtime": runtime, "reranker_runtime": reranker_runtime,
        "prompt_version": "grounded_v1", "renderer": "citation_facing_v1", "canonicalization_enabled": True,
        "decorated_e_normalization_enabled": True, "frozen_context_configuration": config.to_dict(),
        "reranker_cache": str(TEST_RERANK_CACHE), "context_cache": str(TEST_CONTEXT_CACHE),
        "review_artifact": str(review_path), "progress_artifact": str(progress_path),
        "counts": {"expected_total": len(query_ids), "recorded": len(ordered), "answerable": sum(case.answerable for case in cases_list),
                   "no_answer": sum(not case.answerable for case in cases_list), "provider_failures": len(provider_failures),
                   "remaining": len(query_ids) - len(ordered)},
        "provider_failure_query_ids": provider_failures, "stopped_on_rate_limit": rate_limited,
        "answerability": _answerability(ordered, cases, cache) if len(ordered) == len(query_ids) else {},
        "citation": {**base_metrics.get("citation", {}), **_canonicalization_metrics(ordered)},
        "multi_evidence": _multi_evidence(ordered, cases, cache) if ordered else {},
        "language": base_metrics.get("arabic_language", {}), "latency_seconds": base_metrics.get("latency_seconds", {}),
        "token_usage": token_usage,
        "unknown_identifier_diagnosis": _unknown_identifier_diagnosis(ordered, cache) if ordered else {},
        "upstream_retrieval_context_misses": {query_id: _upstream_miss(query_id, cases, cache, reranks) for query_id in query_ids if cases[query_id].answerable},
        "generation_error_query_ids": [row["query_id"] for row in ordered if row.get("error_type") and not (row.get("error_type") or "").startswith("LLM")],
    }
    report["dev_vs_test"] = _dev_comparison(report) if len(ordered) == len(query_ids) else {}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacing-seconds", type=float, default=12.0)
    args = parser.parse_args()
    report = run(pacing_seconds=args.pacing_seconds)
    print(json.dumps({"counts": report["counts"], "stopped_on_rate_limit": report["stopped_on_rate_limit"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
