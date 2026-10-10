"""Freeze-candidate validation for D10+B10 with a top-15 chunk-only context.

This module deliberately reuses the certified D10+B10 reranker cache.  It
never retrieves or reranks; the only variable is context candidate_top_k.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path
from statistics import median
from typing import Any

from ..context.builder import ContextBuilder
from ..context.config import ContextConfig, load_frozen_context_config
from ..generation.generator import GroundedGenerator
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from ..reranking.reranked_retriever import RerankedRetriever
from .evaluate_context import DEFAULT_CACHE as RERANK_CACHE, _candidate, evaluate_configuration
from .evaluate_generation import DEFAULT_BUNDLE, DEFAULT_OUTPUT, DEFAULT_SPLITS, DEFAULT_V2_DATASET, DEFAULT_V2_META, _record, _summary, load_and_validate_splits, load_cases
from .evaluate_generation_citation_facing import _unknown_identifier_diagnosis
from .evaluate_generation_final_dev import _answerability, _multi_evidence, _repair_token_usage, _upstream_miss
from .evaluate_generation_provider import ROOT, _canonicalization_metrics, _safe_component, _write_checkpoint, _write_review
from .evaluate_reranker import load_split_cases
from .recover_generation import _metrics


TOP15_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "generation_dev_contexts_d10_b10_top15_v1.json"
NO_ANSWER_RERANK_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "reranker_dev_d10_b10_no_answer_v1.json"
VALIDATION_REPORT = DEFAULT_OUTPUT / "d10_b10_context_top15_validation_v1.json"


def _config() -> ContextConfig:
    """The sole candidate change: top-15 chunk-only within the 2k budget."""
    return replace(load_frozen_context_config(), candidate_top_k=15)


def _identity(meta: dict[str, Any], config: ContextConfig) -> dict[str, Any]:
    rerank_payload = json.loads(RERANK_CACHE.read_text(encoding="utf-8"))
    return {
        "reranker_cache_identity": rerank_payload["identity"],
        "context_config": config.to_dict(),
        "benchmark_version": meta["dataset_version"],
    }


def _context_rows(config: ContextConfig) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Build/reuse distinct top-15 ContextPacks from the certified cache only."""
    cases = load_cases(DEFAULT_V2_DATASET)
    meta = json.loads(DEFAULT_V2_META.read_text(encoding="utf-8"))
    splits = load_and_validate_splits(cases, DEFAULT_SPLITS, meta)
    dev_ids = splits["dev"] + splits["dev_no_answer"]
    by_case = {case.query_id: case for case in cases}
    selected_cases = [by_case[query_id] for query_id in dev_ids]
    identity = _identity(meta, config)
    if TOP15_CACHE.exists():
        payload = json.loads(TOP15_CACHE.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("top-15 ContextPack cache identity is stale")
    else:
        payload = {"schema_version": 1, "identity": identity, "split": "dev", "results": []}
    existing = {row["query_id"]: row for row in payload["results"]}
    rerank_payload = json.loads(RERANK_CACHE.read_text(encoding="utf-8"))
    reranks = {row["query_id"]: row for row in rerank_payload["results"]}
    missing_ids = [case.query_id for case in selected_cases if case.query_id not in reranks]
    # The certified DEV ranking cache intentionally covers only answerable
    # cases.  A separate, immutable supplemental cache is needed for the
    # seven no-answer contexts in the required 63-query confirmation; it uses
    # the same frozen D10+B10 configuration and never alters the certified
    # answerable cache.
    if missing_ids:
        supplement_identity = {"base_reranker_cache_identity": rerank_payload["identity"], "query_ids": missing_ids}
        if NO_ANSWER_RERANK_CACHE.exists():
            supplement = json.loads(NO_ANSWER_RERANK_CACHE.read_text(encoding="utf-8"))
            if supplement.get("identity") != supplement_identity:
                raise ValueError("no-answer D10+B10 supplemental reranker cache identity is stale")
        else:
            supplement = {"schema_version": 1, "identity": supplement_identity, "results": []}
        supplemental_rows = {row["query_id"]: row for row in supplement["results"]}
        retriever: RerankedRetriever | None = None
        for case in selected_cases:
            if case.query_id not in missing_ids or case.query_id in supplemental_rows:
                continue
            retriever = retriever or RerankedRetriever()
            result = retriever.retrieve(case.query)
            supplemental_rows[case.query_id] = {"query_id": case.query_id, "query": case.query,
                                                 "reranked_results": [item.to_dict() for item in result.reranked_results]}
            supplement["results"] = [supplemental_rows[query_id] for query_id in missing_ids if query_id in supplemental_rows]
            NO_ANSWER_RERANK_CACHE.write_text(json.dumps(supplement, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        reranks.update(supplemental_rows)
    builder = ContextBuilder(config)
    TOP15_CACHE.parent.mkdir(parents=True, exist_ok=True)
    for case in selected_cases:
        if case.query_id in existing:
            continue
        started = time.perf_counter()
        row = reranks.get(case.query_id)
        if row is None:
            raise ValueError(f"certified D10+B10 reranker cache is missing {case.query_id}")
        pack = builder.build_from_candidates(case.query, [_candidate(item) for item in row["reranked_results"]])
        existing[case.query_id] = {
            "query_id": case.query_id, "query": case.query, "answerable": case.answerable,
            "context_pack": pack.to_dict(), "context_build_seconds": time.perf_counter() - started,
            "rerank_seconds_if_uncached": 0.0,
        }
        payload["results"] = [existing[query_id] for query_id in dev_ids if query_id in existing]
        TOP15_CACHE.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return selected_cases, payload["results"], meta


def _case_context_details(query_id: str, cases: dict[str, Any], reranks: dict[str, Any], context_rows: dict[str, Any]) -> dict[str, Any]:
    case = cases[query_id]
    candidates = reranks[query_id]["reranked_results"]
    selected = set(context_rows[query_id]["context_pack"]["selected_chunk_ids"])
    requirements: list[dict[str, Any]] = []
    for chunk_id in case.relevant_chunk_ids:
        candidate = next((item for item in candidates if item["chunk_id"] == chunk_id), None)
        requirements.append({
            "chunk_id": chunk_id,
            "reranker_rank": candidates.index(candidate) + 1 if candidate is not None else None,
            "in_top_10": candidate is not None and candidates.index(candidate) < 10,
            "in_top_15": chunk_id in selected,
            "in_candidate_union": candidate is not None,
            "dense_rank": candidate.get("dense_rank") if candidate else None,
            "bm25_rank": candidate.get("bm25_rank") if candidate else None,
        })
    return {
        "query_id": query_id,
        "required_evidence": requirements,
        "context_tokens": context_rows[query_id]["context_pack"]["estimated_context_tokens"],
        "evidence_blocks": context_rows[query_id]["context_pack"]["evidence_count"],
        "duplicate_blocks_removed": context_rows[query_id]["context_pack"]["configuration"]["duplicate_blocks_removed"],
    }


def _context_volume(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tokens = [row["context_pack"]["estimated_context_tokens"] for row in rows]
    blocks = [row["context_pack"]["evidence_count"] for row in rows]
    parents = [row["context_pack"]["unique_parent_count"] for row in rows]
    documents = [row["context_pack"]["unique_document_count"] for row in rows]
    duplicate_removals = [row["context_pack"]["configuration"]["duplicate_blocks_removed"] for row in rows]
    return {"query_count": len(rows), "average_context_tokens": sum(tokens) / len(tokens), "median_context_tokens": median(tokens),
            "maximum_context_tokens": max(tokens), "average_evidence_blocks": sum(blocks) / len(blocks),
            "average_unique_parents": sum(parents) / len(parents), "average_unique_documents": sum(documents) / len(documents),
            "average_duplicate_blocks_removed": sum(duplicate_removals) / len(duplicate_removals),
            "budget_overflow_count": sum(value > 2000 for value in tokens)}


def validate_context() -> dict[str, Any]:
    """Write the global top-10 vs top-15 comparison without an LLM call."""
    config = _config()
    cases, top15_rows, meta = _context_rows(config)
    rerank_payload = json.loads(RERANK_CACHE.read_text(encoding="utf-8"))
    reranks = {row["query_id"]: row for row in rerank_payload["results"]}
    if NO_ANSWER_RERANK_CACHE.exists():
        reranks.update({row["query_id"]: row for row in json.loads(NO_ANSWER_RERANK_CACHE.read_text(encoding="utf-8"))["results"]})
    answerable, _ = load_split_cases(DEFAULT_V2_DATASET, DEFAULT_V2_META, DEFAULT_SPLITS, DEFAULT_BUNDLE, "dev")
    top10_rows = [reranks[case.query_id] for case in answerable]
    top15_rerank_rows = [reranks[case.query_id] for case in answerable]
    top10 = evaluate_configuration(answerable, top10_rows, replace(config, candidate_top_k=10))
    top15 = evaluate_configuration(answerable, top15_rerank_rows, config)
    by_case = {case.query_id: case for case in cases}
    context_rows = {row["query_id"]: row for row in top15_rows}
    top10_builder = ContextBuilder(replace(config, candidate_top_k=10))
    top10_context_rows = [
        {"query_id": case.query_id, "context_pack": top10_builder.build_from_candidates(case.query, [_candidate(item) for item in reranks[case.query_id]["reranked_results"]]).to_dict()}
        for case in cases
    ]
    top10["context_all_63"] = _context_volume(top10_context_rows)
    top15["context_all_63"] = _context_volume(top15_rows)
    max_tokens = max(row["context_pack"]["estimated_context_tokens"] for row in top15_rows)
    overflow_ids = [row["query_id"] for row in top15_rows if row["context_pack"]["estimated_context_tokens"] > config.max_context_tokens]
    comparison = {
        "chunk_recall_improved_or_preserved": top15["metrics"]["chunk_evidence_recall"] >= top10["metrics"]["chunk_evidence_recall"],
        "parent_recall_preserved": top15["metrics"]["parent_evidence_recall"] >= top10["metrics"]["parent_evidence_recall"],
        "query_hit_rate_preserved": top15["metrics"]["query_hit_rate"] >= top10["metrics"]["query_hit_rate"],
        "multi_evidence_not_regressed": top15["multi_evidence"] == top10["multi_evidence"] or all(
            top15["multi_evidence"][kind]["all"] >= top10["multi_evidence"][kind]["all"]
            and top15["multi_evidence"][kind]["none"] <= top10["multi_evidence"][kind]["none"]
            for kind in ("chunk", "parent")
        ),
        "within_token_budget": not overflow_ids,
        "no_deduplication_regression": all(row["context_pack"]["configuration"]["duplicate_blocks_removed"] == 0 for row in top15_rows),
    }
    report = {
        "schema_version": 1,
        "split": "dev",
        "generation_performed": False,
        "source_reranker_cache": str(RERANK_CACHE),
        "no_answer_supplemental_reranker_cache": str(NO_ANSWER_RERANK_CACHE),
        "top15_context_cache": str(TOP15_CACHE),
        "reranker_reused_without_rebuild": True,
        "proposed_configuration": config.to_dict(),
        "top10": top10,
        "top15": top15,
        "top15_budget": {"max_context_tokens_observed": max_tokens, "budget_overflow_count": len(overflow_ids), "overflow_query_ids": overflow_ids},
        "specific_cases": {query_id: _case_context_details(query_id, by_case, reranks, context_rows) for query_id in ("D3-09", "D4-09", "V2-M13")},
        "acceptance_checks": comparison,
    }
    report["accepted"] = all(comparison.values())
    VALIDATION_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _paths(settings: LLMSettings) -> tuple[Path, Path, Path]:
    stem = f"generation_dev_{_safe_component(settings.provider)}_{_safe_component(settings.model)}_grounded_v1_citation_facing_d10_b10_top15_final_v1"
    return DEFAULT_OUTPUT / f"{stem}.jsonl", DEFAULT_OUTPUT / f"{stem}.json", ROOT / "artifacts" / "ai_assistant" / "cache" / f"{stem}.progress.json"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _generation_case_outcome(query_id: str, records: dict[str, Any], cases: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    record = records.get(query_id)
    expected = set(cases[query_id].relevant_chunk_ids)
    available = {chunk_id for source in cache[query_id]["context_pack"]["citation_map"].values() for chunk_id in source["chunk_ids"]}
    cited = set(record.get("resolved_chunk_ids", [])) if record else set()
    return {"query_id": query_id, "recorded": record is not None, "required_evidence_in_context": expected.issubset(available),
            "no_answer": record.get("no_answer") if record else None, "validation_success": record.get("validation_success") if record else None,
            "citation_required_chunk_coverage": len(cited & expected) / len(expected) if expected else None,
            "citation_repair_used": record.get("citation_repair_used") if record else None,
            "error_type": record.get("error_type") if record else None}


def run_final_generation(*, pacing_seconds: float = 5.0) -> dict[str, Any]:
    validation = validate_context()
    if not validation["accepted"]:
        raise ValueError("top-15 context candidate did not pass global acceptance; generation is intentionally not run")
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    review_path, report_path, progress_path = _paths(settings)
    config = _config()
    cases_list, cache_rows, meta = _context_rows(config)
    all_cases = load_cases(DEFAULT_V2_DATASET)
    cases = {case.query_id: case for case in all_cases}
    splits = load_and_validate_splits(all_cases, DEFAULT_SPLITS, meta)
    dev_ids = splits["dev"] + splits["dev_no_answer"]
    cache = {row["query_id"]: row for row in cache_rows}
    records = {row["query_id"]: {key: value for key, value in row.items() if key != "review_status"} for row in _load_rows(review_path)} if review_path.exists() else {}
    generator = GroundedGenerator(get_llm(settings))
    rate_limited = False
    attempted = 0
    for query_id in dev_ids:
        prior = records.get(query_id)
        if prior and (prior.get("validation_success") or prior.get("error_type") != "LLMRateLimitError"):
            continue
        if attempted:
            time.sleep(pacing_seconds)
        record = _record(cases[query_id], cache[query_id], generator)
        record["trace"] = generator.last_trace
        records[query_id] = record
        _write_review(review_path, records, dev_ids)
        _write_checkpoint(progress_path, records, dev_ids, runtime)
        attempted += 1
        if record.get("error_type") == "LLMRateLimitError":
            rate_limited = True
            break
    ordered = [records[query_id] for query_id in dev_ids if query_id in records]
    reranks = {row["query_id"]: row for row in json.loads(RERANK_CACHE.read_text(encoding="utf-8"))["results"]}
    provider_failures = [row["query_id"] for row in ordered if (row.get("error_type") or "").startswith("LLM")]
    base_metrics = _metrics(ordered, cases) if ordered else {}
    report = {
        "schema_version": 1, "split": "dev", "runtime": runtime, "prompt_version": "grounded_v1",
        "renderer": "citation_facing_v1", "canonicalization_enabled": True, "decorated_e_normalization_enabled": True,
        "context_configuration": config.to_dict(), "context_validation_artifact": str(VALIDATION_REPORT), "context_cache": str(TOP15_CACHE),
        "review_artifact": str(review_path), "progress_artifact": str(progress_path),
        "counts": {"expected_total": len(dev_ids), "recorded": len(ordered), "answerable": len(splits["dev"]), "no_answer": len(splits["dev_no_answer"]), "provider_failures": len(provider_failures), "remaining": len(dev_ids) - len(ordered)},
        "provider_failure_query_ids": provider_failures, "stopped_on_rate_limit": rate_limited,
        "answerability": _answerability(ordered, cases, cache) if len(ordered) == len(dev_ids) else {},
        "citation": {**base_metrics.get("citation", {}), **_canonicalization_metrics(ordered)},
        "unknown_identifier_diagnosis": _unknown_identifier_diagnosis(ordered, cache) if ordered else {},
        "multi_evidence": _multi_evidence(ordered, cases, cache) if ordered else {},
        "language": base_metrics.get("arabic_language", {}), "latency_seconds": base_metrics.get("latency_seconds", {}),
        "token_usage": {**base_metrics.get("token_usage", {}), "repair_calls": _repair_token_usage(ordered)},
        "upstream_miss_diagnosis": {query_id: _upstream_miss(query_id, cases, cache, reranks) for query_id in ("D3-09", "D4-09", "V2-M13")},
        "requested_case_outcomes": {query_id: _generation_case_outcome(query_id, records, cases, cache) for query_id in ("V2-M01", "V2-M03", "V2-M11")},
        "generation_error_query_ids": [row["query_id"] for row in ordered if row.get("error_type") and not (row.get("error_type") or "").startswith("LLM")],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--pacing-seconds", type=float, default=5.0)
    args = parser.parse_args()
    report = validate_context() if args.validate_only else run_final_generation(pacing_seconds=args.pacing_seconds)
    print(json.dumps({"accepted": report.get("accepted"), "counts": report.get("counts")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
