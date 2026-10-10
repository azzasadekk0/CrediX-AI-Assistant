"""Resumable final DEV confirmation for grounded_v1 citation-facing rendering."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from statistics import median
from typing import Any

from ..generation.generator import GroundedGenerator
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from .evaluate_generation import CONTEXT_CACHE, DEFAULT_OUTPUT, DEFAULT_SPLITS, DEFAULT_V2_DATASET, DEFAULT_V2_META, _context_from_dict, _context_identity, _record, _summary, load_and_validate_splits, load_cases
from .evaluate_generation_citation_facing import _unknown_identifier_diagnosis
from .evaluate_generation_provider import ROOT, _canonicalization_metrics, _safe_component, _write_checkpoint, _write_review
from .recover_generation import _metrics


def _paths(settings: LLMSettings) -> tuple[Path, Path, Path]:
    stem = f"generation_dev_{_safe_component(settings.provider)}_{_safe_component(settings.model)}_grounded_v1_citation_facing_final_v1"
    return (DEFAULT_OUTPUT / f"{stem}.jsonl", DEFAULT_OUTPUT / f"{stem}.json",
            ROOT / "artifacts" / "ai_assistant" / "cache" / f"{stem}.progress.json")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _pack_chunks(cache_row: dict[str, Any]) -> set[str]:
    return {chunk_id for source in cache_row["context_pack"]["citation_map"].values() for chunk_id in source["chunk_ids"]}


def _multi_evidence(records: list[dict[str, Any]], cases: dict[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    result = {"chunk": {"all": 0, "partial": 0, "none": 0}, "parent": {"all": 0, "partial": 0, "none": 0},
              "partial_or_none_attribution": []}
    for row in records:
        case = cases[row["query_id"]]
        if not case.answerable or len(case.relevant_chunk_ids) <= 1:
            continue
        expected_chunks, expected_parents = set(case.relevant_chunk_ids), set(case.relevant_parent_ids)
        found_chunks, found_parents = set(row["resolved_chunk_ids"]), set(row["resolved_parent_ids"])
        chunk_state = "all" if expected_chunks.issubset(found_chunks) else "partial" if expected_chunks & found_chunks else "none"
        parent_state = "all" if expected_parents.issubset(found_parents) else "partial" if expected_parents & found_parents else "none"
        result["chunk"][chunk_state] += 1
        result["parent"][parent_state] += 1
        if chunk_state != "all":
            present = expected_chunks.issubset(_pack_chunks(cache[row["query_id"]]))
            result["partial_or_none_attribution"].append({"query_id": row["query_id"], "chunk_state": chunk_state,
                                                            "required_evidence_present_in_context_pack": present,
                                                            "cause": "generation_failed_to_cite_present_evidence" if present else "upstream_context_missing_required_evidence"})
    return result


def _upstream_miss(query_id: str, cases: dict[str, Any], cache: dict[str, dict[str, Any]], reranks: dict[str, Any]) -> dict[str, Any]:
    case = cases[query_id]
    selected = _pack_chunks(cache[query_id])
    candidates = reranks[query_id]["reranked_results"]
    details: list[dict[str, Any]] = []
    for chunk_id in case.relevant_chunk_ids:
        candidate = next((item for item in candidates if item["chunk_id"] == chunk_id), None)
        details.append({"chunk_id": chunk_id, "in_context_pack": chunk_id in selected,
                        "in_reranker_candidate_output": candidate is not None,
                        "reranker_rank": (candidates.index(candidate) + 1) if candidate is not None else None,
                        "dense_rank": candidate.get("dense_rank") if candidate else None,
                        "bm25_rank": candidate.get("bm25_rank") if candidate else None,
                        "missing_from_candidate_union_entirely": candidate is None})
    return {"query_id": query_id, "required_chunks": details}


def _answerability(records: list[dict[str, Any]], cases: dict[str, Any], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in records if row["gold_answerable"]]
    no_answer = [row for row in records if not row["gold_answerable"]]
    required_absent = []
    for row in answerable:
        case = cases[row["query_id"]]
        if not set(case.relevant_chunk_ids).issubset(_pack_chunks(cache[row["query_id"]])):
            required_absent.append(row["query_id"])
    answer_rate = sum(row["no_answer"] is False for row in answerable) / len(answerable)
    correct_abstention = sum(row["no_answer"] is True for row in no_answer) / len(no_answer)
    return {"answer_rate": answer_rate, "false_abstention_rate": sum(row["no_answer"] is True for row in answerable) / len(answerable),
            "correct_abstention_rate": correct_abstention, "validated_false_answer_rate": sum(row["no_answer"] is False for row in no_answer) / len(no_answer),
            "balanced_accuracy": (answer_rate + correct_abstention) / 2,
            "answerable_required_evidence_absent_from_context_pack": required_absent}


def _repair_token_usage(records: list[dict[str, Any]]) -> dict[str, Any]:
    repair = [row.get("repair_token_usage") for row in records if row.get("repair_token_usage")]
    return {key: _summary(float(item[key]) for item in repair if item.get(key) is not None) for key in ("input", "output", "total")} | {"repair_call_count": len(repair)}


def run(*, pacing_seconds: float = 5.0) -> dict[str, Any]:
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    review_path, report_path, progress_path = _paths(settings)
    cache_payload = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))
    meta = json.loads(DEFAULT_V2_META.read_text(encoding="utf-8"))
    if cache_payload.get("identity") != _context_identity(ROOT / "src" / "rag_data" / "current", meta):
        raise ValueError("frozen DEV ContextPack cache identity does not match the current frozen branch")
    cases_list = load_cases(DEFAULT_V2_DATASET)
    cases = {case.query_id: case for case in cases_list}
    splits = load_and_validate_splits(cases_list, DEFAULT_SPLITS, meta)
    dev_ids = splits["dev"] + splits["dev_no_answer"]
    cache = {row["query_id"]: row for row in cache_payload["results"]}
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
    provider_failures = [row["query_id"] for row in ordered if (row.get("error_type") or "").startswith("LLM")]
    reranks = {row["query_id"]: row for row in json.loads((ROOT / "artifacts" / "ai_assistant" / "cache" / "context_eval_reranked_v1.json").read_text(encoding="utf-8"))["results"]}
    base_metrics = _metrics(ordered, cases) if ordered else {}
    report = {
        "schema_version": 1, "split": "dev", "runtime": runtime, "prompt_version": "grounded_v1", "renderer": "citation_facing_v1", "canonicalization_enabled": True,
        "context_cache": str(CONTEXT_CACHE), "review_artifact": str(review_path), "progress_artifact": str(progress_path),
        "counts": {"expected_total": len(dev_ids), "recorded": len(ordered), "answerable": len(splits["dev"]), "no_answer": len(splits["dev_no_answer"]), "provider_failures": len(provider_failures), "remaining": len(dev_ids) - len(ordered)},
        "provider_failure_query_ids": provider_failures, "stopped_on_rate_limit": rate_limited,
        "answerability": _answerability(ordered, cases, cache) if len(ordered) == len(dev_ids) else {},
        "citation": {**base_metrics.get("citation", {}), **_canonicalization_metrics(ordered)},
        "unknown_identifier_diagnosis": _unknown_identifier_diagnosis(ordered, cache) if ordered else {},
        "multi_evidence": _multi_evidence(ordered, cases, cache) if ordered else {},
        "language": base_metrics.get("arabic_language", {}), "latency_seconds": base_metrics.get("latency_seconds", {}),
        "token_usage": {**base_metrics.get("token_usage", {}), "repair_calls": _repair_token_usage(ordered)},
        "upstream_miss_diagnosis": {query_id: _upstream_miss(query_id, cases, cache, reranks) for query_id in ("D4-09", "V2-M13")},
        "generation_error_query_ids": [row["query_id"] for row in ordered if row.get("error_type") and not (row.get("error_type") or "").startswith("LLM")],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacing-seconds", type=float, default=5.0)
    args = parser.parse_args()
    report = run(pacing_seconds=args.pacing_seconds)
    print(json.dumps({"counts": report["counts"], "rate_limited": report["stopped_on_rate_limit"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
