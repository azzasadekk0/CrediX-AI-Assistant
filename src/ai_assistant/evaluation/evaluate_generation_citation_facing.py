"""Resumable targeted DEV comparison for citation-facing context rendering only."""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..generation.generator import GroundedGenerator
from ..generation.citations import StructuredGroundedResult, validate_citations
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from .evaluate_generation import CONTEXT_CACHE, DEFAULT_OUTPUT, DEFAULT_V2_DATASET, _context_from_dict, _record, load_cases
from .evaluate_generation_provider import _canonicalization_metrics, _metrics, _safe_component, _write_review


ROOT = Path(__file__).resolve().parents[3]
FULL_BASELINE_STEM = "generation_dev_{provider}_{model}_grounded_v1_canonicalization_v1"
UPSTREAM_ONLY = {"V2-M13"}
EXTRA_TARGETS = {"D4-09", "V2-M03", "V2-M01", "V2-M05", "V2-M11", "V2-M15", "NA-07"}


def _paths(settings: LLMSettings) -> tuple[Path, Path, Path]:
    stem = f"generation_dev_{_safe_component(settings.provider)}_{_safe_component(settings.model)}_grounded_v1_citation_facing_v1"
    return (DEFAULT_OUTPUT / f"{stem}.jsonl", DEFAULT_OUTPUT / f"{stem}.json",
            ROOT / "artifacts" / "ai_assistant" / "cache" / f"{stem}.progress.json")


def _baseline_path(settings: LLMSettings) -> Path:
    stem = FULL_BASELINE_STEM.format(provider=_safe_component(settings.provider), model=_safe_component(settings.model))
    return DEFAULT_OUTPUT / f"{stem}.jsonl"


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _target_ids(baseline: list[dict[str, Any]]) -> list[str]:
    selected = {row["query_id"] for row in baseline if row.get("citation_repair_used")}
    selected.update(EXTRA_TARGETS)
    return sorted(selected - UPSTREAM_ONLY)


def _write_progress(path: Path, records: dict[str, dict[str, Any]], target_ids: list[str], runtime: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "runtime": runtime,
                                "completed_query_ids": [query_id for query_id in target_ids if query_id in records]},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _record_with_trace(case: Any, cached: dict[str, Any], generator: GroundedGenerator) -> dict[str, Any]:
    record = _record(case, cached, generator)
    record["trace"] = generator.last_trace
    return record


def _recover_logging_failure(record: dict[str, Any], cached: dict[str, Any], runtime: dict[str, object]) -> dict[str, Any]:
    """Recover a generated response lost only by the pre-write logging defect.

    This consumes no provider output: it validates the persisted canonicalized
    trace against the same frozen ContextPack and leaves unavailable token
    counters as ``None``.
    """
    if record.get("error_type") != "UnboundLocalError" or "cannot access local variable 'exc'" not in (record.get("error") or ""):
        return record
    trace = record.get("trace") or {}
    payload = trace.get("repair_canonicalized") or trace.get("first_pass_canonicalized")
    if not isinstance(payload, dict):
        return record
    pack = _context_from_dict(cached["context_pack"])
    result = StructuredGroundedResult(payload["answer"], list(payload["citations"]), payload["no_answer"])
    try:
        sources = validate_citations(result, pack)
    except Exception:
        return record
    details = trace.get("citation_normalization_details") or []
    latency = sum(value for value in (trace.get("first_pass_latency_seconds"), trace.get("repair_latency_seconds")) if isinstance(value, (int, float)))
    return {**record, "generated_answer": result.answer, "no_answer": result.no_answer, "citations": result.citations,
            "resolved_sources": [asdict(source) for source in sources],
            "resolved_chunk_ids": list(dict.fromkeys(chunk_id for source in sources for chunk_id in source.chunk_ids)),
            "resolved_parent_ids": list(dict.fromkeys(source.parent_id for source in sources)),
            "provider": runtime["provider"], "model": runtime["model"], "schema_success": True, "validation_success": True,
            "citation_repair_used": bool(trace.get("repair_attempted")),
            "citation_normalization_used": bool(trace.get("citation_normalization_count")),
            "citation_normalization_count": int(trace.get("citation_normalization_count") or 0),
            "canonicalization_used": bool(trace.get("citation_normalization_count")), "direct_citation_valid": trace.get("direct_citation_valid"),
            "identifiers_normalized": sum(item.get("status") == "normalized" for item in details),
            "identifiers_unresolved": sum(item.get("status") == "unresolved" for item in details),
            "normalization_details": details, "safety_stop": False, "generation_latency_seconds": latency,
            "input_tokens": None, "output_tokens": None, "total_tokens": None, "error_type": None, "error": None}


def _unknown_identifier_diagnosis(rows: list[dict[str, Any]], cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    corpus = [json.loads(line) for line in (ROOT / "src" / "rag_data" / "current" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()]
    corpus_chunk_ids = {item["chunk_id"] for item in corpus}
    corpus_parent_ids = {item["parent_id"] for item in corpus}
    corpus_document_ids = {item["document_id"] for item in corpus}
    findings: list[dict[str, str]] = []
    for row in rows:
        pack = _context_from_dict(cache[row["query_id"]]["context_pack"])
        valid_handles = set(pack.citation_map)
        current_chunks = {chunk_id for value in pack.citation_map.values() for chunk_id in value["chunk_ids"]}
        for identifier in row.get("raw_first_pass_citations") or []:
            if identifier in valid_handles or identifier in current_chunks:
                continue
            if identifier in corpus_chunk_ids:
                category = "real chunk ID outside current ContextPack"
            elif identifier in corpus_parent_ids:
                category = "parent ID"
            elif identifier in corpus_document_ids:
                category = "document ID"
            elif re.fullmatch(r"E\d+", identifier):
                category = "valid E-handle shape but unavailable in current ContextPack"
            elif re.fullmatch(r"(?:EVIDENCE\s+)?E\d+", identifier):
                category = "other: E-handle decorated with prose"
            elif re.search(r"(?:page|article|provision)\b", identifier, re.IGNORECASE):
                category = "page/article reference"
            else:
                category = "malformed/hallucinated chunk ID or other"
            findings.append({"query_id": row["query_id"], "identifier": identifier, "category": category})
    return {"count": len(findings), "category_counts": dict(Counter(item["category"] for item in findings)),
            "examples": findings[:20], "all_findings": findings}


def _comparison(rows: list[dict[str, Any]], cases: dict[str, Any]) -> dict[str, Any]:
    successful = [row for row in rows if row.get("validation_success")]
    answerable = [row for row in rows if row["gold_answerable"]]
    multi = [row for row in answerable if len(cases[row["query_id"]].relevant_chunk_ids) > 1 and row["query_id"] not in UPSTREAM_ONLY]
    multi_all = sum(set(cases[row["query_id"]].relevant_chunk_ids).issubset(set(row["resolved_chunk_ids"])) for row in multi)
    runtime_metrics = _metrics(rows, cases)
    token_usage = runtime_metrics["token_usage"]
    if not any(row.get("total_tokens") is not None for row in rows):
        token_usage = {key: None for key in token_usage}
        token_usage["unavailable_reason"] = "The initial targeted recorder defect persisted no provider token counters; no calls were repeated."
    return {
        "records": len(rows),
        "unknown_identifier_occurrences": sum(row.get("identifiers_unresolved", 0) for row in rows),
        "unknown_identifier_query_count": sum(row.get("identifiers_unresolved", 0) > 0 for row in rows),
        "repair_calls": sum(row.get("citation_repair_used") for row in rows),
        "repair_rate": sum(row.get("citation_repair_used") for row in rows) / len(rows) if rows else None,
        "direct_citation_validity": sum(row.get("direct_citation_valid") is True for row in rows) / len(rows) if rows else None,
        "final_citation_validity": len(successful) / len(rows) if rows else None,
        "safety_stop_query_ids": [row["query_id"] for row in rows if row.get("safety_stop") or row.get("error_type") == "GroundedGenerationError"],
        "false_abstention_query_ids": [row["query_id"] for row in answerable if row.get("no_answer") is True],
        "multi_evidence": {"evaluated": len(multi), "all": multi_all, "partial_or_none": len(multi) - multi_all},
        "arabic_adherence": runtime_metrics["arabic_language"],
        "latency_seconds": runtime_metrics["latency_seconds"],
        "token_usage": token_usage,
    }


def _blocker_diagnosis(query_id: str, before: dict[str, Any], after: dict[str, Any], cache: dict[str, dict[str, Any]], cases: dict[str, Any]) -> dict[str, Any]:
    pack = _context_from_dict(cache[query_id]["context_pack"])
    trace = after.get("trace") or {}
    required = set(cases[query_id].relevant_chunk_ids)
    available = {chunk_id for source in pack.citation_map.values() for chunk_id in source["chunk_ids"]}
    return {
        "query": cases[query_id].query,
        "available_e_handles": list(pack.citation_map),
        "required_evidence_in_context_pack": required.issubset(available),
        "baseline": {"raw_first_pass_answer": before.get("raw_first_pass_answer"), "raw_first_pass_no_answer": before.get("raw_first_pass_no_answer"),
                     "raw_first_pass_citations": before.get("raw_first_pass_citations"), "final_error": before.get("error"), "final_no_answer": before.get("no_answer")},
        "citation_facing": {"raw_first_pass_answer": after.get("raw_first_pass_answer"), "raw_first_pass_no_answer": after.get("raw_first_pass_no_answer"),
                             "raw_first_pass_citations": after.get("raw_first_pass_citations"), "canonicalized": trace.get("first_pass_canonicalized"),
                             "repair_result": trace.get("repair_output"), "final_error": after.get("error"), "final_no_answer": after.get("no_answer")},
    }


def _v2_m13_diagnosis(cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case = next(item for item in load_cases(DEFAULT_V2_DATASET) if item.query_id == "V2-M13")
    pack = _context_from_dict(cache[case.query_id]["context_pack"])
    selected = {chunk_id for source in pack.citation_map.values() for chunk_id in source["chunk_ids"]}
    missing = next(chunk_id for chunk_id in case.relevant_chunk_ids if chunk_id not in selected)
    chunks = {json.loads(line)["chunk_id"]: json.loads(line) for line in (ROOT / "src" / "rag_data" / "current" / "chunks.jsonl").read_text(encoding="utf-8").splitlines()}
    rerank_rows = {item["query_id"]: item for item in json.loads((ROOT / "artifacts" / "ai_assistant" / "cache" / "context_eval_reranked_v1.json").read_text(encoding="utf-8"))["results"]}
    results = rerank_rows[case.query_id]["reranked_results"]
    rank = next((index + 1 for index, item in enumerate(results) if item["chunk_id"] == missing), None)
    additional_tokens = len(chunks[missing]["text_original"].split())
    return {"missing_chunk_id": missing, "parent_id": chunks[missing]["parent_id"], "reranker_rank": rank,
            "in_top_11": rank is not None and rank <= 11, "in_top_12": rank is not None and rank <= 12,
            "in_top_15": rank is not None and rank <= 15, "in_top_20": rank is not None and rank <= 20,
            "estimated_additional_context_tokens": additional_tokens, "would_remain_within_2000": pack.estimated_context_tokens + additional_tokens <= 2000}


def run(*, pacing_seconds: float = 5.0) -> dict[str, Any]:
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    review_path, report_path, progress_path = _paths(settings)
    baseline_path = _baseline_path(settings)
    baseline = _load_rows(baseline_path)
    target_ids = _target_ids(baseline)
    baseline_by_id = {row["query_id"]: row for row in baseline}
    cases = {case.query_id: case for case in load_cases(DEFAULT_V2_DATASET)}
    cache_rows = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))["results"]
    cache = {row["query_id"]: row for row in cache_rows}
    records = {row["query_id"]: {key: value for key, value in row.items() if key != "review_status"} for row in _load_rows(review_path)} if review_path.exists() else {}
    records = {query_id: _recover_logging_failure(record, cache[query_id], runtime) for query_id, record in records.items()}
    generator = GroundedGenerator(get_llm(settings))
    rate_limited = False
    attempted = 0
    for query_id in target_ids:
        if query_id in records:
            continue
        if attempted:
            time.sleep(pacing_seconds)
        records[query_id] = _record_with_trace(cases[query_id], cache[query_id], generator)
        _write_review(review_path, records, target_ids)
        _write_progress(progress_path, records, target_ids, runtime)
        attempted += 1
        if records[query_id].get("error_type") == "LLMRateLimitError":
            rate_limited = True
            break
    ordered = [records[query_id] for query_id in target_ids if query_id in records]
    _write_review(review_path, records, target_ids)
    _write_progress(progress_path, records, target_ids, runtime)
    before = [baseline_by_id[query_id] for query_id in target_ids]
    report = {
        "schema_version": 1, "runtime": runtime, "prompt_version": "grounded_v1", "renderer": "citation_facing_v1",
        "baseline_artifact": str(baseline_path), "review_artifact": str(review_path), "progress_artifact": str(progress_path),
        "target_query_ids": target_ids, "upstream_only_query_ids": sorted(UPSTREAM_ONLY),
        "counts": {"targeted": len(target_ids), "completed": len(ordered), "remaining": len(target_ids) - len(ordered), "rate_limited": rate_limited},
        "unknown_identifier_diagnosis": _unknown_identifier_diagnosis(baseline, cache),
        "citation_facing_unknown_identifier_diagnosis": _unknown_identifier_diagnosis(ordered, cache),
        "before": _comparison(before, cases), "after": _comparison(ordered, cases) if ordered else {},
        "blocker_diagnostics": {query_id: _blocker_diagnosis(query_id, baseline_by_id[query_id], records[query_id], cache, cases)
                                for query_id in ("D4-09", "V2-M03", "V2-M15", "NA-07") if query_id in records},
        "v2_m13_upstream_diagnosis": _v2_m13_diagnosis(cache),
    }
    if len(ordered) == len(target_ids):
        before_metrics, after_metrics = report["before"], report["after"]
        report["acceptance"] = {
            "unknown_identifiers_decreased": after_metrics["unknown_identifier_occurrences"] < before_metrics["unknown_identifier_occurrences"],
            "repair_requirement_decreased": after_metrics["repair_calls"] < before_metrics["repair_calls"],
            "final_validity_preserved_or_improved": after_metrics["final_citation_validity"] >= before_metrics["final_citation_validity"],
            "no_unsafe_citations_accepted": all(row.get("validation_success") for row in ordered),
            "safety_stops_not_increased": len(after_metrics["safety_stop_query_ids"]) <= len(before_metrics["safety_stop_query_ids"]),
            "false_abstentions_not_increased": len(after_metrics["false_abstention_query_ids"]) <= len(before_metrics["false_abstention_query_ids"]),
            "multi_evidence_not_regressed": after_metrics["multi_evidence"]["all"] >= before_metrics["multi_evidence"]["all"],
        }
        report["acceptance"]["passed"] = all(report["acceptance"].values())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacing-seconds", type=float, default=5.0)
    args = parser.parse_args()
    report = run(pacing_seconds=args.pacing_seconds)
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
