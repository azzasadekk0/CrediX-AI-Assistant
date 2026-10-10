"""Resumable, provider/model-specific DEV generation baseline.

This evaluator reads only frozen ContextPacks. It checkpoints every record and
never mixes generated results between provider/model runtimes.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from ..generation.generator import GroundedGenerator
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from .evaluate_generation import CONTEXT_CACHE, DEFAULT_BUNDLE, DEFAULT_OUTPUT, DEFAULT_SPLITS, DEFAULT_V2_DATASET, DEFAULT_V2_META, _context_identity, _record, load_and_validate_splits, load_cases
from .recover_generation import _metrics


def _safe_component(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "unknown"


def _paths(settings: LLMSettings) -> tuple[Path, Path, Path]:
    stem = f"generation_dev_{_safe_component(settings.provider)}_{_safe_component(settings.model)}_grounded_v1_canonicalization_v1"
    return (DEFAULT_OUTPUT / f"{stem}.jsonl", DEFAULT_OUTPUT / f"{stem}.json", ROOT / "artifacts" / "ai_assistant" / "cache" / f"{stem}.progress.json")


ROOT = Path(__file__).resolve().parents[3]


def _write_checkpoint(path: Path, records: dict[str, dict[str, Any]], dev_ids: list[str], runtime: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "completed_query_ids": [query_id for query_id in dev_ids if query_id in records]}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_review(path: Path, records: dict[str, dict[str, Any]], dev_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for query_id in dev_ids:
            if query_id in records:
                handle.write(json.dumps({**records[query_id], "review_status": "pending"}, ensure_ascii=False) + "\n")


def _repair_reason(record: dict[str, Any]) -> str:
    citations = record.get("raw_first_pass_citations") or []
    if record.get("raw_first_pass_no_answer") and citations:
        return "no-answer/citation contract violation"
    if not citations and not record.get("raw_first_pass_no_answer"):
        return "missing citation"
    if any(re.fullmatch(r"E\d+", identifier) for identifier in citations if isinstance(identifier, str)):
        return "unavailable E-handle"
    if record.get("identifiers_unresolved"):
        return "unknown identifier"
    return "other"


def _canonicalization_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if not (record.get("error_type") or "").startswith("LLM")]
    repairs = [record for record in completed if record.get("citation_repair_used")]
    normalized = sum(record.get("identifiers_normalized", 0) for record in completed)
    avoided = sum(
        record.get("direct_citation_valid") is True
        and record.get("identifiers_normalized", 0) > 0
        and not record.get("citation_repair_used")
        for record in completed
    )
    return {
        "direct_citation_validity_before_repair": sum(record.get("direct_citation_valid") is True for record in completed) / len(completed) if completed else None,
        "canonicalization_rate": sum(record.get("canonicalization_used") for record in completed) / len(completed) if completed else None,
        "exact_chunk_ids_normalized": normalized,
        "unresolved_identifiers": sum(record.get("identifiers_unresolved", 0) for record in completed),
        "citation_repair_rate": len(repairs) / len(completed) if completed else None,
        "citation_repair_calls": len(repairs),
        "repair_calls_avoided_through_canonicalization": avoided,
        "failed_citation_validation_rate": sum(record.get("safety_stop") or record.get("error_type") == "GroundedGenerationError" for record in completed) / len(completed) if completed else None,
        "remaining_safety_stop_query_ids": [record["query_id"] for record in completed if record.get("safety_stop") or record.get("error_type") == "GroundedGenerationError"],
        "remaining_repair_reason_counts": dict(Counter(_repair_reason(record) for record in repairs)),
        "remaining_repair_query_ids_by_reason": {
            reason: [record["query_id"] for record in repairs if _repair_reason(record) == reason]
            for reason in sorted({_repair_reason(record) for record in repairs})
        },
    }


def run_provider_dev_baseline(*, pacing_seconds: float = 8.0) -> dict[str, Any]:
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    review_path, report_path, progress_path = _paths(settings)
    cache = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))
    meta = json.loads(DEFAULT_V2_META.read_text(encoding="utf-8"))
    if cache.get("identity") != _context_identity(DEFAULT_BUNDLE, meta):
        raise ValueError("frozen DEV ContextPack cache identity does not match the current frozen branch")
    cases = load_cases(DEFAULT_V2_DATASET)
    splits = load_and_validate_splits(cases, DEFAULT_SPLITS, meta)
    dev_ids = splits["dev"] + splits["dev_no_answer"]
    by_case = {case.query_id: case for case in cases}
    cached = {row["query_id"]: row for row in cache["results"]}
    missing_contexts = [query_id for query_id in dev_ids if query_id not in cached]
    if missing_contexts:
        raise ValueError(f"frozen DEV ContextPacks missing: {missing_contexts}")

    records: dict[str, dict[str, Any]] = {}
    if review_path.exists():
        for line in review_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            records[row["query_id"]] = {key: value for key, value in row.items() if key != "review_status"}
        for row in records.values():
            if row.get("provider") and (row["provider"] != settings.provider or row.get("model") != settings.model):
                raise ValueError("provider/model-specific checkpoint does not match current ENV runtime")

    generator = GroundedGenerator(get_llm(settings))
    rate_limited = False
    attempted = 0
    for query_id in dev_ids:
        prior = records.get(query_id)
        if prior and (prior.get("validation_success") or prior.get("error_type") != "LLMRateLimitError"):
            continue
        if attempted:
            time.sleep(pacing_seconds)
        record = _record(by_case[query_id], cached[query_id], generator)
        records[query_id] = record
        _write_review(review_path, records, dev_ids)
        _write_checkpoint(progress_path, records, dev_ids, runtime)
        attempted += 1
        if record.get("error_type") == "LLMRateLimitError":
            rate_limited = True
            break

    ordered = [records[query_id] for query_id in dev_ids if query_id in records]
    provider_failed = [record["query_id"] for record in ordered if (record.get("error_type") or "").startswith("LLM")]
    report = {
        "schema_version": 1,
        "split": "dev",
        "runtime": runtime,
        "context_cache": str(CONTEXT_CACHE),
        "review_artifact": str(review_path),
        "progress_artifact": str(progress_path),
        "counts": {"expected_total": len(dev_ids), "recorded": len(ordered), "provider_failed": len(provider_failed), "remaining": len(dev_ids) - len(ordered)},
        "provider_failed_query_ids": provider_failed,
        "stopped_on_rate_limit": rate_limited,
        "baseline_complete": len(ordered) == len(dev_ids) and not provider_failed,
        "metrics": {**_metrics(ordered, by_case), "canonicalization": _canonicalization_metrics(ordered)} if ordered else {},
        "semantic_accuracy_note": "No semantic answer-accuracy metric is claimed; labels cover answerability and evidence alignment only.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def refresh_existing_provider_baseline() -> dict[str, Any]:
    """Recompute local metrics for this runtime's existing checkpoint; no LLM calls."""
    settings = LLMSettings.from_env()
    review_path, report_path, _ = _paths(settings)
    if not review_path.exists() or not report_path.exists():
        raise ValueError("no provider-specific baseline artifacts exist to refresh")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = [
        {key: value for key, value in json.loads(line).items() if key != "review_status"}
        for line in review_path.read_text(encoding="utf-8").splitlines() if line
    ]
    for record in records:
        if record.get("error_type") == "GroundedGenerationError":
            record["safety_stop"] = True
            record["citation_repair_used"] = True
    _write_review(review_path, {record["query_id"]: record for record in records}, [record["query_id"] for record in records])
    cases = {case.query_id: case for case in load_cases(DEFAULT_V2_DATASET)}
    report["metrics"] = {**_metrics(records, cases), "canonicalization": _canonicalization_metrics(records)}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacing-seconds", type=float, default=8.0)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.pacing_seconds < 0:
        parser.error("--pacing-seconds must be non-negative")
    report = refresh_existing_provider_baseline() if args.refresh_existing else run_provider_dev_baseline(pacing_seconds=args.pacing_seconds)
    print(json.dumps({key: report[key] for key in ("counts", "provider_failed_query_ids", "stopped_on_rate_limit", "baseline_complete")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
