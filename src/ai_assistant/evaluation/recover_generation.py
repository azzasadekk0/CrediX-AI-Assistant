"""Recover only Groq-rate-limited records from the Step 6D DEV evaluation.

This utility reads the immutable Step 6D artifacts and frozen ContextPack cache.
It never retrieves, reranks, or regenerates an already completed DEV record.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..generation.generator import GroundedGenerator
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from .evaluate_generation import (
    CONTEXT_CACHE,
    DEFAULT_OUTPUT,
    DEFAULT_SPLITS,
    DEFAULT_V2_DATASET,
    _quality,
    _record,
    _summary,
    load_and_validate_splits,
    load_cases,
)
from .evaluate_retrieval import DEFAULT_V2_META


ORIGINAL_REPORT = DEFAULT_OUTPUT / "generation_dev_evaluation_v1.json"
ORIGINAL_REVIEW = DEFAULT_OUTPUT / "generation_dev_human_review_v1.jsonl"
RECOVERED_REPORT = DEFAULT_OUTPUT / "generation_dev_evaluation_v1_recovered.json"
RECOVERED_MARKDOWN = DEFAULT_OUTPUT / "generation_dev_evaluation_v1_recovered.md"
RECOVERED_REVIEW = DEFAULT_OUTPUT / "generation_dev_human_review_v1_recovered.jsonl"


def _is_provider_failure(record: dict[str, Any]) -> bool:
    return bool(record.get("error_type") and record["error_type"].startswith("LLM"))


def _answerability_over_completed(records: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Do not count provider failures as abstentions, answers, or citation failures."""
    completed = [record for record in records if not _is_provider_failure(record)]
    answerable = [record for record in completed if record["gold_answerable"]]
    no_answer = [record for record in completed if not record["gold_answerable"]]

    def rate(rows: list[dict[str, Any]], prediction: bool) -> float | None:
        return sum(record["no_answer"] is prediction for record in rows) / len(rows) if rows else None

    answer_rate = rate(answerable, False)
    correct_abstention_rate = rate(no_answer, True)
    return {
        "completed_answerable": len(answerable),
        "completed_no_answer": len(no_answer),
        "answer_rate": answer_rate,
        "false_abstention_rate": rate(answerable, True),
        "correct_abstention_rate": correct_abstention_rate,
        "false_answer_rate": rate(no_answer, False),
        "balanced_accuracy": (answer_rate + correct_abstention_rate) / 2 if answer_rate is not None and correct_abstention_rate is not None else None,
    }


def _metrics(records: list[dict[str, Any]], cases: dict[str, Any]) -> dict[str, Any]:
    metrics = _quality(records, cases)
    completed = [record for record in records if not _is_provider_failure(record)]
    metrics["completed_case_metrics"] = _quality(completed, cases)
    successful = [record for record in records if record["validation_success"]]
    metrics["latency_seconds"] = _summary(record["generation_latency_seconds"] for record in successful if record["generation_latency_seconds"] is not None)
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        metrics.setdefault("token_usage", {})[key] = _summary(float(record[key]) for record in successful if record[key] is not None)
    metrics["token_usage"]["dev_total_tokens"] = sum(record["total_tokens"] or 0 for record in successful)
    metrics["token_usage"]["answerable_average_total_tokens"] = sum(record["total_tokens"] or 0 for record in successful if record["gold_answerable"]) / max(1, sum(record["gold_answerable"] for record in successful))
    metrics["token_usage"]["no_answer_average_total_tokens"] = sum(record["total_tokens"] or 0 for record in successful if not record["gold_answerable"]) / max(1, sum(not record["gold_answerable"] for record in successful))
    metrics["answer_length_characters"] = _summary(float(len(record["generated_answer"] or "")) for record in successful)
    metrics["answerability_completed_cases"] = _answerability_over_completed(records)
    return metrics


def _write_artifacts(records: list[dict[str, Any]], report: dict[str, Any]) -> None:
    DEFAULT_OUTPUT.mkdir(parents=True, exist_ok=True)
    with RECOVERED_REVIEW.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({**record, "review_status": "pending"}, ensure_ascii=False) + "\n")
    RECOVERED_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    overview = [
        "# Step 6D.1 DEV provider-failure recovery",
        "",
        "This report preserves the original Step 6D run and retries only Groq rate-limited records using their cached ContextPacks.",
        "",
        "```json",
        json.dumps(report, ensure_ascii=False, indent=2),
        "```",
    ]
    RECOVERED_MARKDOWN.write_text("\n".join(overview) + "\n", encoding="utf-8")


def recover_provider_failures(*, pacing_seconds: float = 120.0) -> dict[str, Any]:
    original_report = json.loads(ORIGINAL_REPORT.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in ORIGINAL_REVIEW.read_text(encoding="utf-8").splitlines() if line]
    targets = [row["query_id"] for row in rows if row.get("error_type") == "LLMRateLimitError"]
    if not targets:
        raise ValueError("original Step 6D review has no Groq rate-limit records to recover")

    cache = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))
    cached = {row["query_id"]: row for row in cache["results"]}
    missing = [query_id for query_id in targets if query_id not in cached]
    if missing:
        raise ValueError(f"frozen ContextPack cache is missing recovery cases: {missing}")

    cases = load_cases(DEFAULT_V2_DATASET)
    meta = json.loads(DEFAULT_V2_META.read_text(encoding="utf-8"))
    splits = load_and_validate_splits(cases, DEFAULT_SPLITS, meta)
    dev_ids = splits["dev"] + splits["dev_no_answer"]
    if {row["query_id"] for row in rows} != set(dev_ids):
        raise ValueError("original review does not contain exactly the frozen DEV IDs")
    by_case = {case.query_id: case for case in cases}

    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    if runtime != original_report["runtime"]:
        raise ValueError("current ENV runtime differs from the original Step 6D runtime")
    generator = GroundedGenerator(get_llm(settings))
    merged = {row["query_id"]: row for row in rows}
    recovered: list[str] = []
    stopped_on_rate_limit = False

    # No reset header is available in the safe stored errors. One provider-limit
    # response therefore ends this invocation instead of issuing blind retries.
    for index, query_id in enumerate(targets):
        if index:
            time.sleep(pacing_seconds)
        result = _record(by_case[query_id], cached[query_id], generator)
        if result.get("error_type") == "LLMRateLimitError":
            stopped_on_rate_limit = True
            break
        merged[query_id] = result
        recovered.append(query_id)
        if _is_provider_failure(result):
            break

    records = [merged[query_id] for query_id in dev_ids]
    remaining = [row["query_id"] for row in records if row.get("error_type") == "LLMRateLimitError"]
    report = {
        "schema_version": 1,
        "split": "dev",
        "original_artifacts": {"report": str(ORIGINAL_REPORT), "review": str(ORIGINAL_REVIEW)},
        "context_cache": str(CONTEXT_CACHE),
        "runtime": runtime,
        "original_provider_failed_count": len(targets),
        "recovered_query_ids": recovered,
        "remaining_rate_limited_query_ids": remaining,
        "recovery_blocked_by_groq_rate_limit": stopped_on_rate_limit,
        "baseline_complete": not remaining,
        "citation_safety_stop_count": sum(row.get("error_type") == "GroundedGenerationError" for row in records),
        "metrics": _metrics(records, by_case),
    }
    _write_artifacts(records, report)
    return report


def refresh_recovered_artifacts() -> None:
    """Refresh report calculations from an existing recovery JSONL without runtime calls."""
    report = json.loads(RECOVERED_REPORT.read_text(encoding="utf-8"))
    records = [json.loads(line) for line in RECOVERED_REVIEW.read_text(encoding="utf-8").splitlines() if line]
    cases = {case.query_id: case for case in load_cases(DEFAULT_V2_DATASET)}
    report["metrics"] = _metrics(records, cases)
    _write_artifacts(records, report)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pacing-seconds", type=float, default=120.0)
    parser.add_argument("--refresh-existing", action="store_true")
    args = parser.parse_args()
    if args.pacing_seconds < 0:
        parser.error("--pacing-seconds must be non-negative")
    if args.refresh_existing:
        refresh_recovered_artifacts()
        return 0
    report = recover_provider_failures(pacing_seconds=args.pacing_seconds)
    print(json.dumps({key: report[key] for key in ("original_provider_failed_count", "recovered_query_ids", "remaining_rate_limited_query_ids", "recovery_blocked_by_groq_rate_limit", "baseline_complete")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
