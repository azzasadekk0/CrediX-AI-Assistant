"""Replay the Step 6E diagnostic subset with grounded_v1 citation canonicalization."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..generation.config import GenerationConfig
from ..generation.generator import GroundedGenerator
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from .evaluate_generation import CONTEXT_CACHE, DEFAULT_OUTPUT, DEFAULT_V2_DATASET, _record, load_cases
from .evaluate_generation_tuning import BASELINE, _selection


DIAGNOSTIC = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_citation_canonicalization_v1.jsonl"
REPORT = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_citation_canonicalization_v1.json"


def _write(records: dict[str, dict[str, Any]], order: list[str], runtime: dict[str, object]) -> None:
    with DIAGNOSTIC.open("w", encoding="utf-8") as handle:
        for query_id in order:
            if query_id in records:
                handle.write(json.dumps({**records[query_id], "review_status": "pending"}, ensure_ascii=False) + "\n")
    REPORT.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "prompt_version": "grounded_v1", "query_ids": order,
                                  "completed": len(records), "review_artifact": str(DIAGNOSTIC),
                                  "semantic_accuracy_note": "No semantic correctness is automatically claimed."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(*, pacing_seconds: float = 8.0) -> dict[str, Any]:
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    baseline = [json.loads(line) for line in BASELINE.read_text(encoding="utf-8").splitlines() if line]
    cases = {case.query_id: case for case in load_cases(DEFAULT_V2_DATASET)}
    selected = _selection(baseline, cases)
    order = list(selected)
    cached = {row["query_id"]: row for row in json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))["results"]}
    records: dict[str, dict[str, Any]] = {}
    if DIAGNOSTIC.exists():
        records = {row["query_id"]: {key: value for key, value in row.items() if key != "review_status"}
                   for row in (json.loads(line) for line in DIAGNOSTIC.read_text(encoding="utf-8").splitlines() if line)}
    generator = GroundedGenerator(get_llm(settings), GenerationConfig(prompt_version="grounded_v1"), citation_canonicalization=True)
    attempted = 0
    for query_id in order:
        if records.get(query_id, {}).get("validation_success") or records.get(query_id, {}).get("error_type") != "LLMRateLimitError" and query_id in records:
            continue
        if attempted:
            time.sleep(pacing_seconds)
        record = _record(cases[query_id], cached[query_id], generator)
        record["trace"] = generator.last_trace
        record["selection_reasons"] = selected[query_id]
        records[query_id] = record
        _write(records, order, runtime)
        attempted += 1
        if record.get("error_type") == "LLMRateLimitError":
            break
    _write(records, order, runtime)
    return {"selected": len(order), "completed": len(records), "rate_limited": next((query_id for query_id in order if records.get(query_id, {}).get("error_type") == "LLMRateLimitError"), None), "artifact": str(DIAGNOSTIC)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
