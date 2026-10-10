"""DEV-only v1/v2 grounding diagnostic over frozen Gemini ContextPacks."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..generation.config import GenerationConfig
from ..generation.generator import GroundedGenerator
from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from .evaluate_generation import CONTEXT_CACHE, DEFAULT_OUTPUT, DEFAULT_V2_DATASET, _context_from_dict, _record, load_cases


BASELINE = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_v1.jsonl"
DIAGNOSTIC = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_grounding_tuning_v1.jsonl"
REPORT = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_grounding_tuning_v1.json"


def _selection(rows: list[dict[str, Any]], cases: dict[str, Any]) -> dict[str, list[str]]:
    selected: dict[str, list[str]] = {}
    for row in rows:
        query_id = row["query_id"]
        reasons: list[str] = []
        if row["citation_repair_used"]:
            reasons.append("baseline_citation_repair")
        if row["error_type"] == "GroundedGenerationError":
            reasons.append("citation_safety_stop")
        if row["gold_answerable"] and row["no_answer"] is True:
            reasons.append("false_abstention")
        if not row["gold_answerable"] and row["no_answer"] is False:
            reasons.append("false_answer")
        case = cases[query_id]
        if row["gold_answerable"] and len(case.relevant_chunk_ids) > 1:
            found, expected = set(row["resolved_chunk_ids"]), set(case.relevant_chunk_ids)
            if not expected.issubset(found):
                reasons.append("multi_evidence_partial" if found & expected else "multi_evidence_none")
        if query_id == "NA-06":
            reasons.append("arabic_language_mismatch")
        if reasons:
            selected[query_id] = reasons
    return selected


def _write(rows: dict[str, dict[str, Any]], order: list[str], runtime: dict[str, object]) -> None:
    with DIAGNOSTIC.open("w", encoding="utf-8") as handle:
        for query_id in order:
            if query_id in rows:
                handle.write(json.dumps({**rows[query_id], "review_status": "pending"}, ensure_ascii=False) + "\n")
    REPORT.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "query_ids": order,
                                  "completed": len(rows), "review_artifact": str(DIAGNOSTIC),
                                  "semantic_accuracy_note": "No semantic correctness is automatically claimed."}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run(*, pacing_seconds: float = 8.0) -> dict[str, Any]:
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    baseline_rows = [json.loads(line) for line in BASELINE.read_text(encoding="utf-8").splitlines() if line]
    cases = {case.query_id: case for case in load_cases(DEFAULT_V2_DATASET)}
    selected = _selection(baseline_rows, cases)
    order = list(selected)
    cache = json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))
    cached = {row["query_id"]: row for row in cache["results"]}
    existing: dict[str, dict[str, Any]] = {}
    if DIAGNOSTIC.exists():
        existing = {row["query_id"]: {key: value for key, value in row.items() if key != "review_status"}
                    for row in (json.loads(line) for line in DIAGNOSTIC.read_text(encoding="utf-8").splitlines() if line)}
    for query_id in order:
        prior = existing.get(query_id)
        if prior and prior.get("v1") and prior.get("v2"):
            continue
        pack = cached[query_id]
        versions: dict[str, Any] = prior or {"query_id": query_id, "query": cases[query_id].query, "selection_reasons": selected[query_id],
                                             "baseline_v1": next(row for row in baseline_rows if row["query_id"] == query_id)}
        for version in ("grounded_v1", "grounded_v2"):
            if versions.get(version):
                continue
            generator = GroundedGenerator(get_llm(settings), GenerationConfig(prompt_version=version))
            record = _record(cases[query_id], pack, generator)
            record["trace"] = generator.last_trace
            versions[version] = record
            existing[query_id] = versions
            _write(existing, order, runtime)
            if record.get("error_type") == "LLMRateLimitError":
                return {"completed": len(existing), "selected": len(order), "rate_limited": query_id, "artifact": str(DIAGNOSTIC)}
            time.sleep(pacing_seconds)
    _write(existing, order, runtime)
    return {"completed": len(existing), "selected": len(order), "rate_limited": None, "artifact": str(DIAGNOSTIC)}


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
