"""Offline causal validation of strict decorated E-handle normalization."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..generation.citations import StructuredGroundedResult, canonicalize_citations_with_details, validate_citations
from ..generation.errors import CitationValidationError
from .evaluate_generation import CONTEXT_CACHE, DEFAULT_OUTPUT, _context_from_dict


SOURCE = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_grounded_v1_citation_facing_final_v1.jsonl"
OUTPUT = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_decorated_citation_offline_v1.json"


def _valid(result: StructuredGroundedResult, pack: Any) -> bool:
    try:
        validate_citations(result, pack)
        return True
    except CitationValidationError:
        return False


def run() -> dict[str, Any]:
    rows = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines() if line]
    cache = {row["query_id"]: row for row in json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))["results"]}
    records: list[dict[str, Any]] = []
    for row in rows:
        raw = StructuredGroundedResult(row["raw_first_pass_answer"], list(row["raw_first_pass_citations"] or []), row["raw_first_pass_no_answer"])
        pack = _context_from_dict(cache[row["query_id"]]["context_pack"])
        current, _, _ = canonicalize_citations_with_details(raw, pack, include_decorated=False)
        decorated, count, details = canonicalize_citations_with_details(raw, pack, include_decorated=True)
        records.append({"query_id": row["query_id"], "current_valid": _valid(current, pack), "decorated_valid": _valid(decorated, pack),
                        "current_citations": current.citations, "decorated_citations": decorated.citations, "normalization_count": count,
                        "details": details, "answer_unchanged": raw.answer == decorated.answer, "no_answer_unchanged": raw.no_answer == decorated.no_answer})
    before_repairs = sum(not row["current_valid"] for row in records)
    after_repairs = sum(not row["decorated_valid"] for row in records)
    report = {"schema_version": 1, "source_artifact": str(SOURCE), "generation_performed": False, "records": len(records),
              "metrics": {"repair_required_before": before_repairs, "repair_required_after": after_repairs,
                          "repairs_avoided": before_repairs - after_repairs, "direct_validity_before": sum(row["current_valid"] for row in records) / len(records),
                          "direct_validity_after": sum(row["decorated_valid"] for row in records) / len(records),
                          "decorated_identifiers_normalized": sum(row["normalization_count"] for row in records),
                          "ambiguous_mappings": sum(item["status"] == "ambiguous" for row in records for item in row["details"]),
                          "unsafe_mappings": [row["query_id"] for row in records if row["decorated_valid"] and any(item["status"] == "unresolved" for item in row["details"])],
                          "answer_changed_cases": [row["query_id"] for row in records if not row["answer_unchanged"]],
                          "no_answer_changed_cases": [row["query_id"] for row in records if not row["no_answer_unchanged"]]},
              "acceptance": {"answers_unchanged": all(row["answer_unchanged"] for row in records), "no_answer_unchanged": all(row["no_answer_unchanged"] for row in records),
                             "repairs_decreased": after_repairs < before_repairs, "validity_not_decreased": sum(row["decorated_valid"] for row in records) >= sum(row["current_valid"] for row in records)}}
    report["acceptance"]["passed"] = all(report["acceptance"].values()) and not report["metrics"]["unsafe_mappings"] and not report["metrics"]["ambiguous_mappings"]
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
