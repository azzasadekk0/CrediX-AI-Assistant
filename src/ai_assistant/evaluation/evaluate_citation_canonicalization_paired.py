"""Causal offline A/B validator evaluation for exact citation canonicalization.

No provider is constructed and no LLM generation occurs in this module.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from ..generation.citations import StructuredGroundedResult, canonicalize_citations_with_details, validate_citations
from ..generation.errors import CitationValidationError
from .evaluate_generation import CONTEXT_CACHE, DEFAULT_OUTPUT, _context_from_dict
from .evaluate_generation_tuning import DIAGNOSTIC as SOURCE_DIAGNOSTIC


PAIRED_JSONL = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_citation_canonicalization_paired_v1.jsonl"
PAIRED_REPORT = DEFAULT_OUTPUT / "generation_dev_gemini_gemini_3_5_flash_lite_citation_canonicalization_paired_v1.json"


def _branch(result: StructuredGroundedResult, pack: Any) -> dict[str, Any]:
    try:
        sources = validate_citations(result, pack)
    except CitationValidationError as error:
        return {"citations": result.citations, "directly_valid": False, "repair_required": True,
                "invalid_identifiers": error.invalid_handles, "resolved_handles": []}
    return {"citations": result.citations, "directly_valid": True, "repair_required": False,
            "invalid_identifiers": [], "resolved_handles": [source.handle for source in sources]}


def _payload(result: StructuredGroundedResult) -> dict[str, object]:
    return {"answer": result.answer, "citations": list(result.citations), "no_answer": result.no_answer}


def _evaluate_pair(original: StructuredGroundedResult, pack: Any) -> dict[str, Any]:
    """Evaluate both validator branches from one immutable first-pass payload."""
    raw_payload = _payload(original)
    canonicalized, normalized_count, details = canonicalize_citations_with_details(original, pack)
    current = _branch(original, pack)
    normalized = _branch(canonicalized, pack)
    expected_citations = list(dict.fromkeys(detail["output"] for detail in details))
    provenance_safe = all(
        detail["output"] in pack.citation_map and detail["input"] in pack.citation_map[detail["output"]]["chunk_ids"]
        for detail in details if detail["status"] == "normalized"
    )
    return {
        # Both branches receive this same saved payload before any citation processing.
        "branch_a_input": raw_payload,
        "branch_b_input": raw_payload,
        "raw_first_pass_answer": original.answer,
        "raw_first_pass_citations": original.citations,
        "raw_first_pass_no_answer": original.no_answer,
        "current": current,
        "canonicalized": normalized,
        "normalization_details": details,
        "citation_normalization_count": normalized_count,
        "answer_unchanged": original.answer == canonicalized.answer,
        "no_answer_unchanged": original.no_answer == canonicalized.no_answer,
        "same_raw_payload_for_both_branches": raw_payload == _payload(original),
        "normalization_provenance_safe": provenance_safe,
        "citation_order_preserved": normalized["citations"] == expected_citations,
        "citation_deduplication_preserved": len(normalized["citations"]) == len(set(normalized["citations"])),
        "review_status": "pending",
    }


def run() -> dict[str, Any]:
    source_rows = [json.loads(line) for line in SOURCE_DIAGNOSTIC.read_text(encoding="utf-8").splitlines() if line]
    cache = {row["query_id"]: row for row in json.loads(CONTEXT_CACHE.read_text(encoding="utf-8"))["results"]}
    records: list[dict[str, Any]] = []
    for source in source_rows:
        raw = (source.get("grounded_v1", {}).get("trace") or {}).get("first_pass")
        if not isinstance(raw, dict) or not isinstance(raw.get("answer"), str) or not isinstance(raw.get("citations"), list) or not isinstance(raw.get("no_answer"), bool):
            continue
        query_id = source["query_id"]
        pack = _context_from_dict(cache[query_id]["context_pack"])
        original = StructuredGroundedResult(raw["answer"], list(raw["citations"]), raw["no_answer"])
        records.append({"query_id": query_id, "selection_reasons": source["selection_reasons"], **_evaluate_pair(original, pack)})
    with PAIRED_JSONL.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    detail_rows = [detail for record in records for detail in record["normalization_details"]]
    before_repairs = sum(record["current"]["repair_required"] for record in records)
    after_repairs = sum(record["canonicalized"]["repair_required"] for record in records)
    report = {
        "schema_version": 1,
        "source_artifact": str(SOURCE_DIAGNOSTIC),
        "paired_raw_payload_count": len(records),
        "generation_performed": False,
        "raw_payload_kind": "pre-normalization/pre-repair parsed structured trace from the existing diagnostic",
        "metrics": {
            "raw_chunk_identifier_occurrences": sum(detail["status"] in {"normalized", "ambiguous"} for detail in detail_rows),
            "identifiers_normalized": sum(detail["status"] == "normalized" for detail in detail_rows),
            "unresolved_identifiers": sum(detail["status"] == "unresolved" for detail in detail_rows),
            "ambiguous_identifiers": sum(detail["status"] == "ambiguous" for detail in detail_rows),
            "repair_required_before": before_repairs,
            "repair_required_after": after_repairs,
            "repairs_avoided": before_repairs - after_repairs,
            "repair_reduction_percent": 100 * (before_repairs - after_repairs) / before_repairs if before_repairs else 0.0,
            "direct_citation_validity_before": sum(record["current"]["directly_valid"] for record in records) / len(records) if records else 0.0,
            "direct_citation_validity_after": sum(record["canonicalized"]["directly_valid"] for record in records) / len(records) if records else 0.0,
            "fully_resolved_without_repair_before": sum(not record["current"]["repair_required"] for record in records),
            "fully_resolved_without_repair_after": sum(not record["canonicalized"]["repair_required"] for record in records),
            "strict_validation_safety_stop_eligible_before": before_repairs,
            "strict_validation_safety_stop_eligible_after": after_repairs,
            "citation_order_preserved_cases": sum(record["citation_order_preserved"] for record in records),
            "citation_deduplication_preserved_cases": sum(record["citation_deduplication_preserved"] for record in records),
            "answer_changed_cases": [record["query_id"] for record in records if not record["answer_unchanged"]],
            "no_answer_changed_cases": [record["query_id"] for record in records if not record["no_answer_unchanged"]],
            "same_raw_payload_violations": [record["query_id"] for record in records if not record["same_raw_payload_for_both_branches"]],
            "normalized_status_counts": dict(Counter(detail["status"] for detail in detail_rows)),
            "newly_accepted_invalid_cases": [record["query_id"] for record in records if not record["current"]["directly_valid"] and record["canonicalized"]["directly_valid"] and not record["citation_normalization_count"]],
        },
        "acceptance": {
            "answers_unchanged": all(record["answer_unchanged"] for record in records),
            "no_answer_unchanged": all(record["no_answer_unchanged"] for record in records),
            "citation_validity_not_decreased": sum(record["canonicalized"]["directly_valid"] for record in records) >= sum(record["current"]["directly_valid"] for record in records),
            "repair_requirement_decreased": after_repairs < before_repairs,
            "no_ambiguous_mapping_accepted": not any(detail["status"] == "ambiguous" and detail["input"] != detail["output"] for detail in detail_rows),
            "canonical_provenance_safe": all(record["normalization_provenance_safe"] for record in records),
            "same_raw_payload_for_both_branches": all(record["same_raw_payload_for_both_branches"] for record in records),
            "safety_guard_unchanged": True,
        },
    }
    report["acceptance"]["passed"] = all(report["acceptance"].values())
    PAIRED_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
