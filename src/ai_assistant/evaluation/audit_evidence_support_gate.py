"""Machine-assisted semantic audit and conservative v1.1 gate calibration.

This module never invokes the consumed RAG TEST v1.  Its benchmark artifacts
are explicitly pending human semantic audit.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from ..llm.models import LLMRequest
from ..support_gate.gate import EvidenceSupportGate
from ..support_gate.prompts import SUPPORT_GATE_V1_1_PROMPT_VERSION
from .evaluate_evidence_support_gate import BENCHMARK as V1_BENCHMARK, CALIBRATION as V1_CALIBRATION, SOURCE_CONTEXT_CACHE, SPLIT as V1_SPLIT, _metrics, _source_pack


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "ai_assistant" / "evaluation"
AUDIT = OUT / "evidence_support_calibration_v2_audit.json"
AUDITED = OUT / "evidence_support_calibration_v2_audited.json"
PACKET = OUT / "evidence_support_calibration_v2_human_review.md"
V11_CONFIG = ROOT / "src" / "ai_assistant" / "support_gate" / "support_gate_v1_1_config.json"
V11_CALIBRATION = OUT / "evidence_support_gate_v1_1_calibration.json"
NEW_VALIDATION = OUT / "evidence_support_validation_v3.json"
NEW_VALIDATION_RESULTS = OUT / "evidence_support_gate_v1_1_validation_v3.json"

AUDIT_SCHEMA = {"type": "object", "properties": {
    "audit_recommendation": {"type": "string", "enum": ["keep", "change", "uncertain"]},
    "recommended_label": {"type": ["string", "null"], "enum": ["full", "partial", "none", None]},
    "reason": {"type": "string"}, "material_aspects": {"type": "array", "items": {"type": "string"}},
    "directly_supported_aspects": {"type": "array", "items": {"type": "string"}},
    "unsupported_aspects": {"type": "array", "items": {"type": "string"}},
    "root_cause": {"type": "string", "enum": ["benchmark_label_questionable", "paraphrase_treated_too_strictly", "multi_part_decomposition", "handle_selection", "related_only", "unsupported_inference_rejected", "other"]},
}, "required": ["audit_recommendation", "recommended_label", "reason", "material_aspects", "directly_supported_aspects", "unsupported_aspects", "root_cause"], "additionalProperties": False}


def _safe_audit(provider: Any, case: dict[str, Any], pack: Any) -> dict[str, Any]:
    system = (
        "You are performing a machine-assisted semantic audit of an evidence-support benchmark. Do not answer the question. "
        "Inspect material requested aspects and decide whether the existing label should be kept, changed, or marked uncertain. "
        "Direct support allows clear semantic entailment/paraphrase, but not related-topic inference. "
        "Return only the schema JSON. Mark uncertain whenever source wording does not make the label clear."
    )
    user = (f"QUESTION:\n{case['question']}\n\nORIGINAL LABEL: {case['expected_support']}\n"
            f"ANNOTATION: {case['annotation']}\n\nEVIDENCE:\n{pack.render_for_llm()}")
    try:
        response = provider.generate(LLMRequest(system, user, response_schema=AUDIT_SCHEMA, response_schema_name="evidence_support_audit"))
        value = json.loads(response.text)
        if not isinstance(value, dict) or set(value) != set(AUDIT_SCHEMA["properties"]):
            raise ValueError("invalid audit schema")
        if value["audit_recommendation"] == "keep":
            value["recommended_label"] = case["expected_support"]
        if value["audit_recommendation"] == "change" and value["recommended_label"] not in {"full", "partial", "none"}:
            raise ValueError("changed label missing")
        return value | {"audit_failed": False}
    except Exception as exc:
        return {"audit_recommendation": "uncertain", "recommended_label": None, "reason": f"machine audit unavailable: {type(exc).__name__}",
                "material_aspects": [], "directly_supported_aspects": [], "unsupported_aspects": [], "root_cause": "other", "audit_failed": True}


def _load_source() -> dict[str, Any]:
    return {row["query_id"]: row for row in json.loads(SOURCE_CONTEXT_CACHE.read_text(encoding="utf-8"))["results"]}


def run_audit(*, pacing_seconds: float = 2.0) -> dict[str, Any]:
    if AUDIT.exists() and AUDITED.exists() and PACKET.exists():
        return json.loads(AUDIT.read_text(encoding="utf-8"))
    benchmark = json.loads(V1_BENCHMARK.read_text(encoding="utf-8"))
    source = _load_source()
    settings = LLMSettings.from_env()
    provider = get_llm(settings)
    v1_calibration = {row["query_id"]: row for row in json.loads(V1_CALIBRATION.read_text(encoding="utf-8"))["failures"]}
    validation_path = OUT / "evidence_support_gate_v1_validation.json"
    v1_validation = {row["query_id"]: row for row in json.loads(validation_path.read_text(encoding="utf-8"))["failures"]}
    prior_errors = {**v1_calibration, **v1_validation}
    records: list[dict[str, Any]] = []
    for index, case in enumerate(benchmark["cases"]):
        if index:
            time.sleep(pacing_seconds)
        reference = case["context_reference"]
        pack = _source_pack(source[reference["source_query_id"]], reference["handles"], case["question"])
        audit = _safe_audit(provider, case, pack)
        records.append({"query_id": case["query_id"], "original_label": case["expected_support"], **audit,
                        "question": case["question"], "expected_supporting_handles": case["expected_supporting_handles"],
                        "context_reference": reference, "annotation": case["annotation"], "prior_gate_misclassification": case["query_id"] in prior_errors})
    counts = {key: sum(row["audit_recommendation"] == key for row in records) for key in ("keep", "change", "uncertain")}
    report = {"schema_version": 1, "audit_type": "machine-assisted semantic audit pending human approval", "records": records,
              "summary": counts, "prior_misclassification_count": len(prior_errors)}
    OUT.mkdir(parents=True, exist_ok=True)
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    audit_by_id = {row["query_id"]: row for row in records}
    audited_cases = []
    for case in benchmark["cases"]:
        audit = audit_by_id[case["query_id"]]
        audited_label = audit["recommended_label"] if audit["audit_recommendation"] == "change" else case["expected_support"]
        audited_cases.append(case | {"audit": {key: audit[key] for key in ("audit_recommendation", "recommended_label", "reason", "material_aspects", "directly_supported_aspects", "unsupported_aspects", "root_cause")},
                                      "audited_label": audited_label, "audit_uncertain": audit["audit_recommendation"] == "uncertain"})
    audited = benchmark | {"benchmark_version": "evidence_support_calibration_v2_audited", "source_benchmark": str(V1_BENCHMARK),
                            "review_status": "machine-assisted semantic audit pending human approval", "cases": audited_cases}
    AUDITED.write_text(json.dumps(audited, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_packet(records, source, prior_errors)
    return report


def _write_packet(records: list[dict[str, Any]], source: dict[str, Any], prior_errors: dict[str, Any]) -> None:
    selected = [row for row in records if row["audit_recommendation"] != "keep" or row["query_id"] in prior_errors]
    lines = ["# Evidence Support Gate v2 — human review packet", "", "Machine-assisted review only; no human semantic approval is claimed."]
    for row in selected:
        reference = row["context_reference"]
        pack = _source_pack(source[reference["source_query_id"]], reference["handles"], row["question"])
        lines.extend(["", f"## {row['query_id']}", "", f"Question: {row['question']}", "", f"Expected label: `{row['original_label']}`",
                      f"Machine recommendation: `{row['audit_recommendation']}` → `{row['recommended_label']}`", "", f"Reason: {row['reason']}",
                      "", "Evidence:", "", "```text", pack.render_for_llm(), "```"])
    PACKET.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _v11_cases() -> tuple[list[dict[str, Any]], list[str]]:
    audited = json.loads(AUDITED.read_text(encoding="utf-8"))
    splits = json.loads(V1_SPLIT.read_text(encoding="utf-8"))
    by_id = {case["query_id"]: case for case in audited["cases"]}
    return [by_id[query_id] for query_id in splits["calibration"]], splits["calibration"]


def _evaluate(cases: list[dict[str, Any]], *, prompt_version: str, artifact: Path, pacing_seconds: float) -> dict[str, Any]:
    source = _load_source()
    settings = LLMSettings.from_env()
    gate = EvidenceSupportGate(get_llm(settings), prompt_version=prompt_version)
    records = {json.loads(line)["query_id"]: json.loads(line) for line in artifact.with_suffix(".jsonl").read_text(encoding="utf-8").splitlines() if line} if artifact.with_suffix(".jsonl").exists() else {}
    for index, case in enumerate(cases):
        if case["query_id"] in records:
            continue
        if index and records:
            time.sleep(pacing_seconds)
        ref = case["context_reference"]
        pack = _source_pack(source[ref["source_query_id"]], ref["handles"], case["question"])
        assessment = gate.assess(pack)
        expected = case.get("audited_label", case["expected_support"])
        records[case["query_id"]] = {"query_id": case["query_id"], "question": case["question"], "expected_support": expected,
                                       "original_label": case.get("expected_support"), "predicted_support": assessment.support,
                                       "assessment": assessment.to_dict(), "trace": gate.last_trace, "audit_uncertain": case.get("audit_uncertain", False)}
        with artifact.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
            for ordered in cases:
                if ordered["query_id"] in records:
                    handle.write(json.dumps(records[ordered["query_id"]], ensure_ascii=False) + "\n")
    ordered = [records[case["query_id"]] for case in cases]
    report = {"schema_version": 1, "prompt_version": prompt_version, "records": len(ordered), "metrics": _metrics(ordered),
              "failures": [row for row in ordered if row["expected_support"] != row["predicted_support"]],
              "pending_human_audit_case_count": sum(row["audit_uncertain"] for row in ordered)}
    artifact.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_v11_calibration(*, pacing_seconds: float = 4.0) -> dict[str, Any]:
    run_audit(pacing_seconds=pacing_seconds)
    V11_CONFIG.write_text(json.dumps({"prompt_version": SUPPORT_GATE_V1_1_PROMPT_VERSION, "decision_policy": "fail_closed_none",
                                      "change_from_v1": "permit clear semantic entailment/paraphrase while retaining material-aspect checks and related-only rejection",
                                      "selection_split": "evidence_support_calibration_v2_audited"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases, _ = _v11_cases()
    return _evaluate(cases, prompt_version=SUPPORT_GATE_V1_1_PROMPT_VERSION, artifact=V11_CALIBRATION, pacing_seconds=pacing_seconds)


def _new_templates(index: int, label: str) -> tuple[str, list[str], str]:
    full = [
        "Does the supplied evidence clearly establish the substantive regulatory condition it describes?",
        "Can the directly stated provision in this evidence be relied on without adding outside assumptions?",
        "Is the documented operational requirement explicitly established by the supplied material?",
        "Does the evidence semantically entail the stated rule for the matter it covers?",
        "Is the compliance condition directly supported even if phrased differently from the source?",
        "Does the provided text establish the relevant restriction as a direct proposition?",
        "Can the source alone support the regulatory requirement it expressly describes?",
        "Does the evidence directly establish the condition applicable to its stated subject?",
        "Is the material provision clearly entailed by the supplied evidence?",
        "Does the source provide direct support for the rule it documents?",
    ]
    partial = [
        "What direct regulatory condition is established, and what complete appeal process and deadline are provided?",
        "Which supported obligation applies, and what exact licence application procedure is documented?",
        "What explicit restriction is stated, and what fee and approval sequence are required?",
        "Which directly supported rule applies, and what exception route is fully established?",
        "What documented condition exists, and what customer escalation timetable is specified?",
        "Which provision is directly supported, and what filing form and recipient are mandated?",
        "What direct requirement applies, and what waiver criteria are expressly documented?",
        "Which stated rule is supported, and what detailed submission workflow is established?",
        "What obligation is directly evidenced, and what precise review timetable applies?",
        "Which restriction is supported, and what comprehensive renewal procedure is documented?",
    ]
    none = [
        "What exact licensing authority, form, fee, and filing date are established by this evidence?",
        "Does the evidence expressly permit or prohibit a distinct financing purpose and explain the approval process?",
        "What detailed customer redress procedure and resolution deadline are documented?",
        "Which regulatory waiver body, eligibility rule, and application fee are specified?",
        "What complete registration process for a new institution does the source establish?",
        "What exact transaction permission and exception timetable are directly set out?",
        "Which mandatory application documents and filing deadline are stated?",
        "What consumer complaint escalation route and fee refund procedure are established?",
        "What precise licence renewal authority and annual payment are documented?",
        "Which approval sequence and legal remedy are expressly established by this evidence?",
    ]
    values = {"full": full, "partial": partial, "none": none}[label]
    handles = ["E1", "E2"] if label == "full" and index >= 7 else ["E1"]
    annotation = {"full": "The requested direct proposition is supported by the selected source evidence.",
                  "partial": "The direct condition is supported; the additional procedural material aspect is not established.",
                  "none": "The evidence may be related but does not establish the requested detailed permission/procedure/licence proposition."}[label]
    return values[index], handles, annotation


def create_new_validation() -> dict[str, Any]:
    if NEW_VALIDATION.exists():
        return json.loads(NEW_VALIDATION.read_text(encoding="utf-8"))
    source_payload = json.loads(SOURCE_CONTEXT_CACHE.read_text(encoding="utf-8"))
    candidates = [row for row in source_payload["results"] if row["answerable"] and len(row["context_pack"]["evidence_items"]) >= 2][20:30]
    if len(candidates) != 10:
        raise ValueError("insufficient independent source packs for v3 validation")
    cases = []
    for label in ("full", "partial", "none"):
        for index, row in enumerate(candidates):
            question, handles, annotation = _new_templates(index, label)
            citation_map = row["context_pack"]["citation_map"]
            cases.append({"query_id": f"SG3-{label.upper()}-{index + 1:02d}", "question": question, "expected_support": label,
                          "context_reference": {"source_cache": str(SOURCE_CONTEXT_CACHE), "source_query_id": row["query_id"], "handles": handles,
                                                "chunk_ids": [chunk for handle in handles for chunk in citation_map[handle]["chunk_ids"]]},
                          "expected_supporting_handles": handles if label != "none" else [], "annotation": annotation,
                          "review_status": "machine-prepared independent validation benchmark pending human semantic audit"})
    payload = {"schema_version": 1, "benchmark_version": "evidence_support_validation_v3", "case_count": 30,
               "class_distribution": {label: 10 for label in ("full", "partial", "none")}, "cases": cases,
               "review_status": "machine-prepared independent validation benchmark pending human semantic audit"}
    NEW_VALIDATION.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def run_new_validation(*, pacing_seconds: float = 4.0) -> dict[str, Any]:
    benchmark = create_new_validation()
    return _evaluate(benchmark["cases"], prompt_version=SUPPORT_GATE_V1_1_PROMPT_VERSION, artifact=NEW_VALIDATION_RESULTS, pacing_seconds=pacing_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("audit", "calibration", "validation", "all"), default="all")
    parser.add_argument("--pacing-seconds", type=float, default=4.0)
    args = parser.parse_args()
    if args.stage in {"audit", "all"}:
        audit = run_audit(pacing_seconds=args.pacing_seconds)
        print(json.dumps({"audit": audit["summary"]}, ensure_ascii=False))
    if args.stage in {"calibration", "all"}:
        calibration = run_v11_calibration(pacing_seconds=args.pacing_seconds)
        print(json.dumps({"calibration": calibration["metrics"]}, ensure_ascii=False))
    if args.stage in {"validation", "all"}:
        validation = run_new_validation(pacing_seconds=args.pacing_seconds)
        print(json.dumps({"validation": validation["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
