"""Independent calibration and untouched validation for Evidence Support Gate v1.

The benchmark is machine-prepared from frozen source evidence and is explicitly
marked pending human semantic audit.  It is not the consumed RAG TEST v1 set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from statistics import median
from typing import Any

from ..llm.config import LLMSettings
from ..llm.factory import get_llm
from ..models import ContextPack, ExpandedEvidence
from ..support_gate.gate import EvidenceSupportGate
from ..support_gate.prompts import SUPPORT_GATE_PROMPT_VERSION
from .evaluate_generation import DEFAULT_OUTPUT


ROOT = Path(__file__).resolve().parents[3]
SOURCE_CONTEXT_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "generation_dev_contexts_d10_b10_top15_v1.json"
BENCHMARK = DEFAULT_OUTPUT / "evidence_support_calibration_v1.json"
SPLIT = DEFAULT_OUTPUT / "evidence_support_calibration_v1.splits.json"
CALIBRATION = DEFAULT_OUTPUT / "evidence_support_gate_v1_calibration.json"
VALIDATION = DEFAULT_OUTPUT / "evidence_support_gate_v1_validation.json"
RESULT_DIR = ROOT / "artifacts" / "ai_assistant" / "cache"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_pack(value: dict[str, Any], handles: list[str], question: str) -> ContextPack:
    pack = value["context_pack"]
    by_handle = {item["citation_handle"]: item for item in pack["evidence_items"]}
    items = [ExpandedEvidence(**by_handle[handle]) for handle in handles]
    citation_map = {handle: pack["citation_map"][handle] for handle in handles}
    return ContextPack(question, items, [chunk for item in items for chunk in item.chunk_ids],
                       list(dict.fromkeys(item.parent_id for item in items)), list(dict.fromkeys(item.document_id for item in items)),
                       citation_map, len(items), len({item.parent_id for item in items}), len({item.document_id for item in items}),
                       sum(len(item.expanded_text.split()) for item in items), {"benchmark_context": True, "source_query_id": value["query_id"]})


def _benchmark_templates(index: int, label: str) -> tuple[str, list[str], str]:
    full = [
        "What direct requirement is stated in the supplied regulatory evidence?",
        "Which condition does the supplied provision explicitly establish?",
        "What rule or restriction is directly set out by this evidence?",
        "What directly stated obligation applies in the supplied material?",
        "Which explicit provision should be taken from the evidence without inference?",
        "What factual requirement is expressly described in the provided text?",
        "What direct condition does this source impose for the matter it addresses?",
        "What is the directly stated regulatory position in the supplied evidence?",
        "Which requirement can be stated solely from this provision?",
        "What explicit limitation or condition appears in the evidence?",
        "What direct operational rule is documented by the supplied source?",
        "Which stated requirement is supported without relying on outside knowledge?",
        "What is the expressly documented condition in this evidence block?",
        "What direct rule follows from the text as written?",
        "What explicitly stated provision is available from the supplied material?",
        "What direct compliance condition does the evidence set out?",
        "Which directly documented rule applies to the issue in the evidence?",
        "What direct factual requirement is established by this source?",
        "What limitation is expressly described in the provided evidence?",
        "What requirement is explicitly supported by the supplied provision?",
    ]
    partial = [
        "What direct provision is stated in the evidence, and what appeal route and filing deadline does it establish?",
        "Which explicit condition applies, and which authority must approve an exception under this evidence?",
        "What rule is directly stated, and what exact fee does the evidence require for compliance?",
        "What documented obligation applies, and what detailed step-by-step submission process is specified?",
        "Which stated restriction applies, and what review timetable does the evidence establish?",
        "What direct condition is provided, and which form and submission deadline are required?",
        "What requirement is explicitly mentioned, and what appeal procedure is available?",
        "Which provision is supported, and what exception criteria are documented in detail?",
        "What direct rule appears here, and what licensing authority and fee are specified?",
        "What restriction is stated, and what exact notification procedure must be followed?",
        "Which obligation is documented, and what evidence establishes a waiver process?",
        "What direct condition is supported, and what statutory deadline is imposed?",
        "Which rule is stated, and what application procedure is fully described?",
        "What requirement is direct, and what complaint escalation process is specified?",
        "Which limitation is in the text, and what approval workflow is established?",
        "What provision is supported, and what exact reporting template is required?",
        "Which documented condition applies, and what filing authority is named?",
        "What rule is stated, and what exception appeal and timetable are established?",
        "Which explicit requirement exists, and what detailed licence process is provided?",
        "What direct regulatory point is supported, and what fee schedule is specified?",
    ]
    none = [
        "Which authority issues the licence, what application fee applies, and what filing deadline governs this matter?",
        "What detailed appeal procedure and statutory deadline does the evidence establish?",
        "Which exact form must be submitted, to whom, and by what date?",
        "What customer complaint steps, escalation route, and resolution period are set out?",
        "What permission to finance a transaction does the evidence explicitly grant or prohibit?",
        "What licensing eligibility criteria and annual fee are directly established here?",
        "Which authority approves an exemption and what application process applies?",
        "What mandatory notification timeline and filing format does this source provide?",
        "What exact registration procedure is stated for a new institution?",
        "Which legal remedy, appeal body, and filing deadline are specified?",
        "What precise numerical cap and calculation method does this evidence establish?",
        "What process is required to dispute a customer charge under this material?",
        "Which permission is expressly given for a new financial product?",
        "What complete procedure is documented for obtaining a regulatory waiver?",
        "What licence renewal timetable and payment amount are specified?",
        "What exact consumer disclosure form and approval sequence are required?",
        "Which authority receives the application and what supporting documents are mandatory?",
        "What detailed process is set out for approving a transaction exception?",
        "What exact deadline and fee apply to the requested regulatory filing?",
        "What explicit permission or prohibition concerning a financing purpose is established?",
    ]
    if label == "full":
        return full[index], ["E1", "E2"] if index >= 15 else ["E1"], "The requested provision is directly stated by the referenced evidence block(s)."
    if label == "partial":
        return partial[index], ["E1"], "The evidence directly supports a stated provision but does not establish the additional requested procedural/detail aspect."
    return none[index], ["E1"], "The evidence is source-grounded and topic-adjacent, but it does not directly establish the requested licence/procedure/permission/detail proposition."


def create_benchmark() -> dict[str, Any]:
    """Create exactly 60 deterministic, source-referenced machine-prepared cases."""
    if BENCHMARK.exists() and SPLIT.exists():
        return json.loads(BENCHMARK.read_text(encoding="utf-8"))
    source = json.loads(SOURCE_CONTEXT_CACHE.read_text(encoding="utf-8"))
    candidates = [row for row in source["results"] if row["answerable"] and len(row["context_pack"]["evidence_items"]) >= 2]
    if len(candidates) < 20:
        raise ValueError("frozen source cache does not have enough evidence packs for calibration benchmark")
    cases: list[dict[str, Any]] = []
    for label in ("full", "partial", "none"):
        for index, source_row in enumerate(candidates[:20]):
            question, handles, annotation = _benchmark_templates(index, label)
            reference = source_row["context_pack"]["citation_map"]
            cases.append({"query_id": f"SG-{label.upper()}-{index + 1:02d}", "question": question, "expected_support": label,
                          "context_reference": {"source_cache": str(SOURCE_CONTEXT_CACHE), "source_query_id": source_row["query_id"],
                                                "handles": handles, "chunk_ids": [chunk for handle in handles for chunk in reference[handle]["chunk_ids"]]},
                          "expected_supporting_handles": handles if label != "none" else [], "annotation": annotation,
                          "review_status": "machine-prepared calibration benchmark pending human semantic audit"})
    payload = {"schema_version": 1, "benchmark_version": "evidence_support_calibration_v1", "source_context_cache_sha256": _sha256(SOURCE_CONTEXT_CACHE),
               "case_count": len(cases), "class_distribution": {label: 20 for label in ("full", "partial", "none")}, "cases": cases,
               "review_status": "machine-prepared calibration benchmark pending human semantic audit"}
    BENCHMARK.parent.mkdir(parents=True, exist_ok=True)
    BENCHMARK.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Fixed 14/6 class balance, recorded before any model execution.
    split = {"schema_version": 1, "benchmark_version": payload["benchmark_version"], "method": "class-stratified fixed order before tuning",
             "calibration": [case["query_id"] for label in ("full", "partial", "none") for case in [item for item in cases if item["expected_support"] == label][:14]],
             "validation": [case["query_id"] for label in ("full", "partial", "none") for case in [item for item in cases if item["expected_support"] == label][14:]]}
    split["class_distribution"] = {"calibration": {label: 14 for label in ("full", "partial", "none")}, "validation": {label: 6 for label in ("full", "partial", "none")}}
    SPLIT.write_text(json.dumps(split, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = ("full", "partial", "none")
    confusion = {expected: {predicted: 0 for predicted in labels} for expected in labels}
    for row in rows:
        confusion[row["expected_support"]][row["predicted_support"]] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        predicted = sum(confusion[expected][label] for expected in labels)
        actual = sum(confusion[label].values())
        per_class[label] = {"precision": tp / predicted if predicted else 0.0, "recall": tp / actual if actual else 0.0, "support": actual}
    return {"overall_accuracy": sum(confusion[label][label] for label in labels) / len(rows) if rows else 0.0,
            "per_class": per_class, "confusion_matrix": confusion,
            "safety_errors": {"none_to_full": confusion["none"]["full"], "none_to_partial": confusion["none"]["partial"],
                              "partial_to_full": confusion["partial"]["full"], "full_to_none": confusion["full"]["none"]}}


def _result_paths(split_name: str, settings: LLMSettings) -> tuple[Path, Path]:
    stem = f"evidence_support_gate_v1_{split_name}_{settings.provider}_{settings.model}".replace("/", "_")
    return RESULT_DIR / f"{stem}.jsonl", CALIBRATION if split_name == "calibration" else VALIDATION


def run_split(split_name: str, *, pacing_seconds: float = 4.0) -> dict[str, Any]:
    benchmark = create_benchmark()
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    ids = split[split_name]
    by_case = {case["query_id"]: case for case in benchmark["cases"]}
    source = json.loads(SOURCE_CONTEXT_CACHE.read_text(encoding="utf-8"))
    source_rows = {row["query_id"]: row for row in source["results"]}
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    jsonl_path, report_path = _result_paths(split_name, settings)
    records = {json.loads(line)["query_id"]: json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line} if jsonl_path.exists() else {}
    gate = EvidenceSupportGate(get_llm(settings))
    for index, query_id in enumerate(ids):
        if query_id in records:
            continue
        if index and records:
            time.sleep(pacing_seconds)
        case = by_case[query_id]
        reference = case["context_reference"]
        pack = _source_pack(source_rows[reference["source_query_id"]], reference["handles"], case["question"])
        assessment = gate.assess(pack)
        records[query_id] = {"query_id": query_id, "question": case["question"], "expected_support": case["expected_support"],
                             "expected_supporting_handles": case["expected_supporting_handles"], "context_reference": reference,
                             "predicted_support": assessment.support, "assessment": assessment.to_dict(), "gate_trace": gate.last_trace}
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for identifier in ids:
                if identifier in records:
                    handle.write(json.dumps(records[identifier], ensure_ascii=False) + "\n")
    ordered = [records[query_id] for query_id in ids]
    report = {"schema_version": 1, "benchmark_version": benchmark["benchmark_version"], "split": split_name,
              "prompt_version": SUPPORT_GATE_PROMPT_VERSION, "runtime": runtime, "records": len(ordered), "result_artifact": str(jsonl_path),
              "metrics": _metrics(ordered), "failures": [row for row in ordered if row["expected_support"] != row["predicted_support"]],
              "calibration_tuning": "none; support_gate_v1 prompt was frozen before untouched validation" if split_name == "validation" else "none; baseline prompt was retained",
              "policy": {"full": "future GroundedGenerator candidate", "partial": "do not silently answer unsupported aspects", "none": "grounded no-answer", "classifier_failure": "fail closed to none"}}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("calibration", "validation", "all"), default="all")
    parser.add_argument("--pacing-seconds", type=float, default=4.0)
    args = parser.parse_args()
    if args.split in {"calibration", "all"}:
        calibration = run_split("calibration", pacing_seconds=args.pacing_seconds)
        print(json.dumps({"calibration": calibration["metrics"]}, ensure_ascii=False))
    if args.split in {"validation", "all"}:
        validation = run_split("validation", pacing_seconds=args.pacing_seconds)
        print(json.dumps({"validation": validation["metrics"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
