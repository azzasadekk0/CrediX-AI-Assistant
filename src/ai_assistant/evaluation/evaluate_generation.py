"""DEV-only baseline evaluation for the frozen grounded RAG branch.

Gold labels are used exclusively after generation for answerability and citation
measurement; they are never passed to the LLM prompt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable

from ..context.builder import ContextBuilder
from ..context.config import load_frozen_context_config
from ..generation.errors import CitationValidationError, GroundedGenerationError, StructuredOutputError
from ..generation.generator import GroundedGenerator
from ..llm.config import LLMSettings
from ..llm.errors import LLMProviderError
from ..llm.factory import get_llm
from ..models import ContextPack, ExpandedEvidence
from ..reranking.reranked_retriever import RerankedRetriever
from .evaluate_context import _candidate, cache_identity
from .evaluate_reranker import load_split_cases
from .evaluate_retrieval import DEFAULT_BUNDLE, DEFAULT_OUTPUT, DEFAULT_SPLITS, DEFAULT_V2_DATASET, DEFAULT_V2_META, load_and_validate_splits, load_cases
from .schema import GoldenRetrievalCase


ROOT = Path(__file__).resolve().parents[3]
RERANK_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "context_eval_reranked_v1.json"
CONTEXT_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "generation_dev_contexts_v1.json"
PROGRESS_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "generation_dev_progress_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _context_identity(bundle_dir: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {"corpus": {"chunks_sha256": _sha256(bundle_dir / "chunks.jsonl"),
                        "release_manifest_sha256": _sha256(bundle_dir / "release_manifest.json")},
            "reranker": cache_identity(bundle_dir, meta)["reranker"],
            "context_config": load_frozen_context_config().to_dict(), "benchmark_version": meta["dataset_version"]}


def _context_from_dict(value: dict[str, Any]) -> ContextPack:
    fields = {name: value[name] for name in ContextPack.__dataclass_fields__ if name != "evidence_items"}
    return ContextPack(evidence_items=[ExpandedEvidence(**item) for item in value["evidence_items"]], **fields)


def _rerank_rows(bundle_dir: Path, meta: dict[str, Any]) -> dict[str, dict[str, Any]]:
    payload = json.loads(RERANK_CACHE.read_text(encoding="utf-8"))
    if payload.get("identity") != cache_identity(bundle_dir, meta):
        raise ValueError("Step 5 rerank cache is stale; cannot reuse it for generation evaluation")
    return {row["query_id"]: row for row in payload["results"]}


def build_dev_context_cache(dataset: Path = DEFAULT_V2_DATASET, meta_path: Path = DEFAULT_V2_META,
                            splits_path: Path = DEFAULT_SPLITS, bundle_dir: Path = DEFAULT_BUNDLE,
                            cache_path: Path = CONTEXT_CACHE) -> dict[str, Any]:
    """Create each frozen DEV ContextPack once, reusing the certified rerank cache."""
    answerable_cases, meta = load_split_cases(dataset, meta_path, splits_path, bundle_dir, "dev")
    all_cases = load_cases(dataset)
    splits = load_and_validate_splits(all_cases, splits_path, meta)
    by_id = {case.query_id: case for case in all_cases}
    cases = answerable_cases + [by_id[query_id] for query_id in splits["dev_no_answer"]]
    identity = _context_identity(bundle_dir, meta)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity:
            raise ValueError("generation DEV context cache is stale")
    else:
        payload = {"schema_version": 1, "identity": identity, "split": "dev", "results": []}
    existing = {row["query_id"]: row for row in payload["results"]}
    rerank_rows = _rerank_rows(bundle_dir, meta)
    builder = ContextBuilder(load_frozen_context_config())
    reranked_retriever: RerankedRetriever | None = None
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    for case in cases:
        if case.query_id in existing:
            continue
        started = time.perf_counter()
        cached = rerank_rows.get(case.query_id)
        rerank_seconds = 0.0
        if cached is not None:
            candidates = [_candidate(item) for item in cached["reranked_results"]]
        else:
            reranked_retriever = reranked_retriever or RerankedRetriever()
            rerank_started = time.perf_counter()
            candidates = reranked_retriever.retrieve(case.query).reranked_results
            rerank_seconds = time.perf_counter() - rerank_started
        context_pack = builder.build_from_candidates(case.query, candidates)
        existing[case.query_id] = {"query_id": case.query_id, "query": case.query, "answerable": case.answerable,
                                   "context_pack": context_pack.to_dict(), "context_build_seconds": time.perf_counter() - started,
                                   "rerank_seconds_if_uncached": rerank_seconds}
        payload["results"] = [existing[case.query_id] for case in cases if case.query_id in existing]
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def _summary(values: Iterable[float]) -> dict[str, float | None]:
    numbers = list(values)
    return {"mean": sum(numbers) / len(numbers) if numbers else None, "median": median(numbers) if numbers else None,
            "p95": _percentile(numbers, 0.95), "min": min(numbers) if numbers else None, "max": max(numbers) if numbers else None}


def _is_arabic(text: str) -> bool:
    return any("\u0600" <= character <= "\u06ff" for character in text)


def _record(case: GoldenRetrievalCase, cached: dict[str, Any], generator: GroundedGenerator) -> dict[str, Any]:
    pack = _context_from_dict(cached["context_pack"])
    base = {"query_id": case.query_id, "query": case.query, "gold_answerable": case.answerable,
            "context_handles": list(pack.citation_map), "context_evidence_count": pack.evidence_count,
            "context_tokens": pack.estimated_context_tokens, "context_build_seconds": cached["context_build_seconds"],
            "rerank_seconds_if_uncached": cached["rerank_seconds_if_uncached"]}
    try:
        answer = generator.generate(pack)
        sources = [asdict(source) for source in answer.resolved_sources]
        raw = generator.last_trace.get("raw_first_pass") or {}
        trace = generator.last_trace
        normalization_details = trace.get("citation_normalization_details") or []
        return {**base, "generated_answer": answer.answer, "no_answer": answer.no_answer, "citations": answer.citations,
                "raw_first_pass_answer": raw.get("answer"), "raw_first_pass_citations": raw.get("citations"), "raw_first_pass_no_answer": raw.get("no_answer"),
                "canonicalized_first_pass_citations": (trace.get("first_pass_canonicalized") or {}).get("citations"),
                "canonicalized_repair_citations": (trace.get("repair_canonicalized") or {}).get("citations"),
                "first_pass_token_usage": trace.get("first_pass_token_usage"), "repair_token_usage": trace.get("repair_token_usage"),
                "resolved_sources": sources, "resolved_chunk_ids": list(dict.fromkeys(chunk_id for source in sources for chunk_id in source["chunk_ids"])),
                "resolved_parent_ids": list(dict.fromkeys(source["parent_id"] for source in sources)),
                "provider": answer.provider, "model": answer.model, "prompt_version": answer.prompt_version,
                "schema_success": True, "validation_success": True, "citation_repair_used": answer.citation_repair_attempted,
                "citation_normalization_used": answer.citation_normalization_used, "citation_normalization_count": answer.citation_normalization_count,
                "canonicalization_used": answer.citation_normalization_used, "direct_citation_valid": trace.get("direct_citation_valid"),
                "identifiers_normalized": sum(item.get("status") in {"normalized", "decorated_normalized"} for item in normalization_details),
                "identifiers_unresolved": sum(item.get("status") == "unresolved" for item in normalization_details),
                "normalization_details": normalization_details, "safety_stop": bool(trace.get("citation_safety_stop")),
                "generation_latency_seconds": answer.generation_latency_seconds, "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens, "total_tokens": answer.total_tokens, "error_type": None, "error": None}
    except Exception as exc:
        raw = generator.last_trace.get("raw_first_pass") or {}
        trace = generator.last_trace
        normalization_details = trace.get("citation_normalization_details") or []
        return {**base, "generated_answer": None, "no_answer": None, "citations": [],
                "raw_first_pass_answer": raw.get("answer"), "raw_first_pass_citations": raw.get("citations"), "raw_first_pass_no_answer": raw.get("no_answer"), "resolved_sources": [],
                "canonicalized_first_pass_citations": (trace.get("first_pass_canonicalized") or {}).get("citations"),
                "canonicalized_repair_citations": (trace.get("repair_canonicalized") or {}).get("citations"),
                "first_pass_token_usage": trace.get("first_pass_token_usage"), "repair_token_usage": trace.get("repair_token_usage"),
                "resolved_chunk_ids": [], "resolved_parent_ids": [], "provider": None, "model": None, "prompt_version": generator.config.prompt_version,
                "schema_success": not isinstance(exc, StructuredOutputError), "validation_success": False,
                "citation_repair_used": bool(trace.get("repair_attempted")), "citation_normalization_used": bool(trace.get("citation_normalization_count")),
                "citation_normalization_count": int(trace.get("citation_normalization_count") or 0),
                "canonicalization_used": False, "direct_citation_valid": trace.get("direct_citation_valid"),
                "identifiers_normalized": sum(item.get("status") in {"normalized", "decorated_normalized"} for item in normalization_details),
                "identifiers_unresolved": sum(item.get("status") == "unresolved" for item in normalization_details),
                "normalization_details": normalization_details, "safety_stop": bool(trace.get("citation_safety_stop") or isinstance(exc, GroundedGenerationError)),
                "generation_latency_seconds": None, "input_tokens": None,
                "output_tokens": None, "total_tokens": None, "error_type": type(exc).__name__, "error": str(exc)}


def _quality(records: list[dict[str, Any]], cases: dict[str, GoldenRetrievalCase]) -> dict[str, Any]:
    answerable = [record for record in records if record["gold_answerable"]]
    no_answer = [record for record in records if not record["gold_answerable"]]
    successful = [record for record in records if record["validation_success"]]
    predicted_chunks = gold_chunks = chunk_matches = predicted_parents = gold_parents = parent_matches = 0
    multi = {"chunk": {"all": 0, "partial": 0, "none": 0}, "parent": {"all": 0, "partial": 0, "none": 0}}
    multi_citations: list[int] = []
    for record in answerable:
        case = cases[record["query_id"]]
        found_chunks, expected_chunks = set(record["resolved_chunk_ids"]), set(case.relevant_chunk_ids)
        found_parents, expected_parents = set(record["resolved_parent_ids"]), set(case.relevant_parent_ids)
        predicted_chunks += len(found_chunks); gold_chunks += len(expected_chunks); chunk_matches += len(found_chunks & expected_chunks)
        predicted_parents += len(found_parents); gold_parents += len(expected_parents); parent_matches += len(found_parents & expected_parents)
        if len(expected_chunks) > 1:
            for name, found, expected in (("chunk", found_chunks, expected_chunks), ("parent", found_parents, expected_parents)):
                state = "all" if expected.issubset(found) else "partial" if found & expected else "none"
                multi[name][state] += 1
            multi_citations.append(len(record["citations"]))
    failures = [record for record in records if not record["validation_success"]]
    factual = [record for record in successful if not record["no_answer"]]
    selectivity = [len(record["citations"]) / record["context_evidence_count"] for record in factual if record["context_evidence_count"]]
    # Benchmark metadata can be stale (NA-06 is an English query). Evaluate
    # language adherence from the actual prompt language sent to the provider.
    arabic = [record for record in successful if _is_arabic(cases[record["query_id"]].query)]
    return {
        "counts": {"answerable": len(answerable), "no_answer": len(no_answer), "total": len(records)},
        "answerability": {"answer_rate": sum(record["no_answer"] is False for record in answerable) / len(answerable) if answerable else None,
                            "false_abstention_rate": sum(record["no_answer"] is True for record in answerable) / len(answerable) if answerable else None,
                            "correct_abstention_rate": sum(record["no_answer"] is True for record in no_answer) / len(no_answer) if no_answer else None,
                            "false_answer_rate": sum(record["no_answer"] is False for record in no_answer) / len(no_answer) if no_answer else None},
        "citation": {"validity_rate": len(successful) / len(records), "repair_rate": sum(record["citation_repair_used"] for record in successful) / len(records),
                     "failed_validation_rate": sum(record["error_type"] == "CitationValidationError" for record in failures) / len(records),
                     "chunk_precision_micro": chunk_matches / predicted_chunks if predicted_chunks else 0.0,
                     "chunk_recall_micro": chunk_matches / gold_chunks if gold_chunks else 0.0,
                     "parent_precision_micro": parent_matches / predicted_parents if predicted_parents else 0.0,
                     "parent_recall_micro": parent_matches / gold_parents if gold_parents else 0.0},
        "multi_evidence": {**multi, "average_citations_per_multi_evidence_answer": sum(multi_citations) / len(multi_citations) if multi_citations else 0.0},
        "selectivity": {"mean_citations_per_answer": sum(len(record["citations"]) for record in factual) / len(factual) if factual else 0.0,
                         "median_citations_per_answer": median([len(record["citations"]) for record in factual]) if factual else 0.0,
                         "mean_available_evidence_percent_cited": 100 * sum(selectivity) / len(selectivity) if selectivity else 0.0,
                         "high_citation_cases": [record["query_id"] for record in factual if record["context_evidence_count"] and len(record["citations"]) / record["context_evidence_count"] >= 0.8]},
        "reliability": {"schema_success_rate": sum(record["schema_success"] for record in records) / len(records),
                        "validation_success_rate": len(successful) / len(records), "malformed_response_rate": sum(record["error_type"] == "StructuredOutputError" for record in failures) / len(records),
                        "generation_failure_rate": len(failures) / len(records), "provider_failure_rate": sum(record["error_type"] and record["error_type"].startswith("LLM") for record in failures) / len(records)},
        "arabic_language": {"arabic_query_response_rate": sum(_is_arabic(record["generated_answer"] or "") for record in arabic) / len(arabic) if arabic else 0.0,
                            "unexpected_non_arabic_query_ids": [record["query_id"] for record in arabic if not _is_arabic(record["generated_answer"] or "")]},
    }


def _markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"]
    samples = report["sample_query_ids"]
    return "\n".join(["# Grounded RAG DEV Baseline", "", "This is answerability/citation alignment only—not semantic answer accuracy.",
                      "", "## Metrics", "", "```json", json.dumps(metrics, ensure_ascii=False, indent=2), "```",
                      "", "## Review sample query IDs", "", json.dumps(samples, ensure_ascii=False, indent=2)]) + "\n"


def _review_sample_markdown(records: list[dict[str, Any]], cases: dict[str, Any], sample_ids: dict[str, str | None]) -> str:
    """Render representative recorded outcomes without making runtime calls."""
    by_id = {record["query_id"]: record for record in records}

    def first(predicate: Any, excluded: set[str] | None = None) -> str | None:
        excluded = excluded or set()
        return next((record["query_id"] for record in records if record["query_id"] not in excluded and predicate(record)), None)

    normal = first(lambda row: row["gold_answerable"] and row["validation_success"])
    single = first(
        lambda row: row["gold_answerable"] and row["validation_success"] and len(cases[row["query_id"]].relevant_chunk_ids) == 1,
        {normal} if normal else set(),
    )
    categories = [
        ("Normal answerable", normal),
        ("No-answer", sample_ids.get("no_answer")),
        ("Single-evidence", single),
        ("Multi-evidence", sample_ids.get("multi_evidence")),
        ("High-citation", sample_ids.get("high_citation")),
        ("Citation repair", sample_ids.get("citation_repair")),
        ("False abstention", sample_ids.get("false_abstention")),
        ("False answer", sample_ids.get("false_answer")),
    ]
    lines = [
        "# Grounded RAG DEV review sample",
        "",
        "Representative recorded DEV outcomes for human review. Gold labels are evaluation-only; this document makes no semantic-correctness claim.",
    ]
    for label, query_id in categories:
        lines.extend(["", f"## {label}"])
        if not query_id:
            lines.append("No matching case was observed in this baseline.")
            continue
        record = by_id[query_id]
        answer = record.get("generated_answer")
        if answer and len(answer) > 700:
            answer = answer[:697].rstrip() + "..."
        lines.extend([
            f"Query ID: `{query_id}`",
            "", f"Question: {record['query']}",
            "", f"Gold answerable (evaluation only): `{record['gold_answerable']}`",
            "", f"No-answer: `{record.get('no_answer')}` | schema: `{record['schema_success']}` | validation: `{record['validation_success']}` | citation repair: `{record['citation_repair_used']}`",
            "", f"Citations: `{', '.join(record['citations']) if record['citations'] else 'none'}`",
        ])
        if answer:
            lines.extend(["", f"Recorded answer: {answer}"])
        elif record.get("error"):
            lines.extend(["", f"Recorded outcome: `{record['error_type']}` - {record['error']}"])
    return "\n".join(lines) + "\n"


def _load_successful_progress(path: Path, runtime: dict[str, object]) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("runtime") != runtime:
        raise ValueError("generation progress cache belongs to a different runtime configuration")
    # A rate-limit response is transport noise, not a baseline outcome.  Keep
    # every other completed record (including deterministic validation
    # failures), then retry only those requests once the provider window clears.
    return {
        row["query_id"]: row
        for row in payload.get("records", [])
        if row.get("validation_success") or row.get("error_type") != "LLMRateLimitError"
    }


def _write_progress(path: Path, runtime: dict[str, object], records: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": 1, "runtime": runtime, "records": list(records.values())}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_dev_evaluation(output_dir: Path = DEFAULT_OUTPUT, cache_path: Path = CONTEXT_CACHE,
                       progress_path: Path = PROGRESS_CACHE, *, inter_request_delay_seconds: float = 40.0) -> dict[str, Any]:
    """Execute exactly one DEV baseline for the current frozen branch."""
    cache = build_dev_context_cache(cache_path=cache_path)
    cases = load_cases(DEFAULT_V2_DATASET)
    meta = json.loads(DEFAULT_V2_META.read_text(encoding="utf-8"))
    splits = load_and_validate_splits(cases, DEFAULT_SPLITS, meta)
    dev_ids = splits["dev"] + splits["dev_no_answer"]
    by_case = {case.query_id: case for case in cases}
    cached = {row["query_id"]: row for row in cache["results"]}
    settings = LLMSettings.from_env()
    runtime = settings.diagnostics()
    generator = GroundedGenerator(get_llm(settings))
    completed = _load_successful_progress(progress_path, runtime)
    # Seed the checkpoint from the interrupted first baseline. Failures are intentionally retried only after the
    # provider's rate window has cleared; successful outputs are never regenerated.
    prior_human = output_dir / "generation_dev_human_review_v1.jsonl"
    if not completed and prior_human.exists():
        for line in prior_human.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("validation_success"):
                completed[row["query_id"]] = {key: value for key, value in row.items() if key != "review_status"}
        _write_progress(progress_path, runtime, completed)
    for query_id in dev_ids:
        if query_id in completed:
            continue
        if inter_request_delay_seconds:
            time.sleep(inter_request_delay_seconds)
        completed[query_id] = _record(by_case[query_id], cached[query_id], generator)
        _write_progress(progress_path, runtime, completed)
    records = [completed[query_id] for query_id in dev_ids]
    metrics = _quality(records, by_case)
    metrics["answerability"]["balanced_accuracy"] = (metrics["answerability"]["answer_rate"] + metrics["answerability"]["correct_abstention_rate"]) / 2
    success = [record for record in records if record["validation_success"]]
    metrics["latency_seconds"] = _summary(record["generation_latency_seconds"] for record in success if record["generation_latency_seconds"] is not None)
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        metrics.setdefault("token_usage", {})[key] = _summary(float(record[key]) for record in success if record[key] is not None)
    metrics["token_usage"]["dev_total_tokens"] = sum(record["total_tokens"] or 0 for record in success)
    metrics["token_usage"]["answerable_average_total_tokens"] = sum(record["total_tokens"] or 0 for record in success if record["gold_answerable"]) / max(1, sum(record["gold_answerable"] for record in success))
    metrics["token_usage"]["no_answer_average_total_tokens"] = sum(record["total_tokens"] or 0 for record in success if not record["gold_answerable"]) / max(1, sum(not record["gold_answerable"] for record in success))
    metrics["answer_length_characters"] = _summary(float(len(record["generated_answer"] or "")) for record in success)
    metrics["cache_preparation_seconds"] = _summary(float(record["context_build_seconds"]) for record in cache["results"])
    human_path = output_dir / "generation_dev_human_review_v1.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with human_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({**record, "review_status": "pending"}, ensure_ascii=False) + "\n")
    sample_ids = {"answerable": next((row["query_id"] for row in records if row["gold_answerable"]), None),
                  "no_answer": next((row["query_id"] for row in records if not row["gold_answerable"]), None),
                  "multi_evidence": next((case.query_id for case in by_case.values() if case.query_id in dev_ids and len(case.relevant_chunk_ids) > 1), None),
                  "high_citation": metrics["selectivity"]["high_citation_cases"][0] if metrics["selectivity"]["high_citation_cases"] else None,
                  "citation_repair": next((row["query_id"] for row in records if row["citation_repair_used"]), None),
                  "false_abstention": next((row["query_id"] for row in records if row["gold_answerable"] and row["no_answer"] is True), None),
                  "false_answer": next((row["query_id"] for row in records if not row["gold_answerable"] and row["no_answer"] is False), None)}
    report = {"schema_version": 1, "split": "dev", "runtime": runtime, "context_cache": str(cache_path),
              "metrics": metrics, "sample_query_ids": sample_ids, "semantic_accuracy_note": "No semantic answer-accuracy metric is claimed; labels cover evidence retrieval only."}
    (output_dir / "generation_dev_evaluation_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "generation_dev_evaluation_v1.md").write_text(_markdown(report), encoding="utf-8")
    (output_dir / "generation_dev_review_sample_v1.md").write_text(_review_sample_markdown(records, by_case, sample_ids), encoding="utf-8")
    return report


def write_review_sample_from_artifact(output_dir: Path = DEFAULT_OUTPUT) -> None:
    """Regenerate the sample from completed artifacts without LLM or retrieval work."""
    rows = [json.loads(line) for line in (output_dir / "generation_dev_human_review_v1.jsonl").read_text(encoding="utf-8").splitlines() if line]
    report = json.loads((output_dir / "generation_dev_evaluation_v1.json").read_text(encoding="utf-8"))
    cases = {case.query_id: case for case in load_cases(DEFAULT_V2_DATASET)}
    (output_dir / "generation_dev_review_sample_v1.md").write_text(_review_sample_markdown(rows, cases, report["sample_query_ids"]), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inter-request-delay-seconds", type=float, default=40.0)
    parser.add_argument("--write-review-sample-only", action="store_true")
    args = parser.parse_args()
    if args.inter_request_delay_seconds < 0:
        parser.error("--inter-request-delay-seconds must be non-negative")
    if args.write_review_sample_only:
        write_review_sample_from_artifact()
        return 0
    report = run_dev_evaluation(inter_request_delay_seconds=args.inter_request_delay_seconds)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
