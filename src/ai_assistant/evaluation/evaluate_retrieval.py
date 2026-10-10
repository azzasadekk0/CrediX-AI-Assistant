"""Evaluate independent Dense and BM25 retrieval against Golden Retrieval v1."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from ..retrieval.hybrid_retriever import HybridRetriever
from .metrics import mean_metrics, ranking_metrics, relevant_ranks
from .schema import GoldenRetrievalCase

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = Path(__file__).resolve().parent / "datasets" / "golden_retrieval_v1.jsonl"
DEFAULT_META = Path(__file__).resolve().parent / "datasets" / "golden_retrieval_v1.meta.json"
DEFAULT_V2_DATASET = Path(__file__).resolve().parent / "datasets" / "golden_retrieval_v2.jsonl"
DEFAULT_V2_META = Path(__file__).resolve().parent / "datasets" / "golden_retrieval_v2.meta.json"
DEFAULT_SPLITS = Path(__file__).resolve().parent / "datasets" / "golden_retrieval_v2.splits.json"
DEFAULT_BUNDLE = ROOT / "src" / "rag_data" / "current"
DEFAULT_OUTPUT = ROOT / "artifacts" / "ai_assistant" / "evaluation"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_cases(path: Path) -> list[GoldenRetrievalCase]:
    cases = [GoldenRetrievalCase.from_dict(value) for value in _read_jsonl(path)]
    query_ids = [case.query_id for case in cases]
    if len(query_ids) != len(set(query_ids)):
        duplicates = sorted(key for key, count in Counter(query_ids).items() if count > 1)
        raise ValueError(f"duplicate query_id values: {', '.join(duplicates)}")
    return cases


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_cases(cases: Iterable[GoldenRetrievalCase], bundle_dir: Path) -> None:
    """Fail clearly when a reviewed label no longer points into this release."""
    chunks = _read_jsonl(bundle_dir / "chunks.jsonl")
    parents = _read_jsonl(bundle_dir / "parents.jsonl")
    chunks_by_id = {row["chunk_id"]: row for row in chunks}
    parents_by_id = {row["parent_id"]: row for row in parents}
    for case in cases:
        for chunk_id in case.relevant_chunk_ids:
            if chunk_id not in chunks_by_id:
                raise ValueError(f"{case.query_id}: unknown relevant chunk {chunk_id}")
            chunk = chunks_by_id[chunk_id]
            if chunk["document_id"] not in case.expected_document_ids:
                raise ValueError(f"{case.query_id}: expected_document_ids does not include {chunk['document_id']}")
            if chunk["parent_id"] not in case.relevant_parent_ids:
                raise ValueError(f"{case.query_id}: relevant_parent_ids does not include {chunk['parent_id']}")
        for parent_id in case.relevant_parent_ids:
            if parent_id not in parents_by_id:
                raise ValueError(f"{case.query_id}: unknown relevant parent {parent_id}")
            parent = parents_by_id[parent_id]
            if parent["document_id"] not in case.expected_document_ids:
                raise ValueError(
                    f"{case.query_id}: expected_document_ids does not include parent document "
                    f"{parent['document_id']}"
                )


def validate_metadata(cases: list[GoldenRetrievalCase], meta: dict[str, Any], bundle_dir: Path) -> None:
    answerable = [case for case in cases if case.answerable]
    unanswerable = [case for case in cases if not case.answerable]
    expected = {
        "number_of_answerable_queries": len(answerable),
        "number_of_unanswerable_queries": len(unanswerable),
        "query_type_counts": dict(sorted(Counter(case.query_type for case in cases).items())),
        "difficulty_counts": dict(sorted(Counter(case.difficulty for case in cases).items())),
        "document_coverage_counts": dict(sorted(Counter(doc for case in answerable for doc in case.expected_document_ids).items())),
        "chunks_file_sha256": _sha256(bundle_dir / "chunks.jsonl"),
        "release_manifest_sha256": _sha256(bundle_dir / "release_manifest.json"),
    }
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f"dataset metadata mismatch for {key}: expected {value!r}, got {meta.get(key)!r}")


def load_and_validate_splits(cases: list[GoldenRetrievalCase], path: Path, meta: dict[str, Any]) -> dict[str, list[str]]:
    splits = json.loads(path.read_text(encoding="utf-8"))
    required = {"dev", "test", "dev_no_answer", "test_no_answer"}
    if not isinstance(splits, dict) or set(splits) != required:
        raise ValueError("split file must contain dev, test, dev_no_answer, and test_no_answer")
    answerable = {case.query_id for case in cases if case.answerable}
    no_answer = {case.query_id for case in cases if not case.answerable}
    for name, expected in (("dev", answerable), ("test", answerable), ("dev_no_answer", no_answer), ("test_no_answer", no_answer)):
        values = splits[name]
        if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
            raise ValueError(f"{name} must be a list of non-empty query IDs")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} contains duplicate query IDs")
        if not set(values).issubset(expected):
            raise ValueError(f"{name} contains unknown or wrong-scope query IDs")
    if set(splits["dev"]) & set(splits["test"]) or set(splits["dev_no_answer"]) & set(splits["test_no_answer"]):
        raise ValueError("split assignments overlap")
    if set(splits["dev"]) | set(splits["test"]) != answerable or set(splits["dev_no_answer"]) | set(splits["test_no_answer"]) != no_answer:
        raise ValueError("split assignments are missing query IDs")
    split_counts = meta.get("split_counts", {})
    if not isinstance(split_counts, dict):
        raise ValueError("metadata split_counts must be an object")
    unknown_count_keys = set(split_counts) - required
    if unknown_count_keys:
        raise ValueError(f"metadata split_counts contains unknown keys: {', '.join(sorted(unknown_count_keys))}")
    for key, value in split_counts.items():
        if len(splits[key]) != value:
            raise ValueError(f"split metadata mismatch for {key}")
    return splits


def _group_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        for key in row[field] if isinstance(row[field], list) else [row[field]]:
            grouped[key].append(row["metrics"])
    return {key: mean_metrics(values) for key, values in sorted(grouped.items())}


def _run_retriever(cases: list[GoldenRetrievalCase]) -> tuple[dict[str, Any], dict[str, Any]]:
    retriever = HybridRetriever()
    dense_rows: list[dict[str, Any]] = []
    bm25_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for case in cases:
        if not case.answerable:
            continue
        result = retriever.retrieve(case.query, dense_top_k=10, bm25_top_k=10)
        dense_ids = [candidate.chunk_id for candidate in result.dense_results]
        bm25_ids = [candidate.chunk_id for candidate in result.bm25_results]
        dense_ranks = relevant_ranks(dense_ids, case.relevant_chunk_ids, 10)
        bm25_ranks = relevant_ranks(bm25_ids, case.relevant_chunk_ids, 10)
        shared = {"query_id": case.query_id, "query": case.query, "documents": case.expected_document_ids,
                  "query_type": case.query_type, "difficulty": case.difficulty, "language": case.language}
        dense_rows.append({**shared, "metrics": ranking_metrics(dense_ids, case.relevant_chunk_ids)})
        bm25_rows.append({**shared, "metrics": ranking_metrics(bm25_ids, case.relevant_chunk_ids)})
        diagnostics.append({"query_id": case.query_id, "query": case.query, "relevant_chunk_ids": case.relevant_chunk_ids,
                            "dense_top_10_ids": dense_ids, "dense_relevant_ranks": dense_ranks,
                            "dense_first_relevant_rank": dense_ranks[0] if dense_ranks else None,
                            "bm25_top_10_ids": bm25_ids, "bm25_relevant_ranks": bm25_ranks,
                            "bm25_first_relevant_rank": bm25_ranks[0] if bm25_ranks else None})
    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return {"aggregate_metrics": mean_metrics([row["metrics"] for row in rows]),
                "by_document": _group_metrics(rows, "documents"), "by_query_type": _group_metrics(rows, "query_type"),
                "by_difficulty": _group_metrics(rows, "difficulty"), "by_language": _group_metrics(rows, "language")}
    return {"dense": summary(dense_rows), "bm25": summary(bm25_rows)}, {"per_query": diagnostics}


def _markdown(report: dict[str, Any]) -> str:
    def metric_table(metrics: dict[str, float]) -> str:
        return "\n".join(["| Metric | Value |", "|---|---:|"] + [f"| {key} | {value:.4f} |" for key, value in metrics.items()])
    lines = ["# Dataset Summary", "", f"Evaluation split: {report['split']}",
             f"Answerable queries: {report['dataset_summary']['answerable_queries']}",
             f"No-answer cases reserved for later: {report['dataset_summary']['unanswerable_queries']}", "", "# Dense Baseline", "",
             metric_table(report["dense"]["aggregate_metrics"]), "", "# BM25 Baseline", "", metric_table(report["bm25"]["aggregate_metrics"])]
    for heading, key in (("Metrics by Document", "by_document"), ("Metrics by Query Type", "by_query_type")):
        lines.extend(["", f"# {heading}", ""])
        for name, metrics in report["dense"][key].items():
            lines.extend([f"## Dense — {name}", metric_table(metrics), ""])
        for name, metrics in report["bm25"][key].items():
            lines.extend([f"## BM25 — {name}", metric_table(metrics), ""])
    failures = report["failure_cases"]
    lines.extend(["# Failure Cases", "", f"Dense top-10 failures: {len(failures['dense'])}", f"BM25 top-10 failures: {len(failures['bm25'])}",
                  "", "# No-Answer Cases Reserved for Later", "", "These cases are excluded from ordinary retrieval metrics."])
    return "\n".join(lines) + "\n"


def evaluate(dataset: Path = DEFAULT_DATASET, meta_path: Path = DEFAULT_META, bundle_dir: Path = DEFAULT_BUNDLE,
             output_dir: Path = DEFAULT_OUTPUT, splits_path: Path | None = None, split: str = "all") -> dict[str, Any]:
    cases = load_cases(dataset)
    validate_cases(cases, bundle_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_metadata(cases, meta, bundle_dir)
    if split not in {"all", "dev", "test"}:
        raise ValueError("split must be dev, test, or all")
    if split != "all":
        splits_path = splits_path or DEFAULT_SPLITS
        ids = set(load_and_validate_splits(cases, splits_path, meta)[split])
        cases = [case for case in cases if case.query_id in ids]
    summaries, diagnostics = _run_retriever(cases)
    failures = {name: [row["query_id"] for row in diagnostics["per_query"] if row[f"{name}_first_relevant_rank"] is None] for name in ("dense", "bm25")}
    report = {"split": split, "dataset_summary": {"dataset_version": meta["dataset_version"],
              "answerable_queries": sum(case.answerable for case in cases),
              # Keep this count scoped to the selected evaluation split.  The
              # previous implementation copied the full-dataset value (10)
              # into DEV/TEST reports, even though those scopes contain only
              # answerable cases.
              "unanswerable_queries": sum(not case.answerable for case in cases),
              "unanswerable_queries_total": meta["number_of_unanswerable_queries"],
              "metadata": meta}, **summaries,
              "diagnostics": diagnostics, "failure_cases": failures,
              "dense_succeeds_bm25_fails": sorted(set(failures["bm25"]) - set(failures["dense"])),
              "bm25_succeeds_dense_fails": sorted(set(failures["dense"]) - set(failures["bm25"]))}
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"v2_{split}" if meta["dataset_version"] == "golden_retrieval_v2" else "v1"
    (output_dir / f"baseline_retrieval_{suffix}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / f"baseline_retrieval_{suffix}.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_V2_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_V2_META)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="all")
    args = parser.parse_args()
    report = evaluate(args.dataset, args.metadata, args.bundle_dir, args.output_dir, args.splits, args.split)
    print(json.dumps({"dense": report["dense"]["aggregate_metrics"], "bm25": report["bm25"]["aggregate_metrics"],
                      "failure_cases": report["failure_cases"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
