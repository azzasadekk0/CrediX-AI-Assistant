"""DEV-only RRF tuning with one cached Dense/BM25 retrieval pass per query."""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..models import RetrievalCandidate
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.rrf import RRFConfig, fuse_ranked_candidates
from .evaluate_retrieval import (
    DEFAULT_BUNDLE,
    DEFAULT_OUTPUT,
    DEFAULT_SPLITS,
    DEFAULT_V2_DATASET,
    DEFAULT_V2_META,
    load_and_validate_splits,
    load_cases,
    validate_cases,
    validate_metadata,
)
from .metrics import hit_rate_at_k, mean_metrics, ranking_metrics, recall_at_k
from .schema import GoldenRetrievalCase


MAX_CANDIDATES = 50
CANDIDATE_SIZES = (10, 20, 30, 50)
RRF_KS = (10, 20, 40, 60, 80, 100)
BM25_WEIGHTS = (0.25, 0.50, 0.75, 1.00, 1.25)


@dataclass(frozen=True, slots=True)
class CachedRanking:
    """One answerable case and its only Dense/BM25 retrieval observations."""

    case: GoldenRetrievalCase
    dense_results: list[RetrievalCandidate]
    bm25_results: list[RetrievalCandidate]


def load_dev_cases(dataset: Path, meta_path: Path, splits_path: Path, bundle_dir: Path) -> tuple[list[GoldenRetrievalCase], dict[str, Any]]:
    """Load and validate the benchmark, returning DEV answerable cases only."""
    cases = load_cases(dataset)
    validate_cases(cases, bundle_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_metadata(cases, meta, bundle_dir)
    dev_ids = set(load_and_validate_splits(cases, splits_path, meta)["dev"])
    return [case for case in cases if case.query_id in dev_ids], meta


def cache_rankings(cases: Iterable[GoldenRetrievalCase], retriever: HybridRetriever, max_candidates: int = MAX_CANDIDATES) -> tuple[list[CachedRanking], dict[str, float]]:
    """Retrieve each source exactly once per query at the maximum experiment pool."""
    if max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    cached: list[CachedRanking] = []
    dense_seconds = bm25_seconds = 0.0
    total_started = time.perf_counter()
    for case in cases:
        dense_started = time.perf_counter()
        dense = retriever.retrieve_dense(case.query, top_k=max_candidates)
        dense_seconds += time.perf_counter() - dense_started
        bm25_started = time.perf_counter()
        bm25 = retriever.retrieve_bm25(case.query, top_k=max_candidates)
        bm25_seconds += time.perf_counter() - bm25_started
        cached.append(CachedRanking(case, dense, bm25))
    total_seconds = time.perf_counter() - total_started
    count = len(cached)
    return cached, {
        "query_count": count,
        "dense_seconds": dense_seconds,
        "bm25_seconds": bm25_seconds,
        "total_seconds": total_seconds,
        "dense_ms_per_query": 1000 * dense_seconds / count if count else 0.0,
        "bm25_ms_per_query": 1000 * bm25_seconds / count if count else 0.0,
        "total_ms_per_query": 1000 * total_seconds / count if count else 0.0,
    }


def candidate_slices(row: CachedRanking, dense_k: int, bm25_k: int) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
    """Pure, deterministic slices of cached maximum-pool results."""
    return row.dense_results[:dense_k], row.bm25_results[:bm25_k]


def extended_metrics(retrieved_ids: Iterable[str], relevant_ids: Iterable[str]) -> dict[str, float]:
    """Keep baseline metrics intact while adding downstream pool diagnostics."""
    retrieved = list(retrieved_ids)
    relevant = list(relevant_ids)
    return {
        **ranking_metrics(retrieved, relevant),
        "recall_at_20": recall_at_k(retrieved, relevant, 20),
        "recall_at_30": recall_at_k(retrieved, relevant, 30),
    }


def evaluate_cached_configuration(cached: Iterable[CachedRanking], config: RRFConfig) -> tuple[dict[str, float], dict[str, list[str]], list[dict[str, Any]], float]:
    """Evaluate RRF only in memory; no retrieval or embedding happens here."""
    rows = list(cached)
    metric_rows: list[dict[str, float]] = []
    diagnostics: list[dict[str, Any]] = []
    started = time.perf_counter()
    for row in rows:
        dense, bm25 = candidate_slices(row, config.dense_candidate_k, config.bm25_candidate_k)
        fused = fuse_ranked_candidates(dense, bm25, config)
        fused_ids = [candidate.chunk_id for candidate in fused]
        metric_rows.append(extended_metrics(fused_ids, row.case.relevant_chunk_ids))
        diagnostics.append({
            "query_id": row.case.query_id,
            "relevant_chunk_ids": row.case.relevant_chunk_ids,
            "dense_ids": [candidate.chunk_id for candidate in dense],
            "bm25_ids": [candidate.chunk_id for candidate in bm25],
            "fused_ids": fused_ids,
            "bm25_only_relevant_promoted": [candidate.chunk_id for candidate in fused[:10]
                                             if candidate.dense_rank is None and candidate.bm25_rank is not None
                                             and candidate.chunk_id in row.case.relevant_chunk_ids],
        })
    elapsed = time.perf_counter() - started
    groups = {
        "single_evidence": [metrics for row, metrics in zip(rows, metric_rows) if len(row.case.relevant_chunk_ids) == 1],
        "multi_evidence": [metrics for row, metrics in zip(rows, metric_rows) if len(row.case.relevant_chunk_ids) >= 2],
    }
    return mean_metrics(metric_rows), {name: [] for name in groups}, diagnostics, elapsed


def _aggregate_group_metrics(cached: list[CachedRanking], config: RRFConfig) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in cached:
        dense, bm25 = candidate_slices(row, config.dense_candidate_k, config.bm25_candidate_k)
        fused_ids = [candidate.chunk_id for candidate in fuse_ranked_candidates(dense, bm25, config)]
        grouped["single_evidence" if len(row.case.relevant_chunk_ids) == 1 else "multi_evidence"].append(
            extended_metrics(fused_ids, row.case.relevant_chunk_ids)
        )
    return {name: mean_metrics(values) for name, values in grouped.items()}


def _oracle_counts(retrieved_ids: Iterable[str], relevant_ids: Iterable[str]) -> str:
    found = len(set(retrieved_ids) & set(relevant_ids))
    return "all" if found == len(set(relevant_ids)) else "some" if found else "none"


def candidate_union_coverage(cached: Iterable[CachedRanking]) -> dict[str, Any]:
    """Candidate coverage and oracle headroom before any RRF rank-fusion choice."""
    rows = list(cached)
    report: dict[str, Any] = {}
    for size in CANDIDATE_SIZES:
        coverage: dict[str, list[float]] = {"dense": [], "bm25": [], "union": []}
        hits: dict[str, list[float]] = {"dense": [], "bm25": [], "union": []}
        oracle: dict[str, dict[str, int]] = {
            "single_evidence": {"all": 0, "some": 0, "none": 0},
            "multi_evidence": {"all": 0, "some": 0, "none": 0},
        }
        for row in rows:
            dense, bm25 = candidate_slices(row, size, size)
            rankings = {
                "dense": [candidate.chunk_id for candidate in dense],
                "bm25": [candidate.chunk_id for candidate in bm25],
            }
            rankings["union"] = list(dict.fromkeys(rankings["dense"] + rankings["bm25"]))
            for name, ids in rankings.items():
                coverage[name].append(recall_at_k(ids, row.case.relevant_chunk_ids, len(ids) or 1))
                hits[name].append(hit_rate_at_k(ids, row.case.relevant_chunk_ids, len(ids) or 1))
            evidence_group = "single_evidence" if len(row.case.relevant_chunk_ids) == 1 else "multi_evidence"
            oracle[evidence_group][_oracle_counts(rankings["union"], row.case.relevant_chunk_ids)] += 1
        report[str(size)] = {
            "candidate_recall": {name: sum(values) / len(values) for name, values in coverage.items()},
            "candidate_hit_rate": {name: sum(values) / len(values) for name, values in hits.items()},
            "union_oracle": oracle,
        }
    return report


def baseline_from_cache(cached: Iterable[CachedRanking], source: str) -> dict[str, float]:
    rows = list(cached)
    return mean_metrics([
        extended_metrics([candidate.chunk_id for candidate in (row.dense_results if source == "dense" else row.bm25_results)[:30]], row.case.relevant_chunk_ids)
        for row in rows
    ])


def grid_search(cached: list[CachedRanking]) -> list[dict[str, Any]]:
    """Evaluate the prescribed 96 standard plus 480 weighted DEV configurations."""
    entries: list[dict[str, Any]] = []
    for kind, weights in (("standard", (1.0,)), ("weighted", BM25_WEIGHTS)):
        for dense_k in CANDIDATE_SIZES:
            for bm25_k in CANDIDATE_SIZES:
                for rrf_k in RRF_KS:
                    for bm25_weight in weights:
                        config = RRFConfig(dense_k, bm25_k, 50, rrf_k, 1.0, bm25_weight)
                        metrics, _, _, _ = evaluate_cached_configuration(cached, config)
                        entries.append({"kind": kind, "configuration": config.to_dict(), "metrics": metrics})
    return entries


def selection_key(entry: dict[str, Any]) -> tuple[float, ...]:
    """DEV-only selection order with smaller/simpler pools as deterministic ties."""
    metrics = entry["metrics"]
    config = entry["configuration"]
    return (
        metrics["recall_at_10"], metrics["ndcg_at_10"], metrics["mrr_at_10"],
        metrics["recall_at_20"], metrics["recall_at_30"],
        -(config["dense_candidate_k"] + config["bm25_candidate_k"]),
        -abs(config["bm25_weight"] - 1.0), -config["rrf_k"],
    )


def choose_configuration(entries: list[dict[str, Any]], dense_baseline: dict[str, float]) -> tuple[dict[str, Any], bool]:
    """Prefer configurations that preserve Dense Recall@10, then apply DEV priorities."""
    preserving = [entry for entry in entries if entry["metrics"]["recall_at_10"] >= dense_baseline["recall_at_10"] - 1e-12]
    pool = preserving or entries
    return max(pool, key=selection_key), bool(preserving)


def _failure_analysis(cached: list[CachedRanking], config: RRFConfig) -> dict[str, Any]:
    dense_failed_recovered: list[str] = []
    dense_succeeded_lost: list[str] = []
    promoted: dict[str, list[str]] = {}
    both_miss: list[str] = []
    union_but_fused_miss: list[str] = []
    multi: dict[str, dict[str, list[str]]] = {str(k): {"all": [], "partial": [], "none": []} for k in (10, 20, 30)}
    for row in cached:
        dense, bm25 = candidate_slices(row, config.dense_candidate_k, config.bm25_candidate_k)
        fused = fuse_ranked_candidates(dense, bm25, config)
        relevant = set(row.case.relevant_chunk_ids)
        dense_top10 = {candidate.chunk_id for candidate in dense[:10]}
        fused_top10 = {candidate.chunk_id for candidate in fused[:10]}
        union = {candidate.chunk_id for candidate in dense + bm25}
        if not dense_top10 & relevant and fused_top10 & relevant:
            dense_failed_recovered.append(row.case.query_id)
        if dense_top10 & relevant and not fused_top10 & relevant:
            dense_succeeded_lost.append(row.case.query_id)
        promoted_ids = [candidate.chunk_id for candidate in fused[:10] if candidate.dense_rank is None
                        and candidate.bm25_rank is not None and candidate.chunk_id in relevant]
        if promoted_ids:
            promoted[row.case.query_id] = promoted_ids
        if not union & relevant:
            both_miss.append(row.case.query_id)
        elif not fused_top10 & relevant:
            union_but_fused_miss.append(row.case.query_id)
        if len(relevant) >= 2:
            for k in (10, 20, 30):
                found = len({candidate.chunk_id for candidate in fused[:k]} & relevant)
                multi[str(k)]["all" if found == len(relevant) else "partial" if found else "none"].append(row.case.query_id)
    return {
        "dense_top_10_failed_rrf_succeeded": dense_failed_recovered,
        "dense_top_10_succeeded_rrf_failed": dense_succeeded_lost,
        "bm25_only_relevant_promoted_into_fused_top_10": promoted,
        "both_candidate_retrievers_miss_evidence": both_miss,
        "union_contains_evidence_but_fused_top_10_misses": union_but_fused_miss,
        "multi_evidence_recovery": multi,
    }


def _markdown(report: dict[str, Any]) -> str:
    def metrics(values: dict[str, float]) -> str:
        return "\n".join(["| Metric | Value |", "|---|---:|"] + [f"| {key} | {value:.4f} |" for key, value in values.items()])

    selected = report["selected"]
    union_rows = []
    for size, values in report["candidate_union_coverage"].items():
        recall = values["candidate_recall"]
        hit = values["candidate_hit_rate"]
        union_rows.append(f"| {size} | {recall['dense']:.4f} | {recall['bm25']:.4f} | {recall['union']:.4f} | {hit['union']:.4f} |")
    top_rows = sorted(report["search_results"], key=selection_key, reverse=True)[:10]
    lines = [
        "# Dense Baseline", "", metrics(report["dense_baseline"]), "", "# BM25 Baseline", "", metrics(report["bm25_baseline"]),
        "", "# Candidate Union Coverage", "", "| Pool | Dense Recall | BM25 Recall | Union Recall | Union HitRate |", "|---:|---:|---:|---:|---:|", *union_rows,
        "", "# Best Standard RRF", "", json.dumps(report["best_standard"], ensure_ascii=False, indent=2),
        "", "# Best Weighted RRF", "", json.dumps(report["best_weighted"], ensure_ascii=False, indent=2),
        "", "# Selected Configuration", "", json.dumps(selected, ensure_ascii=False, indent=2),
        "", "# Top Configurations", "", "| Type | Dense pool | BM25 pool | k | BM25 weight | Recall@10 | nDCG@10 | MRR@10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in top_rows:
        config, values = entry["configuration"], entry["metrics"]
        lines.append(f"| {entry['kind']} | {config['dense_candidate_k']} | {config['bm25_candidate_k']} | {config['rrf_k']} | {config['bm25_weight']:.2f} | {values['recall_at_10']:.4f} | {values['ndcg_at_10']:.4f} | {values['mrr_at_10']:.4f} |")
    lines.extend(["", "# Single vs Multi Evidence", "", json.dumps(report["single_vs_multi_evidence"], ensure_ascii=False, indent=2),
                  "", "# Failure Recovery", "", json.dumps(report["failure_analysis"], ensure_ascii=False, indent=2)])
    if report.get("test_evaluation"):
        lines.extend(["", "# Frozen Held-Out TEST", "", metrics(report["test_evaluation"]["metrics"])])
    return "\n".join(lines) + "\n"


def run_tuning(dataset: Path = DEFAULT_V2_DATASET, meta_path: Path = DEFAULT_V2_META, splits_path: Path = DEFAULT_SPLITS,
               bundle_dir: Path = DEFAULT_BUNDLE, output_dir: Path = DEFAULT_OUTPUT, retriever: HybridRetriever | None = None) -> dict[str, Any]:
    """Run the complete DEV-only experiment and write its frozen selection inputs."""
    cases, meta = load_dev_cases(dataset, meta_path, splits_path, bundle_dir)
    cached, timings = cache_rankings(cases, retriever or HybridRetriever())
    union = candidate_union_coverage(cached)
    dense_baseline = baseline_from_cache(cached, "dense")
    bm25_baseline = baseline_from_cache(cached, "bm25")
    entries = grid_search(cached)
    best_standard = max((entry for entry in entries if entry["kind"] == "standard"), key=selection_key)
    best_weighted = max((entry for entry in entries if entry["kind"] == "weighted"), key=selection_key)
    selected_entry, preserves_dense_recall = choose_configuration(entries, dense_baseline)
    frozen_config = {**selected_entry["configuration"], "fusion_top_k": 30,
                     "golden_dataset_version": meta["dataset_version"], "selection_split": "dev"}
    selected_config = RRFConfig(**{key: frozen_config[key] for key in RRFConfig.__dataclass_fields__})
    _, _, _, fusion_seconds = evaluate_cached_configuration(cached, selected_config)
    report = {
        "dataset_version": meta["dataset_version"], "selection_split": "dev", "dev_query_count": len(cached),
        "dense_baseline": dense_baseline, "bm25_baseline": bm25_baseline,
        "candidate_union_coverage": union, "search_configuration_count": len(entries), "search_results": entries,
        "best_standard": best_standard, "best_weighted": best_weighted,
        "selected": {"configuration": frozen_config, "metrics": selected_entry["metrics"],
                     "preserves_dense_recall_at_10": preserves_dense_recall},
        "single_vs_multi_evidence": _aggregate_group_metrics(cached, selected_config),
        "failure_analysis": _failure_analysis(cached, selected_config),
        "performance": {**timings, "rrf_fusion_seconds_for_dev": fusion_seconds,
                        "rrf_fusion_ms_per_query": 1000 * fusion_seconds / len(cached) if cached else 0.0},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rrf_search_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "rrf_search_v1.md").write_text(_markdown(report), encoding="utf-8")
    config_path = Path(__file__).resolve().parents[1] / "retrieval" / "rrf_config.json"
    config_path.write_text(json.dumps(frozen_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def evaluate_frozen_test(report: dict[str, Any], dataset: Path = DEFAULT_V2_DATASET, meta_path: Path = DEFAULT_V2_META,
                         splits_path: Path = DEFAULT_SPLITS, bundle_dir: Path = DEFAULT_BUNDLE,
                         retriever: HybridRetriever | None = None) -> dict[str, Any]:
    """Run one held-out TEST evaluation after DEV configuration selection is written."""
    cases = load_cases(dataset)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    test_ids = set(load_and_validate_splits(cases, splits_path, meta)["test"])
    cached, timings = cache_rankings([case for case in cases if case.query_id in test_ids], retriever or HybridRetriever())
    config = RRFConfig(**{key: report["selected"]["configuration"][key] for key in RRFConfig.__dataclass_fields__})
    metrics, _, _, fusion_seconds = evaluate_cached_configuration(cached, config)
    return {"split": "test", "query_count": len(cached), "metrics": metrics,
            "performance": {**timings, "rrf_fusion_seconds": fusion_seconds}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_V2_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_V2_META)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-test", action="store_true", help="Run the one post-freeze held-out TEST evaluation.")
    args = parser.parse_args()
    report = run_tuning(args.dataset, args.metadata, args.splits, args.bundle_dir, args.output_dir)
    if args.run_test:
        report["test_evaluation"] = evaluate_frozen_test(report, args.dataset, args.metadata, args.splits, args.bundle_dir)
        (args.output_dir / "rrf_search_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "rrf_search_v1.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"selected": report["selected"], "test_evaluation": report.get("test_evaluation")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
