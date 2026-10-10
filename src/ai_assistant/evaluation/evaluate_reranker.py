"""DEV-only candidate-strategy selection for the pinned BGE cross encoder."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import RetrievalCandidate
from ..reranking.config import RerankerConfig
from ..reranking.cross_encoder import CrossEncoderReranker
from ..reranking.reranked_retriever import build_candidate_strategy, rerank_candidates
from ..retrieval.hybrid_retriever import HybridRetriever
from .evaluate_retrieval import DEFAULT_BUNDLE, DEFAULT_OUTPUT, DEFAULT_SPLITS, DEFAULT_V2_DATASET, DEFAULT_V2_META, load_and_validate_splits, load_cases, validate_cases, validate_metadata
from .metrics import hit_rate_at_k, mean_metrics, ranking_metrics, recall_at_k
from .schema import GoldenRetrievalCase


@dataclass(frozen=True, slots=True)
class CachedRerankQuery:
    case: GoldenRetrievalCase
    dense_results: list[RetrievalCandidate]
    bm25_results: list[RetrievalCandidate]


STRATEGIES = (
    RerankerConfig(candidate_strategy="dense_top_20", dense_candidate_k=20, bm25_candidate_k=1, rerank_candidate_limit=20, output_top_k=20),
    RerankerConfig(candidate_strategy="rrf_top_30", dense_candidate_k=10, bm25_candidate_k=10, rerank_candidate_limit=30, output_top_k=30),
    RerankerConfig(candidate_strategy="union_d10_b10", dense_candidate_k=10, bm25_candidate_k=10, rerank_candidate_limit=20, output_top_k=20),
    RerankerConfig(candidate_strategy="union_d20_b10", dense_candidate_k=20, bm25_candidate_k=10, rerank_candidate_limit=30, output_top_k=30),
    RerankerConfig(candidate_strategy="union_d20_b20", dense_candidate_k=20, bm25_candidate_k=20, rerank_candidate_limit=40, output_top_k=30),
)


def load_split_cases(dataset: Path, meta_path: Path, splits_path: Path, bundle_dir: Path, split: str) -> tuple[list[GoldenRetrievalCase], dict[str, Any]]:
    """Validate the frozen dataset and return only the explicitly requested split."""
    cases = load_cases(dataset)
    validate_cases(cases, bundle_dir)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    validate_metadata(cases, meta, bundle_dir)
    ids = set(load_and_validate_splits(cases, splits_path, meta)[split])
    return [case for case in cases if case.query_id in ids], meta


def cache_retrieval(cases: list[GoldenRetrievalCase], retriever: HybridRetriever) -> tuple[list[CachedRerankQuery], dict[str, float]]:
    """Make exactly one Dense-20 and BM25-20 call for each query in the split."""
    dense_seconds = bm25_seconds = 0.0
    started = time.perf_counter()
    cached: list[CachedRerankQuery] = []
    for case in cases:
        before = time.perf_counter()
        dense = retriever.retrieve_dense(case.query, top_k=20)
        dense_seconds += time.perf_counter() - before
        before = time.perf_counter()
        bm25 = retriever.retrieve_bm25(case.query, top_k=20)
        bm25_seconds += time.perf_counter() - before
        cached.append(CachedRerankQuery(case, dense, bm25))
    total = time.perf_counter() - started
    count = len(cached)
    return cached, {"query_count": count, "dense_seconds": dense_seconds, "bm25_seconds": bm25_seconds,
                    "retrieval_seconds": total, "retrieval_ms_per_query": 1000 * total / count if count else 0.0}


def _metrics(ids: list[str], relevant: list[str]) -> dict[str, float]:
    return {**ranking_metrics(ids, relevant), "hit_rate_at_3": hit_rate_at_k(ids, relevant, 3)}


def _score_strategy(cached: list[CachedRerankQuery], config: RerankerConfig, scorer: CrossEncoderReranker,
                    score_cache: dict[tuple[str, str, str, int], float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score only unseen query/chunk pairs and rank each full candidate strategy."""
    rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, float]] = []
    scoring_seconds = 0.0
    before_dedup = after_dedup = batches = 0
    for row in cached:
        candidates = build_candidate_strategy(row.dense_results, row.bm25_results, config)
        before_dedup += len(row.dense_results[:config.dense_candidate_k]) + (0 if config.candidate_strategy == "dense_top_20" else len(row.bm25_results[:config.bm25_candidate_k]))
        after_dedup += len(candidates)
        keys = [(row.case.query_id, candidate.chunk_id, config.model_revision, config.max_length) for candidate in candidates]
        missing = [(key, candidate) for key, candidate in zip(keys, candidates) if key not in score_cache]
        if missing:
            started = time.perf_counter()
            values = scorer.score(row.case.query, [candidate.text_original for _, candidate in missing])
            scoring_seconds += time.perf_counter() - started
            if len(values) != len(missing):
                raise ValueError("reranker score count does not match unseen candidates")
            score_cache.update({key: score for (key, _), score in zip(missing, values)})
            batches += (len(missing) + config.batch_size - 1) // config.batch_size
        reranked = rerank_candidates(candidates, [score_cache[key] for key in keys])
        ids = [candidate.chunk_id for candidate in reranked]
        metric_rows.append(_metrics(ids, row.case.relevant_chunk_ids))
        rows.append({"query_id": row.case.query_id, "relevant_chunk_ids": row.case.relevant_chunk_ids,
                     "candidate_ids": [candidate.chunk_id for candidate in candidates], "reranked_ids": ids,
                     "dense_ids": [candidate.chunk_id for candidate in row.dense_results[:10]],
                     "rrf_ids": [candidate.chunk_id for candidate in build_candidate_strategy(row.dense_results, row.bm25_results, STRATEGIES[1])],
                     "candidate_count": len(candidates)})
    return {"configuration": config.to_dict(), "metrics": mean_metrics(metric_rows),
            "candidate_count_before_dedup": before_dedup / len(cached), "candidate_count_after_dedup": after_dedup / len(cached),
            "new_scoring_seconds": scoring_seconds, "new_scoring_batches": batches,
            "new_scoring_ms_per_query": 1000 * scoring_seconds / len(cached) if cached else 0.0}, rows


def _stratified_metrics(cached: list[CachedRerankQuery], rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, float]]] = {"single_evidence": [], "multi_evidence": []}
    for cached_row, row in zip(cached, rows):
        groups["single_evidence" if len(cached_row.case.relevant_chunk_ids) == 1 else "multi_evidence"].append(_metrics(row["reranked_ids"], cached_row.case.relevant_chunk_ids))
    return {name: mean_metrics(values) for name, values in groups.items()}


def _coverage(cached: list[CachedRerankQuery], config: RerankerConfig) -> dict[str, Any]:
    recalls: list[float] = []
    hits: list[float] = []
    groups = {"single_evidence": {"all": 0, "some": 0, "none": 0}, "multi_evidence": {"all": 0, "some": 0, "none": 0}}
    for row in cached:
        ids = [candidate.chunk_id for candidate in build_candidate_strategy(row.dense_results, row.bm25_results, config)]
        relevant = set(row.case.relevant_chunk_ids)
        found = len(set(ids) & relevant)
        recalls.append(recall_at_k(ids, relevant, len(ids) or 1))
        hits.append(hit_rate_at_k(ids, relevant, len(ids) or 1))
        group = "single_evidence" if len(relevant) == 1 else "multi_evidence"
        groups[group]["all" if found == len(relevant) else "some" if found else "none"] += 1
    return {"candidate_recall": sum(recalls) / len(recalls), "candidate_hit_rate": sum(hits) / len(hits), "evidence_coverage": groups}


def _selection_key(entry: dict[str, Any]) -> tuple[float, ...]:
    metrics, config = entry["metrics"], entry["configuration"]
    return (metrics["ndcg_at_10"], metrics["mrr_at_10"], metrics["recall_at_5"], metrics["hit_rate_at_5"],
            metrics["recall_at_10"], -entry["candidate_count_after_dedup"], -entry["new_scoring_ms_per_query"],
            -config["dense_candidate_k"], -config["bm25_candidate_k"])


def _failure_analysis(cached: list[CachedRerankQuery], selected_rows: list[dict[str, Any]]) -> dict[str, Any]:
    recovered_dense: list[str] = []
    recovered_rrf: list[str] = []
    lost_dense: list[str] = []
    lost_rrf: list[str] = []
    outside: list[str] = []
    rank_buckets = {"1": [], "2-3": [], "4-5": [], "6-10": [], ">10": []}
    multi = {"all": [], "partial": [], "none": []}
    for cached_row, row in zip(cached, selected_rows):
        relevant = set(cached_row.case.relevant_chunk_ids)
        dense_hit = bool(set(row["dense_ids"]) & relevant)
        rrf_hit = bool(set(row["rrf_ids"][:10]) & relevant)
        reranked_top = row["reranked_ids"][:10]
        reranked_hit = bool(set(reranked_top) & relevant)
        if not dense_hit and reranked_hit:
            recovered_dense.append(cached_row.case.query_id)
        if not rrf_hit and reranked_hit:
            recovered_rrf.append(cached_row.case.query_id)
        if dense_hit and not reranked_hit:
            lost_dense.append(cached_row.case.query_id)
        if rrf_hit and not reranked_hit:
            lost_rrf.append(cached_row.case.query_id)
        first_rank = next((index for index, chunk_id in enumerate(row["reranked_ids"], start=1) if chunk_id in relevant), None)
        bucket = "1" if first_rank == 1 else "2-3" if first_rank and first_rank <= 3 else "4-5" if first_rank and first_rank <= 5 else "6-10" if first_rank and first_rank <= 10 else ">10"
        rank_buckets[bucket].append(cached_row.case.query_id)
        if set(row["candidate_ids"]) & relevant and not reranked_hit:
            outside.append(cached_row.case.query_id)
        if len(relevant) > 1:
            found = len(set(reranked_top) & relevant)
            multi["all" if found == len(relevant) else "partial" if found else "none"].append(cached_row.case.query_id)
    return {"dense_failures_recovered": recovered_dense, "rrf_failures_recovered": recovered_rrf,
            "dense_successes_lost": lost_dense, "rrf_successes_lost": lost_rrf,
            "relevant_candidate_outside_reranked_top_10": outside, "first_relevant_rank": rank_buckets,
            "multi_evidence_top_10": multi}


def _markdown(report: dict[str, Any]) -> str:
    def table(metrics: dict[str, float]) -> str:
        return "\n".join(["| Metric | Value |", "|---|---:|"] + [f"| {name} | {value:.4f} |" for name, value in metrics.items()])
    lines = ["# Existing Baselines", "", "## Dense", table(report["dense_baseline"]), "", "## Selected RRF", table(report["rrf_baseline"]),
             "", "# Candidate Strategies", "", "| Strategy | Candidates | Recall@10 | nDCG@10 | MRR@10 |", "|---|---:|---:|---:|---:|"]
    for entry in report["strategies"]:
        metric = entry["metrics"]
        lines.append(f"| {entry['configuration']['candidate_strategy']} | {entry['candidate_count_after_dedup']:.1f} | {metric['recall_at_10']:.4f} | {metric['ndcg_at_10']:.4f} | {metric['mrr_at_10']:.4f} |")
    lines.extend(["", "# BGE Reranker DEV Results", "", json.dumps(report["strategies"], ensure_ascii=False, indent=2),
                  "", "# Selected Configuration", "", json.dumps(report["selected"], ensure_ascii=False, indent=2),
                  "", "# Single vs Multi Evidence", "", json.dumps(report["single_vs_multi_evidence"], ensure_ascii=False, indent=2),
                  "", "# Failure Recovery", "", json.dumps(report["failure_analysis"], ensure_ascii=False, indent=2),
                  "", "# Latency", "", json.dumps(report["latency"], ensure_ascii=False, indent=2)])
    if report.get("test_evaluation"):
        lines.extend(["", "# Held-out TEST Result", "", table(report["test_evaluation"]["metrics"]), "", json.dumps(report["test_evaluation"]["candidate_coverage"], ensure_ascii=False, indent=2)])
    return "\n".join(lines) + "\n"


def run_dev_experiment(dataset: Path = DEFAULT_V2_DATASET, meta_path: Path = DEFAULT_V2_META, splits_path: Path = DEFAULT_SPLITS,
                       bundle_dir: Path = DEFAULT_BUNDLE, output_dir: Path = DEFAULT_OUTPUT, retriever: HybridRetriever | None = None,
                       scorer: CrossEncoderReranker | None = None) -> dict[str, Any]:
    """Choose a reranker candidate strategy using DEV only."""
    cases, meta = load_split_cases(dataset, meta_path, splits_path, bundle_dir, "dev")
    cached, retrieval_timing = cache_retrieval(cases, retriever or HybridRetriever())
    scorer = scorer or CrossEncoderReranker(STRATEGIES[0])
    score_cache: dict[tuple[str, str, str, int], float] = {}
    strategies: list[dict[str, Any]] = []
    row_sets: dict[str, list[dict[str, Any]]] = {}
    for config in STRATEGIES:
        entry, rows = _score_strategy(cached, config, scorer, score_cache)
        strategies.append(entry)
        row_sets[config.candidate_strategy] = rows
    dense_baseline = mean_metrics([_metrics([candidate.chunk_id for candidate in row.dense_results], row.case.relevant_chunk_ids) for row in cached])
    rrf_rows = row_sets["rrf_top_30"]
    rrf_baseline = mean_metrics([_metrics(row["rrf_ids"], cached_row.case.relevant_chunk_ids) for cached_row, row in zip(cached, rrf_rows)])
    acceptable = [entry for entry in strategies if entry["metrics"]["recall_at_10"] >= dense_baseline["recall_at_10"] - 1e-12]
    selected = max(acceptable or strategies, key=_selection_key)
    selected_config = next(config for config in STRATEGIES if config.candidate_strategy == selected["configuration"]["candidate_strategy"])
    selected_rows = row_sets[selected_config.candidate_strategy]
    frozen = {**selected_config.to_dict(), **scorer.model_identity(), "selection_split": "dev", "golden_dataset_version": meta["dataset_version"]}
    report = {"dataset_version": meta["dataset_version"], "selection_split": "dev", "model": scorer.model_identity(),
              "dev_query_count": len(cached), "dense_baseline": dense_baseline, "rrf_baseline": rrf_baseline,
              "strategies": strategies, "selected": {"configuration": frozen, "metrics": selected["metrics"],
              "constraint_preserves_dense_recall_at_10": bool(acceptable)},
              "single_vs_multi_evidence": _stratified_metrics(cached, selected_rows),
              "failure_analysis": _failure_analysis(cached, selected_rows),
              "latency": {**retrieval_timing, "score_cache_entries": len(score_cache)}}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reranker_search_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "reranker_search_v1.md").write_text(_markdown(report), encoding="utf-8")
    (Path(__file__).resolve().parents[1] / "reranking" / "reranker_config.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def run_frozen_test(report: dict[str, Any], dataset: Path = DEFAULT_V2_DATASET, meta_path: Path = DEFAULT_V2_META,
                    splits_path: Path = DEFAULT_SPLITS, bundle_dir: Path = DEFAULT_BUNDLE, retriever: HybridRetriever | None = None,
                    scorer: CrossEncoderReranker | None = None) -> dict[str, Any]:
    """Evaluate the already-frozen DEV selection once on held-out TEST."""
    cases, _ = load_split_cases(dataset, meta_path, splits_path, bundle_dir, "test")
    cached, timing = cache_retrieval(cases, retriever or HybridRetriever())
    config_data = report["selected"]["configuration"]
    config = RerankerConfig(**{name: config_data[name] for name in RerankerConfig.__dataclass_fields__})
    scorer = scorer or CrossEncoderReranker(config)
    entry, _ = _score_strategy(cached, config, scorer, {})
    return {"split": "test", "query_count": len(cached), "metrics": entry["metrics"],
            "candidate_coverage": _coverage(cached, config), "latency": {**timing, "reranking_seconds": entry["new_scoring_seconds"],
            "reranking_ms_per_query": entry["new_scoring_ms_per_query"], "scoring_batches": entry["new_scoring_batches"],
            "candidate_count_before_dedup": entry["candidate_count_before_dedup"], "candidate_count_after_dedup": entry["candidate_count_after_dedup"]}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_V2_DATASET)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_V2_META)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-test", action="store_true")
    args = parser.parse_args()
    report = run_dev_experiment(args.dataset, args.metadata, args.splits, args.bundle_dir, args.output_dir)
    if args.run_test:
        report["test_evaluation"] = run_frozen_test(report, args.dataset, args.metadata, args.splits, args.bundle_dir)
        (args.output_dir / "reranker_search_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (args.output_dir / "reranker_search_v1.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps({"selected": report["selected"], "test_evaluation": report.get("test_evaluation")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
