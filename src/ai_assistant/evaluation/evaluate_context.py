"""DEV-only selection and held-out evaluation of source-grounded context packs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from typing import Any

from ..context.builder import ContextBuilder
from ..context.config import ContextConfig, FROZEN_CONTEXT_CONFIG_PATH
from ..models import RerankedCandidate
from ..reranking.config import load_frozen_reranker_config
from ..reranking.reranked_retriever import RerankedRetriever
from .evaluate_reranker import load_split_cases
from .evaluate_retrieval import DEFAULT_BUNDLE, DEFAULT_OUTPUT, DEFAULT_SPLITS, DEFAULT_V2_DATASET, DEFAULT_V2_META
from .schema import GoldenRetrievalCase


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CACHE = ROOT / "artifacts" / "ai_assistant" / "cache" / "context_eval_reranked_v1.json"
STRATEGIES = ("chunk_only", "parent", "chunk_plus_parent", "parent_deduplicated")
TOP_KS = (3, 5, 8, 10)
BUDGETS = (2000, 4000, 6000)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cache_identity(bundle_dir: Path, dataset_meta: dict[str, Any]) -> dict[str, Any]:
    reranker = load_frozen_reranker_config()
    return {
        "corpus": {"chunks_sha256": _sha256(bundle_dir / "chunks.jsonl"),
                   "release_manifest_sha256": _sha256(bundle_dir / "release_manifest.json")},
        "reranker": {**reranker.to_dict(), "model_id": reranker.model_id, "model_revision": reranker.model_revision},
        "benchmark_version": dataset_meta["dataset_version"],
    }


def _candidate(data: dict[str, Any]) -> RerankedCandidate:
    fields = RerankedCandidate.__dataclass_fields__
    return RerankedCandidate(**{name: data[name] for name in fields})


def load_or_create_reranked_cache(cases: list[GoldenRetrievalCase], identity: dict[str, Any], cache_path: Path = DEFAULT_CACHE,
                                  *, refresh: bool = False) -> list[dict[str, Any]]:
    """Rerank each requested query once; reject cache data from any different release/config."""
    requested_ids = [case.query_id for case in cases]
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("identity") != identity:
            raise ValueError("context rerank cache is stale; rerun with --refresh-cache")
        by_id = {row["query_id"]: row for row in payload.get("results", [])}
        if all(query_id in by_id for query_id in requested_ids):
            return [by_id[query_id] for query_id in requested_ids]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    existing: dict[str, dict[str, Any]] = {}
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("identity") == identity:
            existing = {row["query_id"]: row for row in payload.get("results", [])}
    retriever = RerankedRetriever()
    for case in cases:
        if case.query_id not in existing:
            result = retriever.retrieve(case.query)
            existing[case.query_id] = {"query_id": case.query_id, "query": case.query,
                                       "reranked_results": [item.to_dict() for item in result.reranked_results]}
            cache_path.write_text(json.dumps({"schema_version": 1, "identity": identity,
                                               "results": list(existing.values())}, ensure_ascii=False, indent=2) + "\n",
                                  encoding="utf-8")
    return [existing[query_id] for query_id in requested_ids]


def _classify(found: set[str], expected: set[str]) -> str:
    if not expected or not found:
        return "none"
    return "all" if expected.issubset(found) else "partial"


def evaluate_configuration(cases: list[GoldenRetrievalCase], cached: list[dict[str, Any]], config: ContextConfig) -> dict[str, Any]:
    """Pure in-memory context evaluation; reranker is deliberately never called here."""
    builder = ContextBuilder(config)
    chunk_recalls: list[float] = []
    parent_recalls: list[float] = []
    hits: list[float] = []
    tokens: list[int] = []
    blocks: list[int] = []
    parents: list[int] = []
    documents: list[int] = []
    duplicate_removals: list[int] = []
    multi_chunk = {"all": 0, "partial": 0, "none": 0}
    multi_parent = {"all": 0, "partial": 0, "none": 0}
    for case, row in zip(cases, cached):
        pack = builder.build_from_candidates(case.query, [_candidate(item) for item in row["reranked_results"]])
        tokens.append(pack.estimated_context_tokens)
        blocks.append(pack.evidence_count)
        parents.append(pack.unique_parent_count)
        documents.append(pack.unique_document_count)
        duplicate_removals.append(int(pack.configuration["duplicate_blocks_removed"]))
        if not case.answerable:
            continue
        found_chunks, found_parents = set(pack.selected_chunk_ids), set(pack.selected_parent_ids)
        relevant_chunks, relevant_parents = set(case.relevant_chunk_ids), set(case.relevant_parent_ids)
        chunk_recalls.append(len(found_chunks & relevant_chunks) / len(relevant_chunks))
        parent_recalls.append(len(found_parents & relevant_parents) / len(relevant_parents))
        hits.append(float(bool(found_chunks & relevant_chunks or found_parents & relevant_parents)))
        if len(relevant_chunks) > 1:
            multi_chunk[_classify(found_chunks & relevant_chunks, relevant_chunks)] += 1
            multi_parent[_classify(found_parents & relevant_parents, relevant_parents)] += 1
    count = len(tokens)
    answerable = len(chunk_recalls)
    return {
        "configuration": config.to_dict(), "query_count": count, "answerable_query_count": answerable,
        "metrics": {"chunk_evidence_recall": sum(chunk_recalls) / answerable if answerable else 0.0,
                    "parent_evidence_recall": sum(parent_recalls) / answerable if answerable else 0.0,
                    "query_hit_rate": sum(hits) / answerable if answerable else 0.0},
        "multi_evidence": {"chunk": multi_chunk, "parent": multi_parent},
        "context": {"average_tokens": sum(tokens) / count if count else 0.0, "median_tokens": median(tokens) if tokens else 0.0,
                    "average_evidence_blocks": sum(blocks) / count if count else 0.0,
                    "average_unique_parents": sum(parents) / count if count else 0.0,
                    "average_unique_documents": sum(documents) / count if count else 0.0,
                    "average_duplicate_blocks_removed": sum(duplicate_removals) / count if count else 0.0},
    }


def _multi_all_rate(entry: dict[str, Any]) -> float:
    values = entry["multi_evidence"]["parent"]
    total = sum(values.values())
    return values["all"] / total if total else 0.0


def _selection_key(entry: dict[str, Any]) -> tuple[float, ...]:
    metrics, context, config = entry["metrics"], entry["context"], entry["configuration"]
    strategy_order = {name: index for index, name in enumerate(STRATEGIES)}
    return (metrics["parent_evidence_recall"], metrics["chunk_evidence_recall"], _multi_all_rate(entry),
            -context["average_tokens"], -context["average_evidence_blocks"], -config["candidate_top_k"],
            -strategy_order[config["expansion_strategy"]])


def _markdown(report: dict[str, Any]) -> str:
    lines = ["# Context Expansion DEV Search", "", f"Configurations evaluated: {len(report['configurations'])}", "",
             "| Strategy | Top K | Budget | Chunk recall | Parent recall | Hit rate | Avg tokens | Avg blocks |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report["configurations"]:
        c, m, x = row["configuration"], row["metrics"], row["context"]
        lines.append(f"| {c['expansion_strategy']} | {c['candidate_top_k']} | {c['max_context_tokens']} | {m['chunk_evidence_recall']:.4f} | {m['parent_evidence_recall']:.4f} | {m['query_hit_rate']:.4f} | {x['average_tokens']:.1f} | {x['average_evidence_blocks']:.2f} |")
    lines.extend(["", "## Selected frozen configuration", "", "```json", json.dumps(report["selected"], ensure_ascii=False, indent=2), "```"])
    if "test" in report:
        lines.extend(["", "## Held-out TEST", "", "```json", json.dumps(report["test"], ensure_ascii=False, indent=2), "```"])
    return "\n".join(lines) + "\n"


def _coverage_gain(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Diagnostic marginal coverage changes between the evaluated budget levels."""
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in entries:
        config = entry["configuration"]
        grouped.setdefault((config["expansion_strategy"], config["candidate_top_k"]), []).append(entry)
    diagnostics: list[dict[str, Any]] = []
    for (strategy, top_k), rows in grouped.items():
        rows.sort(key=lambda row: row["configuration"]["max_context_tokens"])
        for before, after in zip(rows, rows[1:]):
            token_delta = after["context"]["average_tokens"] - before["context"]["average_tokens"]
            if token_delta <= 0:
                continue
            diagnostics.append({"expansion_strategy": strategy, "candidate_top_k": top_k,
                                "from_budget": before["configuration"]["max_context_tokens"],
                                "to_budget": after["configuration"]["max_context_tokens"],
                                "parent_coverage_gain_per_1000_tokens": 1000 * (after["metrics"]["parent_evidence_recall"] - before["metrics"]["parent_evidence_recall"]) / token_delta,
                                "chunk_coverage_gain_per_1000_tokens": 1000 * (after["metrics"]["chunk_evidence_recall"] - before["metrics"]["chunk_evidence_recall"]) / token_delta})
    return diagnostics


def run_dev_experiment(dataset: Path = DEFAULT_V2_DATASET, meta_path: Path = DEFAULT_V2_META, splits_path: Path = DEFAULT_SPLITS,
                       bundle_dir: Path = DEFAULT_BUNDLE, output_dir: Path = DEFAULT_OUTPUT, cache_path: Path = DEFAULT_CACHE,
                       *, refresh_cache: bool = False) -> dict[str, Any]:
    cases, meta = load_split_cases(dataset, meta_path, splits_path, bundle_dir, "dev")
    cached = load_or_create_reranked_cache(cases, cache_identity(bundle_dir, meta), cache_path, refresh=refresh_cache)
    entries = [evaluate_configuration(cases, cached, ContextConfig(candidate_top_k=top_k, expansion_strategy=strategy,
               max_context_tokens=budget, deduplicate_parents=strategy == "parent_deduplicated"))
               for top_k in TOP_KS for strategy in STRATEGIES for budget in BUDGETS]
    selected_entry = max(entries, key=_selection_key)
    selected_config = ContextConfig(**selected_entry["configuration"])
    FROZEN_CONTEXT_CONFIG_PATH.write_text(json.dumps(selected_config.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    example_pack = ContextBuilder(selected_config).build_from_candidates(cases[0].query, [_candidate(item) for item in cached[0]["reranked_results"]])
    report = {"schema_version": 1, "selection_split": "dev", "golden_dataset_version": meta["dataset_version"],
              "neighbor_expansion": {"supported": True, "reason": "unique canonical (document_id, chunk_order) pairs permit same-document adjacency"},
              "configurations": entries, "coverage_gain_per_additional_1000_tokens": _coverage_gain(entries),
              "selected": {"configuration": selected_config.to_dict(), "dev": selected_entry},
              "example_context_pack": example_pack.to_dict(), "example_structured_context": example_pack.render_structured()}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "context_search_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "context_search_v1.md").write_text(_markdown(report), encoding="utf-8")
    return report


def run_frozen_test(report: dict[str, Any], dataset: Path = DEFAULT_V2_DATASET, meta_path: Path = DEFAULT_V2_META,
                    splits_path: Path = DEFAULT_SPLITS, bundle_dir: Path = DEFAULT_BUNDLE, cache_path: Path = DEFAULT_CACHE) -> dict[str, Any]:
    cases, meta = load_split_cases(dataset, meta_path, splits_path, bundle_dir, "test")
    cached = load_or_create_reranked_cache(cases, cache_identity(bundle_dir, meta), cache_path)
    config = ContextConfig(**report["selected"]["configuration"])
    return evaluate_configuration(cases, cached, config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-cache", action="store_true")
    parser.add_argument("--run-test", action="store_true")
    args = parser.parse_args()
    report = run_dev_experiment(refresh_cache=args.refresh_cache)
    if args.run_test:
        report["test"] = run_frozen_test(report)
        DEFAULT_OUTPUT.joinpath("context_search_v1.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        DEFAULT_OUTPUT.joinpath("context_search_v1.md").write_text(_markdown(report), encoding="utf-8")
    print(json.dumps(report["selected"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
