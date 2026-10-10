"""Inspect the independent dense and BM25 retrieval result lists."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..retrieval.hybrid_retriever import HybridRetriever


def _print_candidate(rank: int, candidate, score_label: str) -> None:
    print(f"Rank: {rank}")
    print(f"Chunk ID: {candidate.chunk_id}")
    print(f"Document: {candidate.document_id}")
    print(f"Title: {candidate.document_title or ''}")
    print(f"{score_label}: {candidate.score}")
    print(f"PDF Page: {candidate.pdf_page_start}-{candidate.pdf_page_end}")
    print(f"Article: {candidate.article_number or ''}")
    print(f"Provision: {candidate.provision_number or ''}")
    print(f"Parent ID: {candidate.parent_id}")
    print("\nText:")
    print(candidate.text_original)
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", help="question to retrieve without an interactive prompt")
    parser.add_argument("--dense-top-k", type=int, default=10)
    parser.add_argument("--bm25-top-k", type=int, default=10)
    parser.add_argument("--json-output", type=Path, help="optional path for separate raw retrieval lists")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    query = args.query if args.query is not None else input("Enter your question: ")
    retriever = HybridRetriever()
    result = retriever.retrieve(query, dense_top_k=args.dense_top_k, bm25_top_k=args.bm25_top_k)

    print("=" * 40)
    print("QUERY")
    print("=" * 40)
    print(result.original_query)
    print("\n" + "=" * 40)
    print("DENSE — BGE-M3 + FAISS")
    print("=" * 40)
    for rank, candidate in enumerate(result.dense_results, start=1):
        _print_candidate(rank, candidate, "Dense Score")

    print("=" * 40)
    print("BM25")
    print("=" * 40)
    for rank, candidate in enumerate(result.bm25_results, start=1):
        _print_candidate(rank, candidate, "BM25 Score")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
