"""Exercise BM25 and dense retrieval against a built local bundle."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from . import retrieval_adapter as retrieval
except (ImportError, ValueError):
    try:
        import retrieval_adapter as retrieval
    except ImportError:
        from rag import retrieval_adapter as retrieval


QUERIES = [
    ("DOC1", "ما هي شروط منح تسهيلات ائتمانية للعميل بالنقد الأجنبي؟"),
    ("DOC2", "قواعد تصنيف الجدارة وتكوين مخصصات الديون غير المنتظمة"),
    ("DOC3", "قواعد تسجيل الائتمان بالبنك المركزي المصري"),
    ("DOC4", "تعريف المنشآت المتوسطة والصغيرة والحد الأقصى للمبيعات"),
    ("DOC5", "عقوبة استخدام التمويل في غير الغرض المخصص له"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/rag/release/current"))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--bm25-only", action="store_true", help="skip dense search when no FAISS index is installed")
    parser.add_argument("--require-citations", action="store_true", help="fail if any returned result lacks a non-null PDF page range")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ["CREDIX_RAG_INDEX_DIR"] = str(args.index_dir)
    manifest = json.loads((args.index_dir / "release_manifest.json").read_text(encoding="utf-8"))
    results = []
    candidate_only = manifest.get("status") == "candidate_only"
    for expected_doc_id, query in QUERIES:
        lexical = retrieval.bm25_search(query, args.top_k)
        dense = [] if args.bm25_only else retrieval.dense_search(retrieval.embed_query(query), args.top_k)
        if not lexical or lexical[0]["document_id"] != expected_doc_id:
            raise SystemExit(f"BM25 top hit for {expected_doc_id} smoke query was not from {expected_doc_id}")
        if any(row["score"] == 0 for row in lexical + dense):
            raise SystemExit("A search result returned a zero score")
        for rows in (lexical, dense):
            for row in rows:
                if not row["text_original"] or not row["chunk_id"] or not {"pdf_page_start", "pdf_page_end"}.issubset(row["metadata"]):
                    raise SystemExit("Search result is missing text, ID, or page metadata")
                has_page_range = row["metadata"]["pdf_page_start"] is not None and row["metadata"]["pdf_page_end"] is not None
                if args.require_citations and not has_page_range:
                    raise SystemExit(f"A search result lacks a non-null PDF page range: {row['chunk_id']}")
                if not candidate_only and not has_page_range:
                    raise SystemExit("A production smoke result lacks its verified PDF page reference")
                retrieval.get_parent_context(row["metadata"]["parent_id"])
        results.append({"query": query, "expected_document_id": expected_doc_id, "bm25": lexical, "dense": dense})
    output = {"release_status": manifest["status"], "production_release_eligible": manifest.get("production_release_eligible", False),
              "query_results": results}
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
