"""Verify release bundle integrity and its five-query hybrid smoke results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .release_integrity import ReleaseValidationError, validate_runtime_bundle
except ImportError:
    from release_integrity import ReleaseValidationError, validate_runtime_bundle

try:
    from .bm25 import read_jsonl
    from .eligibility import release_blocking_chunks
except (ImportError, ValueError):
    try:
        from bm25 import read_jsonl
        from eligibility import release_blocking_chunks
    except ImportError:
        from rag.bm25 import read_jsonl
        from rag.eligibility import release_blocking_chunks


def validate(index_dir: Path) -> dict[str, Any]:
    manifest = json.loads((index_dir / "release_manifest.json").read_text(encoding="utf-8"))
    status = manifest.get("release_status", manifest.get("status"))
    if status == "candidate_only":
        if manifest.get("production_release_eligible") is not False:
            raise ValueError("candidate bundle must remain production-ineligible")
    elif status == "production_released":
        if manifest.get("production_release_eligible") is not True:
            raise ValueError("production release must be marked eligible")
    else:
        raise ValueError(f"unsupported release status: {status!r}")

    hash_results = {}
    for filename, expected in manifest.get("artifacts_sha256", {}).items():
        actual = hashlib.sha256((index_dir / filename).read_bytes()).hexdigest()
        hash_results[filename] = {"expected": expected, "actual": actual, "matches": actual == expected}
        if actual != expected:
            raise ValueError(f"artifact hash mismatch: {filename}")

    chunks = read_jsonl(index_dir / "chunks.jsonl")
    eligibility_blockers = release_blocking_chunks(chunks)
    if status == "production_released" and eligibility_blockers:
        raise ValueError(f"production bundle has {len(eligibility_blockers)} eligibility-blocked chunks")
    parents = read_jsonl(index_dir / "parents.jsonl")
    bm25 = read_jsonl(index_dir / "bm25_corpus.jsonl")
    id_map = json.loads((index_dir / "faiss_id_map.json").read_text(encoding="utf-8"))
    chunk_ids = [row["chunk_id"] for row in chunks]
    if len(chunks) != manifest.get("chunk_count") or len(parents) != manifest.get("parent_count"):
        raise ValueError("chunk/parent counts do not match the manifest")
    if chunk_ids != id_map or chunk_ids != [row["chunk_id"] for row in bm25]:
        raise ValueError("dense ID map and BM25 corpus do not match chunk records")

    missing_page_count = sum(row.get("pdf_page_start") is None or row.get("pdf_page_end") is None for row in chunks)
    if status == "production_released":
        allowed_provenance = {"human_source_verified", "ai_assisted_ocr_page_matched"}
        ineligible = [row["chunk_id"] for row in chunks if
                      row.get("verification_status") != "verified"
                      or row.get("text_quality_status") != "reviewed"
                      or row.get("pdf_page_start") is None or row.get("pdf_page_end") is None
                      or row.get("page_provenance") not in allowed_provenance]
        if ineligible:
            raise ValueError(f"production bundle has {len(ineligible)} ineligible chunks")
        if missing_page_count != 0 or manifest.get("pending_verification_count") != 0:
            raise ValueError("production manifest has unresolved page or verification counts")

    smoke = json.loads((index_dir / "smoke_test.json").read_text(encoding="utf-8"))
    query_results = smoke.get("query_results", [])
    if len(query_results) != 5:
        raise ValueError(f"expected five smoke queries, found {len(query_results)}")
    top_hit_results = []
    for case in query_results:
        expected = case["expected_document_id"]
        dense_doc = case["dense"][0]["document_id"] if case.get("dense") else None
        bm25_doc = case["bm25"][0]["document_id"] if case.get("bm25") else None
        top_hit_results.append({"expected_document_id": expected, "dense_top_document_id": dense_doc,
                                "bm25_top_document_id": bm25_doc,
                                "both_expected": dense_doc == expected and bm25_doc == expected})
    if not all(row["both_expected"] for row in top_hit_results):
        raise ValueError("one or more dense/BM25 smoke queries missed the expected document at rank 1")

    default_path = json.loads((index_dir / "default_path_smoke.json").read_text(encoding="utf-8"))
    if not default_path.get("retrieval_passed") or default_path.get("environment_override_used"):
        raise ValueError("default-path smoke did not pass without an environment override")

    return {
        "status": "production_release_passed" if status == "production_released" else "candidate_mvp_integration_passed",
        "release_status": status,
        "production_release_eligible": manifest["production_release_eligible"],
        "chunk_count": len(chunks),
        "verified_chunk_count": sum(row.get("verification_status") == "verified" and row.get("text_quality_status") == "reviewed" for row in chunks),
        "eligibility_blocker_count": len(eligibility_blockers),
        "parent_count": len(parents),
        "artifact_hashes_match": all(row["matches"] for row in hash_results.values()),
        "artifact_hashes": hash_results,
        "five_query_dense_and_bm25_expected_top_document_passed": True,
        "top_document_checks": top_hit_results,
        "default_adapter_path_passed_without_environment_override": True,
        "chunks_with_missing_page_references": missing_page_count,
        "page_citation_complete": missing_page_count == 0,
        "pending_verification_count": manifest.get("pending_verification_count"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index-dir", type=Path, default=Path("artifacts/rag/release/current"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.index_dir)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
