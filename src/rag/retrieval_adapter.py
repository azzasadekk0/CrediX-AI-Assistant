"""Local retrieval adapter for a built CrediX release directory."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .bm25 import BM25Index, read_jsonl
    from .release_integrity import validate_runtime_bundle
except (ImportError, ValueError):
    try:
        from bm25 import BM25Index, read_jsonl
        from release_integrity import validate_runtime_bundle
    except ImportError:
        from rag.bm25 import BM25Index, read_jsonl
        from rag.release_integrity import validate_runtime_bundle


def _index_dir() -> Path:
    configured = os.environ.get("CREDIX_RAG_INDEX_DIR")
    if configured:
        return Path(configured).resolve()
    sibling_data = Path(__file__).resolve().parent.parent / "rag_data" / "current"
    if sibling_data.exists():
        return sibling_data.resolve()
    cwd_data = Path("src/rag_data/current").resolve()
    if cwd_data.exists():
        return cwd_data
    return Path("artifacts/rag/release/current").resolve()


@lru_cache(maxsize=4)
def _corpus(index_dir: str):
    root = Path(index_dir).resolve()
    validate_runtime_bundle(root)
    chunks = read_jsonl(root / "chunks.jsonl")
    parents = read_jsonl(root / "parents.jsonl")
    chunk_by_id = {row["chunk_id"]: row for row in chunks}
    parent_by_id = {row["parent_id"]: row for row in parents}
    bm25_rows = read_jsonl(root / "bm25_corpus.jsonl")
    if len(chunk_by_id) != len(chunks) or len(parent_by_id) != len(parents):
        raise ValueError("Index bundle has duplicate chunk or parent IDs")
    if [r["chunk_id"] for r in bm25_rows] != [r["chunk_id"] for r in chunks]:
        raise ValueError("BM25 corpus order/IDs do not match chunks.jsonl")
    return root, chunks, parents, chunk_by_id, parent_by_id, bm25_rows, BM25Index(bm25_rows)


def _result(chunk: dict[str, Any], score: float) -> dict[str, Any]:
    metadata = {key: value for key, value in chunk.items() if key not in {"chunk_id", "document_id", "text_original"}}
    return {"chunk_id": chunk["chunk_id"], "document_id": chunk["document_id"],
            "text_original": chunk["text_original"], "metadata": metadata, "score": float(score)}


def embed_query(query: str) -> np.ndarray:
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a nonempty string")
    root = _index_dir()
    validate_runtime_bundle(root)
    config = json.loads((root / "embedding_config.json").read_text(encoding="utf-8"))
    model_key = (config["model_name"], config["model_version"], config.get("device", "cpu"), config.get("max_sequence_length", 8192))
    return _encode_query(model_key, query.strip())


@lru_cache(maxsize=2)
def _load_model(model_name: str, revision: str, device: str, max_sequence_length: int):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for embed_query; install rag/requirements.txt") from exc
    model = SentenceTransformer(model_name, revision=revision, device=device)
    model.max_seq_length = max_sequence_length
    return model


def _encode_query(model_key: tuple[str, str, str, int], query: str) -> np.ndarray:
    model = _load_model(*model_key)
    vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    vector = np.asarray(vector, dtype=np.float32)
    if vector.ndim != 2 or vector.shape[0] != 1:
        raise ValueError("query encoder returned an invalid shape")
    return vector[0]


def dense_search(query_embedding: np.ndarray, top_k: int = 20,
                 filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if top_k < 1:
        raise ValueError("top_k must be positive")
    root, chunks, _, chunk_by_id, _, _, _ = _corpus(str(_index_dir()))
    config = json.loads((root / "embedding_config.json").read_text(encoding="utf-8"))
    query = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    if query.shape[1] != int(config["dimensions"]):
        raise ValueError(f"query dimension {query.shape[1]} does not match {config['dimensions']}")
    norm = np.linalg.norm(query, axis=1)
    if not np.isfinite(query).all() or norm[0] == 0:
        raise ValueError("query embedding must be finite and nonzero")
    query /= norm[:, None]
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("faiss-cpu is required for dense_search; see the dedicated environment instructions") from exc
    index = faiss.read_index(str(root / "index.faiss"))
    scores, indices = index.search(np.ascontiguousarray(query), len(chunks))
    id_map = json.loads((root / "faiss_id_map.json").read_text(encoding="utf-8"))
    results = []
    for score, row_idx in zip(scores[0], indices[0]):
        if row_idx < 0:
            continue
        chunk_id = id_map[row_idx]
        chunk = chunk_by_id[chunk_id]
        if filters and any(chunk.get(key) != value for key, value in filters.items()):
            continue
        results.append(_result(chunk, float(score)))
        if len(results) >= top_k:
            break
    return results


def bm25_search(query: str, top_k: int = 20) -> list[dict[str, Any]]:
    root, _, _, chunk_by_id, _, bm25_rows, index = _corpus(str(_index_dir()))
    del root
    return [_result(chunk_by_id[bm25_rows[idx]["chunk_id"]], score) for idx, score in index.search(query, top_k)]


def get_chunk(chunk_id: str) -> dict[str, Any]:
    _, _, _, chunk_by_id, _, _, _ = _corpus(str(_index_dir()))
    try:
        return chunk_by_id[chunk_id]
    except KeyError as exc:
        raise KeyError(f"Unknown chunk_id: {chunk_id}") from exc


def get_parent_context(parent_id: str) -> dict[str, Any]:
    _, _, _, _, parent_by_id, _, _ = _corpus(str(_index_dir()))
    try:
        return parent_by_id[parent_id]
    except KeyError as exc:
        raise KeyError(f"Unknown parent_id: {parent_id}") from exc
