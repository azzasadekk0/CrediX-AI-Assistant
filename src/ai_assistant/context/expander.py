"""Canonical, source-only expansion of reranked chunk evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from ..models import ExpandedEvidence, RerankedCandidate
from ..evaluation.evaluate_retrieval import DEFAULT_BUNDLE


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


class CanonicalBundleContext:
    """Small read-only indexes over the already-certified chunks and parents."""

    def __init__(self, bundle_dir: Path = DEFAULT_BUNDLE) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.parents = {row["parent_id"]: row for row in _read_jsonl(self.bundle_dir / "parents.jsonl")}
        self.chunks = {row["chunk_id"]: row for row in _read_jsonl(self.bundle_dir / "chunks.jsonl")}
        orders: dict[tuple[str, int], str] = {}
        invalid: set[tuple[str, int]] = set()
        for chunk in self.chunks.values():
            order = chunk.get("chunk_order")
            document_id = chunk.get("document_id")
            if not isinstance(document_id, str) or not isinstance(order, int):
                continue
            key = (document_id, order)
            if key in orders:
                invalid.add(key)
            else:
                orders[key] = chunk["chunk_id"]
        self._orders = {key: value for key, value in orders.items() if key not in invalid}

    def parent(self, parent_id: str) -> dict[str, Any]:
        try:
            return self.parents[parent_id]
        except KeyError as exc:
            raise ValueError(f"canonical parent not found: {parent_id}") from exc

    def neighbors(self, chunk_id: str) -> list[dict[str, Any]]:
        """Return only safe same-document adjacent chunks, in canonical order."""
        chunk = self.chunks.get(chunk_id)
        if not chunk:
            return []
        document_id, order = chunk.get("document_id"), chunk.get("chunk_order")
        if not isinstance(document_id, str) or not isinstance(order, int):
            return []
        rows = []
        for adjacent in (order - 1, order, order + 1):
            adjacent_id = self._orders.get((document_id, adjacent))
            if adjacent_id is None:
                continue
            row = self.chunks[adjacent_id]
            if row.get("document_id") == document_id:
                rows.append(row)
        return rows


class ContextExpander:
    """Expand evidence without modifying or synthesizing its source text."""

    def __init__(self, source: CanonicalBundleContext | None = None) -> None:
        self.source = source or CanonicalBundleContext()

    def expand(self, candidate: RerankedCandidate, reranker_rank: int, strategy: str) -> ExpandedEvidence:
        parent = self.source.parent(candidate.parent_id)
        chunk_text = candidate.text_original
        if strategy == "chunk_only":
            text, chunk_ids, expansion_type = chunk_text, [candidate.chunk_id], strategy
        elif strategy in {"parent", "parent_deduplicated"}:
            text, chunk_ids, expansion_type = parent["text_original"], [candidate.chunk_id], strategy
        elif strategy == "chunk_plus_parent":
            parent_text = parent["text_original"]
            if chunk_text.strip() and chunk_text.strip() in parent_text:
                text, expansion_type = parent_text, "parent_containing_chunk"
            else:
                text, expansion_type = f"{chunk_text}\n\n{parent_text}", strategy
            chunk_ids = [candidate.chunk_id]
        elif strategy == "neighbor_1":
            neighbors = self.source.neighbors(candidate.chunk_id)
            if not neighbors:
                text, chunk_ids, expansion_type = chunk_text, [candidate.chunk_id], "chunk_only_no_safe_neighbors"
            else:
                text = "\n\n".join(str(row["text_original"]) for row in neighbors)
                chunk_ids = [str(row["chunk_id"]) for row in neighbors]
                expansion_type = strategy
        else:
            raise ValueError(f"unsupported expansion strategy: {strategy}")
        metadata = dict(candidate.metadata)
        metadata.update({"canonical_parent_id": parent["parent_id"], "source_text_only": True})
        if strategy == "neighbor_1" and chunk_ids:
            # Adjacent chunks can cross a parent boundary (for example, a
            # legal article split across consecutive chunks).  Preserve the
            # parent provenance for every included chunk instead of attributing
            # the whole window to the primary candidate's parent.
            chunk_parent_ids = {
                str(row["chunk_id"]): str(row.get("parent_id"))
                for row in self.source.neighbors(candidate.chunk_id)
                if row.get("chunk_id") is not None and row.get("parent_id") is not None
            }
            metadata["chunk_parent_ids"] = chunk_parent_ids
            metadata["source_parent_ids"] = list(dict.fromkeys(chunk_parent_ids.values()))
        return ExpandedEvidence(
            citation_handle=None, chunk_ids=chunk_ids, primary_chunk_id=candidate.chunk_id, parent_id=candidate.parent_id,
            document_id=candidate.document_id, document_title=candidate.document_title,
            text_original=chunk_text, expanded_text=text, reranker_rank=reranker_rank,
            reranker_score=candidate.reranker_score, dense_rank=candidate.dense_rank, bm25_rank=candidate.bm25_rank,
            rrf_score=candidate.rrf_score, pdf_page_start=candidate.pdf_page_start, pdf_page_end=candidate.pdf_page_end,
            article_number=candidate.article_number, provision_number=candidate.provision_number,
            expansion_type=expansion_type, metadata=metadata,
        )
