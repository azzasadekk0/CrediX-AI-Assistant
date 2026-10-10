"""Rank-preserving, complete-block evidence selection."""
from __future__ import annotations

from dataclasses import replace

from ..models import ExpandedEvidence
from .config import ContextConfig


def estimate_tokens(text: str, method: str = "whitespace_approx_v1") -> int:
    """Deterministic approximation: one token per non-empty whitespace token."""
    if method != "whitespace_approx_v1":
        raise ValueError(f"unsupported token estimation method: {method}")
    return len(text.split())


class EvidenceSelector:
    """Deduplicate source blocks, then retain full ranked blocks within budget."""

    def select(self, items: list[ExpandedEvidence], config: ContextConfig) -> tuple[list[ExpandedEvidence], int, int]:
        deduplicated: list[ExpandedEvidence] = []
        # Text equality alone is insufficient: unrelated documents can legitimately
        # repeat boilerplate and must retain separate source boundaries.
        seen_text: set[tuple[str, str]] = set()
        parent_positions: dict[str, int] = {}
        duplicates_removed = 0
        for item in items:
            parent_dedup = config.deduplicate_parents or config.expansion_strategy == "parent_deduplicated"
            position = parent_positions.get(item.parent_id) if parent_dedup else None
            if position is not None:
                existing = deduplicated[position]
                chunk_ids = list(dict.fromkeys(existing.chunk_ids + item.chunk_ids))
                deduplicated[position] = replace(existing, chunk_ids=chunk_ids)
                duplicates_removed += 1
                continue
            text_key = (item.document_id, item.expanded_text)
            if text_key in seen_text:
                duplicates_removed += 1
                continue
            parent_positions[item.parent_id] = len(deduplicated)
            seen_text.add(text_key)
            deduplicated.append(item)

        selected: list[ExpandedEvidence] = []
        total = 0
        for item in deduplicated:
            tokens = estimate_tokens(item.expanded_text, config.token_estimation_method)
            if total + tokens > config.max_context_tokens:
                continue  # explicit complete-block policy: never silently truncate.
            selected.append(replace(item, citation_handle=f"E{len(selected) + 1}"))
            total += tokens
        return selected, total, duplicates_removed
