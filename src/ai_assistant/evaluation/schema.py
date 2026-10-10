"""Typed schema and validation for reviewed retrieval-ground-truth cases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GoldenRetrievalCase:
    """A reviewed query and its chunk-level retrieval evidence labels."""

    query_id: str
    query: str
    language: str
    query_type: str
    difficulty: str
    answerable: bool
    expected_document_ids: list[str]
    relevant_chunk_ids: list[str]
    primary_chunk_id: str | None
    relevant_parent_ids: list[str]
    label_note: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GoldenRetrievalCase":
        required = {
            "query_id", "query", "language", "query_type", "difficulty", "answerable",
            "expected_document_ids", "relevant_chunk_ids", "primary_chunk_id",
            "relevant_parent_ids", "label_note",
        }
        missing = sorted(required.difference(value))
        if missing:
            raise ValueError(f"golden case is missing required fields: {', '.join(missing)}")
        case = cls(**{key: value[key] for key in required})
        case.validate_shape()
        return case

    def validate_shape(self) -> None:
        if not self.query_id or not isinstance(self.query_id, str):
            raise ValueError("query_id must be a non-empty string")
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError(f"{self.query_id}: query must be a non-empty string")
        if self.difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"{self.query_id}: invalid difficulty {self.difficulty!r}")
        for name, values in (
            ("expected_document_ids", self.expected_document_ids),
            ("relevant_chunk_ids", self.relevant_chunk_ids),
            ("relevant_parent_ids", self.relevant_parent_ids),
        ):
            if not isinstance(values, list) or not all(isinstance(item, str) and item for item in values):
                raise ValueError(f"{self.query_id}: {name} must contain non-empty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{self.query_id}: {name} contains duplicates")
        if self.answerable:
            if not self.expected_document_ids or not self.relevant_chunk_ids or not self.relevant_parent_ids:
                raise ValueError(f"{self.query_id}: answerable cases require document, chunk, and parent labels")
            if self.primary_chunk_id is not None and self.primary_chunk_id not in self.relevant_chunk_ids:
                raise ValueError(f"{self.query_id}: primary_chunk_id must belong to relevant_chunk_ids")
        elif any((self.expected_document_ids, self.relevant_chunk_ids, self.relevant_parent_ids)) or self.primary_chunk_id is not None:
            raise ValueError(f"{self.query_id}: unanswerable cases must not contain evidence labels")
