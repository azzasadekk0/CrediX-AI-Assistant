"""Typed contracts for claim-level Auto routing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

from ..generation.hybrid import HybridSegment
from ..generation.response import ResolvedSource


ClaimKind = Literal["source", "general", "ambiguous"]
RouteMode = Literal["grounded", "general", "hybrid", "insufficient_evidence"]


@dataclass(frozen=True, slots=True)
class Claim:
    text: str
    kind: ClaimKind
    source_requested: bool
    general_requested: bool


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    claim: Claim
    evidence_count: int
    support: str
    supported_handles: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    mode: RouteMode
    claims: list[Claim]
    assessments: list[ClaimAssessment]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RoutedAnswer:
    """Uniform API-facing answer contract for all Auto routes."""

    original_query: str
    answer: str
    answer_mode: RouteMode
    provenance: str
    segments: list[HybridSegment]
    citations: list[str]
    resolved_sources: list[ResolvedSource]
    provider: str
    model: str
    generation_latency_seconds: float
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    disclaimer: str | None = None
    no_answer: bool = False
    validation_status: str = "validated"
    routing_decision: RoutingDecision | None = None

    def __post_init__(self) -> None:
        if self.answer_mode == "general" and (self.provenance != "ai_generated" or self.citations):
            raise ValueError("general route must be AI-generated without citations")
        if self.answer_mode == "grounded" and self.provenance != "retrieved":
            raise ValueError("grounded route must be retrieved")
        if self.answer_mode == "hybrid" and self.provenance != "mixed":
            raise ValueError("hybrid route must use mixed provenance")
        if self.answer_mode == "insufficient_evidence" and self.citations:
            raise ValueError("insufficient evidence cannot contain citations")
        for segment in self.segments:
            if segment.source_type == "ai_generated" and segment.citation_handles:
                raise ValueError("AI-generated segments cannot contain citations")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

