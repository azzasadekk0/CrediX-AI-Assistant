"""Typed, conservative result model for the Evidence Support Gate."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .errors import SupportGateError


SupportLevel = Literal["full", "partial", "none"]
SourceRequirement = Literal["required", "not_required", "unclear"]
ReasonCode = Literal["direct_support", "partial_support", "related_only", "missing_detail", "missing_required_fact", "other"]


@dataclass(frozen=True, slots=True)
class EvidenceSupportAssessment:
    """A source-only assessment; it must never contain an answer to the user."""

    support: SupportLevel
    supported_handles: list[str]
    supported_aspects: list[str]
    unsupported_aspects: list[str]
    reason_code: ReasonCode
    classifier_failed: bool = False

    def __post_init__(self) -> None:
        if self.support not in {"full", "partial", "none"}:
            raise SupportGateError("support must be full, partial, or none")
        if self.reason_code not in {"direct_support", "partial_support", "related_only", "missing_detail", "missing_required_fact", "other"}:
            raise SupportGateError("unsupported reason_code")
        fields = (self.supported_handles, self.supported_aspects, self.unsupported_aspects)
        if not all(isinstance(values, list) and all(isinstance(value, str) and value.strip() for value in values) for values in fields):
            raise SupportGateError("support details must be lists of non-empty strings")
        if len(self.supported_handles) != len(set(self.supported_handles)):
            raise SupportGateError("supported_handles must be de-duplicated")
        if self.support == "full" and (not self.supported_handles or not self.supported_aspects or self.unsupported_aspects):
            raise SupportGateError("full support requires supported handles/aspects and no unsupported aspects")
        if self.support == "partial" and (not self.supported_handles or not self.supported_aspects or not self.unsupported_aspects):
            raise SupportGateError("partial support requires both supported and unsupported material aspects")
        if self.support == "none" and (self.supported_handles or self.supported_aspects):
            raise SupportGateError("none support cannot claim supported handles or aspects")
        if not isinstance(self.classifier_failed, bool):
            raise SupportGateError("classifier_failed must be boolean")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
