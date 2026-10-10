"""Provider-neutral, fail-closed Evidence Support Gate."""
from __future__ import annotations

import json

from ..llm.base import LLMProvider
from ..llm.models import LLMRequest
from ..models import ContextPack
from .errors import SupportGateError
from .models import EvidenceSupportAssessment
from .prompts import SUPPORT_GATE_PROMPT_VERSION, SUPPORT_GATE_SCHEMA, support_gate_system_prompt, support_gate_user_prompt


def parse_assessment(text: str) -> EvidenceSupportAssessment:
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise SupportGateError("provider did not return valid support-gate JSON") from exc
    required = {"support", "supported_handles", "supported_aspects", "unsupported_aspects", "reason_code"}
    if not isinstance(value, dict) or set(value) != required:
        raise SupportGateError("support-gate result must contain exactly the configured fields")
    for key in ("supported_handles", "supported_aspects", "unsupported_aspects"):
        if not isinstance(value[key], list):
            raise SupportGateError(f"{key} must be an array")
        value[key] = list(dict.fromkeys(value[key]))
    return EvidenceSupportAssessment(**value)


class EvidenceSupportGate:
    """Classify direct evidence support without answering or routing the query."""

    def __init__(self, provider: LLMProvider, *, prompt_version: str = SUPPORT_GATE_PROMPT_VERSION) -> None:
        self.provider = provider
        self.prompt_version = prompt_version
        self.last_trace: dict[str, object] = {}

    def assess(self, context_pack: ContextPack) -> EvidenceSupportAssessment:
        if not context_pack.evidence_items:
            result = EvidenceSupportAssessment("none", [], [], ["no supplied evidence"], "other")
            self.last_trace = {"conservative_default": True, "reason": "empty_context"}
            return result
        try:
            response = self.provider.generate(LLMRequest(support_gate_system_prompt(self.prompt_version), support_gate_user_prompt(context_pack),
                                                          response_schema=SUPPORT_GATE_SCHEMA, response_schema_name="evidence_support"))
            result = parse_assessment(response.text)
            unknown = [handle for handle in result.supported_handles if handle not in context_pack.citation_map]
            if unknown:
                raise SupportGateError("support-gate result referenced unavailable evidence handles")
            self.last_trace = {"provider": response.provider, "model": response.model, "latency_seconds": response.latency_seconds,
                               "input_tokens": response.input_tokens, "output_tokens": response.output_tokens, "total_tokens": response.total_tokens,
                               "conservative_default": False}
            return result
        except Exception as exc:
            self.last_trace = {"conservative_default": True, "reason": type(exc).__name__, "error": str(exc)}
            return EvidenceSupportAssessment("none", [], [], ["support classifier unavailable or invalid"], "other", classifier_failed=True)
