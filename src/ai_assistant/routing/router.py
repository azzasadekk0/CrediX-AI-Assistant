"""Evidence-first Auto router with bounded decomposition and fail-closed policy."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

from ..context.builder import ContextBuilder
from ..generation.general import GeneralAssistant
from ..generation.generator import GroundedRAGAssistant
from ..generation.hybrid import HybridSegment
from ..llm.base import LLMProvider
from ..llm.factory import get_llm
from ..models import ContextPack
from ..support_gate.gate import EvidenceSupportGate
from ..support_gate.models import EvidenceSupportAssessment
from .models import Claim, ClaimAssessment, RoutedAnswer, RoutingDecision


_GENERAL_INTENT = re.compile(
    r"\b(?:h+i+|hello|hey|thanks?|thank\s+you|explain|define|meaning of|what is|what does\s+[^?]*\s+mean|in general|generally|concept of)\b"
    r"|(?:\u0627\u0647\u0644\u0627|\u0623\u0647\u0644\u0627|\u0645\u0631\u062d\u0628\u0627|\u0645\u0631\u062d\u0628\u0627\u064b|\u0633\u0644\u0627\u0645|\u0634\u0643\u0631\u0627|\u0634\u0643\u0631\u064b\u0627|\u0627\u0634\u0631\u062d|\u064a\u0639\u0646\u064a\s*\u0627\u064a\u0647|\u0645\u0627\s+\u0645\u0639\u0646\u0649|\u0628\u0634\u0643\u0644\s+\u0639\u0627\u0645|\u0645\u0641\u0647\u0648\u0645)",
    re.IGNORECASE,
)
_SOURCE_INTENT = re.compile(
    r"\b(?:according to|per the|in the document|in our|credi[xs]|policy|policies|regulation|source|evidence|file|case|application|record|limit)\b"
    r"|(?:\u062d\u0633\u0628|\u0648\u0641\u0642|\u0627\u0644\u0644\u0627\u0626\u062d\u0629|\u0627\u0644\u0633\u064a\u0627\u0633\u0629|\u0627\u0644\u0645\u0633\u062a\u0646\u062f|\u0627\u0644\u0648\u062b\u064a\u0642\u0629|\u0627\u0644\u0645\u0644\u0641|\u0627\u0644\u062d\u0627\u0644\u0629|\u0627\u0644\u0637\u0644\u0628|\u0627\u0644\u0633\u062c\u0644|\u0643\u0634\u0641|\u0627\u0644\u062d\u0633\u0627\u0628)",
    re.IGNORECASE,
)
_HYBRID_SPLIT = re.compile(
    r"\s+(?:and|then)\s+(?=(?:explain|tell\s+me|why|how|what\s+does)\b)"
    r"|\s+(?:\u0648|\u062b\u0645)\s*(?=(?:\u0627\u0634\u0631\u062d|\u0648\u0636\u062d|\u0644\u0645\u0627\u0630\u0627|\u0644\u064a\u0647|\u0645\u0627\s+\u0645\u0639\u0646\u0649|\u0627\u0632\u0627\u064a)\b)",
    re.IGNORECASE,
)


def decompose_query(query: str, *, max_claims: int = 3) -> list[Claim]:
    """Bounded, deterministic decomposition; preserve the original text."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    text = query.strip()
    match = _HYBRID_SPLIT.search(text)
    if match:
        parts = [text[: match.start()].strip(" ,;؟?"), text[match.end() :].strip(" ,;؟?")]
    else:
        parts = [text]
    split_detected = match is not None
    parts = [part for part in parts if part][:max_claims]
    claims: list[Claim] = []
    for index, part in enumerate(parts):
        # An explicit ``<document claim> and/و explain`` split defines the
        # first clause as source-backed even when colloquial Arabic does not
        # contain one of the lexical source markers.  Without this flag the
        # router could label the whole request ``general`` and then violate
        # the typed provenance contract while merging a grounded segment.
        source = (split_detected and index == 0) or bool(_SOURCE_INTENT.search(part))
        general = bool(_GENERAL_INTENT.search(part))
        kind = "source" if source else "general" if general else "ambiguous"
        claims.append(Claim(part, kind, source, general))
    return claims


def _assessment(context: ContextPack, gate: EvidenceSupportGate | None) -> EvidenceSupportAssessment:
    if gate is None:
        return EvidenceSupportAssessment("none", [], [], ["support gate unavailable"], "other", classifier_failed=True)
    return gate.assess(context)


def route_query(claims: list[Claim], assessments: list[ClaimAssessment]) -> RoutingDecision:
    """Apply deterministic policy after retrieval/support checks."""

    if not claims:
        raise ValueError("at least one claim is required")
    if any(item.claim.kind in {"source", "ambiguous"} and item.support != "full" for item in assessments):
        return RoutingDecision("insufficient_evidence", claims, assessments, "required claim lacks full evidence support")
    has_source = any(item.claim.kind == "source" for item in assessments)
    has_general = any(item.claim.kind == "general" for item in assessments)
    if has_source and has_general:
        return RoutingDecision("hybrid", claims, assessments, "supported source and explicit general claims")
    if has_general:
        return RoutingDecision("general", claims, assessments, "explicit general intent")
    return RoutingDecision("grounded", claims, assessments, "supported source or conservative fallback")


class RoutedAssistant:
    """Probe first, support-check source claims, then call isolated generators."""

    def __init__(
        self,
        *,
        provider: LLMProvider | None = None,
        context_builder: ContextBuilder | None = None,
        support_gate: EvidenceSupportGate | None = None,
        grounded: GroundedRAGAssistant | None = None,
        general: GeneralAssistant | None = None,
    ) -> None:
        # Resolve one provider for every stage.  The probe is deterministic,
        # while the support gate and both generators must share the same
        # ENV-selected runtime.  Supplying a provider is preferred in tests;
        # otherwise use the canonical factory so direct library users cannot
        # accidentally run with a missing support gate.
        resolved_provider = provider
        if resolved_provider is None and (support_gate is None or grounded is None or general is None):
            resolved_provider = get_llm()
        self._context_builder = context_builder or ContextBuilder()
        self._support_gate = support_gate or EvidenceSupportGate(resolved_provider)  # type: ignore[arg-type]
        # Reuse the probe builder in the grounded generator.  Both stages
        # need the same frozen retrieval/context pipeline; sharing it avoids
        # loading a second CUDA cross-encoder for the first Auto request.
        self._grounded = grounded or GroundedRAGAssistant(
            context_builder=self._context_builder,
            provider=resolved_provider,
        )
        self._general = general or GeneralAssistant(provider=resolved_provider)

    def decide(self, query: str) -> RoutingDecision:
        claims = decompose_query(query)
        assessments: list[ClaimAssessment] = []
        for claim in claims:
            if claim.kind == "general":
                assessments.append(ClaimAssessment(claim, 0, "not_required", [], "explicit general intent"))
                continue
            try:
                context = self._context_builder.build(claim.text)
            except Exception as exc:
                # Retrieval/index/runtime failures are a safe abstention, never
                # permission to fall back silently to general generation.
                assessments.append(ClaimAssessment(claim, 0, "none", [], f"probe_failed:{type(exc).__name__}"))
                continue
            assessment = (
                EvidenceSupportAssessment("none", [], [], ["no supplied evidence"], "other")
                if context.evidence_count <= 0
                else _assessment(context, self._support_gate)
            )
            support = "none" if assessment.classifier_failed else assessment.support
            assessments.append(ClaimAssessment(claim, context.evidence_count, support,
                                                list(assessment.supported_handles) if support != "none" else [],
                                                "classifier_failed" if assessment.classifier_failed else assessment.reason_code))
        return route_query(claims, assessments)

    def answer(self, query: str) -> RoutedAnswer:
        decision = self.decide(query)
        if decision.mode == "insufficient_evidence":
            return self._insufficient(query, decision)
        generated: list[tuple[Claim, Any]] = []
        for claim in decision.claims:
            generated.append((claim, self._general.answer(claim.text) if claim.kind == "general" else self._grounded.answer(claim.text)))
        return self._merge(query, decision, generated)

    @staticmethod
    def _insufficient(query: str, decision: RoutingDecision) -> RoutedAnswer:
        arabic = any("\u0600" <= char <= "\u06ff" for char in query)
        text = "المصادر المتاحة لا تحتوي على أدلة كافية للإجابة عن هذا السؤال." if arabic else "The available sources do not contain enough evidence to answer this question."
        segment = HybridSegment(text, "retrieved", [], "unsupported")
        return RoutedAnswer(query, text, "insufficient_evidence", "retrieved", [segment], [], [], "router", "router", 0.0, None, None, None, no_answer=True, routing_decision=decision)

    @staticmethod
    def _merge(query: str, decision: RoutingDecision, generated: list[tuple[Claim, Any]]) -> RoutedAnswer:
        segments: list[HybridSegment] = []
        citations: list[str] = []
        sources = []
        texts = []
        latency = 0.0
        input_tokens = output_tokens = total_tokens = None
        provider_names: list[str] = []
        model_names: list[str] = []
        for claim, result in generated:
            texts.append(result.answer)
            latency += result.generation_latency_seconds
            input_tokens = _add_optional(input_tokens, result.input_tokens)
            output_tokens = _add_optional(output_tokens, result.output_tokens)
            total_tokens = _add_optional(total_tokens, result.total_tokens)
            provider_names.append(result.provider)
            model_names.append(result.model)
            if claim.kind == "general":
                segments.append(HybridSegment(result.answer, "ai_generated", [], "inference"))
            else:
                if result.no_answer:
                    return RoutedAssistant._insufficient(query, replace(decision, mode="insufficient_evidence"))
                segments.append(HybridSegment(result.answer, "retrieved", list(result.citations), "supported"))
                citations.extend(result.citations)
                sources.extend(result.resolved_sources)
        mode = decision.mode
        provenance = "mixed" if mode == "hybrid" else "ai_generated" if mode == "general" else "retrieved"
        disclaimer = next((getattr(result, "disclaimer", None) for _, result in generated if getattr(result, "disclaimer", None)), None)
        return RoutedAnswer(query, "\n\n".join(texts), mode, provenance, segments, list(dict.fromkeys(citations)), sources,
                            "+".join(provider_names), "+".join(model_names), latency,
                            input_tokens, output_tokens, total_tokens, disclaimer=disclaimer, routing_decision=decision)


def _add_optional(first: int | None, second: int | None) -> int | None:
    """Sum optional token counters without turning missing data into zero."""

    if first is None or second is None:
        return None
    return first + second
