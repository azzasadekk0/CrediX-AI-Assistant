"""Versioned prompts and schema for source-only evidence-support classification."""
from __future__ import annotations

from ..models import ContextPack


SUPPORT_GATE_PROMPT_VERSION = "evidence_support_gate_v1"
SUPPORT_GATE_V1_1_PROMPT_VERSION = "evidence_support_gate_v1_1"
SUPPORT_GATE_SCHEMA = {
    "type": "object",
    "properties": {
        "support": {"type": "string", "enum": ["full", "partial", "none"]},
        "supported_handles": {"type": "array", "items": {"type": "string"}},
        "supported_aspects": {"type": "array", "items": {"type": "string"}},
        "unsupported_aspects": {"type": "array", "items": {"type": "string"}},
        "reason_code": {"type": "string", "enum": ["direct_support", "partial_support", "related_only", "missing_detail", "missing_required_fact", "other"]},
    },
    "required": ["support", "supported_handles", "supported_aspects", "unsupported_aspects", "reason_code"],
    "additionalProperties": False,
}


def support_gate_system_prompt(version: str = SUPPORT_GATE_PROMPT_VERSION) -> str:
    base = (
        "You are an evidence-support classifier for a source-grounded banking and regulatory assistant. "
        "Do not answer the user's question and do not use outside knowledge. Use only the supplied evidence. "
        "Classify whether the evidence directly supports every material aspect requested by the question: "
        "full means every material aspect is directly established; partial means at least one material aspect is directly established and at least one is not; "
        "none means no material requested aspect is directly established. Topic similarity, related rules, plausible inference, or a nearby process do not establish a requested fact. "
        "Use only E-handles present in the evidence. For full or partial, name concise material aspects, not an answer. "
        "For none, provide no supported handles or supported aspects. Return only the requested JSON object."
    )
    if version == SUPPORT_GATE_PROMPT_VERSION:
        return base
    if version == SUPPORT_GATE_V1_1_PROMPT_VERSION:
        return base + (
            " Treat clear semantic entailment and ordinary paraphrase as direct support even when the question does not repeat the evidence wording. "
            "First identify only material requested aspects, then check each against the evidence; do not create minor artificial sub-aspects. "
            "Do not upgrade a related topic, an unstated implication, or a missing procedure/detail into direct support."
        )
    raise ValueError(f"unsupported support-gate prompt version: {version}")


def support_gate_user_prompt(context_pack: ContextPack) -> str:
    return f"QUESTION:\n{context_pack.original_query}\n\nEVIDENCE:\n{context_pack.render_for_llm()}"
