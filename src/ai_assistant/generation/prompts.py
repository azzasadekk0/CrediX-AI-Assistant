"""Versioned source-only prompts for a ContextPack."""
from __future__ import annotations

from ..models import ContextPack
from .config import GenerationConfig


GROUNDED_RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {"type": "array", "items": {"type": "string"}},
        "no_answer": {"type": "boolean"},
    },
    "required": ["answer", "citations", "no_answer"],
    "additionalProperties": False,
}


def grounded_system_prompt(config: GenerationConfig) -> str:
    if config.prompt_version not in {"grounded_v1", "grounded_v2"}:
        raise ValueError(f"unsupported prompt version: {config.prompt_version}")
    base = (
        "You are the CrediX grounded banking and regulatory assistant. Use ONLY the supplied evidence and no outside factual knowledge. "
        "Do not invent regulations, requirements, limits, percentages, monetary values, dates, article numbers, document names, or exceptions. "
        "Preserve supported numbers, conditions, exceptions, mandatory/optional wording, and legal qualifiers. "
        "Answer in the user's language. Every factual answer must cite one or more supplied E-handles only. "
        "If the evidence is insufficient, return no_answer=true with an empty citations list and a concise abstention. "
        "Return only the requested JSON structure."
    )
    if config.prompt_version == "grounded_v1":
        return base
    return base + (
        " For a factual answer, select only exact E-handles from the supplied evidence that directly support claims you state; never invent, alter, or guess handles. "
        "Do not infer the requested fact from related evidence: if direct support is missing, return no_answer=true and citations=[]. "
        "For a multi-part answer, include every directly supporting E-handle needed for the stated parts. "
        "For an Arabic question, the answer must be Arabic. Before responding, verify that no_answer=true has no citations and no_answer=false has at least one exact supplied E-handle."
    )


def grounded_user_prompt(context_pack: ContextPack) -> str:
    return (f"RESPONSE LANGUAGE: {_response_language(context_pack.original_query)}\n\n"
            f"QUESTION:\n{context_pack.original_query}\n\nEVIDENCE:\n{context_pack.render_for_llm()}")


def _response_language(query: str) -> str:
    """Small deterministic policy signal; source evidence never changes language selection."""
    arabic = sum("\u0600" <= character <= "\u06ff" for character in query)
    latin = sum(character.isascii() and character.isalpha() for character in query)
    return "Arabic" if arabic > latin else "English"


def citation_repair_prompt(answer: str, citations: list[str], invalid_handles: list[str], context_pack: ContextPack) -> str:
    return (
        "Correct only citation selection/format in the JSON. Preserve the answer text exactly and do not add or change facts. "
        f"Answer text that must remain unchanged: {answer!r}\n"
        f"Invalid or missing citations: {', '.join(invalid_handles) or 'missing citations'}\n"
        f"Available handles: {', '.join(context_pack.citation_map)}\n"
        f"Current citations: {citations!r}"
    )
