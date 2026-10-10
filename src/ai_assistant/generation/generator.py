"""Grounded generation over a frozen ContextPack and generic LLM provider."""
from __future__ import annotations

from dataclasses import replace

from ..context.builder import ContextBuilder
from ..llm.base import LLMProvider
from ..llm.factory import get_llm
from ..llm.models import LLMRequest
from ..models import ContextPack
from .citations import StructuredGroundedResult, canonicalize_citations_with_details, parse_grounded_result, raw_first_pass_payload, validate_citations
from .config import GenerationConfig, load_generation_config
from .errors import CitationValidationError, GroundedGenerationError, StructuredOutputError
from .prompts import GROUNDED_RESULT_SCHEMA, citation_repair_prompt, grounded_system_prompt, grounded_user_prompt
from .response import GroundedAnswer


class GroundedGenerator:
    """Use only a supplied ContextPack; it never searches or accesses the corpus."""

    def __init__(self, provider: LLMProvider, config: GenerationConfig | None = None, *, citation_canonicalization: bool = True) -> None:
        self.provider = provider
        self.config = config or load_generation_config()
        self.citation_canonicalization = citation_canonicalization
        self.last_trace: dict[str, object] = {}

    def generate(self, context_pack: ContextPack) -> GroundedAnswer:
        system = grounded_system_prompt(self.config)
        response = self.provider.generate(LLMRequest(system, grounded_user_prompt(context_pack), response_schema=GROUNDED_RESULT_SCHEMA,
                                                     response_schema_name="grounded_answer"))
        total_latency_seconds = response.latency_seconds
        total_input_tokens = response.input_tokens
        total_output_tokens = response.output_tokens
        total_tokens = response.total_tokens
        self.last_trace = {"raw_first_pass": raw_first_pass_payload(response.text)}
        result = parse_grounded_result(response.text)
        self.last_trace.update({"first_pass": {"answer": result.answer, "citations": result.citations, "no_answer": result.no_answer},
                           "repair_attempted": False, "repair_reason": None, "repair_output": None,
                           "citation_normalization_count": 0, "citation_normalization_details": [],
                           "direct_citation_valid": None, "citation_safety_stop": False,
                           "first_pass_latency_seconds": response.latency_seconds, "repair_latency_seconds": None,
                           "first_pass_token_usage": {"input": response.input_tokens, "output": response.output_tokens, "total": response.total_tokens},
                           "repair_token_usage": None})
        if result.no_answer and not result.answer:
            result = replace(result, answer=self._abstention(context_pack.original_query))
        if self.citation_canonicalization:
            result, normalization_count, normalization_details = canonicalize_citations_with_details(result, context_pack)
        else:
            normalization_count, normalization_details = 0, []
        self.last_trace["citation_normalization_count"] = normalization_count
        self.last_trace["citation_normalization_details"] = normalization_details
        self.last_trace["first_pass_canonicalized"] = {"answer": result.answer, "citations": result.citations, "no_answer": result.no_answer}
        repair_attempted = False
        try:
            sources = validate_citations(result, context_pack)
            self.last_trace["direct_citation_valid"] = True
        except CitationValidationError as error:
            self.last_trace["direct_citation_valid"] = False
            repair_attempted = True
            self.last_trace["repair_attempted"] = True
            self.last_trace["repair_reason"] = str(error)
            repaired_response = self.provider.generate(LLMRequest(system, citation_repair_prompt(result.answer, result.citations,
                                                                        error.invalid_handles, context_pack), response_schema=GROUNDED_RESULT_SCHEMA,
                                                               response_schema_name="grounded_answer_repair"))
            total_latency_seconds += repaired_response.latency_seconds
            total_input_tokens = self._sum_tokens(total_input_tokens, repaired_response.input_tokens)
            total_output_tokens = self._sum_tokens(total_output_tokens, repaired_response.output_tokens)
            total_tokens = self._sum_tokens(total_tokens, repaired_response.total_tokens)
            repaired = parse_grounded_result(repaired_response.text)
            self.last_trace["repair_latency_seconds"] = repaired_response.latency_seconds
            self.last_trace["repair_token_usage"] = {"input": repaired_response.input_tokens, "output": repaired_response.output_tokens, "total": repaired_response.total_tokens}
            self.last_trace["repair_output"] = {"answer": repaired.answer, "citations": repaired.citations, "no_answer": repaired.no_answer}
            if repaired.answer != result.answer or repaired.no_answer != result.no_answer:
                self.last_trace["citation_safety_stop"] = True
                raise GroundedGenerationError("citation repair attempted to alter answer facts")
            if self.citation_canonicalization:
                repaired, repair_normalization_count, repair_details = canonicalize_citations_with_details(repaired, context_pack)
            else:
                repair_normalization_count, repair_details = 0, []
            normalization_count += repair_normalization_count
            self.last_trace["citation_normalization_count"] = normalization_count
            self.last_trace["citation_normalization_details"] = normalization_details + repair_details
            self.last_trace["repair_canonicalized"] = {"answer": repaired.answer, "citations": repaired.citations, "no_answer": repaired.no_answer}
            try:
                sources = validate_citations(repaired, context_pack)
            except (CitationValidationError, StructuredOutputError) as repair_error:
                self.last_trace["citation_safety_stop"] = True
                raise GroundedGenerationError("citation repair failed") from repair_error
            result, response = repaired, repaired_response
        return GroundedAnswer(context_pack.original_query, result.answer, result.no_answer, result.citations, sources,
                              response.provider, response.model, total_latency_seconds, total_input_tokens,
                              total_output_tokens, total_tokens, "validated", self.config.prompt_version,
                              self._context_metadata(context_pack), repair_attempted, normalization_count > 0, normalization_count)

    @staticmethod
    def _context_metadata(context_pack: ContextPack) -> dict[str, object]:
        return {"evidence_count": context_pack.evidence_count, "estimated_context_tokens": context_pack.estimated_context_tokens,
                "document_ids": list(context_pack.document_ids)}

    @staticmethod
    def _abstention(query: str) -> str:
        """A source-safe fallback only for an explicit model abstention with empty text."""
        return ("المصادر المتاحة لا تحتوي على معلومات كافية للإجابة عن هذا السؤال."
                if any("\u0600" <= character <= "\u06ff" for character in query)
                else "The available sources do not contain enough information to answer this question.")

    @staticmethod
    def _sum_tokens(first: int | None, second: int | None) -> int | None:
        return first + second if first is not None and second is not None else None


class GroundedRAGAssistant:
    """Standalone orchestration that reuses frozen context and ENV-driven LLM runtime."""

    def __init__(self, *, context_builder: ContextBuilder | None = None, generator: GroundedGenerator | None = None,
                 provider: LLMProvider | None = None, config: GenerationConfig | None = None) -> None:
        self._context_builder = context_builder or ContextBuilder()
        self._generator = generator or GroundedGenerator(provider or get_llm(), config)

    def answer(self, query: str) -> GroundedAnswer:
        return self._generator.generate(self._context_builder.build(query))
