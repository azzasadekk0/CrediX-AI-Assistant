"""Central provider construction without silent provider fallback."""
from __future__ import annotations

from .base import LLMProvider
from .config import LLMSettings
from .errors import LLMConfigurationError
from .gemini_provider import GeminiProvider
from .groq_provider import GroqProvider


def get_llm(settings: LLMSettings | None = None) -> LLMProvider:
    settings = settings or LLMSettings.from_env()
    if settings.provider == "groq":
        return GroqProvider(settings)
    if settings.provider == "gemini":
        return GeminiProvider(settings)
    if settings.provider == "openai":
        raise LLMConfigurationError(f"LLM provider is recognized but not implemented: {settings.provider}")
    raise LLMConfigurationError(f"unsupported LLM provider: {settings.provider}")
