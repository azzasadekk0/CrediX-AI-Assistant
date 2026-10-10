"""Google Gemini adapter behind the provider-neutral LLM interface."""
from __future__ import annotations

import os
import time
from typing import Any

from google import genai
from google.genai import errors, types

from .base import LLMProvider
from .config import LLMSettings
from .errors import LLMAuthenticationError, LLMConfigurationError, LLMProviderError, LLMRateLimitError, LLMResponseError, LLMTimeoutError
from .models import LLMRequest, LLMResponse


class GeminiProvider(LLMProvider):
    """Official Google Gen AI SDK implementation with native JSON-schema output."""

    def __init__(self, settings: LLMSettings, *, client: Any | None = None, api_key: str | None = None) -> None:
        if settings.provider != "gemini":
            raise LLMConfigurationError("GeminiProvider requires LLM_PROVIDER=gemini")
        self.settings = settings
        key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise LLMAuthenticationError("GEMINI_API_KEY or GOOGLE_API_KEY must be configured for the Gemini provider")
        self._client = client or genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(settings.timeout_seconds * 1000)))

    def generate(self, request: LLMRequest) -> LLMResponse:
        config = self._config(request)
        started = time.perf_counter()
        for attempt in range(self.settings.max_retries + 1):
            try:
                response = self._client.models.generate_content(model=self.settings.model, contents=request.user_prompt, config=config)
                return self._response(response, time.perf_counter() - started)
            except Exception as exc:  # Map SDK errors without preserving request or credential data.
                mapped, transient = self._map_error(exc)
                if transient and attempt < self.settings.max_retries:
                    continue
                raise mapped from exc
        raise LLMProviderError("Gemini request failed")  # defensive guard

    def _config(self, request: LLMRequest) -> types.GenerateContentConfig:
        values: dict[str, Any] = {
            "system_instruction": request.system_prompt,
            "max_output_tokens": self.settings.max_output_tokens if request.max_output_tokens is None else request.max_output_tokens,
        }
        temperature = request.temperature if request.temperature is not None else self.settings.temperature
        if temperature is not None:
            values["temperature"] = temperature
        effort = request.reasoning_effort if request.reasoning_effort is not None else self.settings.reasoning_effort
        if effort is not None:
            values["thinking_config"] = {"thinking_budget": 0} if effort == "none" else {"thinking_level": effort}
        if request.response_schema is not None:
            values["response_mime_type"] = "application/json"
            values["response_json_schema"] = request.response_schema
        return types.GenerateContentConfig(**values)

    @staticmethod
    def _response(response: Any, latency_seconds: float) -> LLMResponse:
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("Gemini response contains no text content")
        usage = getattr(response, "usage_metadata", None)
        candidates = getattr(response, "candidates", None) or []
        finish_reason = getattr(candidates[0], "finish_reason", None) if candidates else None
        return LLMResponse(
            text=text,
            provider="gemini",
            model=getattr(response, "model_version", None) or "",
            latency_seconds=latency_seconds,
            input_tokens=getattr(usage, "prompt_token_count", None),
            output_tokens=getattr(usage, "candidates_token_count", None),
            total_tokens=getattr(usage, "total_token_count", None),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
        )

    @staticmethod
    def _map_error(exc: Exception) -> tuple[LLMProviderError, bool]:
        if isinstance(exc, errors.APIError):
            status = getattr(exc, "code", None)
            if status in {401, 403}:
                return LLMAuthenticationError("Gemini authentication failed"), False
            if status == 429:
                return LLMRateLimitError("Gemini rate limit reached"), True
            if status in {408, 504}:
                return LLMTimeoutError("Gemini request timed out"), True
            if isinstance(status, int) and status >= 500:
                return LLMProviderError("Gemini service returned an error"), True
            return LLMProviderError("Gemini service returned an error"), False
        if isinstance(exc, (TimeoutError,)):
            return LLMTimeoutError("Gemini request timed out"), True
        if isinstance(exc, LLMProviderError):
            return exc, False
        return LLMProviderError("Gemini provider request failed"), False
