"""Real Groq adapter using a single official-SDK client per provider instance."""
from __future__ import annotations

import os
import time
from typing import Any

import groq

from .base import LLMProvider
from .config import LLMSettings
from .errors import LLMAuthenticationError, LLMConfigurationError, LLMProviderError, LLMRateLimitError, LLMResponseError, LLMTimeoutError
from .models import LLMRequest, LLMResponse


_GROQ_REASONING_EFFORTS = {"none", "default", "low", "medium", "high"}


class GroqProvider(LLMProvider):
    def __init__(self, settings: LLMSettings, *, client: Any | None = None, api_key: str | None = None) -> None:
        if settings.provider != "groq":
            raise LLMConfigurationError("GroqProvider requires LLM_PROVIDER=groq")
        self.settings = settings
        key = api_key if api_key is not None else os.getenv("GROQ_API_KEY")
        if not key:
            raise LLMAuthenticationError("GROQ_API_KEY must be configured for the Groq provider")
        self._client = client or groq.Groq(api_key=key, timeout=settings.timeout_seconds, max_retries=0)

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload = self._payload(request)
        started = time.perf_counter()
        for attempt in range(self.settings.max_retries + 1):
            try:
                completion = self._client.chat.completions.create(**payload)
                return self._response(completion, time.perf_counter() - started)
            except Exception as exc:  # map only known SDK classes below; never leak request headers/key.
                mapped, transient = self._map_error(exc)
                if transient and attempt < self.settings.max_retries:
                    continue
                raise mapped from exc
        raise LLMProviderError("Groq request failed")  # unreachable defensive guard

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        effort = request.reasoning_effort if request.reasoning_effort is not None else self.settings.reasoning_effort
        if effort is not None and effort not in _GROQ_REASONING_EFFORTS:
            raise LLMConfigurationError("configured reasoning effort is not supported by the Groq chat interface")
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [{"role": "system", "content": request.system_prompt}, {"role": "user", "content": request.user_prompt}],
            "max_completion_tokens": self.settings.max_output_tokens if request.max_output_tokens is None else request.max_output_tokens,
        }
        temperature = request.temperature if request.temperature is not None else self.settings.temperature
        if temperature is not None:
            payload["temperature"] = temperature
        if effort is not None:
            payload["reasoning_effort"] = effort
        if request.response_schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {"name": request.response_schema_name,
                                          "strict": True, "schema": request.response_schema}}
        return payload

    @staticmethod
    def _response(completion: Any, latency_seconds: float) -> LLMResponse:
        try:
            choice = completion.choices[0]
            text = choice.message.content
        except (AttributeError, IndexError, TypeError) as exc:
            raise LLMResponseError("Groq response lacks completion content") from exc
        if not isinstance(text, str) or not text.strip():
            raise LLMResponseError("Groq response contains empty completion text")
        usage = getattr(completion, "usage", None)
        return LLMResponse(text=text, provider="groq", model=getattr(completion, "model", None) or "",
                           latency_seconds=latency_seconds, input_tokens=getattr(usage, "prompt_tokens", None),
                           output_tokens=getattr(usage, "completion_tokens", None), total_tokens=getattr(usage, "total_tokens", None),
                           finish_reason=getattr(choice, "finish_reason", None))

    @staticmethod
    def _map_error(exc: Exception) -> tuple[LLMProviderError, bool]:
        if isinstance(exc, groq.AuthenticationError):
            return LLMAuthenticationError("Groq authentication failed"), False
        if isinstance(exc, groq.RateLimitError):
            return LLMRateLimitError("Groq rate limit reached"), True
        if isinstance(exc, groq.APITimeoutError):
            return LLMTimeoutError("Groq request timed out"), True
        if isinstance(exc, groq.BadRequestError):
            return LLMConfigurationError("Groq rejected the configured request"), False
        if isinstance(exc, groq.APIConnectionError):
            return LLMProviderError("Groq network connection failed"), True
        if isinstance(exc, groq.APIStatusError):
            return LLMProviderError("Groq service returned an error"), getattr(exc, "status_code", 0) >= 500
        if isinstance(exc, LLMProviderError):
            return exc, False
        return LLMProviderError("Groq provider request failed"), False
