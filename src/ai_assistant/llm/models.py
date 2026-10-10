"""Provider-neutral request and response value objects."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LLMRequest:
    system_prompt: str
    user_prompt: str
    temperature: float | None = None
    max_output_tokens: int | None = None
    reasoning_effort: str | None = None
    response_schema: dict[str, Any] | None = None
    response_schema_name: str = "response"

    def __post_init__(self) -> None:
        if not isinstance(self.system_prompt, str) or not isinstance(self.user_prompt, str):
            raise ValueError("system_prompt and user_prompt must be strings")
        if self.temperature is not None and (not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or self.temperature < 0):
            raise ValueError("temperature must be a non-negative number")
        if self.max_output_tokens is not None and (not isinstance(self.max_output_tokens, int) or isinstance(self.max_output_tokens, bool) or self.max_output_tokens < 1):
            raise ValueError("max_output_tokens must be a positive integer")
        if self.reasoning_effort is not None and not isinstance(self.reasoning_effort, str):
            raise ValueError("reasoning_effort must be a string when supplied")
        if self.response_schema is not None and not isinstance(self.response_schema, dict):
            raise ValueError("response_schema must be an object when supplied")
        if not self.response_schema_name or not isinstance(self.response_schema_name, str):
            raise ValueError("response_schema_name must be non-empty")


@dataclass(frozen=True, slots=True)
class LLMResponse:
    text: str
    provider: str
    model: str
    latency_seconds: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    finish_reason: str | None = None
