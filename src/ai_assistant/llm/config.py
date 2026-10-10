"""Environment-only immutable settings for standalone LLM runtime selection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

from .errors import LLMConfigurationError


_SUPPORTED_PROVIDERS = {"groq", "openai", "gemini"}
_CREDENTIAL_ENV_NAMES = {"groq": ("GROQ_API_KEY",), "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY")}


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value or not value.strip():
        raise LLMConfigurationError(f"{name} must be configured for the real LLM runtime")
    return value.strip()


def _number_env(name: str, default: float | None, *, integer: bool = False) -> float | int | None:
    raw = os.getenv(name)
    if raw is None:
        return int(default) if integer and default is not None else default
    try:
        value = int(raw) if integer else float(raw)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be a valid {'integer' if integer else 'number'}") from exc
    if value < 0 or (not integer and value != value):
        raise LLMConfigurationError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class LLMSettings:
    provider: str
    model: str
    temperature: float | None = None
    max_output_tokens: int = 1000
    timeout_seconds: float = 45.0
    max_retries: int = 2
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        if self.provider not in _SUPPORTED_PROVIDERS or not self.model:
            raise LLMConfigurationError("LLM_PROVIDER and LLM_MODEL must be configured")
        if self.temperature is not None and (not isinstance(self.temperature, (int, float)) or isinstance(self.temperature, bool) or self.temperature < 0):
            raise LLMConfigurationError("invalid LLM_TEMPERATURE")
        if self.max_output_tokens < 1 or self.timeout_seconds <= 0 or self.max_retries < 0:
            raise LLMConfigurationError("invalid LLM numeric runtime setting")
        if self.reasoning_effort is not None and (not isinstance(self.reasoning_effort, str) or not self.reasoning_effort.strip()):
            raise LLMConfigurationError("LLM_REASONING_EFFORT must be a non-empty string when configured")

    @classmethod
    def from_env(cls, *, dotenv_path: Path | None = None) -> "LLMSettings":
        path = dotenv_path or Path.cwd() / ".env"
        if path.is_file():
            load_dotenv(path, override=False)
        effort = os.getenv("LLM_REASONING_EFFORT")
        return cls(provider=_required_env("LLM_PROVIDER").lower(), model=_required_env("LLM_MODEL"),
                   temperature=_number_env("LLM_TEMPERATURE", None),
                   max_output_tokens=int(_number_env("LLM_MAX_OUTPUT_TOKENS", 1000, integer=True)),
                   timeout_seconds=float(_number_env("LLM_TIMEOUT_SECONDS", 45.0)),
                   max_retries=int(_number_env("LLM_MAX_RETRIES", 2, integer=True)),
                   reasoning_effort=effort.strip().lower() if effort is not None else None)

    def diagnostics(self) -> dict[str, object]:
        credential_names = _CREDENTIAL_ENV_NAMES.get(self.provider, ())
        return {**asdict(self), "credential_configured": any(bool(os.getenv(name)) for name in credential_names)}
