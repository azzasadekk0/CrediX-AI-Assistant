"""Frozen CrediX generation behavior, intentionally separate from LLM runtime settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .errors import GroundedGenerationError


PROMPT_VERSIONS = {"grounded_v1", "grounded_v2"}
FROZEN_GENERATION_CONFIG_PATH = Path(__file__).with_name("generation_config.json")


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    prompt_version: str = "grounded_v1"
    citation_required: bool = True
    answer_language_mode: str = "match_query"
    max_citation_repair_attempts: int = 1
    grounding_policy: str = "context_pack_only"

    def __post_init__(self) -> None:
        if self.prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"unsupported prompt_version: {self.prompt_version}")
        if not self.citation_required or self.answer_language_mode != "match_query" or self.grounding_policy != "context_pack_only":
            raise ValueError("GenerationConfig permits only strict grounded behavior")
        if self.max_citation_repair_attempts != 1:
            raise ValueError("max_citation_repair_attempts must be exactly one")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_generation_config(path: Path = FROZEN_GENERATION_CONFIG_PATH) -> GenerationConfig:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GroundedGenerationError(f"generation configuration not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise GroundedGenerationError(f"generation configuration is invalid JSON: {path}") from exc
    fields = GenerationConfig.__dataclass_fields__
    missing = sorted(set(fields).difference(data))
    if missing:
        raise GroundedGenerationError(f"generation configuration is missing: {', '.join(missing)}")
    return GenerationConfig(**{name: data[name] for name in fields})
