"""Bridge between the public API and the bundled frozen CrediX runtime."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

SERVICE_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(SERVICE_ROOT / ".env", override=False)
os.environ.setdefault("CREDIX_RAG_INDEX_DIR", str(SERVICE_ROOT / "src" / "rag_data" / "current"))
os.environ.setdefault("CREDIX_RERANKER_DEVICE", "auto")


@lru_cache(maxsize=1)
def _settings_and_provider() -> tuple[Any, Any]:
    from src.ai_assistant.llm.config import LLMSettings
    from src.ai_assistant.llm.factory import get_llm

    settings = LLMSettings.from_env(dotenv_path=SERVICE_ROOT / ".env")
    return settings, get_llm(settings)


@lru_cache(maxsize=1)
def _general() -> Any:
    from src.ai_assistant.generation.general import GeneralAssistant
    return GeneralAssistant(provider=_settings_and_provider()[1])


@lru_cache(maxsize=1)
def _grounded() -> Any:
    from src.ai_assistant.generation.generator import GroundedRAGAssistant
    return GroundedRAGAssistant(provider=_settings_and_provider()[1])


@lru_cache(maxsize=1)
def _auto() -> Any:
    from src.ai_assistant.routing.router import RoutedAssistant
    from src.ai_assistant.support_gate.gate import EvidenceSupportGate
    provider = _settings_and_provider()[1]
    return RoutedAssistant(provider=provider, support_gate=EvidenceSupportGate(provider))


def answer(query: str, mode: str) -> Any:
    if mode == "general":
        return _general().answer(query)
    if mode == "grounded":
        return _grounded().answer(query)
    return _auto().answer(query)


def citations_for(answer_result: Any) -> list[dict[str, Any]]:
    try:
        from src.rag.retrieval_adapter import get_chunk
    except Exception:
        get_chunk = None
    citations = []
    for source in getattr(answer_result, "resolved_sources", []):
        title = source.document_title or source.document_id
        excerpts = []
        if get_chunk is not None:
            for chunk_id in source.chunk_ids:
                try:
                    text = str(get_chunk(chunk_id).get("text_original", "")).strip()
                except Exception:
                    continue
                if text:
                    excerpts.append(text)
        citations.append({
            "documentName": title,
            "documentNameEn": title,
            "page": source.page_start or 0,
            "quote": " ".join(excerpts)[:500] or source.handle,
        })
    return citations
