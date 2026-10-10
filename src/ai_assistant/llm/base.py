"""The only interface future grounded generation may depend on."""
from __future__ import annotations

from abc import ABC, abstractmethod

from .models import LLMRequest, LLMResponse


class LLMProvider(ABC):
    @abstractmethod
    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate one provider-neutral response."""
