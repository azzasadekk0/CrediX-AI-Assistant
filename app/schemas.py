"""Stable public API schemas for the Railway AI Assistant service."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


AnswerMode = Literal["grounded", "general", "hybrid", "insufficient_evidence"]
Provenance = Literal["retrieved", "ai_generated", "mixed", "unavailable"]


class SessionCreate(BaseModel):
    title: str
    title_en: str | None = Field(None, alias="titleEn")
    model_config = ConfigDict(populate_by_name=True)


class SessionResponse(BaseModel):
    id: str
    title: str
    title_en: str = Field(alias="titleEn")
    active: bool
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ChatMessageCreate(BaseModel):
    text: str
    text_en: str | None = Field(None, alias="textEn")
    mode: Literal["auto", "grounded", "general"] = "auto"
    model_config = ConfigDict(populate_by_name=True)


class Citation(BaseModel):
    document_name: str = Field(alias="documentName")
    document_name_en: str = Field(alias="documentNameEn")
    page: int
    quote: str
    model_config = ConfigDict(populate_by_name=True)


class Segment(BaseModel):
    text: str
    source_type: Literal["retrieved", "ai_generated"] = Field(alias="sourceType")
    citation_handles: list[str] = Field(default_factory=list, alias="citationHandles")
    support_status: Literal["supported", "inference", "unsupported"] = Field(alias="supportStatus")
    model_config = ConfigDict(populate_by_name=True)


class MessageResponse(BaseModel):
    id: str
    sender: Literal["user", "assistant"]
    text: str
    text_en: str | None = Field(None, alias="textEn")
    timestamp: str
    citations: list[Citation] = Field(default_factory=list)
    answer_mode: AnswerMode | None = Field(None, alias="answerMode")
    provenance: Provenance | None = None
    disclaimer: str | None = None
    segments: list[Segment] = Field(default_factory=list)
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    @model_validator(mode="after")
    def validate_provenance(self) -> "MessageResponse":
        if self.answer_mode == "general" and (self.provenance != "ai_generated" or self.citations):
            raise ValueError("general responses must be AI-generated and citation-free")
        if self.answer_mode == "insufficient_evidence" and self.citations:
            raise ValueError("insufficient-evidence responses cannot contain citations")
        return self
