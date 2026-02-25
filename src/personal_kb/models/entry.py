"""Knowledge entry models."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class EntryType(StrEnum):
    """Classification of knowledge entries."""

    FACTUAL_REFERENCE = "factual_reference"
    DECISION = "decision"
    PATTERN_CONVENTION = "pattern_convention"
    LESSON_LEARNED = "lesson_learned"


class KnowledgeEntry(BaseModel):
    """A single knowledge entry with metadata and versioning."""

    id: str
    project_ref: str | None = None
    short_title: str
    long_title: str
    knowledge_details: str
    entry_type: EntryType
    source_context: str | None = None
    confidence_level: float = Field(default=0.9, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    hints: dict[str, object] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    superseded_by: str | None = None
    is_active: bool = True
    has_embedding: bool = False
    version: int = 1

    @property
    def embedding_text(self) -> str:
        """Text used for generating embeddings."""
        return f"{self.short_title} {self.long_title} {self.knowledge_details}"
