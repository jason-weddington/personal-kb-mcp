"""Knowledge entry models."""

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


def normalize_tags(tags: Iterable[str]) -> list[str]:
    """Collapse whitespace runs in each tag to "-", drop empties, de-duplicate in order."""
    seen: dict[str, None] = {}
    for tag in tags:
        norm = "-".join(tag.split())
        if norm:
            seen.setdefault(norm, None)
    return list(seen)


class EntryType(StrEnum):
    """Classification of knowledge entries."""

    FACTUAL_REFERENCE = "factual_reference"
    DECISION = "decision"
    PATTERN_CONVENTION = "pattern_convention"
    LESSON_LEARNED = "lesson_learned"
    MENTAL_MAP = "mental_map"


class KnowledgeEntry(BaseModel):
    """A single knowledge entry with metadata and versioning."""

    id: str
    project_ref: str | None = None
    short_title: str
    long_title: str
    knowledge_details: str
    entry_type: EntryType
    source_context: str | None = None
    contributor: str | None = None
    team: str | None = None
    updated_by: str | None = None
    sensitivity: Literal["internal", "restricted", "public"] | None = None
    confidence_level: float = Field(default=0.9, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    hints: dict[str, object] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_accessed: datetime | None = None
    expires_at: datetime | None = None

    @field_validator("tags", mode="after")
    @classmethod
    def _normalize_tags(cls, value: list[str]) -> list[str]:
        return normalize_tags(value)

    superseded_by: str | None = None
    is_active: bool = True
    has_embedding: bool = False
    version: int = 1

    @staticmethod
    def compose_embedding_text(short_title: str, long_title: str, knowledge_details: str) -> str:
        """The single definition of the text that is embedded for an entry."""
        return f"{short_title} {long_title} {knowledge_details}"

    @property
    def embedding_text(self) -> str:
        """Text used for generating embeddings."""
        return KnowledgeEntry.compose_embedding_text(
            self.short_title, self.long_title, self.knowledge_details
        )
