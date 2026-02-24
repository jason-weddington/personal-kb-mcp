"""Entry version models."""

from datetime import datetime

from pydantic import BaseModel, Field


class EntryVersion(BaseModel):
    """A versioned snapshot of a knowledge entry."""

    entry_id: str
    version_number: int
    knowledge_details: str
    change_reason: str | None = None
    confidence_level: float = Field(ge=0.0, le=1.0)
    created_at: datetime | None = None
