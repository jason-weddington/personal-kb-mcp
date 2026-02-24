"""Search-related models."""

from pydantic import BaseModel, Field

from personal_kb.models.entry import EntryType, KnowledgeEntry


class SearchQuery(BaseModel):
    """Parameters for a knowledge base search."""

    query: str
    project_ref: str | None = None
    entry_type: EntryType | None = None
    tags: list[str] | None = None
    limit: int = Field(default=10, ge=1, le=50)
    include_stale: bool = False


class SearchResult(BaseModel):
    """A single search result with scoring and staleness info."""

    entry: KnowledgeEntry
    score: float
    effective_confidence: float
    staleness_warning: str | None = None
    match_source: str  # "hybrid", "fts", "vector"
