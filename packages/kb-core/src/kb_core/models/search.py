"""Search-related models."""

from pydantic import BaseModel, Field

from kb_core.models.entry import EntryType, KnowledgeEntry


class SearchQuery(BaseModel):
    """Parameters for a knowledge base search."""

    query: str = ""
    project_ref: str | None = None
    entry_type: EntryType | None = None
    tags: list[str] | None = None
    contributor: str | None = None
    team: str | None = None
    limit: int = Field(default=10, ge=1, le=50)
    include_stale: bool = False
    include_expired: bool = False
    min_score_ratio: float = Field(default=0.5, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    """A single search result with scoring and staleness info."""

    entry: KnowledgeEntry
    score: float
    effective_confidence: float
    staleness_warning: str | None = None
    match_source: str  # "hybrid", "fts", "vector"
    # Raw per-leg relevance signals. Populated by hybrid_search; None/False
    # when the entry did not appear in that leg or the leg was unavailable.
    vector_similarity: float | None = None  # cosine similarity (0..1, higher = more similar)
    fts_matched: bool = False  # True when the entry appeared in the FTS leg
    fts_rank: int | None = None  # 1-based rank in the FTS leg (None if not in FTS)
