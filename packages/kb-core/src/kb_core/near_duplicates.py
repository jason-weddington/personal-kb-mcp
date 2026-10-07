"""Near-duplicate lookup: same-project active entries whose vector is close.

Read-only: opens no transaction and writes nothing.
"""

from dataclasses import dataclass
from typing import Literal

from kb_core.db.backend import Database

NEAR_DUPLICATE_NEIGHBOURS = 20


@dataclass(frozen=True)
class NearDuplicateCandidate:
    """One existing entry at or above the similarity floor."""

    id: str
    short_title: str
    entry_type: str
    similarity: float
    updated_at: str | None


@dataclass(frozen=True)
class NearDuplicateCheck:
    """Result of a near-duplicate check (see ``status`` for fail-open cases)."""

    status: Literal["checked", "embedder_unavailable", "search_failed"]
    candidates: tuple[NearDuplicateCandidate, ...]
    top_similarity: float | None
    raw_hits: int = 0
    eligible_count: int = 0
    embed_ms: int = 0
    search_ms: int = 0


async def find_near_duplicates(
    db: Database,
    embedding: list[float],
    *,
    project_ref: str,
    floor: float,
    limit: int = 5,
) -> NearDuplicateCheck:
    """Return eligible same-project neighbours with similarity >= *floor*.

    Eligible means non-mental_map and ``superseded_by IS NULL``; the vector
    search itself already restricts to active rows of the project.
    """
    hits = await db.vector_search(
        embedding, limit=NEAR_DUPLICATE_NEIGHBOURS, project_ref=project_ref
    )
    if not hits:
        return NearDuplicateCheck(status="checked", candidates=(), top_similarity=None)
    distances = dict(hits)
    placeholders = ", ".join("?" for _ in distances)
    cursor = await db.execute(
        "SELECT id, short_title, entry_type, updated_at, superseded_by"  # noqa: S608
        f" FROM knowledge_entries WHERE id IN ({placeholders})",
        tuple(distances),
    )
    rows = await cursor.fetchall()
    eligible: list[NearDuplicateCandidate] = []
    for row in rows:
        if row["entry_type"] == "mental_map" or row["superseded_by"] is not None:
            continue
        updated = row["updated_at"]
        eligible.append(
            NearDuplicateCandidate(
                id=str(row["id"]),
                short_title=str(row["short_title"]),
                entry_type=str(row["entry_type"]),
                similarity=round(1.0 - distances[str(row["id"])], 4),
                updated_at=str(updated) if updated is not None else None,
            )
        )
    top = max((c.similarity for c in eligible), default=None)
    candidates = sorted(
        (c for c in eligible if c.similarity >= floor), key=lambda c: (-c.similarity, c.id)
    )[:limit]
    return NearDuplicateCheck(
        status="checked",
        candidates=tuple(candidates),
        top_similarity=top,
        raw_hits=len(hits),
        eligible_count=len(eligible),
    )
