"""KB-aware dedup agent for ingestion — checks chunks against existing KB."""

import logging
from dataclasses import dataclass, field

from kb_core.db.backend import Database
from kb_core.ingest.chunker import Chunk
from kb_core.llm.json_parser import parse_json_object
from kb_core.llm.provider import LLMProvider
from kb_core.search.embedder_protocol import Embedder

logger = logging.getLogger(__name__)
# Log marker for the dedup similarity gate (mirrors the ``op=`` log convention in ``kb_core.search.hybrid``).  # noqa: E501
DEDUP_GATE_MARKER = "dedup-gate"

# Default mirrors ``personal_kb.config.get_ingest_dedup_similarity_floor`` /
# ``IngestConfig.dedup_similarity_floor``. Used only when the construction
# site does not pass an explicit floor.
_DEFAULT_DEDUP_SIMILARITY_FLOOR = 0.85

_DEDUP_SYSTEM_PROMPT = """\
You are a dedup checker for a knowledge base ingestion pipeline.

Given a chunk of text and a list of existing KB entries that might overlap, \
decide whether the chunk should be extracted, skipped, or partially extracted.

Rules:
- "skip": The chunk's knowledge is ALREADY fully covered by the existing entries. \
  Only use this for near-total overlap — if even 20% of the chunk is new, extract it.
- "partial": Some content overlaps with existing entries, but there is also new \
  knowledge worth extracting. List the titles of existing entries that overlap.
- "extract": The chunk contains mostly new knowledge not in the existing entries.

Be AGGRESSIVE about extracting — when in doubt, extract. Missing new knowledge \
is worse than a small amount of duplication.

Return a JSON object:
{"verdict": "extract"|"skip"|"partial", "reason": "brief explanation", \
"existing_titles": ["title1", "title2"]}

existing_titles should only be populated for "partial" verdicts — list the \
titles of overlapping entries so the extractor can avoid re-extracting them.\
"""


@dataclass
class DedupResult:
    """Result of a dedup check on a chunk."""

    action: str  # "extract", "skip", or "partial"
    reason: str = ""
    existing_titles: list[str] = field(default_factory=list)


class DedupAgent:
    """Checks chunks against existing KB to avoid duplicate extraction."""

    def __init__(
        self,
        db: Database,
        embedder: Embedder,
        llm: LLMProvider,
        similarity_floor: float | None = None,
    ) -> None:
        """Initialize with DB, embedder, and LLM for search + dedup.

        ``similarity_floor`` is the minimum cosine similarity (1 - cosine
        distance, 0..1) between the chunk and an existing entry for the
        chunk to be a dedup candidate. The construction site passes
        ``IngestConfig.dedup_similarity_floor``; ``None`` falls back to the
        kb_core default that mirrors the env-driven default.
        """
        self._db = db
        self._embedder = embedder
        self._llm = llm
        self._similarity_floor = (
            similarity_floor if similarity_floor is not None else _DEFAULT_DEDUP_SIMILARITY_FLOOR
        )
        if self._similarity_floor > 1.0:
            logger.warning(
                "%s tripwire=floor_unreachable floor=%.4f",
                DEDUP_GATE_MARKER,
                self._similarity_floor,
            )

    async def check_chunk(
        self,
        chunk: Chunk,
        previously_extracted: list[str],
    ) -> DedupResult:
        """Check if a chunk duplicates existing KB entries.

        1. Build search query from chunk heading + first ~500 chars
        2. Search KB via hybrid_search
        3. If no results, no result carries a vector similarity, or the best similarity is below
        the floor → extract
        4. If at or above the floor → ask LLM to confirm
        5. On any failure → extract (graceful degradation)
        """
        try:
            return await self._do_check(chunk, previously_extracted)
        except Exception:
            logger.warning("Dedup check failed, defaulting to extract", exc_info=True)
            return DedupResult(action="extract", reason="dedup check failed")

    async def _do_check(
        self,
        chunk: Chunk,
        previously_extracted: list[str],
    ) -> DedupResult:
        """Internal check — exceptions propagate to check_chunk for handling."""
        from kb_core.models.search import SearchQuery
        from kb_core.search.hybrid import hybrid_search

        # Build search query from heading + first ~500 chars
        query_parts: list[str] = []
        if chunk.heading:
            query_parts.append(chunk.heading)
        # Take first ~500 chars of content, stripping the heading if present
        snippet = chunk.text[:500].strip()
        if snippet:
            query_parts.append(snippet)
        query_text = " ".join(query_parts)
        if not query_text:
            return DedupResult(action="extract", reason="empty chunk")

        # Search KB
        search_query = SearchQuery(query=query_text, limit=5, include_superseded=True)
        results, _ = await hybrid_search(self._db, self._embedder, search_query)

        if not results:
            return DedupResult(action="extract", reason="no existing entries found")

        similarities = [r.vector_similarity for r in results if r.vector_similarity is not None]
        if not similarities:
            logger.warning(
                "%s op=gate tripwire=vector_leg_unavailable query=%r results=%d floor=%.4f",
                DEDUP_GATE_MARKER,
                query_text[:80],
                len(results),
                self._similarity_floor,
            )
            return DedupResult(action="extract", reason="vector leg unavailable")

        top_similarity = max(similarities)
        logger.info(
            "%s op=gate query=%r results=%d with_similarity=%d top_similarity=%.4f floor=%.4f "
            "above_floor=%s",
            DEDUP_GATE_MARKER,
            query_text[:80],
            len(results),
            len(similarities),
            top_similarity,
            self._similarity_floor,
            top_similarity >= self._similarity_floor,
        )
        if top_similarity < self._similarity_floor:
            return DedupResult(
                action="extract",
                reason=(
                    f"top similarity {top_similarity:.4f} below floor {self._similarity_floor}"
                ),
            )

        # At or above the floor — ask the LLM to confirm
        if not await self._llm.is_available():
            return DedupResult(action="extract", reason="LLM unavailable for dedup")

        # Format existing entries for LLM
        entry_summaries = []
        for r in results:
            entry_summaries.append(f"- [{r.entry.id}] {r.entry.short_title}: {r.entry.long_title}")
        entries_text = "\n".join(entry_summaries)

        prompt = (
            f"## Chunk to check\n"
            f"Heading: {chunk.heading or '(none)'}\n\n"
            f"{chunk.text[:2000]}\n\n"
            f"## Existing KB entries with potential overlap\n"
            f"{entries_text}"
        )

        raw = await self._llm.generate(prompt, purpose="dedup_agent", system=_DEDUP_SYSTEM_PROMPT)
        if raw is None:
            return DedupResult(action="extract", reason="LLM returned no response")

        return _parse_dedup_response(raw)


def _parse_dedup_response(raw: str) -> DedupResult:
    """Parse LLM JSON response into DedupResult."""
    data = parse_json_object(raw)
    if data is None:
        logger.warning("No JSON object in dedup response")
        return DedupResult(action="extract", reason="malformed response")

    verdict = data.get("verdict", "extract")
    if verdict not in ("extract", "skip", "partial"):
        verdict = "extract"

    reason = str(data.get("reason", ""))

    existing_titles: list[str] = []
    raw_titles = data.get("existing_titles", [])
    if isinstance(raw_titles, list):
        existing_titles = [str(t) for t in raw_titles if isinstance(t, str)]

    return DedupResult(action=verdict, reason=reason, existing_titles=existing_titles)
