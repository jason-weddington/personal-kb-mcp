"""Quick script to test what qwen3:4b actually returns for enrichment prompts."""

import asyncio
import json
import re

import httpx

OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3:4b"

# --- Prompt variants to test ---

SYSTEM_V1 = """\
You are a knowledge graph builder. Given a knowledge entry, extract entities \
and their relationships to this entry.

Return ONLY a JSON array. Each object has:
- "entity": entity name (lowercase, hyphens for spaces)
- "entity_type": one of: person, tool, concept, technology
- "relationship": how the entry relates (uses, depends_on, implements, solves, \
replaces, configures, learned_from, caused_by, related_to, etc.)

Rules:
- Extract 0-8 relationships. Return [] if none.
- Focus on non-obvious relationships (tags and project already captured).
- Do NOT extract tags or project references (already handled).
- Prefer specific relationships over generic "related_to".
- entity_type MUST be one of: person, tool, concept, technology.\
"""

SYSTEM_V2 = """\
You are a knowledge graph builder. Given a knowledge entry, extract entities \
and their relationships to this entry.

Return ONLY a JSON array. Each object has:
- "entity": entity name (lowercase, hyphens for spaces)
- "entity_type": one of: person, tool, concept, technology
- "relationship": how the entry relates to the entity

Good entities are SPECIFIC enough to connect related entries:
- "thread-safety", "connection-pooling", "dependency-injection" (good concepts)
- "error", "problem", "pattern" (too vague — avoid these)
- "postgresql", "redis", "aiosqlite" (good tools/technologies)

Good relationships describe HOW, not just that a link exists:
- uses, depends_on, implements, solves, replaces, configures, learned_from, caused_by

Rules:
- Extract 2-6 entities. Return [] if the entry is too generic.
- Skip tags and project references (already captured separately).
- entity_type MUST be one of: person, tool, concept, technology.

Example input:
  Title: Chose FastAPI over Flask for the new service
  Type: decision
  Content: We chose FastAPI because we need async support and automatic OpenAPI docs.

Example output:
[
  {"entity": "fastapi", "entity_type": "tool", "relationship": "uses"},
  {"entity": "flask", "entity_type": "tool", "relationship": "replaces"},
  {"entity": "openapi", "entity_type": "technology", "relationship": "depends_on"},
  {"entity": "async-http", "entity_type": "concept", "relationship": "implements"}
]\
"""

# --- Test entries ---

ENTRIES = [
    {
        "name": "aiosqlite threading",
        "prompt": """\
Title: aiosqlite cross-thread safety
Full title: Never use db._conn.execute() directly with aiosqlite
Type: lesson_learned
Tags: python, sqlite, async
Project: personal-kb

Content:
When using aiosqlite, never call db._conn.execute() directly from the async \
context. The underlying sqlite3.Connection lives on a worker thread. Calling \
it directly from the event loop thread causes "ProgrammingError: SQLite objects \
created in a thread can only be used in that same thread". Always use \
await db.execute() which properly dispatches to the worker thread.\
""",
    },
    {
        "name": "FTS5 content-sync",
        "prompt": """\
Title: FTS5 content-sync triggers
Full title: Using FTS5 content-sync triggers for automatic full-text indexing
Type: decision
Tags: sqlite, search
Project: personal-kb

Content:
We use FTS5 with content='knowledge_entries' and auto-sync triggers rather \
than a contentless FTS table. This means the FTS index stays in sync with the \
main table automatically on INSERT/UPDATE/DELETE. The trade-off is slightly \
more disk usage, but we avoid the complexity of manual index management and \
get reliable search results without needing to rebuild the index.\
""",
    },
    {
        "name": "Hybrid search with RRF",
        "prompt": """\
Title: Hybrid search ranking
Full title: Reciprocal Rank Fusion for combining FTS5 and vector search results
Type: pattern_convention
Tags: search, sqlite
Project: personal-kb

Content:
Search results from FTS5 (keyword) and sqlite-vec (semantic) are combined \
using Reciprocal Rank Fusion (RRF). Each result gets a score of 1/(k+rank) \
where k=60. Scores from both sources are summed. This avoids the need to \
normalize different score scales and produces good results without tuning.\
""",
    },
]


async def run_test(
    client: httpx.AsyncClient,
    system: str,
    label: str,
    entry: dict[str, str],
) -> None:
    """Run one prompt variant against one entry."""
    print(f"\n{'=' * 60}")
    print(f"[{label}] Entry: {entry['name']}")
    print(f"{'=' * 60}")

    resp = await client.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": MODEL,
            "prompt": entry["prompt"],
            "system": system,
            "stream": False,
        },
        timeout=300,
    )
    resp.raise_for_status()
    data = resp.json()
    raw = data["response"]

    # Check for thinking tags
    if "<think>" in raw:
        think_match = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
            print(f"  [thinking: {len(thinking)} chars] {thinking[:100]}...")
        raw = re.sub(r"<think>.*?</think>", "", raw, count=0, flags=re.DOTALL).strip()

    # Parse
    fence_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, re.DOTALL)
    if fence_match:
        raw = fence_match.group(1)

    array_match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not array_match:
        print(f"  NO JSON ARRAY FOUND. Raw: {raw[:200]}")
        return

    try:
        parsed = json.loads(array_match.group(0))
    except json.JSONDecodeError as e:
        print(f"  JSON PARSE FAILED: {e}")
        print(f"  Raw: {raw[:200]}")
        return

    print(f"  {len(parsed)} entities:")
    for item in parsed:
        entity = item.get("entity", "?")
        etype = item.get("entity_type", "?")
        rel = item.get("relationship", "?")
        print(f"    {etype}:{entity} — {rel}")


async def main() -> None:
    """Run prompt variants against test entries."""
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(f"{OLLAMA_URL}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            print(f"Ollama not available: {e}")
            return

        print(f"Model: {MODEL}")

        # Test V1 (current prompt) against all entries
        for entry in ENTRIES:
            await run_test(client, SYSTEM_V1, "V1-current", entry)

        # Test V2 (improved prompt) against all entries
        for entry in ENTRIES:
            await run_test(client, SYSTEM_V2, "V2-examples", entry)


if __name__ == "__main__":
    asyncio.run(main())
