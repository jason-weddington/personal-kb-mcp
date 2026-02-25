"""LLM-based file summarization and structured entry extraction."""

import json
import logging
import re
from dataclasses import dataclass

from personal_kb.llm.provider import LLMProvider

logger = logging.getLogger(__name__)

_MAX_CONTENT_CHARS = 100_000  # ~25K tokens

_MAX_ENTRIES_PER_FILE = 10

_VALID_ENTRY_TYPES = {"factual_reference", "decision", "pattern_convention", "lesson_learned"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_SUMMARIZE_SYSTEM = """\
You are a knowledge base assistant. Given a file's path and content, write a \
2-3 sentence summary describing what knowledge this file contains and why it \
might be useful to recall later.

Be specific and factual. Focus on WHAT the file teaches, not how it's formatted. \
Return ONLY the summary text, no JSON, no markdown formatting.\
"""

_EXTRACT_SYSTEM = """\
You are a knowledge extraction system. Given a file, extract discrete knowledge \
entries suitable for a personal knowledge base.

Return ONLY a JSON array. Each object has:
- "short_title": brief identifier (3-8 words)
- "long_title": descriptive title (1 sentence)
- "knowledge_details": the actual knowledge content (detailed, self-contained)
- "entry_type": one of: factual_reference, decision, pattern_convention, lesson_learned
- "tags": list of lowercase tag strings (2-5 tags)

Rules:
- Extract 1-10 entries per file. Only extract genuinely useful knowledge.
- Each entry must be SELF-CONTAINED — understandable without the source file.
- Prefer specific, actionable knowledge over vague summaries.
- entry_type must be one of: factual_reference, decision, pattern_convention, lesson_learned.
- Skip boilerplate, TODOs, and trivial content.
- Return [] if the file has no extractable knowledge.

Example output:
[
  {
    "short_title": "aiosqlite vec loading",
    "long_title": "How to load sqlite-vec extension with aiosqlite",
    "knowledge_details": "Use sqlite_vec.load(db._conn) via a closure passed to db._execute().",
    "entry_type": "lesson_learned",
    "tags": ["sqlite", "aiosqlite", "sqlite-vec"]
  }
]\
"""


@dataclass
class ExtractedEntry:
    """A knowledge entry extracted from a file by the LLM."""

    short_title: str
    long_title: str
    knowledge_details: str
    entry_type: str
    tags: list[str]


async def summarize_file(llm: LLMProvider, file_path: str, content: str) -> str | None:
    """Generate a 2-3 sentence summary of a file's knowledge content.

    Returns None if the LLM is unavailable or fails.
    """
    if not await llm.is_available():
        return None

    truncated = content[:_MAX_CONTENT_CHARS]
    prompt = f"File: {file_path}\n\n{truncated}"

    return await llm.generate(prompt, system=_SUMMARIZE_SYSTEM)


async def extract_entries(llm: LLMProvider, file_path: str, content: str) -> list[ExtractedEntry]:
    """Extract structured knowledge entries from a file.

    Returns an empty list if the LLM is unavailable or extraction fails.
    """
    if not await llm.is_available():
        return []

    truncated = content[:_MAX_CONTENT_CHARS]
    prompt = f"File: {file_path}\n\n{truncated}"

    raw = await llm.generate(prompt, system=_EXTRACT_SYSTEM)
    if raw is None:
        return []

    return _parse_entries(raw)


def _parse_entries(raw: str) -> list[ExtractedEntry]:
    """Parse LLM response into validated ExtractedEntry objects."""
    # Strip markdown fences if present
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        raw = fence_match.group(1)

    # Find JSON array
    array_match = _JSON_ARRAY_RE.search(raw)
    if not array_match:
        logger.warning("No JSON array found in extraction response")
        return []

    try:
        data = json.loads(array_match.group(0))
    except json.JSONDecodeError:
        logger.warning("Malformed JSON in extraction response")
        return []

    if not isinstance(data, list):
        return []

    results: list[ExtractedEntry] = []
    for item in data:
        if not isinstance(item, dict):
            continue

        short_title = item.get("short_title")
        long_title = item.get("long_title")
        knowledge_details = item.get("knowledge_details")
        entry_type = item.get("entry_type")
        tags = item.get("tags", [])

        # Validate required fields
        if not (
            isinstance(short_title, str)
            and isinstance(long_title, str)
            and isinstance(knowledge_details, str)
            and isinstance(entry_type, str)
        ):
            continue

        if entry_type not in _VALID_ENTRY_TYPES:
            continue

        if not isinstance(tags, list):
            tags = []
        tags = [str(t).lower() for t in tags if isinstance(t, str)]

        results.append(
            ExtractedEntry(
                short_title=short_title,
                long_title=long_title,
                knowledge_details=knowledge_details,
                entry_type=entry_type,
                tags=tags,
            )
        )

        if len(results) >= _MAX_ENTRIES_PER_FILE:
            break

    return results
