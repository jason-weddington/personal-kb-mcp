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

_CODE_EXTENSIONS: set[str] = {
    ".py",
    ".js",
    ".ts",
    ".jsx",
    ".tsx",
    ".rb",
    ".go",
    ".rs",
    ".java",
    ".kt",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".swift",
    ".sh",
    ".bash",
    ".zsh",
    ".fish",
    ".sql",
    ".r",
    ".R",
    ".lua",
    ".pl",
    ".pm",
    ".ex",
    ".exs",
    ".scala",
    ".clj",
    ".hs",
    ".erl",
    ".elm",
    ".dart",
    ".v",
    ".zig",
}

_PROSE_EXTENSIONS: set[str] = {
    ".md",
    ".markdown",
    ".txt",
    ".rst",
    ".org",
    ".adoc",
    ".tex",
    ".html",
}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

_SUMMARIZE_SYSTEM = """\
You are a knowledge base assistant. The reader of your summaries is an AI \
coding agent that has effectively memorized public knowledge — documentation, \
tutorials, Stack Overflow, Wikipedia. What it lacks is private, project-specific \
context that doesn't exist online.

Given a file's path and content, write a 2-3 sentence summary focusing on what \
makes this file's knowledge UNIQUE — project-specific decisions, personal \
preferences, undocumented behaviors, or hard-won lessons. Skip restating \
well-known concepts the reader already knows.

Return ONLY the summary text, no JSON, no markdown formatting.\
"""

_SUMMARIZE_CODE_SUPPLEMENT = """

This is a SOURCE CODE file. The reader can access the code directly, so focus \
on high-level purpose and any notable design decisions — not implementation \
details.\
"""

_EXTRACT_SYSTEM = """\
You are a knowledge extraction system. The reader of these entries is an AI \
coding agent powered by a state-of-the-art LLM. It has effectively memorized \
the public internet — documentation, tutorials, Stack Overflow, Wikipedia, \
open-source code. What it LACKS is private context that doesn't exist online: \
project-specific decisions, personal preferences, undocumented bugs, \
hard-won lessons, and internal conventions.

Given a file, extract ONLY knowledge that an internet-trained AI wouldn't \
already know. Return a JSON array. Each object has:
- "short_title": brief identifier (3-8 words)
- "long_title": descriptive title (1 sentence)
- "knowledge_details": the actual knowledge content (detailed, self-contained)
- "entry_type": one of: factual_reference, decision, pattern_convention, lesson_learned
- "tags": list of lowercase tag strings (2-5 tags)

Rules:
- Extract 1-10 entries per file. Fewer high-quality entries are better than many shallow ones.
- Each entry must be SELF-CONTAINED — understandable without the source file.
- Skip anything the reader would already know from public sources.
- entry_type must be one of: factual_reference, decision, pattern_convention, lesson_learned.
- Skip boilerplate, TODOs, and trivial content.
- Return [] if the file has no extractable knowledge beyond public knowledge.

Example output:
[
  {
    "short_title": "aiosqlite vec loading",
    "long_title": "How to load sqlite-vec extension with aiosqlite",
    "knowledge_details": "Use sqlite_vec.load(db._conn) not loadable_path().",
    "entry_type": "lesson_learned",
    "tags": ["sqlite", "aiosqlite", "sqlite-vec"]
  }
]\
"""

_SUMMARIZE_PROSE_SUPPLEMENT = """

This is a NOTES or DOCUMENTATION file. Focus on the key insights, conclusions, \
or decisions — not background context the reader already knows.\
"""

_EXTRACT_CODE_SUPPLEMENT = """

This is a SOURCE CODE file. The AI reader can read the code directly and \
already knows common programming patterns. Do NOT extract what the code does \
or how standard patterns work — the reader can see that.

Instead, focus on COMMENTS and ANNOTATIONS left by the developer — these encode \
context that isn't in the code or on the internet:
- Workaround comments (HACK, WORKAROUND, XXX, NOTE, FIXME with context)
- Decision rationale ("we do X because Y", "chose X over Y because...")
- External system gotchas (API quirks, rate limits, undocumented behaviors)
- Explanations of non-obvious thresholds, magic numbers, or heuristics
- Warnings about things that look wrong but are intentional

If the code has few meaningful comments, extract fewer entries. Return [] if \
there is nothing worth preserving beyond what the code itself communicates.\
"""

_EXTRACT_PROSE_SUPPLEMENT = """

This is a NOTES or DOCUMENTATION file. The AI reader already knows standard \
definitions, common best practices, and textbook explanations. Do NOT extract \
background context that merely sets the stage for the real content.

Instead, extract:
- The author's original arguments, analyses, or conclusions
- Non-obvious cause-and-effect reasoning or myth-busting
- Specific data points, thresholds, or measurements cited as evidence
- Decisions and their rationale ("chose X because Y")
- Hard-won lessons or warnings from experience

If the file is mostly background context with one key insight, extract just \
that insight.\
"""


def _is_code_file(file_path: str) -> bool:
    """Check if a file path refers to a source code file."""
    from pathlib import PurePosixPath

    return PurePosixPath(file_path).suffix.lower() in _CODE_EXTENSIONS


def _is_prose_file(file_path: str) -> bool:
    """Check if a file path refers to a prose/documentation file."""
    from pathlib import PurePosixPath

    return PurePosixPath(file_path).suffix.lower() in _PROSE_EXTENSIONS


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

    system = _SUMMARIZE_SYSTEM
    if _is_code_file(file_path):
        system += _SUMMARIZE_CODE_SUPPLEMENT
    elif _is_prose_file(file_path):
        system += _SUMMARIZE_PROSE_SUPPLEMENT

    return await llm.generate(prompt, system=system)


async def extract_entries(llm: LLMProvider, file_path: str, content: str) -> list[ExtractedEntry]:
    """Extract structured knowledge entries from a file.

    Returns an empty list if the LLM is unavailable or extraction fails.
    """
    if not await llm.is_available():
        return []

    truncated = content[:_MAX_CONTENT_CHARS]
    prompt = f"File: {file_path}\n\n{truncated}"

    system = _EXTRACT_SYSTEM
    if _is_code_file(file_path):
        system += _EXTRACT_CODE_SUPPLEMENT
    elif _is_prose_file(file_path):
        system += _EXTRACT_PROSE_SUPPLEMENT

    raw = await llm.generate(prompt, system=system)
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
