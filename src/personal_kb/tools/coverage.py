"""Lightweight coverage assessment for synthesis retrieval."""

import json
import logging
import re
from dataclasses import dataclass

from personal_kb.llm.provider import LLMProvider
from personal_kb.models.entry import KnowledgeEntry

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)

_COVERAGE_SYSTEM_PROMPT = """\
You are a retrieval quality checker. Given a user question and a list of \
retrieved knowledge entries, assess whether the entries cover the question.

Rules:
- If the entries clearly address the question's core intent, respond with \
has_gaps = false.
- ONLY flag a gap if an obvious, specific concept is missing — not if the \
answer could theoretically be more complete.
- Bias toward "no gaps" — most retrievals are sufficient.
- If you suggest a gap, provide a SHORT, focused search query to fill it.

Return a JSON object:
{"has_gaps": true/false, "suggested_query": "...", "reason": "brief explanation"}

suggested_query should be null when has_gaps is false.\
"""


@dataclass
class CoverageResult:
    """Result of a coverage assessment."""

    has_gaps: bool
    suggested_query: str | None = None
    reason: str = ""


async def assess_coverage(
    llm: LLMProvider,
    question: str,
    entries: list[tuple[KnowledgeEntry, str]],
) -> CoverageResult:
    """Assess whether retrieved entries cover the question.

    Single LLM call. On any failure, returns CoverageResult(has_gaps=False)
    to avoid blocking synthesis.
    """
    try:
        return await _do_assess(llm, question, entries)
    except Exception:
        logger.warning("Coverage assessment failed, assuming no gaps", exc_info=True)
        return CoverageResult(has_gaps=False, reason="assessment failed")


async def _do_assess(
    llm: LLMProvider,
    question: str,
    entries: list[tuple[KnowledgeEntry, str]],
) -> CoverageResult:
    """Internal assessment — exceptions propagate to assess_coverage."""
    # Build compact entry summaries
    summaries = []
    for entry, _ctx in entries:
        detail_preview = entry.knowledge_details[:200]
        tags_str = " ".join(f"#{t}" for t in entry.tags) if entry.tags else ""
        summaries.append(f"[{entry.id}] {entry.short_title} {tags_str}\n  {detail_preview}")
    entries_text = "\n".join(summaries)

    prompt = f"Question: {question}\n\nRetrieved entries ({len(entries)}):\n{entries_text}"

    raw = await llm.generate(prompt, system=_COVERAGE_SYSTEM_PROMPT)
    if raw is None:
        return CoverageResult(has_gaps=False, reason="LLM returned no response")

    return _parse_coverage_response(raw)


def _parse_coverage_response(raw: str) -> CoverageResult:
    """Parse LLM JSON response into CoverageResult."""
    # Strip markdown fences if present
    fence_match = _FENCE_RE.search(raw)
    if fence_match:
        raw = fence_match.group(1)

    obj_match = _JSON_OBJECT_RE.search(raw)
    if not obj_match:
        logger.warning("No JSON object in coverage response")
        return CoverageResult(has_gaps=False, reason="malformed response")

    try:
        data = json.loads(obj_match.group(0))
    except json.JSONDecodeError:
        logger.warning("Malformed JSON in coverage response")
        return CoverageResult(has_gaps=False, reason="malformed JSON")

    if not isinstance(data, dict):
        return CoverageResult(has_gaps=False, reason="response not a dict")

    has_gaps = bool(data.get("has_gaps", False))
    suggested_query = data.get("suggested_query")
    if suggested_query is not None:
        suggested_query = str(suggested_query)
    reason = str(data.get("reason", ""))

    return CoverageResult(has_gaps=has_gaps, suggested_query=suggested_query, reason=reason)
