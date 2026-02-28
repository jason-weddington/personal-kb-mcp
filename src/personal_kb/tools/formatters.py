"""Compact output formatters for MCP tool responses."""

from datetime import UTC, datetime

from personal_kb.confidence.decay import compute_effective_confidence, staleness_warning
from personal_kb.models.entry import KnowledgeEntry


def format_entry_header(entry: KnowledgeEntry, effective_confidence: float) -> str:
    """Format: [kb-00082] lesson_learned | Title (90%)."""
    etype = entry.entry_type.value
    return f"[{entry.id}] {etype} | {entry.short_title} ({effective_confidence:.0%})"


def format_entry_meta(entry: KnowledgeEntry, stale_warning: str | None = None) -> str:
    """Format: #tag1 #tag2 | project  [STALE]."""
    parts: list[str] = []
    if entry.tags:
        parts.append(" ".join(f"#{t}" for t in entry.tags))
    if entry.project_ref:
        parts.append(entry.project_ref)
    line = " | ".join(parts)
    if stale_warning:
        line = f"{line}  [STALE]" if line else "[STALE]"
    return line


def format_entry_compact(
    entry: KnowledgeEntry,
    effective_confidence: float,
    stale_warning: str | None = None,
) -> str:
    """Header + long_title + meta, no details. For kb_search and kb_store."""
    header = format_entry_header(entry, effective_confidence)
    lines = [header]
    if entry.long_title and entry.long_title != entry.short_title:
        lines.append(f"  {entry.long_title}")
    meta = format_entry_meta(entry, stale_warning)
    if meta:
        lines.append(f"  {meta}")
    return "\n".join(lines)


def format_entry_full(
    entry: KnowledgeEntry,
    context: str | None = None,
    effective_confidence: float | None = None,
    stale_warning: str | None = None,
) -> str:
    """Header + meta + optional context + knowledge_details. For kb_get and kb_ask."""
    now = datetime.now(UTC)
    if effective_confidence is None:
        anchor = entry.updated_at or entry.created_at or now
        effective_confidence = compute_effective_confidence(
            entry.confidence_level,
            entry.entry_type,
            anchor,
            now,
            last_accessed=entry.last_accessed,
        )
    if stale_warning is None:
        stale_warning = staleness_warning(effective_confidence, entry.entry_type)

    header = format_entry_header(entry, effective_confidence)
    meta = format_entry_meta(entry, stale_warning)

    lines = [header]
    if meta:
        lines.append(f"  {meta}")
    if context:
        lines.append(f"  \u21b3 {context}")
    lines.append(f"  {entry.knowledge_details}")
    return "\n".join(lines)


def format_graph_hint(entry: KnowledgeEntry, via_node: str) -> str:
    """One-liner hint: See also: [kb-00042] Title (via concept:async-io)."""
    return f"See also: [{entry.id}] {entry.short_title} (via {via_node})"


def format_result_list(
    formatted_entries: list[str],
    header: str | None = None,
    note: str | None = None,
    hints: list[str] | None = None,
) -> str:
    """Count + note + entries joined by blank lines, optional graph hints."""
    if not formatted_entries:
        return "No results found."

    lines: list[str] = []
    if header:
        lines.append(header)
    lines.append(f"{len(formatted_entries)} result(s)")
    if note:
        lines.append(f"Note: {note}")
    lines.append("")
    lines.append("\n\n".join(formatted_entries))
    if hints:
        lines.append("")
        lines.append("Related entries via graph:")
        lines.extend(f"  {h}" for h in hints)
    return "\n".join(lines)
