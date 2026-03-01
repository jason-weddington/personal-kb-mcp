"""Markdown-aware content chunking for large file ingestion."""

import re
from dataclasses import dataclass

from personal_kb.config import get_ingest_chunk_overlap, get_ingest_chunk_size

_HEADING_RE = re.compile(r"^#{1,2}\s+", re.MULTILINE)


@dataclass
class Chunk:
    """A chunk of content split from a larger document."""

    text: str
    index: int
    start_char: int
    heading: str | None


def chunk_content(
    content: str,
    *,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[Chunk]:
    """Split content into chunks at markdown heading boundaries.

    1. If content <= chunk_size → single Chunk (no splitting)
    2. Find all H1/H2 heading positions
    3. Split at heading boundaries → list of sections
    4. Greedily merge adjacent sections up to chunk_size
    5. If a section exceeds chunk_size → sub-split at paragraph breaks
    6. If a paragraph still exceeds → sub-split at any newline
    7. Apply overlap: prepend last N chars of previous chunk (snapped to newline)
    """
    if chunk_size is None:
        chunk_size = get_ingest_chunk_size()
    if overlap is None:
        overlap = get_ingest_chunk_overlap()

    if len(content) <= chunk_size:
        heading = _extract_first_heading(content)
        return [Chunk(text=content, index=0, start_char=0, heading=heading)]

    # Find heading positions
    sections = _split_at_headings(content)

    # Merge small adjacent sections up to chunk_size
    merged = _merge_sections(sections, chunk_size)

    # Sub-split any oversized sections
    final_sections: list[tuple[int, str]] = []
    for start, text in merged:
        if len(text) <= chunk_size:
            final_sections.append((start, text))
        else:
            sub = _subsplit(text, chunk_size, start)
            final_sections.extend(sub)

    # Apply overlap and build Chunks
    chunks: list[Chunk] = []
    for i, (start, text) in enumerate(final_sections):
        if i > 0 and overlap > 0:
            # Get overlap from previous chunk's text
            prev_text = final_sections[i - 1][1]
            overlap_text = _snap_overlap(prev_text, overlap)
            if overlap_text:
                text = overlap_text + text
                start = max(0, start - len(overlap_text))

        heading = _extract_first_heading(text)
        chunks.append(Chunk(text=text, index=i, start_char=start, heading=heading))

    return chunks


def _extract_first_heading(text: str) -> str | None:
    """Extract the first H1/H2 heading text from content."""
    for line in text.split("\n"):
        match = _HEADING_RE.match(line)
        if match:
            return line[match.end() :].strip()
    return None


def _split_at_headings(content: str) -> list[tuple[int, str]]:
    """Split content at H1/H2 heading boundaries.

    Returns list of (start_char, section_text) tuples.
    """
    positions = [m.start() for m in _HEADING_RE.finditer(content)]

    if not positions:
        # No headings — return as single section
        return [(0, content)]

    sections: list[tuple[int, str]] = []

    # Content before first heading (if any)
    if positions[0] > 0:
        sections.append((0, content[: positions[0]]))

    # Each heading starts a new section
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(content)
        sections.append((pos, content[pos:end]))

    return sections


def _merge_sections(sections: list[tuple[int, str]], chunk_size: int) -> list[tuple[int, str]]:
    """Greedily merge adjacent sections up to chunk_size."""
    if not sections:
        return []

    merged: list[tuple[int, str]] = []
    current_start, current_text = sections[0]

    for start, text in sections[1:]:
        if len(current_text) + len(text) <= chunk_size:
            current_text += text
        else:
            merged.append((current_start, current_text))
            current_start = start
            current_text = text

    merged.append((current_start, current_text))
    return merged


def _subsplit(text: str, chunk_size: int, base_offset: int) -> list[tuple[int, str]]:
    """Sub-split an oversized section at paragraph breaks, then newlines."""
    # Try paragraph breaks first
    parts = _split_at_boundaries(text, "\n\n", chunk_size)
    if len(parts) > 1:
        result: list[tuple[int, str]] = []
        offset = 0
        for part in parts:
            result.append((base_offset + offset, part))
            offset += len(part)
        return result

    # Fall back to any newline
    parts = _split_at_boundaries(text, "\n", chunk_size)
    result = []
    offset = 0
    for part in parts:
        result.append((base_offset + offset, part))
        offset += len(part)
    return result


def _split_at_boundaries(text: str, separator: str, chunk_size: int) -> list[str]:
    """Split text at separator boundaries, merging into chunks up to chunk_size."""
    pieces = text.split(separator)
    if len(pieces) <= 1:
        return [text]

    chunks: list[str] = []
    current = pieces[0]

    for piece in pieces[1:]:
        candidate = current + separator + piece
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current + separator)
            current = piece

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def _snap_overlap(prev_text: str, overlap: int) -> str:
    """Get the last `overlap` chars of prev_text, snapped to a newline boundary."""
    if len(prev_text) <= overlap:
        return prev_text

    tail = prev_text[-overlap:]
    # Snap forward to the first newline
    nl_pos = tail.find("\n")
    if nl_pos >= 0 and nl_pos < len(tail) - 1:
        return tail[nl_pos + 1 :]
    return tail
