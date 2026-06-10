"""Listener gate: POST /api/kb/listener.

Surfaces a single KB mental-map pointer when an AI agent's working context
matches an orientation map with high confidence (unanimous-3 Sonnet votes,
rules A + B candidate filtering).

Kill switch: returns ``{"pointer": null}`` immediately when the env var
``KB_LISTENER_ENABLED`` is not set to ``'TRUE'`` (default ``'FALSE'`` — opt-in
pilot only; flip on the dev server when piloting).
"""

import asyncio
import os
from typing import Annotated

from fastapi import APIRouter, Depends, Request
from kb_core.models.entry import EntryType, KnowledgeEntry
from kb_core.models.search import SearchQuery

from kb_service.auth import get_current_user
from kb_service.models import ListenerPointer, ListenerRequest, ListenerResponse, User

router = APIRouter(prefix="/api/kb", tags=["kb"])


def _basic_blocks(entries: list[KnowledgeEntry]) -> str:
    """Format KnowledgeEntry objects as candidate blocks.

    Mirrors ``run_gate15.py::basic_blocks`` (lines 81-82).  Each candidate
    renders as ``'### {id}: {short_title}'`` + newline + ``knowledge_details``
    truncated to 400 chars; candidates are joined by blank lines.
    """
    return "\n\n".join(
        f"### {e.id}: {e.short_title}\n{(e.knowledge_details or '')[:400]}"
        for e in entries
    )


def _build_prompt(text: str, source: str, entries: list[KnowledgeEntry]) -> str:
    """Build the gate prompt (verbatim text of ``prompt_base``, run_gate15.py:102-113).

    Uses the proven Gate-1.5 config A prompt.  Do NOT substitute
    ``prompt_contra`` (lines 116-127) — its richer-evidence variant backfired
    per kb-01725 v6.

    String literals are split across source lines for PEP-8 width compliance;
    the concatenated value is byte-identical to the original f-string.
    """
    # Sentence-wrapped for line-length compliance (E501); runtime value is
    # identical to prompt_base in run_gate15.py:102-113.
    header = (
        "You are a strict relevance gate for a knowledge-surfacing system."
        " A false positive (surfacing an irrelevant map) is far worse than"
        " a false negative (staying silent)."
    )
    context = f'An AI coding agent working in the project "{source}" wrote:'
    quote = f'"""{text}"""'
    instruction = (
        "Candidate orientation maps follow."
        " Pick a map ONLY IF the agent is ASSERTING OR ASSUMING SPECIFIC FACTS"
        " about that map's domain"
        " — facts the map's domain documentation would confirm or correct"
        " (network topology, which machine runs what, how a specific system is wired)."
        " Shared vocabulary, tooling mentions, or topical adjacency are NOT sufficient."
        " Ordinary in-project coding/debugging needs NO map."
    )
    reply = (
        "Reply with AT MOST ONE candidate id (the single most load-bearing),"
        " or exactly NONE. When in doubt: NONE. No other text."
    )
    blocks = _basic_blocks(entries)
    return f"{header}\n\n{context}\n\n{quote}\n\n{instruction}\n\n{blocks}\n\n{reply}"


def _parse_vote(text: str | None, valid_ids: set[str]) -> str | None:
    """Parse a single ``generate()`` response into a valid candidate id or None.

    Port of ``run_gate15.py::sonnet`` lines 158-166.  A ``generate()``
    returning ``None`` (provider failure — anthropic.py swallows exceptions)
    counts as a None vote.
    """
    if text is None:
        return None
    stripped = text.strip()
    if stripped.upper().startswith("NONE"):
        return None
    tokens = [t.strip() for t in stripped.replace("\n", ",").split(",")]
    ids = [t for t in tokens if t.startswith("kb-")]
    valid = [i for i in ids if i in valid_ids]
    return valid[0] if valid else None


@router.post("/listener", response_model=ListenerResponse)
async def listener(
    body: ListenerRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> ListenerResponse:
    """Listener gate: surface a single KB map pointer or return null.

    Steps:
    1. Kill switch — return ``{pointer: null}`` if ``KB_LISTENER_ENABLED != 'TRUE'``.
    2. Search — ``EntryType.MENTAL_MAP``, ``limit=5``, query = ``body.text``.
    3. Rule A — drop candidates whose ``project_ref == cwd_project``.
    4. Rule B — drop candidates whose ``operated_via`` hint is in ``body.operating``.
    5. Short-circuit — return null if no candidates remain or no LLM available.
    6. LLM gate — fire 3 concurrent ``kb.synthesis_llm.generate(prompt)`` calls.
    7. Verdict — all 3 votes identical and non-None → return pointer; else null.
    """
    # ── Kill switch (default OFF) ─────────────────────────────────────────────
    # Mirrors config.py::is_agentic_ingest (config.py:159-161): read per-request.
    if os.environ.get("KB_LISTENER_ENABLED", "FALSE").upper() != "TRUE":
        return ListenerResponse(pointer=None)

    kb = request.app.state.kb

    # ── Candidate retrieval ───────────────────────────────────────────────────
    results, _ = await kb.search(
        SearchQuery(query=body.text, entry_type=EntryType.MENTAL_MAP, limit=5)
    )

    # ── Rule A: cross-project filter ──────────────────────────────────────────
    # Drop every candidate whose project_ref equals cwd_project.
    # When cwd_project is null, rule A drops nothing.
    candidates = list(results)
    if body.cwd_project is not None:
        candidates = [r for r in candidates if r.entry.project_ref != body.cwd_project]

    # ── Rule B: operating context filter ─────────────────────────────────────
    # Drop candidates whose operated_via hint (string) is in body.operating.
    # Maps with no operated_via hint are never dropped by rule B.
    operating_set = set(body.operating)
    candidates = [
        r
        for r in candidates
        if not (
            isinstance(r.entry.hints.get("operated_via"), str)
            and r.entry.hints["operated_via"] in operating_set
        )
    ]

    # ── Short-circuit: no candidates or no LLM ───────────────────────────────
    if not candidates or kb.synthesis_llm is None:
        return ListenerResponse(pointer=None)

    # ── LLM gate: 3 concurrent votes ─────────────────────────────────────────
    source = body.source_label or body.cwd_project or "unknown"
    entries = [r.entry for r in candidates]
    prompt = _build_prompt(body.text, source, entries)
    valid_ids = {e.id for e in entries}

    raw_votes = await asyncio.gather(
        kb.synthesis_llm.generate(prompt),
        kb.synthesis_llm.generate(prompt),
        kb.synthesis_llm.generate(prompt),
    )

    votes = [_parse_vote(v, valid_ids) for v in raw_votes]

    # ── Unanimous verdict ─────────────────────────────────────────────────────
    # All 3 votes must be identical AND non-None.
    # Retrieve-and-cite invariant: winner_id can only come from valid_ids.
    if votes[0] is not None and all(v == votes[0] for v in votes):
        winner_id = votes[0]
        winner_entry = next(e for e in entries if e.id == winner_id)
        return ListenerResponse(
            pointer=ListenerPointer(id=winner_id, short_title=winner_entry.short_title)
        )

    return ListenerResponse(pointer=None)
