"""Shared mental-map write guards, reused by every map write path.

Holds the two checks that ``/api/kb/store`` and ``/api/kb/map-op`` must run
IDENTICALLY.

One is the cardinal orphan rule: a mental_map must point at something.

The other is the machine-principal map lint.

They lived in kb_write_routes.py first; map_op_routes.py needs the same pair.

They moved here, and kb_write_routes.py imports them back.

Same code, same status codes, same 422 detail rendering, one definition.

The lint gate is HARD for the machine principal — see the nightly map design doc.

It is the nightly loop's only mechanical purity check.

For every other user the lint stays ADVISORY and the write succeeds.

Humans keep ``/api/kb/store``, where the MCP channel renders the findings.
"""

import logging
import re
from typing import Any

from fastapi import HTTPException
from kb_core.map_lint import lint_map_body
from kb_core.models.entry import EntryType

from kb_service.attribution import is_machine_principal
from kb_service.models import User

logger = logging.getLogger(__name__)

# kb-XXXXX reference pattern (5 digits, verified at kb_store.py:29).
_KB_ID_RE: re.Pattern[str] = re.compile(r"kb-\d{5}")


def _mental_map_has_pointer(
    knowledge_details: str,
    hints: dict[str, Any] | None,
) -> bool:
    r"""Return True when a mental_map entry has at least one outbound pointer.

    A pointer is present when ANY of:

    * ``re.compile(r"kb-\d{5}").search(knowledge_details)`` finds a match.
    * ``hints["supersedes"]`` (scalar-or-list) contains a string full-matching
      the pattern ``kb-\d{5}``.
    * ``hints["related_entities"]`` (scalar-or-list) contains either a dict
      with a non-empty ``"id"`` or ``"target"`` string key, or a bare non-empty
      string.

    ``tag``, ``project``, ``person``, and ``tool`` hints do NOT count.
    Semantics verified against ``_mental_map_has_pointer`` in the MCP channel
    (kb_store.py:46-87).  Private kb_core helpers are NOT imported.
    """
    if _KB_ID_RE.search(knowledge_details):
        return True
    if not hints:
        return False
    # Check hints["supersedes"]
    supersedes = hints.get("supersedes")
    if supersedes is not None:
        items: list[Any] = supersedes if isinstance(supersedes, list) else [supersedes]
        for item in items:
            if isinstance(item, str) and re.fullmatch(r"kb-\d{5}", item):
                return True
    # Check hints["related_entities"]
    related = hints.get("related_entities")
    if related is not None:
        rels: list[Any] = related if isinstance(related, list) else [related]
        for item in rels:
            if isinstance(item, dict):
                id_val = item.get("id")
                target_val = item.get("target")
                if (isinstance(id_val, str) and id_val) or (
                    isinstance(target_val, str) and target_val
                ):
                    return True
            elif isinstance(item, str) and item:
                return True
    return False


def _check_orphan_mental_map(
    entry_type: EntryType,
    knowledge_details: str,
    hints: dict[str, Any] | None,
) -> None:
    """Raise 422 when a mental_map entry has no outbound pointers."""
    if entry_type is not EntryType.MENTAL_MAP:
        return
    if not _mental_map_has_pointer(knowledge_details, hints):
        raise HTTPException(
            status_code=422,
            detail=(
                "A mental_map entry requires at least one outbound pointer "
                "(a kb-XXXXX reference in knowledge_details, or a "
                "supersedes/related_entities hint)."
            ),
        )


async def _check_machine_principal_map_lint(
    entry_type: EntryType,
    effective_details: str,
    user: User,
    *,
    prefix: str = "",
) -> None:
    """Raise 422 when the machine principal writes a lint-failing mental_map.

    HARD gate, machine principal ONLY (docs/nightly-map-maintenance-design.md,
    Phase 0 "server-side hard lint for the machine principal"): the nightly
    map-maintenance loop writes over HTTP, so the write path is the only
    place its bodies can be mechanically checked — without this gate every
    purity rule in that design reduces to a prompt instruction. The lint is
    kb-core's single-source-of-truth one (``kb_core.map_lint``), the same
    findings the dry-run endpoint (``map_lint_routes``) reports and the MCP
    channel renders as advisories.

    For every OTHER user the lint stays ADVISORY (MCP channel) and the write
    succeeds: hard-gating humans would reject the corpus's best maps, which
    legitimately trip the advisory. With no ``machine_principal_email``
    config row set, ``is_machine_principal`` returns False for EVERYONE
    including admins, so the gate is inert and no write is rejected.

    Follows ``_check_orphan_mental_map`` for shape and status code (422).
    """
    if entry_type is not EntryType.MENTAL_MAP:
        return
    if not await is_machine_principal(user):
        return
    findings = lint_map_body(effective_details)
    if not findings:
        return
    rendered = "; ".join(f"{f.code.value}: {f.message}" for f in findings)
    raise HTTPException(
        status_code=422,
        detail=(
            f"{prefix}Map lint findings reject this mental_map write by the "
            f"machine principal: {rendered}"
        ),
    )
