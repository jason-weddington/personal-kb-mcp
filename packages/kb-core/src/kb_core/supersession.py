"""Supersession: the ``superseded_by`` invariant, its writers, and its checks.

THE INVARIANT. Every ``knowledge_entries`` row's ``superseded_by`` equals the
id of its NEWEST qualifying superseder, or NULL when it has none. A qualifying
superseder of T is a row S with ``is_active = 1``, ``S.entry_type !=
'mental_map'`` and a ``graph_edges`` row ``(source=S.id, target=T.id,
edge_type='supersedes')`` whose ``properties`` JSON does NOT carry
``source == 'llm'``. NEWEST means max ``created_at`` (immutable — never
``updated_at``, which moves on every edit and would flip the pointer); ties
break by id descending. The invariant holds whether T is active or not.

Writers that maintain it: ``GraphBuilder.build_for_entry``,
``KnowledgeStore.deactivate_entry``/``reactivate_entry``,
``KnowledgeBase.deactivate`` and :func:`reconcile_supersession`. Hard delete
and direct SQL rely on the startup reconcile to heal.

This module must NOT import ``kb_core.store.knowledge_store`` (that module
imports this one); the audit INSERT mirrors ``_record_audit_event``'s columns.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

from kb_core.db.queries import get_entry
from kb_core.models.entry import EntryType

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kb_core.db.backend import Database

logger = logging.getLogger(__name__)

_KB_ID_RE = re.compile(r"kb-\d{5}")

Trigger = Literal["build", "deactivate", "reconcile"]


@dataclass(frozen=True)
class RecomputeCounts:
    """Outcome of one :func:`recompute_superseded_by` call.

    ``set_count`` counts rows whose new value is non-NULL and differs from the
    old one; ``cleared_count`` counts rows that went from non-NULL to NULL.
    ``changed`` lists ``(target, old, new)`` for every row actually written.
    """

    set_count: int
    cleared_count: int
    changed: tuple[tuple[str, str | None, str | None], ...]


@dataclass(frozen=True)
class SupersessionReconcileReport:
    """Outcome of one :func:`reconcile_supersession` run."""

    edges_added: int
    set_count: int
    cleared_count: int
    changed: tuple[tuple[str, str | None, str | None], ...]
    edges_added_ids: tuple[tuple[str, str], ...]


def norm_supersedes(value: object) -> list[object]:
    """Normalize a ``hints['supersedes']`` value: None -> [], scalar -> [v], list as-is."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _parse_created_at(value: object) -> datetime:
    """Parse a stored ``created_at``; naive values are treated as UTC."""
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value))
        except ValueError:
            parsed = datetime.min
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _is_llm_edge(properties: object) -> bool:
    """Return True when an edge's properties JSON carries ``source == 'llm'``."""
    if not properties:
        return False
    try:
        parsed = json.loads(properties) if isinstance(properties, str) else properties
    except (TypeError, ValueError):
        return False
    return isinstance(parsed, dict) and parsed.get("source") == "llm"


async def _candidates(db: Database, target_id: str) -> list[str]:
    """Qualifying superseder ids of *target_id*, newest first."""
    cursor = await db.execute(
        "SELECT k.id, k.created_at, e.properties FROM graph_edges e"
        " JOIN knowledge_entries k ON k.id = e.source"
        " WHERE e.target = ? AND e.edge_type = 'supersedes'"
        " AND k.is_active = 1 AND k.entry_type <> 'mental_map'",
        (target_id,),
    )
    rows = await cursor.fetchall()
    found: dict[str, datetime] = {}
    for row in rows:
        source_id = str(row["id"])
        if source_id == target_id or _is_llm_edge(row["properties"]):
            continue
        found[source_id] = _parse_created_at(row["created_at"])
    return [sid for sid, _ in sorted(found.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)]


async def recompute_superseded_by(
    db: Database,
    target_ids: Iterable[str],
    *,
    trigger: Trigger,
) -> RecomputeCounts:
    """Recompute ``superseded_by`` for each target from the supersedes edges.

    Opens NO transaction of its own — it runs inside the caller's. A target
    whose value is unchanged is not written, counted or logged. A changed
    target gets a ``superseded_by``-only UPDATE (no version bump, no
    ``updated_at`` change), a ``superseded_by_changed`` audit row and an INFO
    log line. Targets that do not exist are skipped.
    """
    set_count = 0
    cleared_count = 0
    changed: list[tuple[str, str | None, str | None]] = []
    for target_id in sorted(set(target_ids)):
        cursor = await db.execute(
            "SELECT superseded_by FROM knowledge_entries WHERE id = ?", (target_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            continue
        old: str | None = row["superseded_by"]
        candidates = await _candidates(db, target_id)
        new = candidates[0] if candidates else None
        if new == old:
            continue
        await db.execute(
            "UPDATE knowledge_entries SET superseded_by = ? WHERE id = ?", (new, target_id)
        )
        detail = json.dumps({"old": old, "new": new, "candidates": candidates, "trigger": trigger})
        await db.execute(
            "INSERT INTO audit_events (event_type, entry_id, contributor, detail, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            ("superseded_by_changed", target_id, None, detail, _now_iso()),
        )
        logger.info(
            "supersession target=%s old=%r new=%r candidates=%r trigger=%s",
            target_id,
            old,
            new,
            candidates,
            trigger,
        )
        if new is not None:
            set_count += 1
        elif old is not None:
            cleared_count += 1
        changed.append((target_id, old, new))
    return RecomputeCounts(set_count=set_count, cleared_count=cleared_count, changed=tuple(changed))


async def outgoing_supersedes_targets(db: Database, source_id: str) -> set[str]:
    """Targets of *source_id*'s ``supersedes`` edges."""
    cursor = await db.execute(
        "SELECT target FROM graph_edges WHERE source = ? AND edge_type = 'supersedes'",
        (source_id,),
    )
    return {str(r["target"]) for r in await cursor.fetchall()}


async def check_supersedes_targets(
    db: Database,
    target_ids: list[str],
    *,
    writer_id: str | None,
    writer_entry_type: EntryType,
) -> list[str]:
    """Validate supersedes targets; return problem strings, ``[]`` when all valid.

    Cross-project targets are allowed (live edges already cross projects).
    """
    if writer_entry_type is EntryType.MENTAL_MAP and target_ids:
        return ["a mental_map cannot supersede entries"]
    problems: list[str] = []
    for target_id in target_ids:
        if not isinstance(target_id, str) or not _KB_ID_RE.fullmatch(target_id):
            problems.append(f"{target_id} is not a valid entry id")
            continue
        target = await get_entry(db, target_id)
        if target is None:
            problems.append(f"{target_id} not found")
            continue
        if not target.is_active:
            problems.append(f"{target_id} is inactive")
            continue
        if target_id == writer_id:
            problems.append("an entry cannot supersede itself")
            continue
        if target.entry_type is EntryType.MENTAL_MAP:
            problems.append(f"{target_id} is a mental_map; maps are deleted, not superseded")
            continue
        if writer_id is not None and writer_id in norm_supersedes(target.hints.get("supersedes")):
            problems.append(f"{target_id} already supersedes {writer_id} (cycle)")
    return problems


async def ensure_entry_node(db: Database, node_id: str) -> None:
    """Insert an ``entry`` graph node when absent (never rewrites properties)."""
    await db.execute(
        "INSERT INTO graph_nodes (node_id, node_type, properties, created_at)"
        " VALUES (?, 'entry', '{}', ?) ON CONFLICT (node_id) DO NOTHING",
        (node_id, _now_iso()),
    )


async def insert_supersedes_edge(db: Database, source_id: str, target_id: str) -> None:
    """Ensure both endpoint nodes and insert a ``supersedes`` edge (idempotent)."""
    await ensure_entry_node(db, source_id)
    await ensure_entry_node(db, target_id)
    await db.execute(
        "INSERT INTO graph_edges (source, target, edge_type, properties, created_at)"
        " VALUES (?, ?, 'supersedes', '{}', ?)"
        " ON CONFLICT (source, target, edge_type) DO NOTHING",
        (source_id, target_id, _now_iso()),
    )


async def add_supersedes_hint(db: Database, superseder_id: str, target_id: str) -> None:
    """Append *target_id* to *superseder_id*'s ``hints.supersedes`` (no version bump)."""
    cursor = await db.execute("SELECT hints FROM knowledge_entries WHERE id = ?", (superseder_id,))
    row = await cursor.fetchone()
    if row is None:
        raise ValueError(f"superseded_by {superseder_id} not found")
    hints = _load_hints(row["hints"])
    existing = {s for s in norm_supersedes(hints.get("supersedes")) if isinstance(s, str)}
    hints = {**hints, "supersedes": sorted(existing | {target_id})}
    await db.execute(
        "UPDATE knowledge_entries SET hints = ? WHERE id = ?",
        (json.dumps(hints), superseder_id),
    )
    await db.execute(
        "INSERT INTO audit_events (event_type, entry_id, contributor, detail, created_at)"
        " VALUES (?, ?, ?, ?, ?)",
        (
            "supersedes_hint_appended",
            superseder_id,
            None,
            json.dumps({"added": target_id, "via": "deactivate"}),
            _now_iso(),
        ),
    )


def _load_hints(raw: object) -> dict[str, object]:
    if isinstance(raw, dict):
        return dict(raw)
    try:
        parsed = json.loads(str(raw)) if raw else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


async def reconcile_supersession(db: Database) -> SupersessionReconcileReport:
    """Idempotent startup reconcile: backfill missing edges, then recompute.

    (1) Every active non-map entry whose ``hints.supersedes`` names an
    existing kb id with no supersedes edge yet gets that edge. (2) Every
    supersedes-edge target plus every row with ``superseded_by`` set is
    recomputed with ``trigger='reconcile'``. One transaction.
    """
    async with db.transaction():
        cursor = await db.execute(
            "SELECT id, hints FROM knowledge_entries WHERE is_active = 1"
            " AND entry_type <> 'mental_map' AND hints LIKE '%supersedes%'"
        )
        rows = await cursor.fetchall()
        added: list[tuple[str, str]] = []
        for row in rows:
            source_id = str(row["id"])
            hints = _load_hints(row["hints"])
            for target in norm_supersedes(hints.get("supersedes")):
                if not isinstance(target, str) or not _KB_ID_RE.fullmatch(target):
                    continue
                if target == source_id:
                    continue
                exists_cur = await db.execute(
                    "SELECT 1 FROM knowledge_entries WHERE id = ?", (target,)
                )
                if await exists_cur.fetchone() is None:
                    continue
                edge_cur = await db.execute(
                    "SELECT 1 FROM graph_edges WHERE source = ? AND target = ?"
                    " AND edge_type = 'supersedes'",
                    (source_id, target),
                )
                if await edge_cur.fetchone() is not None:
                    continue
                await insert_supersedes_edge(db, source_id, target)
                added.append((source_id, target))

        targets_cur = await db.execute(
            "SELECT DISTINCT target FROM graph_edges WHERE edge_type = 'supersedes'"
        )
        targets = {str(r["target"]) for r in await targets_cur.fetchall()}
        set_cur = await db.execute(
            "SELECT id FROM knowledge_entries WHERE superseded_by IS NOT NULL"
        )
        targets |= {str(r["id"]) for r in await set_cur.fetchall()}
        counts = await recompute_superseded_by(db, targets, trigger="reconcile")
    return SupersessionReconcileReport(
        edges_added=len(added),
        set_count=counts.set_count,
        cleared_count=counts.cleared_count,
        changed=counts.changed,
        edges_added_ids=tuple(added),
    )
