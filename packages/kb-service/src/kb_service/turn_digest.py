"""Surprise-capture turn digests: mode switch, redaction, anomalies, storage.

Pure helpers plus ``turn_events`` DB helpers shared by the ingest route and
the later drain/worker consumers. ``processed_at`` is the only pending marker
on ``turn_events``; ``capture_mode`` is informational only.
"""

import json
import logging
import os
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, assert_never

from kb_core.cues import resolve_cue_project
from kb_core.ingest.safety import redact_secrets

from kb_service.db_types import DbPool
from kb_service.harness_tools import HARNESS_TOOL_MAPS
from kb_service.models import (
    TURN_EXCERPT_MAX,
    TURN_FINAL_MESSAGE_MAX,
    TURN_TARGET_MAX,
    TURN_TEXT_MAX,
    TURN_USER_PROMPT_MAX,
    StoredTurnDigest,
    SurpriseCaptureMode,
    TurnAssistantTextItem,
    TurnDigestRequest,
    TurnHarnessCorrectionItem,
    TurnReasoningItem,
    TurnToolCallItem,
    TurnToolResultItem,
)
from kb_service.routes.event_routes import _normalize_ts

logger = logging.getLogger(__name__)

TURN_DIGEST_MAX_BYTES = 65536
TURN_EVENTS_RETENTION_DAYS = 30

_INSERT_SQL = (
    "INSERT INTO turn_events (event_id, session_id, harness, mode, engine, host,"
    " hook_version, project, turn_index, ts, user_prompt, items, final_message,"
    " truncated, redactions, anomaly, capture_mode, received_ts)"
    " VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,"
    " $16, $17, $18)"
    " ON CONFLICT (event_id) DO NOTHING"
)

_SELECT_SQL = (
    "SELECT event_id, session_id, harness, mode, engine, host, hook_version, project,"
    " turn_index, ts, user_prompt, items, final_message, truncated, redactions,"
    " anomaly, capture_mode, processed_at, received_ts FROM turn_events"
)


def surprise_capture_mode() -> SurpriseCaptureMode:
    """Read ``KB_SURPRISE_CAPTURE`` (off|shadow|on); anything else is off."""
    v = os.environ.get("KB_SURPRISE_CAPTURE", "").strip().lower()
    if v == "shadow":
        return "shadow"
    if v == "on":
        return "on"
    return "off"


class _Redactor:
    """Accumulates redaction types and re-truncation stats across fields."""

    def __init__(self) -> None:
        self.types: list[str] = []
        self.fields_cut = 0
        self.chars_dropped = 0
        self.unavailable = False

    def __call__(self, text: str | None, cap: int) -> str | None:
        if text is None or self.unavailable:
            return text
        result = redact_secrets(text)
        if result is None:
            self.unavailable = True
            return None
        redacted, found = result
        for t in found:
            if t not in self.types:
                self.types.append(t)
        if len(redacted) > cap:
            self.fields_cut += 1
            self.chars_dropped += len(redacted) - cap
            redacted = redacted[:cap]
        return redacted


def redact_turn_digest(
    body: TurnDigestRequest,
) -> tuple[TurnDigestRequest, list[str]] | None:
    """Redact secrets from the free-text fields; None if redaction unavailable.

    Only user_prompt, final_message and the item fields text (assistant_text and
    reasoning), target, excerpt and detail (harness_correction) are scanned.
    """
    red = _Redactor()
    user_prompt = red(body.user_prompt, TURN_USER_PROMPT_MAX)
    items: list[Any] = []
    for item in body.items:
        if isinstance(item, TurnAssistantTextItem):
            text = red(item.text, TURN_TEXT_MAX)
            items.append(item.model_copy(update={"text": text}))
        elif isinstance(item, TurnToolCallItem):
            target = red(item.target, TURN_TARGET_MAX)
            items.append(item.model_copy(update={"target": target}))
        elif isinstance(item, TurnToolResultItem):
            excerpt = red(item.excerpt, TURN_EXCERPT_MAX)
            items.append(item.model_copy(update={"excerpt": excerpt}))
        elif isinstance(item, TurnReasoningItem):
            before = red.fields_cut
            text = red(item.text, TURN_TEXT_MAX)
            items.append(
                item.model_copy(
                    update={
                        "text": text,
                        "truncated": item.truncated or red.fields_cut > before,
                    }
                )
            )
        elif isinstance(item, TurnHarnessCorrectionItem):
            detail = red(item.detail, TURN_EXCERPT_MAX)
            items.append(item.model_copy(update={"detail": detail}))
        else:  # pragma: no cover
            assert_never(item)
    final_message = red(body.final_message, TURN_FINAL_MESSAGE_MAX)
    if red.unavailable:
        return None
    if red.fields_cut:
        logger.info(
            "turn_event retruncated event_id=%s fields=%d chars_dropped=%d",
            body.event_id,
            red.fields_cut,
            red.chars_dropped,
        )
    redacted = body.model_copy(
        update={
            "user_prompt": user_prompt,
            "items": items,
            "final_message": final_message,
        }
    )
    return redacted, red.types


def turn_digest_anomalies(body: TurnDigestRequest) -> list[str]:
    """Return the anomaly members that apply to *body*, in fixed order."""
    out: list[str] = []
    calls = [i for i in body.items if isinstance(i, TurnToolCallItem)]
    results = [i for i in body.items if isinstance(i, TurnToolResultItem)]
    if any(c.tool == "Bash" and c.target_class == "" for c in calls):
        out.append("empty_bash_target_class")
    if not body.truncated:
        call_ids = {c.tool_use_id for c in calls}
        if any(r.tool_use_id not in call_ids for r in results):
            out.append("orphan_tool_result")
    if not body.items and body.user_prompt is None and body.final_message is None:
        out.append("empty_turn")
    if body.harness in HARNESS_TOOL_MAPS and any(
        c.tool in ("Edit", "Write", "Read") and c.target == "" for c in calls
    ):
        out.append("empty_tool_target")
    return out


def _count(status: str) -> int:
    return int(status.rsplit(" ", 1)[1])


async def insert_turn_digest(
    pool: DbPool,
    body: TurnDigestRequest,
    *,
    capture_mode: Literal["shadow", "on"],
    redactions: list[str],
    received_ts: str,
) -> Literal["recorded", "duplicate", "duplicate-mismatch"]:
    """Insert one digest idempotently on ``event_id`` (first write wins)."""
    items_json = json.dumps([item.model_dump() for item in body.items])
    status = await pool.execute(
        _INSERT_SQL,
        body.event_id,
        body.session_id,
        body.harness,
        body.mode,
        body.engine,
        body.host,
        body.hook_version,
        resolve_cue_project(body.project, None)[0],
        body.turn_index,
        _normalize_ts(body.ts, received_ts),
        body.user_prompt,
        items_json,
        body.final_message,
        1 if body.truncated else 0,
        json.dumps(redactions),
        ",".join(turn_digest_anomalies(body)) or None,
        capture_mode,
        received_ts,
    )
    if status == "INSERT 0 1":
        return "recorded"
    row = await pool.fetchrow(
        "SELECT items, user_prompt, final_message FROM turn_events WHERE event_id = $1",
        body.event_id,
    )
    if (
        row is not None
        and row["items"] == items_json
        and row["user_prompt"] == body.user_prompt
        and row["final_message"] == body.final_message
    ):
        return "duplicate"
    stored_items = 0
    if row is not None:
        try:
            stored_items = len(json.loads(row["items"]))
        except ValueError:
            stored_items = 0
    logger.warning(
        "turn_event duplicate_mismatch event_id=%s session_id=%s"
        " stored_items=%d posted_items=%d",
        body.event_id,
        body.session_id,
        stored_items,
        len(body.items),
    )
    return "duplicate-mismatch"


def _to_stored(row: Any) -> StoredTurnDigest | None:
    try:
        return StoredTurnDigest(
            event_id=row["event_id"],
            session_id=row["session_id"],
            harness=row["harness"],
            mode=row["mode"],
            engine=row["engine"],
            host=row["host"],
            hook_version=row["hook_version"],
            project=row["project"],
            turn_index=row["turn_index"],
            ts=row["ts"],
            user_prompt=row["user_prompt"],
            items=json.loads(row["items"]),
            final_message=row["final_message"],
            truncated=bool(row["truncated"]),
            redactions=json.loads(row["redactions"]),
            anomaly=row["anomaly"],
            capture_mode=row["capture_mode"],
            processed_at=row["processed_at"],
            received_ts=row["received_ts"],
        )
    except Exception:
        logger.warning("turn_event malformed event_id=%s", row["event_id"])
        return None


def _to_stored_list(rows: list[Any]) -> list[StoredTurnDigest]:
    out = [_to_stored(r) for r in rows]
    return [d for d in out if d is not None]


async def get_session_turn_digests(
    pool: DbPool, session_id: str
) -> list[StoredTurnDigest]:
    """Return a session's digests ordered by turn_index, then id."""
    rows = await pool.fetch(
        _SELECT_SQL + " WHERE session_id = $1 ORDER BY turn_index ASC, id ASC",
        session_id,
    )
    return _to_stored_list(rows)


async def list_pending_turn_digests(
    pool: DbPool, limit: int | None = None
) -> list[StoredTurnDigest]:
    """Return digests with ``processed_at IS NULL``, oldest id first."""
    sql = _SELECT_SQL + " WHERE processed_at IS NULL ORDER BY id ASC"
    if limit is None:
        rows = await pool.fetch(sql)
    else:
        rows = await pool.fetch(sql + " LIMIT $1", limit)
    return _to_stored_list(rows)


async def mark_turn_digests_processed(
    pool: DbPool, event_ids: Sequence[str], processed_at: str
) -> int:
    """Mark digests processed; return the number of rows this call flipped.

    ``processed_at`` is the only pending marker and this is the only function
    that sets it. Each UPDATE is conditional on ``processed_at IS NULL``, so a
    single-id call returns 1 only to the caller that flipped the row and 0 to
    every later or concurrent caller: consumers may use it as an atomic
    per-row claim.
    """
    total = 0
    for event_id in event_ids:
        status = await pool.execute(
            "UPDATE turn_events SET processed_at = $1"
            " WHERE event_id = $2 AND processed_at IS NULL",
            processed_at,
            event_id,
        )
        total += _count(status)
    return total


async def prune_turn_events(
    pool: DbPool,
    older_than_days: int = TURN_EVENTS_RETENTION_DAYS,
    now: datetime | None = None,
) -> int:
    """Delete digests (pending or processed) received before the cutoff."""
    cutoff = ((now or datetime.now(UTC)) - timedelta(days=older_than_days)).isoformat(
        timespec="seconds"
    )
    status = await pool.execute(
        "DELETE FROM turn_events WHERE received_ts < $1", cutoff
    )
    return _count(status)
