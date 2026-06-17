"""Whisper-efficacy telemetry sink: POST /api/kb/telemetry/whispers.

Accepts a batch of telemetry rows from the personal-kb-hook (one Stop-flush
per session, carrying that session's accrued roster + listener rows) and
upserts them into the SERVICE/AUTH-side ``whisper_telemetry`` table by the
composite primary key ``(session_id, surface, map_id)``.

The composite PK is the ON CONFLICT conflict target — a re-flush carrying an
updated ``consumed`` flag wins via DO UPDATE on the four mutable columns
(``consumed``, ``consumed_ts``, ``build_engine``, ``flushed_at``,
``trigger_context``). The five "where the row was first seen" columns
(``emitted_ts``, ``source_kb``, ``host``, ``cwd_project``, plus the PK
columns themselves) are NEVER overwritten on conflict.

``trigger_context`` is a TEXT json string in the DB — no jsonb anywhere
(asyncpg has no jsonb codec registered for this pool). ``consumed`` is bound
as an integer (the row schema uses ``INTEGER NOT NULL DEFAULT 0``).
"""

import json
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from kb_service.auth import get_current_user
from kb_service.database import get_db
from kb_service.models import (
    User,
    WhisperTelemetryFlushRequest,
    WhisperTelemetryFlushResponse,
)

router = APIRouter(prefix="/api/kb", tags=["kb"])


_UPSERT_SQL = (
    "INSERT INTO whisper_telemetry ("
    "session_id, host, surface, map_id, source_kb, cwd_project,"
    " trigger_context, emitted_ts, consumed, consumed_ts, build_engine,"
    " flushed_at"
    ") VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)"
    " ON CONFLICT (session_id, surface, map_id) DO UPDATE SET"
    " consumed = EXCLUDED.consumed,"
    " consumed_ts = EXCLUDED.consumed_ts,"
    " build_engine = EXCLUDED.build_engine,"
    " flushed_at = EXCLUDED.flushed_at,"
    " trigger_context = EXCLUDED.trigger_context"
)


@router.post("/telemetry/whispers", response_model=WhisperTelemetryFlushResponse)
async def flush_whisper_telemetry(
    body: WhisperTelemetryFlushRequest,
    _user: Annotated[User, Depends(get_current_user)],
) -> WhisperTelemetryFlushResponse:
    """Upsert a batch of whisper telemetry rows.

    Per-row INSERT ... ON CONFLICT (session_id, surface, map_id) DO UPDATE.
    Loops per-row execute() in a single acquired connection (NOT executemany).
    """
    # Empty batch is a no-op — accept it without acquiring a connection.
    if not body.rows:
        return WhisperTelemetryFlushResponse(upserted=0)

    flushed_at = datetime.now(UTC).isoformat()

    pool = await get_db()
    async with pool.acquire() as conn:
        for row in body.rows:
            await conn.execute(
                _UPSERT_SQL,
                row.session_id,
                row.host,
                row.surface,
                row.map_id,
                row.source_kb,
                row.cwd_project,
                json.dumps(row.trigger_context),
                row.emitted_ts,
                int(row.consumed),
                row.consumed_ts,
                row.build_engine,
                flushed_at,
            )

    return WhisperTelemetryFlushResponse(upserted=len(body.rows))
