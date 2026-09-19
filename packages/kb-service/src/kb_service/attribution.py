"""App-config settings helpers and the resolve_attribution seam.

Provides four async functions (get_setting, set_setting, delete_setting,
resolve_attribution) and one private helper (_normalize) that are consumed by
the settings routes and by the P2 write/ingest items.

All SQL goes through ``kb_service.database.get_db()`` (KB_SERVICE_DATABASE_URL).
Nothing is written to the KB data DB (KB_DATABASE_URL / app.state.kb).
"""

from datetime import UTC, datetime

from kb_core import Attribution

from kb_service.database import get_db
from kb_service.models import User

# app_config key naming the machine-principal user's email. Absent row means
# NO user is the machine principal — see is_machine_principal() below. Set
# via `kb-service set-machine-principal --email ...` (kb_service.cli).
MACHINE_PRINCIPAL_EMAIL_KEY = "machine_principal_email"


async def get_setting(key: str) -> str | None:
    """Return the raw stored value for *key*, or None when no row exists.

    Performs a single ``SELECT value FROM app_config WHERE key = $1`` per call.
    No caching layer — every call hits the service-auth pool (cheap PK lookup).
    """
    db = await get_db()
    row = await db.fetchrow("SELECT value FROM app_config WHERE key = $1", key)
    if row is None:
        return None
    stored: str = row["value"]
    return stored


async def set_setting(key: str, value: str, updated_by: str | None = None) -> None:
    """Upsert *key* = *value* into app_config.

    Writes updated_at as an ISO-8601 string (same convention as auth.py).
    Uses INSERT … ON CONFLICT to handle both initial inserts and updates.
    """
    db = await get_db()
    now = datetime.now(UTC).isoformat()
    await db.execute(
        "INSERT INTO app_config (key, value, updated_at, updated_by)"
        " VALUES ($1, $2, $3, $4)"
        " ON CONFLICT (key) DO UPDATE SET"
        " value = EXCLUDED.value,"
        " updated_at = EXCLUDED.updated_at,"
        " updated_by = EXCLUDED.updated_by",
        key,
        value,
        now,
        updated_by,
    )


async def delete_setting(key: str) -> None:
    """Delete the row for *key* from app_config (no-op when absent)."""
    db = await get_db()
    await db.execute("DELETE FROM app_config WHERE key = $1", key)


def _normalize(value: str | None) -> str | None:
    """Return None when value is None or whitespace-only; else return stripped."""
    if value is None or value.strip() == "":
        return None
    return value.strip()


async def resolve_attribution(user: User) -> Attribution:
    """Build an Attribution for *user* using the stored 'team' setting.

    Reads the 'team' key from app_config via get_setting().  Blank or absent
    values are normalised to None so Attribution.team stays None rather than
    carrying an empty string.

    Used by write/ingest endpoints as the per-request attribution seam.
    """
    team_raw = await get_setting("team")
    return Attribution(contributor=user.email, team=_normalize(team_raw))


async def is_machine_principal(user: User) -> bool:
    """Return True iff *user* is the configured machine principal.

    Reads the 'machine_principal_email' key from app_config via the same
    get_setting() accessor resolve_attribution() uses for 'team'. When the
    config row is absent (the default — no machine principal has been
    designated yet) this returns False for EVERY user, including admins;
    there is no fallback identity that could accidentally promote someone.

    Not enforced anywhere yet — this is a plain accessor other modules can
    call once a later item wires it into route behaviour.
    """
    configured_raw = await get_setting(MACHINE_PRINCIPAL_EMAIL_KEY)
    configured = _normalize(configured_raw)
    if configured is None:
        return False
    return user.email == configured
