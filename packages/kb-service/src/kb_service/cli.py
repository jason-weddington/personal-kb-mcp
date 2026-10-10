"""CLI for the KB service — shell-accessible admin bootstrap.

All commands talk DIRECTLY to the service-auth database (the one pointed at by
``KB_SERVICE_DATABASE_URL``), not through the HTTP API. ``create-admin`` is the
bootstrap for the FIRST admin, since registration is invite-gated. ``create-user``
creates an ordinary non-admin user — e.g. a dedicated machine principal for the
nightly map-maintenance jobs — without over-granting admin's invite/reset/read-all
powers. ``set-machine-principal`` designates which user (by email) is the machine
principal via the ``machine_principal_email`` app_config row (see
``kb_service.attribution.is_machine_principal``); it does not require the user to
already exist. ``list-keys`` and ``set-key-surface`` show and set the
write-policy surface of each API key (``kb_service.write_policy``); the CLI is
the only setter, so a key's surface is always admin-set.
"""

import argparse
import asyncio
import logging
import os
import sys
import uuid
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any


async def _create_admin(email: str, password: str) -> str:
    """Insert a new admin user directly into the service-auth database.

    Returns:
        Human-readable status message.

    Raises:
        ValueError: If a user with that email already exists.
    """
    from kb_service.auth import hash_password
    from kb_service.database import close_db, get_db, init_db

    await init_db()
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1", email
            )
            if existing is not None:
                raise ValueError(f"a user with email {email} already exists")
            now = datetime.now(UTC).isoformat()
            await conn.execute(
                "INSERT INTO users (id, email, hashed_password, is_admin, created_at)"
                " VALUES ($1, $2, $3, $4, $5)",
                str(uuid.uuid4()),
                email,
                hash_password(password),
                1,
                now,
            )
            return f"created admin user {email}"
    finally:
        await close_db()


async def _create_user(email: str, password: str) -> str:
    """Insert a new ordinary (non-admin) user directly into the service-auth database.

    Same insert path as ``_create_admin`` but with ``is_admin = 0`` — for
    principals (e.g. a nightly-job machine account) that must not get
    admin's invite/reset/read-all powers.

    Returns:
        Human-readable status message.

    Raises:
        ValueError: If a user with that email already exists.
    """
    from kb_service.auth import hash_password
    from kb_service.database import close_db, get_db, init_db

    await init_db()
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            existing = await conn.fetchrow(
                "SELECT id FROM users WHERE email = $1", email
            )
            if existing is not None:
                raise ValueError(f"a user with email {email} already exists")
            now = datetime.now(UTC).isoformat()
            await conn.execute(
                "INSERT INTO users (id, email, hashed_password, is_admin, created_at)"
                " VALUES ($1, $2, $3, $4, $5)",
                str(uuid.uuid4()),
                email,
                hash_password(password),
                0,
                now,
            )
            return f"created user {email}"
    finally:
        await close_db()


async def _set_machine_principal(email: str) -> str:
    """Upsert the ``machine_principal_email`` app_config row to *email*.

    Does not require *email* to already exist as a user — the config row is
    the source of truth, resolved lazily by
    ``kb_service.attribution.is_machine_principal`` on every check. Pointing
    it at a nonexistent email simply means no current user is the machine
    principal (same as leaving it unset).

    Returns:
        Human-readable status message.
    """
    from kb_service.attribution import MACHINE_PRINCIPAL_EMAIL_KEY, set_setting
    from kb_service.database import close_db, init_db

    await init_db()
    try:
        await set_setting(MACHINE_PRINCIPAL_EMAIL_KEY, email)
        return f"set machine principal to {email}"
    finally:
        await close_db()


async def _make_admin(email: str) -> str:
    """Set ``is_admin = 1`` on the user with the given email.

    Returns:
        Human-readable status message.

    Raises:
        ValueError: If no user with that email exists.
    """
    from kb_service.database import close_db, get_db, init_db

    await init_db()
    try:
        pool = await get_db()
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT id, is_admin FROM users WHERE email = $1", email
            )
            if row is None:
                raise ValueError(f"no user found with email {email}")
            if row["is_admin"]:
                return f"{email} is already an admin"
            await conn.execute("UPDATE users SET is_admin = 1 WHERE email = $1", email)
            return f"promoted {email} to admin"
    finally:
        await close_db()


async def _list_keys(email: str | None) -> str:
    """List API keys (no secret material) with their write-policy surface.

    Returns:
        A tab-separated table, or ``no API keys``.
    """
    from kb_service.database import close_db, get_db, init_db

    await init_db()
    try:
        pool = await get_db()
        sql = (
            "SELECT k.id, u.email, k.name, k.key_hash, k.surface, k.created_at"
            " FROM api_keys k JOIN users u ON u.id = k.user_id"
        )
        args: list[Any] = []
        if email is not None:
            sql += " WHERE u.email = $1"
            args.append(email)
        sql += " ORDER BY u.email, k.created_at"
        rows = await pool.fetch(sql, *args)
        if not rows:
            return "no API keys"
        lines = ["id\temail\tname\thash_prefix\tsurface\tcreated_at"]
        lines.extend(
            "\t".join(
                [
                    str(r["id"]),
                    str(r["email"]),
                    str(r["name"]),
                    str(r["key_hash"])[:8],
                    str(r["surface"]) if r["surface"] is not None else "default",
                    str(r["created_at"]),
                ]
            )
            for r in rows
        )
        return "\n".join(lines)
    finally:
        await close_db()


async def _set_key_surface(key_id: str, surface: str) -> str:
    """Set (or clear, with ``default``) the write-policy surface of an API key.

    Returns:
        Human-readable status message.

    Raises:
        ValueError: If no API key has that id.
    """
    from kb_service.database import close_db, get_db, init_db

    await init_db()
    try:
        pool = await get_db()
        row = await pool.fetchrow(
            "SELECT k.name, u.email FROM api_keys k JOIN users u ON u.id = k.user_id"
            " WHERE k.id = $1",
            key_id,
        )
        if row is None:
            raise ValueError(f"no API key with id {key_id}")
        email, name = row["email"], row["name"]
        value = None if surface == "default" else surface
        await pool.execute(
            "UPDATE api_keys SET surface = $1 WHERE id = $2", value, key_id
        )
        if value is None:
            return (
                f"cleared surface of API key {key_id} ({email}, {name!r}); it follows"
                " KB_WRITE_POLICY_DEFAULT_SURFACE"
            )
        return f"set surface of API key {key_id} ({email}, {name!r}) to {surface}"
    finally:
        await close_db()


async def _metrics_repeat_rate(
    weeks: int,
    project: str | None,
    min_gap_hours: float,
    as_json: bool,
    now: datetime | None = None,
) -> str:
    """Compute the weekly repeat rate from the service DB and the data DB.

    Returns:
        The response as indented JSON when *as_json*, else markdown.
    """
    import kb_service.main
    from kb_service.database import close_db, get_db, init_db
    from kb_service.repeat_rate import build_repeat_rate, render_repeat_rate_markdown

    await init_db()
    try:
        kb = await kb_service.main._open_kb()
        try:
            resp = await build_repeat_rate(
                await get_db(),
                kb.db,
                weeks=weeks,
                project=project,
                min_gap_hours=min_gap_hours,
                now=now or datetime.now(UTC),
            )
        finally:
            await kb.close()
    finally:
        await close_db()
    if as_json:
        return resp.model_dump_json(indent=2)
    return render_repeat_rate_markdown(resp)


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
_HANDLER_MARKER = "_kb_service_handler"


def configure_logging(level_name: str | None = None) -> None:
    """Install one stderr handler on the root logger; INFO for kb_service/kb_core.

    *level_name* defaults to ``KB_LOG_LEVEL`` (INFO when unset/blank; an unknown
    name falls back to INFO with one WARNING). Root stays at WARNING so
    third-party libraries remain quiet. Idempotent.
    """
    raw = level_name if level_name is not None else os.environ.get("KB_LOG_LEVEL")
    raw = (raw or "").strip()
    level = logging.INFO
    bad: str | None = None
    if raw:
        resolved = logging.getLevelName(raw.upper())
        if isinstance(resolved, int):
            level = resolved
        else:
            bad = raw

    root = logging.getLogger()
    if not any(getattr(h, _HANDLER_MARKER, False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        setattr(handler, _HANDLER_MARKER, True)
        root.addHandler(handler)
    root.setLevel(logging.WARNING)
    for name in ("kb_service", "kb_core"):
        logging.getLogger(name).setLevel(level)
    if bad is not None:
        logging.getLogger(__name__).warning(
            "unknown KB_LOG_LEVEL %r; falling back to INFO", bad
        )


def _serve(host: str, port: int) -> None:
    """Run the FastAPI app under uvicorn in the foreground.

    Console-script entry point so a thin client (e.g. the ``personal_kb`` MCP
    server, installed via a git+ssh dependency) can spawn the daemon with
    ``kb-service serve --port <n>`` without needing a uvicorn invocation of its
    own.
    """
    import uvicorn

    configure_logging()
    # log_config=None keeps uvicorn from replacing our root configuration. Its
    # "uvicorn", "uvicorn.error" and "uvicorn.access" loggers then have no
    # handlers of their own and propagate to the root handler. They would
    # inherit root's WARNING level, so log_level="info" sets them to INFO
    # explicitly so startup and access lines still emit.
    uvicorn.run(
        "kb_service.main:app",
        host=host,
        port=port,
        log_config=None,
        log_level="info",
    )


def _run(coro: Coroutine[Any, Any, str]) -> None:
    """Run an async command coroutine, printing the result or erroring out."""
    try:
        msg = asyncio.run(coro)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(msg)


def main() -> None:
    """Entry point for the kb-service CLI."""
    parser = argparse.ArgumentParser(
        prog="kb-service",
        description="Personal KB web service command-line interface.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")

    ca = subparsers.add_parser(
        "create-admin",
        help="Bootstrap the first admin user (direct DB insert).",
    )
    ca.add_argument("--email", required=True, help="Email of the admin user.")
    ca.add_argument("--password", required=True, help="Password for the admin user.")

    cu = subparsers.add_parser(
        "create-user",
        help="Create an ordinary non-admin user (direct DB insert).",
    )
    cu.add_argument("--email", required=True, help="Email of the new user.")
    cu.add_argument("--password", required=True, help="Password for the new user.")

    ma = subparsers.add_parser(
        "make-admin",
        help="Set is_admin=1 on an existing user (direct DB update).",
    )
    ma.add_argument("--email", required=True, help="Email of the user to promote.")

    smp = subparsers.add_parser(
        "set-machine-principal",
        help="Designate the machine-principal user via app_config (direct DB upsert).",
    )
    smp.add_argument(
        "--email",
        required=True,
        help="Email of the user to designate as machine principal.",
    )

    lk = subparsers.add_parser(
        "list-keys",
        help="List API keys with their write-policy surface (direct DB read).",
    )
    lk.add_argument("--email", default=None, help="Only keys of this user.")

    sks = subparsers.add_parser(
        "set-key-surface",
        help="Set an API key's write-policy surface (direct DB update).",
    )
    sks.add_argument("--key-id", required=True, help="Id of the API key.")
    sks.add_argument(
        "--surface",
        required=True,
        choices=["interactive", "headless", "autonomous", "default"],
        help="The surface; default clears it (KB_WRITE_POLICY_DEFAULT_SURFACE).",
    )

    sv = subparsers.add_parser(
        "serve",
        help="Run the web service with uvicorn (foreground).",
    )
    sv.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)."
    )
    sv.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")

    mt = subparsers.add_parser(
        "metrics",
        help="Experience-loop metrics (reads the service DB and the data DB).",
    )
    mt_sub = mt.add_subparsers(dest="metrics_command", metavar="METRIC")
    rr = mt_sub.add_parser(
        "repeat-rate",
        help="Weekly cross-session repeat-mistake rate from failure_events.",
    )
    rr.add_argument("--weeks", type=int, default=8, help="Weeks to cover (default 8).")
    rr.add_argument("--project", default=None, help="Only failures of this project.")
    rr.add_argument(
        "--min-gap-hours",
        type=float,
        default=24.0,
        help="Minimum hours between a cue's first and a repeat (default 24).",
    )
    rr.add_argument("--json", action="store_true", help="Emit JSON, not markdown.")

    args = parser.parse_args()

    if args.command == "create-admin":
        _run(_create_admin(args.email, args.password))
    elif args.command == "create-user":
        _run(_create_user(args.email, args.password))
    elif args.command == "make-admin":
        _run(_make_admin(args.email))
    elif args.command == "set-machine-principal":
        _run(_set_machine_principal(args.email))
    elif args.command == "list-keys":
        _run(_list_keys(args.email))
    elif args.command == "set-key-surface":
        _run(_set_key_surface(args.key_id, args.surface))
    elif args.command == "serve":
        _serve(args.host, args.port)
    elif args.command == "metrics" and args.metrics_command == "repeat-rate":
        _run(
            _metrics_repeat_rate(
                args.weeks, args.project, args.min_gap_hours, args.json
            )
        )
    elif args.command == "metrics":
        mt.print_help(sys.stderr)
        sys.exit(1)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
