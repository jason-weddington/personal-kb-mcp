"""CLI for the KB service — shell-accessible admin bootstrap.

All commands talk DIRECTLY to the service-auth database (the one pointed at by
``KB_SERVICE_DATABASE_URL``), not through the HTTP API. ``create-admin`` is the
bootstrap for the FIRST admin, since registration is invite-gated. ``create-user``
creates an ordinary non-admin user — e.g. a dedicated machine principal for the
nightly map-maintenance jobs — without over-granting admin's invite/reset/read-all
powers. ``set-machine-principal`` designates which user (by email) is the machine
principal via the ``machine_principal_email`` app_config row (see
``kb_service.attribution.is_machine_principal``); it does not require the user to
already exist.
"""

import argparse
import asyncio
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


def _serve(host: str, port: int) -> None:
    """Run the FastAPI app under uvicorn in the foreground.

    Console-script entry point so a thin client (e.g. the ``personal_kb`` MCP
    server, installed via a git+ssh dependency) can spawn the daemon with
    ``kb-service serve --port <n>`` without needing a uvicorn invocation of its
    own.
    """
    import uvicorn

    uvicorn.run("kb_service.main:app", host=host, port=port)


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

    sv = subparsers.add_parser(
        "serve",
        help="Run the web service with uvicorn (foreground).",
    )
    sv.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)."
    )
    sv.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")

    args = parser.parse_args()

    if args.command == "create-admin":
        _run(_create_admin(args.email, args.password))
    elif args.command == "create-user":
        _run(_create_user(args.email, args.password))
    elif args.command == "make-admin":
        _run(_make_admin(args.email))
    elif args.command == "set-machine-principal":
        _run(_set_machine_principal(args.email))
    elif args.command == "serve":
        _serve(args.host, args.port)
    else:
        parser.print_help(sys.stderr)
        sys.exit(1)
