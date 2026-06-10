# Personal KB Web Service

A hosted home-lab FastAPI service that wraps the `kb-core` engine behind an
authed HTTP API, so the `personal-kb` MCP server can become a thin client.
Modeled on the Agent GTD server.

## Knowledge base — read these first

This repo's KB project is `personal-kb` (see `.kb_project`). Before working here,
`kb_get` the canonical context — the KB holds the *why* and the as-built detail
this file only summarizes:

- **kb-01744** — MVP shape + P1 as-built. **Read this first** — it records the
  current single-KB-per-stack design and supersedes the two-scope/role→DSN design
  in kb-01742.
- **kb-01742** — full architecture + the phased build plan (P1–P6). Authoritative
  for the *plan*, but its scope/topology decisions are superseded by kb-01744.
- **kb-01727** — the `kb-core` engine extraction (the seam this service wraps:
  `from kb_core import KnowledgeBase, create_postgres`).
- **kb-01746** — dev-server runbook (`kb-host-1`) + aarch64 validation: how to
  run/boot the service against the real vm01 KB data DB.
- **kb-01745** — lesson: FastAPI `HTTPBearer` returns 401 (not 403) on missing
  creds in 0.136 — watch for stale exact-status assertions copied from agent_gtd.

Build status: **P1 done** (shell + auth + `/api/kb/search`). Pending: P2 full
`kb_routes` (store/get/ask/summarize/ingest/preflight/maps/graph), P3 explorer,
P4 React/MUI frontend, P5 thin MCP client, P6 deploy/cutover. Plan in kb-01742.

## Commands

- Install: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .` / format: `uv run ruff format .`
- Types: `uv run mypy src`
- Run server: `./serve.sh` (uvicorn on 127.0.0.1:8000)

## Layout

```
src/kb_service/
  main.py            # FastAPI app + lifespan (opens the singleton KnowledgeBase)
  auth.py            # JWT + API-key auth, password hashing, invite registration
  database.py        # service-auth asyncpg pool + schema (4 tables)
  db_types.py        # DbPool Protocol
  models.py          # Pydantic models (auth/admin/invite + Search request/response)
  config.py          # kb-core engine-config adapters (env -> dataclasses)
  cli.py             # `kb-service` admin bootstrap CLI
  routes/
    auth_routes.py   # /api/auth (register, login, me, password, api-keys)
    admin_routes.py  # /api/admin (invites, users, password-reset issue)
    kb_routes.py     # /api/kb/search (the authed read endpoint)
tests/               # hermetic — no live Postgres/Ollama/network
```

## Two-database model

There are TWO distinct Postgres databases behind TWO distinct asyncpg pools:

- **`KB_SERVICE_DATABASE_URL`** — the service/auth DB. Holds `users`,
  `api_keys`, `invites`, `password_resets`. Opened by `database.py::get_db()`.
  The service owns this schema (`init_db()` creates the four tables).
- **`KB_DATABASE_URL`** — the kb-core data DB (the knowledge entries). The
  service adds NOTHING to it; kb-core owns its own schema and pool via
  `create_postgres`. `app.state.kb` is a single `KnowledgeBase` opened from
  this URL. Same DB the personal_kb MCP server reads today.

"Every KB is a team KB" — there is no personal/team branching, no `kb_role`,
no role->DSN map in the backend.

## Admin bootstrap

Registration is invite-gated, so bootstrap the first admin directly:

```bash
uv run kb-service create-admin --email you@example.com --password '...'
# promote an existing user:
uv run kb-service make-admin --email someone@example.com
```

Both talk directly to `KB_SERVICE_DATABASE_URL`.

## kb-core source

For now, `kb-core` is sourced via a LOCAL PATH in `pyproject.toml`
(`[tool.uv.sources] kb-core = { path = "../personal_kb/packages/kb-core" }`).
A git+ssh source for headless dispatch is deferred to a later phase.
