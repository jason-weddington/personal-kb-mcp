# Personal KB Web Service

A hosted home-lab FastAPI service that wraps the `kb-core` engine behind an
authed HTTP API, so the `personal-kb` MCP server can become a thin client.
Modeled on the Agent GTD server.

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
