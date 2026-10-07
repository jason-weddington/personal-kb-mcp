# Personal KB Web Service

A hosted FastAPI service that wraps the `kb-core` engine behind an
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
- **kb-01746** — dev-server runbook + aarch64 validation: how to
  run/boot the service against a real KB data DB.
- **kb-01745** — lesson: FastAPI `HTTPBearer` returns 401 (not 403) on missing
  creds in 0.136 — watch for stale exact-status assertions copied from agent_gtd.

Build status: **P1–P5 done, P6 dev done, listener shipped** — full backend + SPA, thin MCP client (HttpBackend + 16 tool shims in the `personal_kb` repo), and the anticipatory listener: `POST /api/kb/listener` (rules A+B over `operated_via` hints + majority-of-3 Sonnet; kill switch `KB_LISTENER_ENABLED`, default OFF) with whisper-next-turn injection in `personal-kb-hook`. Hosted install: see README "Running the hosted service" (`uv tool install 'personal-kb-web-service[postgres]'`, env vars, `kb-service serve`, sample systemd unit); the maintainer's own deploy glue lives in a separate private ops repo. Listener design **kb-01725**. Plan in kb-01742.
A self-healing embedding retry queue + background worker re-embeds entries that previously failed to vectorize, gated by `KB_EMBED_WORKER_ENABLED` (default TRUE) and inspectable via `GET /api/kb/embedding-queue`. Three more knobs tune the worker (`kb_service/config.py`: `KB_EMBED_WORKER_BATCH_SIZE` default 16, `KB_EMBED_WORKER_POLL_SECONDS` default 60.0, `KB_EMBED_WORKER_TIMEOUT` default 180.0) — set them directly in the service's environment (e.g. the systemd `EnvironmentFile`) and restart the service.

## Commands

- Install: `uv sync`
- Tests: `uv run pytest`
- Lint: `uv run ruff check .` / format: `uv run ruff format .`
- Types: `uv run mypy src`
- Run server: `./serve.sh` (uvicorn on 127.0.0.1:8000); with a built `frontend/dist`,
  FastAPI serves the SPA directly (no nginx needed)

### Frontend

- Install: `npm --prefix frontend install`
- Dev server: `npm --prefix frontend run dev` (Vite on port 5173, proxies `/api` to :8000)
- Build: `npm --prefix frontend run build` (produces `frontend/dist`)
- Test: `npm --prefix frontend run test`
- Lint: `npm --prefix frontend run lint`

## Layout

```
frontend/            # Vite + React 19 + MUI 7 SPA
  src/
    api.ts           # typed fetch client (camelCase<->snake_case, kb-01449 401 guard)
    types.ts         # UserResponse, AuthResponse (camelCase client forms)
    utils.ts         # toSnakeCase / toCamelCase / convertKeys
    theme.ts         # dark/light MUI theme pair
    main.tsx         # app entry: StrictMode > BrowserRouter > AuthProvider > ThemeProvider
    App.tsx          # routes: /login, /register (unprotected); ProtectedRoute+Layout
    contexts/
      AuthContext.tsx   # auth state + kb-01449 guards
      ThemeContext.tsx  # dark/light toggle, persists to localStorage 'kb-theme'
    components/
      Layout.tsx        # AppBar + Drawer sidebar (DRAWER_WIDTH=240) + Outlet
      ProtectedRoute.tsx
    pages/
      registry.tsx   # AppPage interface + appPages array (P4b/P4c append here)
      Home.tsx       # placeholder landing page
      Login.tsx      # email/password card, invite-only helper text
      Register.tsx   # reads ?token= invite param; submit disabled without token
    __tests__/       # vitest suites: utils, api 401-guard, AuthContext, ProtectedRoute, Register
src/kb_service/
  main.py            # FastAPI app + lifespan + mount_frontend(app, FRONTEND_DIST)
  auth.py            # JWT + API-key auth, password hashing, invite registration
  database.py        # service-auth DB: asyncpg pool, or SQLite service.db in no-auth mode
  db_sqlite.py       # SQLite DbPool (local mode: unset KB_SERVICE_DATABASE_URL)
  db_types.py        # DbPool Protocol
  models.py          # Pydantic models (auth/admin/invite + Search request/response)
  config.py          # kb-core engine-config adapters (env -> dataclasses)
  cli.py             # `kb-service` admin bootstrap CLI
  routes/
    auth_routes.py   # /api/auth (register, login, me, password, api-keys)
    admin_routes.py  # /api/admin (invites, users, password-reset issue)
    kb_routes.py     # /api/kb/* (authed read/write endpoints)
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

`kb-core` is a uv workspace member (`kb-core = { workspace = true }` in `[tool.uv.sources]`), so the service always builds against the in-tree `packages/kb-core`. When the service is installed from a published package, `kb-core` resolves as an ordinary dependency. Deploy and publish glue for the maintainer's own hosts lives in a separate private ops repo.
