# personal-kb-web-service

Hosted service for the Personal KB. Wraps the **`kb-core`** engine
(hybrid search + knowledge graph + agentic retrieval) behind an authenticated
HTTP API and a React SPA. The `personal-kb` MCP server is a **thin client** of
this service, and the **anticipatory listener** pushes knowledge pointers into
agent sessions.

## What it is

- **MCP over streamable HTTP** at `/mcp`: the same MCP tool set as the stdio `personal-kb` server (except `kb_ingest`), stateless, authenticated with the same bearer API keys as `/api/*`. Tools call the route handlers in-process. Connect with `claude mcp add --transport http personal-kb https://<host>/mcp --header "Authorization: Bearer <key>"`.
- **FastAPI service** over a singleton `kb_core.KnowledgeBase`: the full KB
  surface under `/api/kb/*` (search/get/store/ask/summarize/ingest/preflight/
  graph/maps-index), SSE-streamed queries and chat, JWT-or-API-key auth with
  invite-gated registration and an admin pane.
- **React 19 / MUI 7 SPA** (`frontend/`): login, search, entry detail,
  force-graph visualization, streaming Ask, chat, settings (API-key minting
  with MCP-config snippet), admin (invites/users).
- **Anticipatory listener** (`POST /api/kb/listener`): maps-only cross-project
  relevance engine — rule A (drop the session project's own maps), rule B
  (drop maps for systems the agent is operating, via `operated_via` hints on
  map entries), majority-of-3 Sonnet retrieve-and-cite. The `personal-kb-hook`
  fans this call out across a **roster of KBs** (personal + team), unions the
  per-KB winners via a client-side suppress-only arbitration step, and whispers
  one pointer per KB into the next agent turn. Multi-KB whisper is gated on
  **0 union false-injections**; the evaluation harnesses are maintained outside
  this repo.
- **Surprise-capture audit**: `GET /api/kb/surprise/candidates` lists recent surprise candidates (filters `since`, `status`, `shape`, `project`, `limit`) with each one's shadow dry run (what mode `on` WOULD have written or merged) and its session mode and host, so an agent can audit shadow output before capture is switched on.
- **Experience-loop metric**: `GET /api/kb/metrics/repeat-rate` and `kb-service metrics repeat-rate [--json]` report the weekly cross-session repeat-mistake rate from `failure_events`, cut by harness, mode, engine and host, with resolution coverage.

## Topology

One **stack** = one KB: a Postgres data DB (kb-core owns it, `KB_DATABASE_URL`)
plus a service DB for auth/app-config (`KB_SERVICE_DATABASE_URL`). The personal
and team KBs are two deployments of identical code pointed at different DBs.

## Running the hosted service

Install the service from PyPI-style metadata with the Postgres extra:

```
uv tool install 'personal-kb-web-service[postgres]'
```

You need two Postgres databases (the kb-core data DB and the service's own auth/app-config DB), with the `pgvector` extension available in the data DB, and an Ollama endpoint for embeddings.
Configure the service with environment variables (see `.env.example` for the full list):

- `JWT_SECRET` — long random string used to sign sessions.
- `KB_SERVICE_DATABASE_URL` — DSN of the service's auth/app-config database.
- `KB_DATABASE_URL` — DSN of the kb-core data database.
- Setting `KB_DB_PATH` together with `KB_DATABASE_URL` or `KB_SERVICE_DATABASE_URL` is a startup error; unset one.
- `KB_SERVICE_PUBLIC_URL` — public base URL, used in invite and password-reset links.
- `KB_WRITE_POLICY_DEFAULT_SURFACE` — write-policy surface for API keys with none set and for no-auth callers: `interactive` (default), `headless` or `autonomous` (an unknown value means headless). Headless and autonomous `kb_store` creates are queued as candidates; updates, deactivations and ingests from those surfaces are refused. Set per-key surfaces with `kb-service set-key-surface` before setting this to `headless`; see the root README's write-policy section for the rollout order.
- `KB_OLLAMA_URL`, `KB_EMBEDDING_MODEL`, `KB_EMBEDDING_DIM` — embeddings.
- `KB_LOG_LEVEL` — log level for the `kb_service`/`kb_core` loggers (default `INFO`; unknown values fall back to `INFO` with a warning). Third-party libraries stay at `WARNING`. Logs go to stderr (the journal under systemd).
- `ANTHROPIC_API_KEY` (or the Bedrock/Ollama provider settings) — enrichment, planning and synthesis.
- `KB_SERVICE_CLIENT_INSTALL_SPEC` — the `uvx --from` spec shown in the Settings page's MCP snippet; defaults to `personal-kb @ git+https://github.com/jason-weddington/personal-kb-mcp`.

Create the first admin, then serve:

```
kb-service create-admin --email you@example.com
kb-service serve --host 127.0.0.1 --port 8000
```

`kb-service serve` binds `127.0.0.1:8000` by default; put a reverse proxy (Caddy, nginx) in front of it for TLS.
The packaged SPA is served by the same process.

A sample systemd unit:

```ini
[Unit]
Description=Personal KB web service
After=network-online.target postgresql.service
Wants=network-online.target

[Service]
User=kb-service
EnvironmentFile=/etc/kb-service/env
ExecStart=/usr/local/bin/kb-service serve --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Keep secrets in the `EnvironmentFile` (mode 0600), not in the unit.
Upgrade with `uv tool upgrade personal-kb-web-service` followed by `systemctl restart kb-service`.
The maintainer's own deploy and publish glue lives in a separate private ops repo and is not part of this tree.

## Development

See `CLAUDE.md` for commands, layout, and the KB entries that hold the
architecture (`kb-01744`, `kb-01742`) and listener design (`kb-01725`).
