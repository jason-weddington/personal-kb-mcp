# personal-kb-web-service

Hosted home-lab service for the Personal KB. Wraps the **`kb-core`** engine
(hybrid search + knowledge graph + agentic retrieval) behind an authenticated
HTTP API and a React SPA. The `personal-kb` MCP server is a **thin client** of
this service, and the **anticipatory listener** pushes knowledge pointers into
agent sessions.

## What it is

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
  map entries), unanimous-3 Sonnet retrieve-and-cite. The `personal-kb-hook`
  fans this call out across a **roster of KBs** (personal + team), unions the
  per-KB winners via a client-side suppress-only arbitration step, and whispers
  one pointer per KB into the next agent turn. Single-KB Gate-1 validated at
  1.00 precision / 0 false injections; the multi-KB go-live gate
  (`evals/listener/run_p3.py`) gates on **0 union false-injections** (precision
  informational) and **passed live 2026-06-15** — multi-KB whisper is live
  (kb-01839).

## Topology

One **stack** = one KB: a Postgres data DB (kb-core owns it, `KB_DATABASE_URL`)
plus a service DB for auth/app-config (`KB_SERVICE_DATABASE_URL`). The personal
and team KBs are two deployments of identical code pointed at different DBs.

## Pi deployment

Three Pis run the hosted service as the `kb-service` systemd system unit: `kb-host-1` (personal KB), `kb-host-2` (team KB) and `kb-host-3` (user2's KB). Each is fronted by Caddy and also answers plain HTTP on `:8000`. Deploy from a dev machine with `KB_DEPLOY_HOST=<host> ./deploy.sh`; provision a new host with `KB_MODE=hosted scripts/provision.sh`, and `KB_MODE=upgrade scripts/provision.sh` is the code-only equivalent of `deploy.sh`. Runbooks: dev server **kb-01746**, production cutover **kb-01765**, per-machine hook/listener setup **kb-01784** (their host-path details predate the layout below).

Since the 2026-10-06 repo merge the service lives in the `personal_kb` monorepo, and a host is laid out like this: the checkout is `~/git/personal_kb` (repo `git@$KB_GIT_HOST:repos/personal_kb`); the venv is at the workspace root, `~/git/personal_kb/.venv`, installed with only what the service needs (`uv sync --frozen --package personal-kb-web-service --extra postgres --no-dev`); the unit's `WorkingDirectory` is `~/git/personal_kb/packages/kb-service`, so relative paths such as `frontend/dist` and `.env` resolve as before, and `ExecStart` is `~/git/personal_kb/.venv/bin/uvicorn kb_service.main:app`. The SPA is still built on the host into `packages/kb-service/frontend/dist`, which is gitignored and wins over the packaged static UI. Secrets stay in `/etc/kb-service/env`.

All host-side work is `scripts/host-deploy.sh`, which `deploy.sh` and `provision.sh` pipe over ssh (`ssh host 'bash -s deploy' < scripts/host-deploy.sh`), so both share one implementation. Its subcommands are `deploy` (pull, sync, build, restart, health-check 200 on `127.0.0.1:8000`), `prepare` (the same minus the restart, used by hosted provisioning before it writes the env file) and `install-unit` (writes a fresh unit for the new layout).

**Old-layout migration.** A host that still has the pre-merge `~/git/personal-kb-web-service` checkout is migrated in place the first time `deploy.sh` or `provision.sh` (hosted or upgrade) runs against it, with no extra flags. The script stops kb-service, then `pg_dump`s the data DB (`KB_DATABASE_URL`) and the service DB (`KB_SERVICE_DATABASE_URL`) to `~/backups/<db>-<timestamp>.sql`, reading the DSNs from the unit's `EnvironmentFile` and the old `.env`. The password is passed in `PGPASSWORD`, never on argv, and no DSN is printed. If `pg_dump` is not installed the backup is skipped with a loud message; if it fails, the migration aborts and the old service is restarted untouched (`KB_MIGRATE_SKIP_BACKUP=1` proceeds without a backup). It then clones `personal_kb`, copies the old checkout's `.env` into the new WorkingDirectory (an existing different file is kept as `.env.bak-<timestamp>`), saves the old unit to `~/backups/kb-service.service.pre-merge-<timestamp>`, rewrites the unit's paths, runs `daemon-reload`, and renames the old checkout to `~/git/personal-kb-web-service.pre-merge-<date>`. It is never deleted. The rest of the run is a normal deploy. Re-running on a migrated host is a plain deploy with no migration steps and no new backup. Postgres data is never modified.

To roll back a migration, copy the saved unit back over `/etc/systemd/system/kb-service.service`, rename the `.pre-merge-<date>` checkout back to `~/git/personal-kb-web-service`, then `sudo systemctl daemon-reload && sudo systemctl restart kb-service`.

## Deploy preflight guard

`./deploy.sh` runs `scripts/deploy-preflight.sh` before it touches anything, and every host-touching step (ssh, git pull, frontend build, `systemctl restart kb-service`) sits behind it — the script can also be run on its own to check the deploy window without deploying.
It queries `agent-gtd list-runs --status pending,running` for in-flight dispatch runs (reporting count, run ids and item titles) and enumerates `/run/user/$UID/cc-socks/*.sock` for live peer Claude sessions, excluding this session's own socket when identifiable.
The prompt came out of the harness-design session of 2026-09-19: the rule "do not bounce kb-service while work is in flight" was operator knowledge living in steering docs, and the agent triggering a deploy at 2am will be a different session with none of that context.

kb-service serves the personal-kb hooks that fire on SessionStart, UserPromptSubmit and Stop for every session on this machine, including every dispatch run, so bouncing it mid-flight degrades map injection and whisper telemetry for anything crossing the window.
Hook timeouts are capped at 3s and hooks fail soft, so this is degradation, not breakage — the guard refuses with that stated plainly rather than with a generic warning.
When either check is non-zero or unanswerable (CLI missing or unauthenticated, socket directory unreadable — reported as UNKNOWN, because an unknown is not a safe answer), the preflight exits non-zero and `deploy.sh` aborts before pulling, building or restarting.

To override the guard deliberately, pass `--ack` (or export `KB_DEPLOY_PREFLIGHT_ACK=1`) after notifying the listed peers or deciding to accept the degradation, and the acknowledgement is echoed into the deploy output so the log records that a human or agent bypassed the guard knowingly.
The override is deliberately loud for the same reason the guard exists: a silent bypass is just operator memory again, one context failure away from being nobody's knowledge.

## Installing somnus on the KB hosts

`somnus`, the nightly map-maintenance loop binary (see
`docs/nightly-map-maintenance-design.md`), is delivered to the KB hosts by `scripts/install-somnus.sh`.
It resolves the target token by fetching `latest` from the artifact host (artifact-host's Caddy, default `https://artifacts.lab.example.com`), verifies the artifact exists at that token, and installs it on each of `kb-host-1`, `kb-host-2` and `kb-host-3` (overridable via `KB_HOSTS`), skipping hosts already current.

Run it as `./scripts/install-somnus.sh` — or pin a token with `--version <TOKEN>` and retarget with `KB_HOSTS="kb-host-2"`.
It is idempotent and safe on a fresh host, attempts every host even when one is powered off, prints a per-host summary (installed / skipped-already-current / unreachable / failed), and exits non-zero unless every host ended at the target token.
It restarts nothing: somnus is a timer-invoked subprocess, so there is no daemon to bounce and the next timer fire picks up the new binary.
On a failed post-install version check it rolls back to the previous binary rather than leaving a broken one, because the run is unattended.

This path is deliberately independent of the dispatch fleet's installer (`agent-gtd-dispatch/talos-update.sh`, same artifact host, different session).
Per Jason's ruling of 2026-09-19, every session owns its own release tooling, so the KB hosts and the dispatch fleet can legitimately run different tokens at different times — nothing may assume fleet-wide version uniformity, and changing one installer must not force a change on the other.

## Development

See `CLAUDE.md` for commands, layout, and the KB entries that hold the
architecture (`kb-01744`, `kb-01742`) and listener design (`kb-01725`).
