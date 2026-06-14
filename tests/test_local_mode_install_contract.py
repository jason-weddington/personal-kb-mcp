"""Pin the documented local-mode install contract from the README.

Both halves of the kb-01789 audit (run wf_de20a229) — "kb-service not on
PATH" and "README has no PERSONAL_KB_URL" — slipped past 1047 green
tests because NOTHING exercised the documented onboarding path. The
existing tests cover units (config parsers, daemon-spawn argv with
mocked subprocess, backend selection); none of them parse the README
local-mode block or inspect the ``[local]`` extra. So a doc/dependency
regression that breaks the very first thing a new user runs passes the
whole suite.

This module fills that gap with cheap parse-only checks that fail when
the doc/reality contract regresses:

A. **README** — every ``mcpServers`` JSON snippet whose ``args`` use the
   ``[local]`` extra MUST set both ``PERSONAL_KB_URL`` and
   ``PERSONAL_KB_API_KEY`` in its ``env`` block. (PERSONAL_KB_URL is the
   loopback URL the MCP server hits and the source of the spawn port;
   PERSONAL_KB_API_KEY is the local-no-auth sentinel — empty/unset
   silently disables the personal-kb-hook.) A README block that omits
   either env var is broken onboarding, full stop.

B. **pyproject** — the ``[local]`` extra MUST pull in
   ``personal-kb-web-service``, the package that ships the ``kb-service``
   console script. Removing or renaming it puts ``kb-service`` off PATH
   for every ``uvx --from '...[local]' personal-kb`` install and makes
   ``ensure_daemon`` raise ``RuntimeError: kb-service not on PATH`` on
   the first MCP session.

C. **Daemon ↔ web-service binary name** — the spawn argv pinned in
   ``personal_kb.daemon._SPAWN_CMD`` must stay ``kb-service`` (the
   ``[project.scripts]`` entry the web-service publishes). If either
   side gets renamed without the other, the documented install is a
   silent breakage we'd only catch by hand-running the README.

These checks have ZERO runtime cost (no spawn, no network, no HTTP
client) so the regular CI gate runs them. The optional "boot the daemon
on loopback and hit /api/health" smoke lives in
``scripts/smoke_install.sh`` (release-time / sandbox-with-uvx, gated on
the binary being installed).
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

import personal_kb.daemon as daemon

# ---------------------------------------------------------------------------
# Paths — anchored at the repo root so the tests work regardless of cwd.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_JSON_FENCE_RE = re.compile(r"```(?:json|jsonc)\s*\n(.*?)\n```", re.DOTALL)


def _strip_jsonc_comments(text: str) -> str:
    """Drop ``//`` line comments so ``jsonc`` fences parse with ``json``.

    The README uses a few ``jsonc`` fences; everywhere else we want strict
    ``json.loads`` so a malformed example trips the test. The stripper
    walks each line as a tiny state machine so it does NOT cut the
    ``//`` inside a string literal (every README example contains
    ``https://...`` URLs, which a naïve ``line.find('//')`` would
    decapitate).
    """
    out_lines: list[str] = []
    for line in text.splitlines():
        in_string = False
        escape = False
        cut: int | None = None
        for idx, ch in enumerate(line):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if not in_string and ch == "/" and idx + 1 < len(line) and line[idx + 1] == "/":
                cut = idx
                break
        out_lines.append(line if cut is None else line[:cut].rstrip())
    return "\n".join(out_lines)


def _iter_readme_mcp_blocks() -> list[tuple[int, dict[str, object]]]:
    """Return ``(fence_line, parsed_json)`` for every ``mcpServers`` example.

    Only fenced ``json`` / ``jsonc`` blocks that successfully parse AND
    contain a top-level ``mcpServers`` key qualify. Other JSON blocks
    (request payloads, etc.) are ignored — they're not user-facing
    install configs.
    """
    text = README_PATH.read_text(encoding="utf-8")
    blocks: list[tuple[int, dict[str, object]]] = []
    for match in _JSON_FENCE_RE.finditer(text):
        body = _strip_jsonc_comments(match.group(1))
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, dict) or "mcpServers" not in parsed:
            continue
        fence_line = text.count("\n", 0, match.start()) + 1
        blocks.append((fence_line, parsed))
    return blocks


def _server_uses_local_extra(server: dict[str, object]) -> bool:
    """Return True if *server*'s ``args`` reference the ``[local]`` extra.

    Matches anything of the shape ``...[local]`` or ``...[local,...]`` —
    e.g. ``git+https://...personal-kb-mcp.git[local]`` or
    ``git+https://...personal-kb-mcp.git[local,aws]``. The bracket
    contents may include additional extras (``local,aws``).
    """
    args = server.get("args")
    if not isinstance(args, list):
        return False
    pat = re.compile(r"\[([^\]]*)\]")
    for raw in args:
        if not isinstance(raw, str):
            continue
        for extras_group in pat.findall(raw):
            extras = {e.strip() for e in extras_group.split(",") if e.strip()}
            if "local" in extras:
                return True
    return False


def _iter_local_server_examples() -> list[tuple[int, str, dict[str, object]]]:
    """Return ``(line, server_name, server_dict)`` for every local-mode block."""
    out: list[tuple[int, str, dict[str, object]]] = []
    for line, block in _iter_readme_mcp_blocks():
        servers = block.get("mcpServers", {})
        if not isinstance(servers, dict):
            continue
        for name, server in servers.items():
            if isinstance(server, dict) and _server_uses_local_extra(server):
                out.append((line, name, server))
    return out


# ---------------------------------------------------------------------------
# A. README — every local-mode mcpServers block sets the required env vars
# ---------------------------------------------------------------------------


# Pinned by README "How local mode works (read this first)":
#   PERSONAL_KB_URL → loopback URL the MCP server hits + source of the
#     spawn port (kb-service serve --port <parsed from URL>).
#   PERSONAL_KB_API_KEY → ``local-no-auth`` sentinel. Empty / unset
#     silently disables the personal-kb-hook.
# A regression that drops either env var is the exact failure shape
# kb-01789 found.
_REQUIRED_LOCAL_ENV_VARS = ("PERSONAL_KB_URL", "PERSONAL_KB_API_KEY")


def test_readme_has_at_least_one_local_mode_example() -> None:
    """Guard the guard: if nobody documents local mode anymore, fail loud.

    Without this, deleting every ``[local]`` example from the README
    would also delete the contract this file pins, and the other tests
    in this module would silently pass on zero examples.
    """
    examples = _iter_local_server_examples()
    assert examples, (
        "README.md no longer contains any mcpServers JSON example using the "
        "[local] extra. Either restore the local-mode quick-start blocks or "
        "delete this whole contract test if local mode has been retired."
    )


@pytest.mark.parametrize("env_var", _REQUIRED_LOCAL_ENV_VARS)
def test_readme_local_examples_set_required_env_var(env_var: str) -> None:
    """Every local-mode README block sets *env_var* in its ``env`` map."""
    failures: list[str] = []
    for line, name, server in _iter_local_server_examples():
        env = server.get("env")
        if not isinstance(env, dict) or env_var not in env:
            failures.append(f"  - line {line}: mcpServers[{name!r}] missing {env_var}")
    assert not failures, (
        f"README local-mode mcpServers block(s) missing required env var {env_var!r} "
        "(the loopback URL the MCP server hits / source of the spawn port; or the "
        "local-no-auth sentinel keeping the hook wired). This is the regression "
        "kb-01789 caught — every [local] install needs both vars set:\n" + "\n".join(failures)
    )


def test_readme_local_url_points_at_loopback_with_explicit_port() -> None:
    """The documented ``PERSONAL_KB_URL`` is loopback AND has the spawn port.

    The daemon spawn parses the port directly out of ``PERSONAL_KB_URL``
    (``personal_kb.daemon.parse_port``) — a missing port raises at MCP
    startup. And a non-loopback host short-circuits ``ensure_daemon``
    entirely (no spawn), which silently breaks local mode. Both shapes
    must hold for every documented local block.
    """
    failures: list[str] = []
    for line, name, server in _iter_local_server_examples():
        env = server.get("env") if isinstance(server, dict) else None
        url = env.get("PERSONAL_KB_URL") if isinstance(env, dict) else None
        if not isinstance(url, str) or not url:
            # The required-env-var test already covers missing-url; here we
            # only check the shape of values that ARE present.
            continue
        if not daemon.is_loopback_url(url):
            failures.append(
                f"  - line {line}: mcpServers[{name!r}] PERSONAL_KB_URL={url!r} is "
                "not loopback — daemon spawn will be skipped (ensure_daemon "
                "short-circuits on non-loopback hosts)."
            )
            continue
        try:
            daemon.parse_port(url)
        except ValueError as exc:
            failures.append(
                f"  - line {line}: mcpServers[{name!r}] PERSONAL_KB_URL={url!r} "
                f"has no port ({exc}) — daemon spawn requires an explicit port."
            )
    assert not failures, "\n".join(
        ["README local-mode PERSONAL_KB_URL values violate the spawn contract:", *failures]
    )


# ---------------------------------------------------------------------------
# B. pyproject — [local] extra installs the package shipping kb-service
# ---------------------------------------------------------------------------


# Pinned: the package that publishes the ``kb-service`` console script
# (see personal-kb-web-service/pyproject.toml [project.scripts]).
_WEB_SERVICE_DIST = "personal-kb-web-service"


def _load_pyproject() -> dict[str, object]:
    return tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))


def _normalize_dist_name(spec: str) -> str:
    """Return the distribution name from a PEP 508 requirement string."""
    # Strip extras (``foo[bar]``), version specifiers, env markers.
    cut = re.split(r"[\[;<>=!~ ]", spec, maxsplit=1)[0]
    return cut.strip().lower()


def test_pyproject_local_extra_pulls_in_web_service() -> None:
    """``[project.optional-dependencies.local]`` lists ``personal-kb-web-service``.

    This is the second half of kb-01789: without this dep the
    ``kb-service`` console script does not land on PATH for the venv
    ``uvx`` builds for ``personal-kb[local]``, and ``ensure_daemon``
    raises ``RuntimeError: kb-service not on PATH`` on the first MCP
    lifespan.
    """
    pyproject = _load_pyproject()
    project = pyproject.get("project", {})
    assert isinstance(project, dict), "[project] table is missing or malformed"
    optional = project.get("optional-dependencies", {})
    assert isinstance(optional, dict), (
        "[project.optional-dependencies] is malformed (expected a table)"
    )
    assert "local" in optional, (
        "[project.optional-dependencies.local] is missing — the documented "
        "`uvx --from '...[local]' personal-kb` install will fail PEP 508 "
        "parsing (`Extra 'local' is not defined`)."
    )
    local_deps = optional["local"]
    assert isinstance(local_deps, list) and local_deps, (
        "[project.optional-dependencies.local] is empty — the [local] extra "
        "must pull in personal-kb-web-service so the `kb-service` console "
        "script lands on PATH for the documented install."
    )
    dist_names = {_normalize_dist_name(d) for d in local_deps}
    assert _WEB_SERVICE_DIST in dist_names, (
        f"[project.optional-dependencies.local] must include {_WEB_SERVICE_DIST!r} "
        "(the package that ships the `kb-service` console script the MCP daemon "
        f"spawn invokes). Found: {sorted(dist_names)}."
    )


# ---------------------------------------------------------------------------
# C. Daemon ↔ web-service binary name agreement
# ---------------------------------------------------------------------------


def test_daemon_spawn_command_is_kb_service() -> None:
    """The spawn argv keeps invoking the ``kb-service`` console script.

    The ``[local]`` extra installs ``personal-kb-web-service``, whose
    ``[project.scripts]`` publishes ``kb-service = kb_service.cli:main``.
    If we rename ``_SPAWN_CMD`` in ``personal_kb.daemon`` without
    coordinating with the web-service entry point — or vice versa — the
    install still succeeds but the first MCP session blows up. Pinning
    the spawn-side name here makes a uni-lateral rename loud.
    """
    assert daemon._SPAWN_CMD == "kb-service", (
        f"personal_kb.daemon._SPAWN_CMD changed to {daemon._SPAWN_CMD!r}; the "
        "documented [local] install bundles personal-kb-web-service, whose "
        "[project.scripts] publishes `kb-service`. Update both sides together "
        "or this is broken onboarding."
    )


def test_readme_documents_kb_service_binary_name() -> None:
    """The README's local-mode prose references the exact spawn binary.

    The README's "How local mode works" section calls the daemon
    binary out by name (``kb-service serve --port <port>``). If we ever
    rename the spawn cmd in ``personal_kb.daemon`` (or the web-service
    script), the prose has to move with it — otherwise the README
    documents a binary that nobody ships.
    """
    text = README_PATH.read_text(encoding="utf-8")
    assert daemon._SPAWN_CMD in text, (
        f"README.md never mentions the daemon binary {daemon._SPAWN_CMD!r} — "
        "the local-mode walkthrough has drifted from the spawn argv pinned "
        "in personal_kb.daemon."
    )


# ---------------------------------------------------------------------------
# D. README local-mode blocks never set KB_DATABASE_URL
# ---------------------------------------------------------------------------


def test_readme_local_examples_never_set_kb_database_url() -> None:
    """Every local-mode README block leaves ``KB_DATABASE_URL`` unset.

    The documented local install relies on the web-service's SQLite
    default (kb-service opens ``~/.local/share/personal_kb/knowledge.db``
    when ``KB_DATABASE_URL`` is unset/empty). A local-mode block that
    set ``KB_DATABASE_URL`` would silently flip the daemon to Postgres
    and break onboarding — the user would need a running Postgres they
    were never told to provision.

    Vacuous pass on zero examples is already guarded by
    ``test_readme_has_at_least_one_local_mode_example`` above; no
    redundant guard here.
    """
    failures: list[str] = []
    for line, name, server in _iter_local_server_examples():
        env = server.get("env") or {}
        if not isinstance(env, dict):
            continue
        if "KB_DATABASE_URL" in env:
            failures.append(
                f"  - line {line}: mcpServers[{name!r}] sets KB_DATABASE_URL — "
                "local-mode blocks must rely on the SQLite default."
            )
    assert not failures, (
        "README local-mode mcpServers block(s) set KB_DATABASE_URL — the "
        "documented local install must rely on the SQLite default so users "
        "don't need a running Postgres they were never told to provision:\n" + "\n".join(failures)
    )
