"""Docs drift guard: routes and environment variables must be documented.

``how_it_works.md`` covers the core KB and ``docs/intelligent-kb.md`` covers the
automation layer. These tests fail when code adds a REST route or reads an
environment variable that neither doc mentions, so the docs cannot silently
fall behind the code.

- Every route registered on the kb-service app, with every method it serves
  (HEAD excluded), must have a row in the route table of section 21 of
  ``how_it_works.md``; every mount must appear in that section.
- Every environment variable the shipped packages read must appear, as a whole
  word, in one of the two docs. A read is a string-literal first argument to
  ``os.environ.get``, ``os.getenv``, ``os.environ.setdefault`` or
  ``os.environ[...]``, plus any whole string literal shaped like ``KB_*`` or
  ``PERSONAL_KB_*`` (these reach ``os.environ`` through helpers and named
  constants, which the call-site check cannot see).
"""

from __future__ import annotations

import ast
import asyncio
import importlib
import re
import typing
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pytest

REPO = Path(__file__).resolve().parent.parent
CORE_DOC = REPO / "how_it_works.md"
AUTOMATION_DOC = REPO / "docs" / "intelligent-kb.md"

SOURCE_ROOTS = (
    REPO / "src",
    REPO / "packages" / "kb-core" / "src",
    REPO / "packages" / "kb-service" / "src",
    REPO / "packages" / "personal-kb-hook" / "src",
)

_ENV_SHAPED = re.compile(r"^(?:KB|PERSONAL_KB)_[A-Z0-9_]*[A-Z0-9]$")
_ROUTE_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*([^|]+)\|")
_READ_CALLS = frozenset({"get", "getenv", "setdefault"})


def _route_section() -> str:
    text = CORE_DOC.read_text()
    start = text.find("\n## 21. REST route inventory")
    end = text.find("\n## 22.", start)
    assert start != -1 and end != -1, "section 21 (REST route inventory) not found"
    return text[start:end]


def _documented_routes(section: str) -> dict[str, set[str]]:
    rows: dict[str, set[str]] = {}
    for line in section.splitlines():
        m = _ROUTE_ROW.match(line)
        if m:
            methods = {part.strip() for part in m.group(2).split(",")}
            rows.setdefault(m.group(1), set()).update(methods)
    return rows


def test_every_route_has_a_row_in_the_route_table() -> None:
    """Each (path, method) the app serves appears in section 21's table."""
    from fastapi.routing import APIRoute
    from kb_service.main import app
    from starlette.routing import Mount, Route

    section = _route_section()
    rows = _documented_routes(section)
    assert len(rows) > 20, "route table parsed to almost nothing; did its format change?"

    missing: list[str] = []
    for route in app.routes:
        if isinstance(route, Mount):
            if f"`{route.path}`" not in section:
                missing.append(f"MOUNT {route.path}")
            continue
        if not isinstance(route, APIRoute | Route):
            missing.append(f"{type(route).__name__} {getattr(route, 'path', route)!r}")
            continue
        for method in sorted((route.methods or set()) - {"HEAD"}):
            if method not in rows.get(route.path, set()):
                missing.append(f"{method} {route.path}")

    assert not missing, (
        "Routes missing from the how_it_works.md section 21 table: "
        f"{missing}. Add a row (path, method, module, auth, doc section)."
    )


def _is_os_environ(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "environ"
        and isinstance(node.value, ast.Name)
        and node.value.id == "os"
    )


def _env_vars_read() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}

    def add(name: str, path: Path) -> None:
        found.setdefault(name, set()).add(str(path.relative_to(REPO)))

    for root in SOURCE_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            fstring_parts = {
                id(value)
                for node in ast.walk(tree)
                if isinstance(node, ast.JoinedStr)
                for value in node.values
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and node.args:
                    func = node.func
                    is_read = isinstance(func, ast.Attribute) and (
                        (func.attr in _READ_CALLS and _is_os_environ(func.value))
                        or (
                            func.attr == "getenv"
                            and isinstance(func.value, ast.Name)
                            and func.value.id == "os"
                        )
                    )
                    first = node.args[0]
                    if is_read and isinstance(first, ast.Constant) and isinstance(first.value, str):
                        add(first.value, path)
                elif isinstance(node, ast.Subscript) and _is_os_environ(node.value):
                    key = node.slice
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        add(key.value, path)
                elif (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in fstring_parts
                    and _ENV_SHAPED.match(node.value)
                ):
                    add(node.value, path)
    return found


def test_every_env_var_read_is_documented() -> None:
    """Each environment variable the shipped code reads appears in a doc."""
    docs = CORE_DOC.read_text() + "\n" + AUTOMATION_DOC.read_text()
    read = _env_vars_read()
    assert len(read) > 20, "found almost no env var reads; did the scan break?"

    undocumented = {
        name: sorted(paths)
        for name, paths in sorted(read.items())
        if not re.search(rf"(?<![A-Z0-9_]){re.escape(name)}(?![A-Z0-9_])", docs)
    }
    assert not undocumented, (
        "Environment variables read by the code but absent from how_it_works.md "
        f"and docs/intelligent-kb.md: {undocumented}. Document each one "
        "(the configuration reference in how_it_works.md section 27, or the "
        "automation doc's own table)."
    )


# --- B1/B2: env var ownership and the inverse check ----------------------------

README = REPO / "README.md"
CLAUDE_MD = REPO / "CLAUDE.md"

# Built with an f-string in surprise_worker, so the literal scan cannot see them.
COMPOSED_ENV_VARS = {"KB_SURPRISE_MIN_CONFIDENCE_SHAPE2", "KB_SURPRISE_MIN_CONFIDENCE_SHAPE3"}
# Read by the shipped code but not shaped KB_* / PERSONAL_KB_*.
EXTRA_ENV_VARS = {
    "ANTHROPIC_API_KEY",
    "JWT_SECRET",
    "HEADLESS_BUILD_ENGINE",
    "AWS_BEARER_TOKEN_BEDROCK",
}
AUTOMATION_ENV_PREFIXES = (
    "KB_SURPRISE_",
    "KB_SOFT_GATE_",
    "KB_LISTENER_",
    "PERSONAL_KB_LISTENER",
    "KB_FAILURE_CONTEXT",
    "KB_GOTCHA_SLICE",
    "KB_DELIVER_OBSERVED_ONCE",
    "KB_ROSTER_OTHER_DOMAINS",
    "KB_TOOL_",
    "KB_WRITE_POLICY_",
)

_ENV_IN_CELL = re.compile(r"`((?:KB|PERSONAL_KB)_[A-Z0-9_]*[A-Z0-9])`")


def _table_rows(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("|")]


def _row_env_names(text: str) -> set[str]:
    """Env names in the first cell of every table row (a row may list several)."""
    names: set[str] = set()
    for row in _table_rows(text):
        first_cell = row.split("|")[1] if row.count("|") >= 2 else ""
        names.update(_ENV_IN_CELL.findall(first_cell))
    return names


def _owner_doc(name: str) -> Path:
    return AUTOMATION_DOC if name.startswith(AUTOMATION_ENV_PREFIXES) else CORE_DOC


def _all_env_names() -> set[str]:
    return {n for n in _env_vars_read() if _ENV_SHAPED.match(n)} | COMPOSED_ENV_VARS


def test_env_vars_documented_in_owning_doc() -> None:
    """Each KB_*/PERSONAL_KB_* name has a table row in the doc that owns it."""
    names = _all_env_names() | EXTRA_ENV_VARS
    assert len(names) > 60, "env var scan found suspiciously few names"
    rows = {doc: _row_env_names(doc.read_text()) for doc in (CORE_DOC, AUTOMATION_DOC)}
    rows_text = {doc: "\n".join(_table_rows(doc.read_text())) for doc in (CORE_DOC, AUTOMATION_DOC)}
    missing: list[str] = []
    for name in sorted(names):
        doc = _owner_doc(name)
        if _ENV_SHAPED.match(name):
            ok = name in rows[doc]
        else:
            ok = re.search(rf"(?<![A-Z0-9_]){name}(?![A-Z0-9_])", rows_text[doc]) is not None
        if not ok:
            missing.append(f"{name} -> {doc.relative_to(REPO)}")
    assert not missing, (
        "Environment variables without a table row in their owning doc: "
        f"{missing}. Add a row to the configuration table there."
    )


def test_documented_env_vars_exist() -> None:
    """Every KB_*/PERSONAL_KB_* name in a doc or README table row is read by the code."""
    real = _all_env_names()
    stale: dict[str, list[str]] = {}
    for doc in (CORE_DOC, AUTOMATION_DOC, README):
        for name in sorted(_row_env_names(doc.read_text()) - real):
            stale.setdefault(name, []).append(doc.name)
    assert not stale, (
        f"Table rows name env vars no shipped code reads: {stale}. "
        "Remove the rows (or the variable was renamed: update them)."
    )


# --- B3-B8: names the code defines must be mentioned ---------------------------


def _word_in(name: str, text: str) -> bool:
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


def test_hook_events_documented() -> None:
    from personal_kb_hook.cli import _SUPPORTED_EVENTS

    rows = "\n".join(_table_rows(AUTOMATION_DOC.read_text()))
    missing = sorted(e for e in _SUPPORTED_EVENTS if f"| `{e}` |" not in rows)
    assert len(_SUPPORTED_EVENTS) >= 6
    assert not missing, (
        f"Hook events without a row in the section 3 table of docs/intelligent-kb.md: {missing}"
    )


def test_mcp_tools_documented(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KB_MANAGER", "TRUE")
    monkeypatch.setenv("KB_CONTRIBUTOR", "x")
    monkeypatch.delenv("KB_INSTANCE_ROLE", raising=False)
    from kb_service.mcp_server.server import create_mcp_server

    from personal_kb.server import create_server

    async def _names() -> set[str]:
        service = {t.name for t in await create_mcp_server().list_tools()}
        stdio = {t.name for t in await create_server().list_tools()}
        return service | ({"kb_ingest"} & stdio)

    names = {n.removeprefix("team_") for n in asyncio.run(_names())}
    assert len(names) > 10, "tool listing returned almost nothing"
    text = CORE_DOC.read_text()
    missing = sorted(n for n in names if not _word_in(n, text))
    assert not missing, f"MCP tools not mentioned in how_it_works.md: {missing}"


def test_routes_documented() -> None:
    from fastapi.routing import APIRoute
    from kb_service.main import app

    text = CORE_DOC.read_text() + "\n" + AUTOMATION_DOC.read_text()
    paths = sorted(
        {r.path for r in app.routes if isinstance(r, APIRoute) and r.path.startswith("/api/")}
    )
    assert len(paths) > 20
    missing = [p for p in paths if p not in text]
    assert not missing, f"API routes not named in either doc: {missing}"


_CREATE_TABLE = re.compile(
    r"CREATE\s+(?:VIRTUAL\s+)?TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?([A-Za-z_][A-Za-z0-9_]*)",
    re.IGNORECASE,
)


def test_tables_documented() -> None:
    files = (
        REPO / "packages/kb-core/src/kb_core/db/schema.py",
        REPO / "packages/kb-core/src/kb_core/db/postgres_backend.py",
        REPO / "packages/kb-service/src/kb_service/database.py",
    )
    tables = {m for f in files for m in _CREATE_TABLE.findall(f.read_text())}
    assert len(tables) > 15, "table scan found almost nothing"
    text = CORE_DOC.read_text() + "\n" + AUTOMATION_DOC.read_text()
    missing = sorted(t for t in tables if not _word_in(t, text))
    assert not missing, f"Tables not named in either doc: {missing}"


def test_llm_purposes_documented() -> None:
    from kb_core.llm.provider import LLMPurpose

    purposes = typing.get_args(LLMPurpose)
    assert purposes
    text = CORE_DOC.read_text()
    missing = [p for p in purposes if not _word_in(p, text)]
    assert not missing, f"LLM purposes not mentioned in how_it_works.md: {missing}"


def test_contract_fixtures_documented() -> None:
    text = AUTOMATION_DOC.read_text()
    fixtures = sorted(p.name for p in (REPO / "contracts").glob("*.json"))
    assert fixtures
    missing = [f for f in fixtures if f not in text]
    assert not missing, f"contracts/ fixtures not named in docs/intelligent-kb.md: {missing}"


# --- B9: code references resolve -----------------------------------------------

_CODE_REF = re.compile(r"`([A-Za-z0-9_./{}*-]+\.py)(?:::?([A-Za-z_][A-Za-z0-9_]*))?`")


def _resolve_py(path: str) -> list[Path]:
    """Bases tried: repo root, each package's src/ and root, the import packages
    inside them, and the test directories (docs abbreviate paths within a package)."""
    bases = [REPO, REPO / "src", REPO / "src" / "personal_kb", REPO / "tests", REPO / "scripts"]
    for pkg in sorted((REPO / "packages").iterdir()):
        bases += [pkg, pkg / "src", pkg / "tests", pkg / "scripts"]
        bases += sorted(d for d in (pkg / "src").glob("*") if d.is_dir())
    found = [b / path for b in bases if (b / path).is_file()]
    if not found and "/" not in path:  # bare file name: any file of that name
        found = [p for b in bases for p in b.rglob(path) if ".venv" not in p.parts]
    return found


def _defines(file: Path, symbol: str) -> bool:
    tree = ast.parse(file.read_text())
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            and node.name == symbol
        ):
            return True
    for node in tree.body:
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == symbol for t in targets):
            return True
    return False


def test_doc_code_references_resolve() -> None:
    problems: list[str] = []
    checked = 0
    for doc in (CORE_DOC, AUTOMATION_DOC, CLAUDE_MD):
        for path, symbol in _CODE_REF.findall(doc.read_text()):
            if "{" in path or "*" in path:
                continue
            checked += 1
            files = _resolve_py(path)
            if not files:
                problems.append(f"{doc.name}: {path} (no such file)")
            elif symbol and not any(_defines(f, symbol) for f in files):
                problems.append(f"{doc.name}: {path}:{symbol} (symbol not defined)")
    assert checked > 50, "found almost no code references; did the pattern break?"
    assert not problems, "Doc code references that do not resolve: " + "; ".join(
        sorted(set(problems))
    )


# --- B10: markdown well-formedness ---------------------------------------------


def _slug(heading: str) -> str:
    text = re.sub(r"`", "", heading.strip().lower())
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s", "-", text)


def test_markdown_well_formed() -> None:
    problems: list[str] = []
    for doc in (CORE_DOC, AUTOMATION_DOC):
        text = doc.read_text()
        anchors: set[str] = set()
        in_fence = False
        for n, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            if re.match(r"#{1,6}\s", line):
                anchors.add(_slug(line.lstrip("#")))
            if line.count("`") % 2:
                problems.append(f"{doc.name}:{n}: odd backtick count")
        for anchor in re.findall(r"\]\(#([^)]+)\)", text):
            if anchor not in anchors:
                problems.append(f"{doc.name}: dangling anchor #{anchor}")
    assert not problems, "Markdown problems: " + "; ".join(problems)


# --- B11: pinned constants -----------------------------------------------------

DOC_PINNED: dict[str, tuple[str, Path]] = {
    "RRF_K": ("kb_core.search.hybrid", CORE_DOC),
    "RRF_LEG_FLOOR_RANK": ("kb_core.search.hybrid", CORE_DOC),
    "FTS_MAX_QUERY_TOKENS": ("kb_core.db.fts_query", CORE_DOC),
    "FTS_MAX_MISSING_TOKENS": ("kb_core.db.fts_query", CORE_DOC),
    "_FAST_PATH_THRESHOLD": ("kb_core.graph.agent", CORE_DOC),
    "NEAR_DUPLICATE_NEIGHBOURS": ("kb_core.near_duplicates", CORE_DOC),
    "CUE_NORMALIZER_VERSION": ("kb_core.cues", AUTOMATION_DOC),
    "SURPRISE_DETECTOR_VERSION": ("kb_service.surprise", AUTOMATION_DOC),
    "SURPRISE_DISTILLER_VERSION": ("kb_service.surprise_distill", AUTOMATION_DOC),
    "SURPRISE_CRITIC_VERSION": ("kb_service.surprise_distill", AUTOMATION_DOC),
    "STORE_DISTILLER_VERSION": ("kb_service.store_distill", AUTOMATION_DOC),
    "SLICE_CAP": ("kb_service.prevention", AUTOMATION_DOC),
    "INDEX_CAP": ("kb_service.prevention", AUTOMATION_DOC),
    "TURN_DIGEST_MAX_BYTES": ("kb_service.turn_digest", AUTOMATION_DOC),
    "TURN_EVENTS_RETENTION_DAYS": ("kb_service.turn_digest", AUTOMATION_DOC),
    "LLM_CALLS_RETENTION_DAYS": ("kb_service.llm_calls", CORE_DOC),
    "SURPRISE_LESSON_TTL_DAYS": ("kb_service.surprise_distill", AUTOMATION_DOC),
    "MAP_BODY_BASE_CHARS": ("kb_core.map_lint", CORE_DOC),
    "STALE_FILTER_THRESHOLD": ("kb_core.confidence.decay", CORE_DOC),
    "BELIEF_CHECK_THRESHOLD": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_MATCHER_VERSION": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_TOP_K": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_MAX_CHUNKS": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_MIN_CHUNK_CHARS": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_MAX_CHUNK_CHARS": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_MAX_INPUT_CHARS": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_MAX_CANDIDATES": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_DEADLINE_S": ("kb_service.belief_check", AUTOMATION_DOC),
    "BELIEF_CHECK_WARM_BATCH": ("kb_service.belief_check", AUTOMATION_DOC),
}


def test_pinned_constants() -> None:
    """Load-bearing constants are written as `NAME = value` in their doc."""
    missing: list[str] = []
    for name, (module, doc) in DOC_PINNED.items():
        value = getattr(importlib.import_module(module), name)
        if f"`{name} = {value}`" not in doc.read_text():
            missing.append(f"`{name} = {value}` -> {doc.relative_to(REPO)}")
    assert not missing, "Pinned constants missing or stale in the docs: " + "; ".join(missing)
