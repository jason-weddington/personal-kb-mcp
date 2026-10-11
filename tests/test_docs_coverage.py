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
import re
from pathlib import Path

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
