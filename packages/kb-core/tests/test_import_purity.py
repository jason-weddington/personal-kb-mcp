"""Import-purity guard for ``kb_core``.

``kb_core`` is the channel-agnostic engine. Two invariants must hold for
every later extraction wave, so the guard is laid down here in wave 1
and later waves keep it green:

1. **No env reads.** ``kb_core`` reads NO ``os.environ`` / ``os.getenv``
   anywhere in its source. Config is passed explicitly via
   :class:`kb_core.config.KbConfig`. Enforced by an AST scan over every
   ``.py`` file in the ``kb_core`` package.

2. **No heavy-deps leak on bare import.** A subprocess that runs
   ``import kb_core`` and ``from kb_core.config import KbConfig`` must
   NOT end up with ``fastmcp``, ``fastapi``, ``uvicorn``, ``anthropic``,
   or ``boto3`` in ``sys.modules``. Those become optional extras
   activated by the consumer or imported lazily inside provider modules.

We deliberately use a subprocess for invariant (2) because the parent
test process may have pulled these in via other tests; the only honest
check is a clean interpreter that does nothing but import ``kb_core``.

Style mirrors ``tests/test_path_drift_guard.py`` in the main repo: small
focused asserts, helpful failure messages, no clever fixtures.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

KB_CORE_SRC = Path(__file__).resolve().parent.parent / "src" / "kb_core"


def _all_python_files() -> list[Path]:
    """Every ``.py`` file shipped under ``src/kb_core/``."""
    return sorted(KB_CORE_SRC.rglob("*.py"))


class _OsEnvVisitor(ast.NodeVisitor):
    """AST visitor that records ``os.environ`` and ``os.getenv`` references.

    Matches the two forms that actually read env vars:

    * ``os.environ[...]`` / ``os.environ.get(...)`` — attribute access on
      a name ``os``.
    * ``os.getenv(...)`` — same shape.

    Does NOT match ``from os import environ`` / ``from os import getenv``
    (no one uses that style in this codebase, and matching it would
    require tracking imports). If a later wave introduces those forms,
    add a second visitor pass for ``ImportFrom`` + ``Name`` lookups.
    """

    def __init__(self) -> None:
        self.hits: list[tuple[int, str]] = []

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Record ``os.environ`` and ``os.getenv`` attribute reads (ast.NodeVisitor API)."""
        if (
            isinstance(node.value, ast.Name)
            and node.value.id == "os"
            and node.attr in {"environ", "getenv"}
        ):
            self.hits.append((node.lineno, f"os.{node.attr}"))
        self.generic_visit(node)


def test_no_env_reads_in_kb_core_source() -> None:
    """AST-scan every kb_core file; assert zero ``os.environ`` / ``os.getenv`` refs.

    This is the load-bearing invariant: kb_core's config surface is
    :class:`kb_core.config.KbConfig`, full stop. If someone reaches for
    an env var inside the engine, the consumer no longer has a single
    source of truth and the package is no longer cleanly embeddable.
    """
    all_hits: list[str] = []
    for path in _all_python_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        visitor = _OsEnvVisitor()
        visitor.visit(tree)
        rel = path.relative_to(KB_CORE_SRC.parent.parent)
        for lineno, name in visitor.hits:
            all_hits.append(f"{rel}:{lineno}: {name}")

    assert not all_hits, (
        "kb_core must not read os.environ / os.getenv anywhere in its source. "
        "Config is explicit via KbConfig. Offending references:\n  " + "\n  ".join(all_hits)
    )


def test_bare_import_does_not_pull_heavy_deps() -> None:
    """A bare ``import kb_core`` must not load fastmcp/fastapi/anthropic/boto3.

    Runs in a fresh subprocess so the parent's already-imported modules
    can't contaminate ``sys.modules``. We also import ``KbConfig`` to
    cover the documented public entry point — instantiating it must not
    trip the guard either.
    """
    script = (
        "import sys\n"
        "import kb_core\n"
        "from kb_core.config import KbConfig\n"
        "_ = KbConfig()\n"
        "forbidden = ['fastmcp', 'fastapi', 'uvicorn', 'anthropic', 'boto3']\n"
        "leaked = sorted(m for m in forbidden if m in sys.modules)\n"
        "if leaked:\n"
        "    sys.stdout.write('LEAKED:' + ','.join(leaked))\n"
        "    sys.exit(1)\n"
        "sys.stdout.write('OK')\n"
    )
    result = subprocess.run(  # noqa: S603 (controlled args, no shell)
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        "Heavy deps leaked through bare `import kb_core`:\n"
        f"  stdout: {result.stdout!r}\n"
        f"  stderr: {result.stderr!r}"
    )
    assert result.stdout == "OK", f"Unexpected subprocess output: {result.stdout!r}"
