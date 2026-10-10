"""kb_service never imports personal_kb (the stdio client package)."""

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src" / "kb_service"


def _offenders() -> list[str]:
    bad: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
            for mod in modules:
                if mod.split(".")[0] == "personal_kb":
                    bad.append(f"{path.relative_to(_SRC)}:{node.lineno} {mod}")
    return bad


def test_kb_service_does_not_import_personal_kb() -> None:
    assert _offenders() == []


def test_scan_covers_mcp_server_package() -> None:
    files = {p.name for p in (_SRC / "mcp_server" / "tools").glob("*.py")}
    assert {"kb_store.py", "kb_search.py", "kb_maintain.py"} <= files
