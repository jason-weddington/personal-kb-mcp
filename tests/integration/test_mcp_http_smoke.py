"""Real-subprocess smoke: kb-service /mcp driven by the official MCP client.

Boots ``kb-service serve`` on a free loopback port in no-auth mode against a
throwaway SQLite DB, then uses ``mcp.client.streamable_http`` +
``mcp.ClientSession`` exactly as a harness would: initialize, list tools,
store an entry and read it back. Skipped when ``kb-service`` is not on PATH
(mirrors ``test_local_mode_daemon_smoke.py``).
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import TYPE_CHECKING

import anyio
import httpx
import pytest

from tests.integration.test_local_mode_daemon_smoke import _free_loopback_port

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    shutil.which("kb-service") is None,
    reason="kb-service binary not on PATH",
)

_STRIP = (
    "ANTHROPIC_API_KEY",
    "KB_DATABASE_URL",
    "KB_SERVICE_DATABASE_URL",
    "KB_SURPRISE_CAPTURE",
    "PERSONAL_KB_URL",
    "PERSONAL_KB_API_KEY",
    "KB_INSTANCE_ROLE",
    "KB_MANAGER",
    "KB_CONTRIBUTOR",
    "KB_WRITE_POLICY_DEFAULT_SURFACE",
)


def _env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in _STRIP}
    env.update(
        {
            "KB_AUTH_MODE": "none",
            "KB_DB_PATH": str(tmp_path / "knowledge.db"),
            "KB_OLLAMA_URL": "http://127.0.0.1:9",
            "KB_EXTRACTION_PROVIDER": "ollama",
            "KB_QUERY_PROVIDER": "ollama",
            "PERSONAL_KB_DAEMON_STATE_DIR": str(tmp_path),
            "KB_AUTO_EXPLORE": "FALSE",
        }
    )
    return env


async def _wait_healthy(base_url: str, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=2.0) as client:
        while time.monotonic() < deadline:
            try:
                resp = await client.get(f"{base_url}/api/health")
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await anyio.sleep(0.5)
    return False


def _text(result: object) -> str:
    content = result.content  # type: ignore[attr-defined]
    text: str = content[0].text
    return text


async def test_official_client_lists_stores_and_gets(tmp_path: Path) -> None:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    port = _free_loopback_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / "kb-service.log"
    kb_service = shutil.which("kb-service")
    assert kb_service is not None
    with log_path.open("wb") as log:
        proc = subprocess.Popen(  # noqa: S603
            [kb_service, "serve", "--host", "127.0.0.1", "--port", str(port)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=_env(tmp_path),
        )
    ok = False
    try:
        assert await _wait_healthy(base_url), "kb-service never became healthy"
        async with (
            streamable_http_client(f"{base_url}/mcp") as (read, write, _sid),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            names = {t.name for t in (await session.list_tools()).tools}
            assert {"kb_store", "kb_search", "kb_get", "kb_preflight"} <= names
            assert "kb_ingest" not in names

            stored = _text(
                await session.call_tool(
                    "kb_store",
                    {
                        "short_title": "Smoke entry",
                        "long_title": "Smoke long",
                        "knowledge_details": "Smoke details.",
                        "entry_type": "factual_reference",
                        "supersedes": "none",
                    },
                )
            )
            match = re.match(r"^Created (kb-\d{5}) \(v1\)", stored)
            assert match, stored
            got = _text(await session.call_tool("kb_get", {"entry_id": match.group(1)}))
            assert "Smoke entry" in got
        ok = True
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        if not ok:
            sys.stderr.write(log_path.read_text(errors="replace"))
