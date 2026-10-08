"""Work-user upgrade smoke test — the release gate for the GitHub install path.

About 60 work users run ``uvx --from git+https://github.com/...personal-kb-mcp
personal-kb`` against a LOCAL SQLite KB, with no PERSONAL_KB_URL and no API key
in their MCP config. This script reproduces that population end to end
(GTD 78e9a45e) and fails loudly if an upgrade would break them:

1. Install the OLD build (default: the GitHub HEAD work users run today) in an
   isolated HOME and seed a KB through it — creates, an update, a
   deactivation, feedback — so the data is shaped by the version that wrote it.
2. Record the old build's read outputs and the DB row counts.
3. Start the NEW build with the IDENTICAL environment (no PERSONAL_KB_URL, no
   key, no extras): it must start, spawn the local daemon, return the same read
   outputs, keep every row, accept writes, and serve the web UI.
4. Re-run the OLD build against the upgraded DB (rollback safety).

Run from the repo root:

    uv run python scripts/smoke_work_user_upgrade.py
    uv run python scripts/smoke_work_user_upgrade.py \
        --new-spec "personal-kb @ git+file:///path/to/personal_kb@my-branch"

Exits 0 only when every check passes. The sandbox lives in a temp dir and is
removed on success (kept with --keep or on failure, for debugging).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

# Unpinned on purpose: the baseline is whatever GitHub HEAD serves work users
# TODAY (756b5c7 until the v1.0.0 release), so after each release the gate
# automatically tests upgrades from the newly shipped version.
OLD_SPEC = "git+https://github.com/jason-weddington/personal-kb-mcp"


def _default_new_spec() -> str:
    """Local checkout at HEAD: ``personal-kb @ git+file://<repo root>@<sha>``."""
    root = Path(__file__).resolve().parent.parent
    sha = subprocess.run(  # noqa: S603
        ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return f"personal-kb @ git+file://{root}@{sha}"


LOCAL_PORT = 8765  # the documented local-mode default; the new build must pick it unaided

# Read calls whose output must be identical before and after the upgrade.
# kb_ask timeline/related are excluded: known regression tracked in GTD a8add5c1.
READS: list[tuple[str, dict[str, Any]]] = [
    ("kb_search", {"query": "presigned URL expiry"}),
    ("kb_search", {"project_ref": "docs-platform", "limit": 20}),
    ("kb_get", {"entry_id": "kb-00001"}),
    ("kb_preflight", {"project_ref": "metadata-service"}),
    ("kb_list_projects", {}),
]

_TOPICS = [
    (
        "S3 presigned URL expiry",
        "Presigned URLs signed with STS creds expire with the session.",
        "docs-platform",
    ),
    (
        "Build fleet cache key",
        "The docs build cache key includes the toolchain hash.",
        "docs-platform",
    ),
    (
        "Use Hugo partials for callouts",
        "Callout boxes live in layouts/partials/callout.html.",
        "docs-platform",
    ),
    (
        "Translation pipeline lag",
        "Localized pages trail English by one publish cycle.",
        "localization",
    ),
    ("CloudFront invalidation cost", "Wildcard invalidations count as one path.", "docs-platform"),
    (
        "Choose DynamoDB for page metadata",
        "DynamoDB over RDS: single-key access, no joins.",
        "metadata-service",
    ),
    (
        "Lambda cold start on Java",
        "SnapStart cut Java cold starts from 3s to 400ms.",
        "metadata-service",
    ),
    (
        "Review rotation convention",
        "Doc PR reviews rotate weekly; on-call owns the queue.",
        "team-process",
    ),
    ("Search index rebuild", "Full rebuild takes 40 min; incremental is the default.", "search"),
    (
        "Never edit generated API refs",
        "API reference pages are generated from Smithy models.",
        "search",
    ),
    (
        "Broken link checker flakiness",
        "Allowlist rate-limited hosts in linkcheck.yaml.",
        "docs-platform",
    ),
    ("Quarterly planning template", "Planning docs use the PR-FAQ template.", "team-process"),
]
_TYPES = ["factual_reference", "decision", "pattern_convention", "lesson_learned"]


def _entries() -> list[dict[str, Any]]:
    """Build the seed entries from _TOPICS."""
    return [
        {
            "short_title": t,
            "long_title": f"{t} (long)",
            "knowledge_details": d,
            "entry_type": _TYPES[i % 4],
            "project_ref": p,
            "tags": [p, "seed"],
        }
        for i, (t, d, p) in enumerate(_TOPICS)
    ]


# SEED runs on the OLD build, which by default is GitHub HEAD: what work users
# run today. Since v1.1.0 every kb_store / kb_store_batch entry there requires
# ``supersedes``, so the seed uses the current contract. An --old-spec older
# than v1.1.0 rejects the extra argument and the smoke fails loudly, which is
# the right outcome: that is not an upgrade path anyone is on.
_NONE = {"supersedes": "none"}

SEED: list[tuple[str, dict[str, Any]]] = [
    ("kb_store_batch", {"entries": [{**e, **_NONE} for e in _entries()[:10]]}),
    ("kb_store", {**_entries()[10], **_NONE}),
    ("kb_store", {**_entries()[11], **_NONE}),
    (
        "kb_store",
        {
            **_NONE,
            "update_entry_id": "kb-00001",
            "knowledge_details": "UPDATED: earlier of session expiry and X-Amz-Expires.",
            "change_reason": "clarify",
        },
    ),
    ("kb_store", {**_NONE, "deactivate_entry_id": "kb-00012", "change_reason": "template retired"}),
    (
        "kb_feedback",
        {
            "feedback_type": "missing",
            "tool_name": "kb_search",
            "query_or_params": "origin shield",
            "detail": "no entry",
        },
    ),
]

# WRITES run on the NEW build, whose kb_store schema requires ``supersedes``.
WRITES: list[tuple[str, dict[str, Any]]] = [
    (
        "kb_store",
        {
            "supersedes": "none",
            "short_title": "Post-upgrade entry",
            "long_title": "Written by the upgraded client",
            "knowledge_details": "Written through the local daemon after upgrade.",
            "entry_type": "factual_reference",
            "project_ref": "docs-platform",
            "tags": ["upgrade-test"],
        },
    ),
    (
        "kb_store",
        {
            "supersedes": "none",
            "update_entry_id": "kb-00002",
            "knowledge_details": "UPDATED post-upgrade.",
            "change_reason": "upgrade test",
        },
    ),
    # kb-00003 is the third seed entry (a pattern_convention in SEED's
    # kb_store_batch): active and not a mental_map, so the new client's
    # deactivate (which now sends change_reason) reaches the new daemon.
    (
        "kb_store",
        {"supersedes": "none", "deactivate_entry_id": "kb-00003", "change_reason": "upgrade test"},
    ),
    (
        "kb_feedback",
        {
            "feedback_type": "friction",
            "tool_name": "kb_search",
            "query_or_params": "x",
            "detail": "post-upgrade feedback",
        },
    ),
]

_COUNTS = {
    "entries": "select count(*) from knowledge_entries",
    "active": "select count(*) from knowledge_entries where is_active=1",
    "versions": "select count(*) from entry_versions",
    "feedback": "select count(*) from agent_feedback",
}

# Write responses that look like success but are errors (the HTTP client turns
# a 500 into a tool *result* string, not a raised error — see break 4).
_ERROR_TEXT = re.compile(r"^Error|returned 5\d\d|Internal Server Error", re.M)


class SmokeFailureError(Exception):
    """A check failed."""


def _env(home: Path) -> dict[str, str]:
    """The work-user environment: NOTHING KB-specific, forced, not inherited."""
    return {
        "HOME": str(home),
        "PATH": os.environ["PATH"],
        "UV_CACHE_DIR": os.environ.get("UV_CACHE_DIR", str(Path.home() / ".cache/uv")),
        "UV_PYTHON_INSTALL_DIR": os.environ.get(
            "UV_PYTHON_INSTALL_DIR", str(Path.home() / ".local/share/uv/python")
        ),
        "GIT_SSH_COMMAND": os.environ.get(
            "GIT_SSH_COMMAND",
            f"ssh -o UserKnownHostsFile={Path.home()}/.ssh/known_hosts"
            f" -i {Path.home()}/.ssh/id_ed25519",
        ),
    }


async def _session(spec: str, home: Path, calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    transport = StdioTransport(
        command="uvx", args=["--python", "3.13", "--from", spec, "personal-kb"], env=_env(home)
    )
    out: list[str] = []
    async with Client(transport, timeout=300) as client:
        for name, args in calls:
            result = await client.call_tool(name, args)
            out.append(result.content[0].text)
    return out


def _run(spec: str, home: Path, calls: list[tuple[str, dict[str, Any]]], label: str) -> list[str]:
    try:
        return asyncio.run(_session(spec, home, calls))
    except Exception as exc:  # every failure is a smoke failure
        log = home / ".local/share/personal_kb/kb-daemon.log"
        tail = log.read_text()[-2000:] if log.exists() else "(no daemon log)"
        raise SmokeFailureError(
            f"{label}: {type(exc).__name__}: {exc}\n--- daemon log tail ---\n{tail}"
        ) from exc


def _counts(db: Path) -> dict[str, int]:
    conn = sqlite3.connect(db)
    try:
        return {k: conn.execute(q).fetchone()[0] for k, q in _COUNTS.items()}
    finally:
        conn.close()


def _port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def _stop_daemon(home: Path) -> None:
    pidfile = home / ".local/share/personal_kb/kb-daemon.pid"
    if pidfile.exists():
        try:
            pid = int(pidfile.read_text().strip() or 0)
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
            if b"kb-service" in cmdline:
                os.kill(pid, signal.SIGTERM)
        except (ValueError, OSError):
            pass
    for _ in range(30):
        if _port_free(LOCAL_PORT):
            return
        time.sleep(0.5)


def _check(cond: bool, msg: str, results: list[tuple[str, bool]]) -> None:
    results.append((msg, cond))
    print(f"  [{'PASS' if cond else 'FAIL'}] {msg}", flush=True)


def main() -> int:
    """Run the upgrade smoke test; return the process exit code."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--old-spec", default=OLD_SPEC)
    ap.add_argument(
        "--new-spec",
        default=None,
        help="default: this checkout at HEAD (git+file, computed at runtime)",
    )
    ap.add_argument("--keep", action="store_true", help="keep the sandbox dir")
    a = ap.parse_args()
    if a.new_spec is None:
        a.new_spec = _default_new_spec()

    if not _port_free(LOCAL_PORT):
        print(
            f"Port {LOCAL_PORT} is in use — stop the running kb-service daemon first.",
            file=sys.stderr,
        )
        return 2

    sandbox = Path(tempfile.mkdtemp(prefix="kb-upgrade-smoke-"))
    home = sandbox / "home"
    db = home / ".local/share/personal_kb/knowledge.db"
    home.mkdir()
    results: list[tuple[str, bool]] = []
    print(f"sandbox: {sandbox}\nold: {a.old_spec}\nnew: {a.new_spec}", flush=True)
    try:
        print("1. seed + baseline (old build)", flush=True)
        seeded = _run(a.old_spec, home, SEED + READS, "old build")
        _check(
            not any(_ERROR_TEXT.search(t) for t in seeded[: len(SEED)]),
            "old build seeds without errors",
            results,
        )
        baseline_reads = seeded[len(SEED) :]
        before = _counts(db)
        print(f"     counts: {before}", flush=True)

        print("2. upgrade (new build, identical env)", flush=True)
        after_reads = _run(a.new_spec, home, READS, "new build startup/reads")
        _check(
            not _port_free(LOCAL_PORT),
            f"new build spawned the local daemon on :{LOCAL_PORT}",
            results,
        )
        for (name, _), old, new in zip(READS, baseline_reads, after_reads, strict=True):
            _check(old == new, f"{name} output identical to baseline", results)
            if old != new:
                print(f"       old: {old[:300]!r}\n       new: {new[:300]!r}")
        _check(_counts(db) == before, "row counts unchanged by upgrade", results)

        writes = _run(a.new_spec, home, WRITES, "new build writes")
        bad = [t[:200] for t in writes if _ERROR_TEXT.search(t)]
        _check(not bad, "writes succeed through the daemon" + (f": {bad}" if bad else ""), results)
        after_w = _counts(db)
        _check(
            after_w["entries"] == before["entries"] + 1
            and after_w["versions"] == before["versions"] + 2
            and after_w["feedback"] == before["feedback"] + 1,
            f"writes landed in the user's knowledge.db ({before} -> {after_w})",
            results,
        )
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{LOCAL_PORT}/", timeout=5) as r:
                html = r.read(4096).decode(errors="replace")
            _check("<html" in html.lower(), "daemon serves the web UI at /", results)
        except OSError as exc:
            _check(False, f"daemon serves the web UI at / ({exc})", results)
        _stop_daemon(home)

        print("3. rollback (old build against upgraded DB)", flush=True)
        rollback = _run(a.old_spec, home, READS[:2], "old build after upgrade")
        _check(
            not any(_ERROR_TEXT.search(t) for t in rollback),
            "old build still reads the upgraded DB",
            results,
        )
    except SmokeFailureError as exc:
        results.append((str(exc), False))
        print(f"  [FAIL] {exc}", flush=True)
    finally:
        _stop_daemon(home)

    failed = [m for m, ok in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed or a.keep:
        print(f"sandbox kept: {sandbox}")
    else:
        shutil.rmtree(sandbox, ignore_errors=True)
    print(json.dumps({"passed": not failed, "failed": failed}))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
