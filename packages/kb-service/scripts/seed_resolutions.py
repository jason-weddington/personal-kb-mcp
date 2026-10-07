#!/usr/bin/env python3
r"""Seed ``hints.resolution`` on existing procedure entries (one-off, idempotent).

Scans active ``pattern_convention`` / ``factual_reference`` entries (override
with ``--entry-types``) for a by-hand shell procedure an agent would get wrong
when it does not know the entry, asks an LLM to draft a resolution for it, runs
the draft through a strict precision guard, and (with ``--apply``) writes it
back over HTTP with ``POST /api/kb/store`` as a hints-only update. The server
validates and stamps every write (``kb_service/resolution_hint.py``).

Dry run by default: nothing is written without ``--apply``. Pinned resolutions
(``--pinned-file``, a JSON object ``{entry_id: resolution_dict}``) are written
exactly as given and never go through the LLM. No pinned content is committed
to the repo; the control plane supplies the file at run time.

Apply pinned (deliberate) resolutions with a NON-machine-principal key: the
server forces ``capture='autonomous'`` for the machine principal, which this
script reports as ``rejected['capture_forced']`` (exit 1).

``--audit`` is read-only: it re-validates every stored resolution and re-checks
each Bash ``target_class`` against the live ``kb_core.cues`` normalizer. Run it
after any cues normalizer change.

Every write carries a ``change_reason`` starting with ``seed_resolutions:``,
the after-the-fact query key over entry history.

Usage:
  PERSONAL_KB_URL=http://127.0.0.1:8765 ANTHROPIC_API_KEY=... \\
    uv run python packages/kb-service/scripts/seed_resolutions.py [--apply]
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import urlparse

import httpx
from kb_core.config import _DEFAULT_ANTHROPIC_MODEL
from kb_core.cues import target_class
from kb_core.llm.json_parser import parse_json_object

if TYPE_CHECKING:
    from collections.abc import Mapping

from kb_service.resolution_hint import (
    ResolutionHintError,
    validate_and_stamp_resolution,
)

DEFAULT_URL = "http://127.0.0.1:8765"
LOCAL_KEY = "local-no-auth"
_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
DEFAULT_ENTRY_TYPES = "pattern_convention,factual_reference"
AUDIT_EXTRA_TYPES = ("decision", "lesson_learned")
CHUNK = 20
FACT_MAX = 300
PREFILTER = re.compile(r"~/scripts/|\.sh\b|`[^`\n]+`")
_SUBCOMMAND = re.compile(r"^[a-z][a-z0-9_-]*$")

DENYLIST = frozenset(
    {
        "git status",
        "git log",
        "git diff",
        "git show",
        "git add",
        "git commit",
        "git fetch",
        "git pull",
        "git checkout",
        "git branch",
        "uv run",
        "uv sync",
        "python3 -m",
        "python -m",
    }
)

PROMPT_TEMPLATE = """\
You review one knowledge-base entry that may describe a shell procedure.
Decide whether an agent that does NOT know this entry would, by habit, type a
WRONG or incomplete shell command for the task the entry describes, and the
entry's procedure is the correction. If so, draft a short corrected-fact
reminder to show the agent just before it types that command.

Reply with ONE JSON object and nothing else, with exactly these keys:
  "propose": bool - true only if you are confident; answer false when unsure.
  "corrected_fact": str - the reminder, at most 300 characters.
  "cue_target_class": str - the two-word command class the agent would type by
      hand, e.g. "git remote" (program plus subcommand, lowercase).
  "example_command": str - the by-hand shell command an agent would type when
      it does NOT know this entry.
  "scope": "project" or "global" - global only if it applies in every project.
  "reason": str - one short sentence.
Optional key "cue_args_prefix": str - when the procedure is a specific
subcommand action (e.g. "add" for "git remote add"), that subcommand word, so
read-only forms are never matched.

Entry project: <<PROJECT>>
Entry title: <<TITLE>>
Entry details:
<<DETAILS>>
"""
PROMPT_SHA256 = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()

_REPLY_STR_KEYS = (
    "corrected_fact",
    "cue_target_class",
    "example_command",
    "scope",
    "reason",
)


class LLM(Protocol):
    """Minimal LLM surface the seeder needs."""

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        """Return the model reply, or None when unavailable."""
        ...


@dataclass
class SeedReport:
    """Everything one run did, written to stdout and ``--report-path``."""

    model: str = ""
    prompt_sha256: str = ""
    started_at: str = ""
    finished_at: str = ""
    apply: bool = False
    scanned: int = 0
    prefiltered_out: int = 0
    already_resolved: int = 0
    llm_calls: int = 0
    llm_total_ms: int = 0
    proposed: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    written: int = 0
    proposals: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def resolve_api_key(url: str, environ: Mapping[str, str]) -> str:
    """Mirror ``personal_kb.config``: env key, else loopback fallback, else exit 2."""
    key = environ.get("PERSONAL_KB_API_KEY")
    if key:
        return key
    if (urlparse(url).hostname or "").lower() in _LOOPBACK:
        return LOCAL_KEY
    print("PERSONAL_KB_API_KEY required for a remote URL", file=sys.stderr)
    raise SystemExit(2)


# ─── HTTP enumeration ────────────────────────────────────────────────────────


async def _enumerate_ids(http: httpx.AsyncClient, entry_types: list[str]) -> list[str]:
    ids: set[str] = set()
    for etype in entry_types:
        resp = await http.get("/api/kb/graph/scope-entries", params={"scope": etype})
        resp.raise_for_status()
        ids.update(str(i) for i in resp.json().get("entry_ids", []))
    return sorted(ids)


async def _fetch_entries(
    http: httpx.AsyncClient, ids: list[str]
) -> dict[str, dict[str, Any] | None]:
    """Fetch entries by id in chunks; missing/inactive ids map to None."""
    out: dict[str, dict[str, Any] | None] = {}
    for start in range(0, len(ids), CHUNK):
        chunk = ids[start : start + CHUNK]
        resp = await http.post("/api/kb/get", json={"ids": chunk})
        resp.raise_for_status()
        for item in resp.json().get("results", []):
            out[str(item["id"])] = item.get("entry") if item.get("found") else None
        for i in chunk:
            out.setdefault(i, None)
    return out


# ─── guard ───────────────────────────────────────────────────────────────────


@dataclass
class _Registry:
    """Bash cues already claimed: ``(target_class, scope, project_ref)``."""

    keys: list[tuple[str, str, str | None]] = field(default_factory=list)

    def add(self, tc: str, scope: str, project: str | None) -> None:
        self.keys.append((tc, scope, project))

    def collides(self, tc: str, scope: str, project: str | None) -> bool:
        for etc, escope, eproject in self.keys:
            if etc != tc:
                continue
            if scope == "global" or escope == "global" or eproject == project:
                return True
        return False


def _bash_cue(res: dict[str, Any]) -> str | None:
    cue = res.get("cue")
    if isinstance(cue, dict) and cue.get("tool") == "Bash":
        tc = cue.get("target_class")
        return tc if isinstance(tc, str) else None
    return None


def _args_prefix_for(example: str, tc: str, proposed: str) -> str:
    """Keep *proposed* only when it is a subcommand word following the class."""
    word = proposed.strip()
    if not word or not _SUBCOMMAND.match(word):
        return ""
    rest = example.split()[len(tc.split()) :]
    return word if rest and rest[0] == word else ""


def _parse_reply(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    data = parse_json_object(raw)
    if data is None or not isinstance(data.get("propose"), bool):
        return None
    for key in _REPLY_STR_KEYS:
        if not isinstance(data.get(key), str):
            return None
    if "cue_args_prefix" in data and not isinstance(data["cue_args_prefix"], str):
        return None
    return data


def _guard(
    data: dict[str, Any], project: str | None, registry: _Registry
) -> tuple[str | None, dict[str, Any] | None]:
    """Return ``(rejection_key, None)`` or ``(None, cue_info)``."""
    if data["propose"] is not True:
        return "declined", None
    cf = data["corrected_fact"]
    if cf.strip() == "" or len(cf) > FACT_MAX:
        return "empty_fact", None
    scope = data["scope"]
    if scope not in ("project", "global"):
        return "bad_scope", None
    tc = data["cue_target_class"]
    if target_class("Bash", tc) != tc:
        return "not_normalized", None
    if tc.count(" ") != 1:
        return "not_two_word", None
    if target_class("Bash", data["example_command"]) != tc:
        return "example_mismatch", None
    if tc in DENYLIST:
        return "denylisted", None
    if registry.collides(tc, scope, project):
        return "class_collision", None
    return None, {"tc": tc, "scope": scope}


# ─── run ─────────────────────────────────────────────────────────────────────


def _row(entry_id: str) -> dict[str, Any]:
    return {
        "entry_id": entry_id,
        "stage": "prefilter",
        "outcome": "skipped",
        "reason": "",
        "cue_target_class": None,
        "example_command": None,
        "scope": None,
        "llm_latency_ms": None,
    }


def _set(row: dict[str, Any], stage: str, outcome: str, reason: str) -> None:
    row["stage"], row["outcome"], row["reason"] = stage, outcome, reason


def _reject(report: SeedReport, key: str) -> None:
    report.rejected[key] = report.rejected.get(key, 0) + 1


def _build_prompt(entry: dict[str, Any]) -> str:
    return (
        PROMPT_TEMPLATE.replace("<<PROJECT>>", str(entry.get("project_ref") or ""))
        .replace("<<TITLE>>", str(entry.get("short_title") or ""))
        .replace("<<DETAILS>>", str(entry.get("knowledge_details") or ""))
    )


async def run(
    http: httpx.AsyncClient,
    llm: LLM | None,
    *,
    apply: bool,
    entry_ids: list[str] | None,
    entry_types: list[str],
    limit: int,
    pinned: dict[str, dict[str, Any]],
    model: str,
) -> SeedReport:
    """Scan, draft, guard and (with *apply*) write resolutions; return the report."""
    report = SeedReport(
        model=model,
        prompt_sha256=PROMPT_SHA256,
        started_at=_now(),
        apply=apply,
    )
    ids = set(entry_ids) if entry_ids else set(await _enumerate_ids(http, entry_types))
    ids |= set(pinned)
    ordered = sorted(ids)
    entries = await _fetch_entries(http, ordered)
    report.scanned = len(ordered)

    rows = {i: _row(i) for i in ordered}
    report.decisions = [rows[i] for i in ordered]
    registry = _Registry()
    pinned_ids: list[str] = []
    llm_ids: list[str] = []

    for eid in ordered:
        entry = entries.get(eid)
        row = rows[eid]
        if entry is None:
            report.prefiltered_out += 1
            _set(row, "prefilter", "skipped", "not_found")
            continue
        if entry.get("superseded_by"):
            report.prefiltered_out += 1
            _set(row, "prefilter", "skipped", "superseded")
            continue
        if entry.get("entry_type") == "mental_map":
            report.prefiltered_out += 1
            _set(row, "prefilter", "skipped", "mental_map")
            continue
        existing = (entry.get("hints") or {}).get("resolution")
        if "resolution" in (entry.get("hints") or {}):
            report.already_resolved += 1
            _set(row, "already_resolved", "skipped", "has_resolution")
            tc = _bash_cue(existing) if isinstance(existing, dict) else None
            if tc:
                scope = existing.get("scope", "project")
                registry.add(tc, str(scope), entry.get("project_ref"))
            continue
        if eid in pinned:
            pinned_ids.append(eid)
            tc = _bash_cue(pinned[eid])
            if tc:
                registry.add(
                    tc,
                    str(pinned[eid].get("scope", "project")),
                    entry.get("project_ref"),
                )
            continue
        if not PREFILTER.search(str(entry.get("knowledge_details") or "")):
            report.prefiltered_out += 1
            _set(row, "prefilter", "skipped", "no_procedure_signal")
            continue
        llm_ids.append(eid)

    # proposals: entry_id -> (resolution, drafted_by)
    kept: dict[str, tuple[dict[str, Any], str]] = {}

    for eid in pinned_ids:
        row = rows[eid]
        res = pinned[eid]
        tc = _bash_cue(res)
        reason = None
        if tc is not None:
            if target_class("Bash", tc) != tc:
                reason = "not_normalized"
            elif tc.count(" ") != 1:
                reason = "not_two_word"
        if reason:
            _reject(report, reason)
            _set(row, "pinned", "rejected", reason)
            continue
        cue = res.get("cue") if isinstance(res.get("cue"), dict) else {}
        row["cue_target_class"] = tc
        row["args_prefix"] = cue.get("args_prefix")
        row["scope"] = res.get("scope", "project")
        _set(row, "pinned", "kept", "pinned")
        kept[eid] = (res, "pinned")
        report.proposed += 1
        report.proposals.append(
            {
                "entry_id": eid,
                "source": "pinned",
                "cue_target_class": tc,
                "args_prefix": cue.get("args_prefix"),
                "scope": row["scope"],
                "corrected_fact": res.get("corrected_fact"),
            }
        )

    llm_budget = limit if limit > 0 else None
    for eid in llm_ids:
        row = rows[eid]
        if llm is None:
            _reject(report, "no_llm")
            _set(row, "llm", "skipped", "no_llm")
            continue
        if llm_budget is not None and report.llm_calls >= llm_budget:
            _set(row, "llm", "skipped", "limit")
            continue
        entry = entries[eid] or {}
        started = time.monotonic()
        try:
            raw = await llm.generate(_build_prompt(entry))
        except Exception as exc:
            print(f"{eid}: llm error: {exc}", file=sys.stderr)
            raw = None
        ms = int((time.monotonic() - started) * 1000)
        report.llm_calls += 1
        report.llm_total_ms += ms
        row["llm_latency_ms"] = ms
        data = _parse_reply(raw)
        if data is None:
            _reject(report, "llm_unparseable")
            _set(row, "guard", "rejected", "llm_unparseable")
            continue
        row["cue_target_class"] = data["cue_target_class"]
        row["example_command"] = data["example_command"]
        row["scope"] = data["scope"]
        key, info = _guard(data, entry.get("project_ref"), registry)
        if key is not None or info is None:
            key = key or "declined"
            _reject(report, key)
            _set(row, "guard", "rejected", key)
            continue
        tc, scope = info["tc"], info["scope"]
        cue: dict[str, Any] = {"tool": "Bash", "target_class": tc}
        prefix = _args_prefix_for(
            data["example_command"], tc, str(data.get("cue_args_prefix") or "")
        )
        if prefix:
            cue["args_prefix"] = prefix
        row["args_prefix"] = prefix or None
        res = {
            "corrected_fact": data["corrected_fact"].strip(),
            "cue": cue,
            "scope": scope,
            "provenance": {"capture": "autonomous", "grounding": "asserted"},
            "evidence": f"seed_resolutions from {eid}",
        }
        registry.add(tc, scope, entry.get("project_ref"))
        kept[eid] = (res, model)
        _set(row, "guard", "kept", "ok")
        report.proposed += 1
        report.proposals.append(
            {
                "entry_id": eid,
                "source": "llm",
                "cue_target_class": tc,
                "args_prefix": prefix or None,
                "scope": scope,
                "corrected_fact": res["corrected_fact"],
                "example_command": data["example_command"],
            }
        )

    if apply:
        await _write_and_verify(http, report, rows, kept)

    report.finished_at = _now()
    return report


async def _write_and_verify(
    http: httpx.AsyncClient,
    report: SeedReport,
    rows: dict[str, dict[str, Any]],
    kept: dict[str, tuple[dict[str, Any], str]],
) -> None:
    sent: list[str] = []
    for eid in sorted(kept):
        res, drafted_by = kept[eid]
        tc = _bash_cue(res) or (res.get("cue") or {}).get("target_class", "")
        scope = res.get("scope", "project")
        body = {
            "update_entry_id": eid,
            "change_reason": (
                f"seed_resolutions: add resolution hint (cue Bash/'{tc}', "
                f"scope {scope}) drafted by {drafted_by or 'pinned'}"
            ),
            "hints": {"resolution": res},
        }
        resp = await http.post("/api/kb/store", json=body)
        if resp.status_code // 100 != 2:
            print(
                f"{eid}: write failed status={resp.status_code} detail={resp.text}",
                file=sys.stderr,
            )
            _reject(report, "write_failed")
            _set(rows[eid], "write", "write_failed", f"status_{resp.status_code}")
            continue
        sent.append(eid)
    if not sent:
        return
    fetched = await _fetch_entries(http, sent)
    for eid in sent:
        res, _ = kept[eid]
        want_tc = (res.get("cue") or {}).get("target_class")
        want_capture = (res.get("provenance") or {}).get("capture", "deliberate")
        stored = ((fetched.get(eid) or {}).get("hints") or {}).get("resolution") or {}
        got_tc = (stored.get("cue") or {}).get("target_class")
        got_capture = (stored.get("provenance") or {}).get("capture")
        if got_tc == want_tc and got_capture == want_capture:
            report.written += 1
            _set(rows[eid], "write", "written", "ok")
        elif got_tc == want_tc:
            print(
                f"{eid}: capture_forced wanted={want_capture} got={got_capture}"
                " (was the write made with a machine-principal key?)",
                file=sys.stderr,
            )
            _reject(report, "capture_forced")
            _set(rows[eid], "verify", "verify_mismatch", "capture_forced")
        else:
            print(f"{eid}: verify mismatch", file=sys.stderr)
            _reject(report, "verify_mismatch")
            _set(rows[eid], "verify", "verify_mismatch", "verify_mismatch")


def exit_code(report: SeedReport) -> int:
    """1 when any write/verify failure was recorded, else 0."""
    bad = ("write_failed", "verify_mismatch", "capture_forced")
    return 1 if any(report.rejected.get(k, 0) > 0 for k in bad) else 0


# ─── audit ───────────────────────────────────────────────────────────────────


async def audit(http: httpx.AsyncClient, entry_types: list[str]) -> dict[str, Any]:
    """Re-validate every stored resolution against the live validator and cues."""
    types = list(dict.fromkeys([*entry_types, *AUDIT_EXTRA_TYPES]))
    ids = await _enumerate_ids(http, types)
    entries = await _fetch_entries(http, ids)
    out: dict[str, Any] = {"audited": 0, "ok": 0, "drift": [], "invalid": []}
    for eid in ids:
        entry = entries.get(eid)
        if entry is None or entry.get("superseded_by"):
            continue
        hints = entry.get("hints") or {}
        if "resolution" not in hints:
            continue
        out["audited"] += 1
        tc = (
            _bash_cue(hints["resolution"])
            if isinstance(hints["resolution"], dict)
            else None
        )
        if tc is not None and target_class("Bash", tc) != tc:
            out["drift"].append(
                {
                    "entry_id": eid,
                    "stored_target_class": tc,
                    "normalized_now": target_class("Bash", tc),
                }
            )
            continue
        try:
            validate_and_stamp_resolution(
                hints, is_machine=False, entry_type=str(entry.get("entry_type"))
            )
        except ResolutionHintError as exc:
            out["invalid"].append({"entry_id": eid, "reason": exc.reason})
            continue
        out["ok"] += 1
    return out


def audit_exit_code(result: dict[str, Any]) -> int:
    """1 when drift or invalid is non-empty, else 0."""
    return 1 if result["drift"] or result["invalid"] else 0


# ─── CLI ─────────────────────────────────────────────────────────────────────


def _build_llm(model: str) -> LLM | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    from kb_core.config import AnthropicProviderConfig
    from kb_core.llm.anthropic import AnthropicLLMClient

    return AnthropicLLMClient(AnthropicProviderConfig(model=model, api_key=api_key))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else "")
    p.add_argument("--apply", action="store_true", help="write (default: dry run)")
    p.add_argument("--pinned-file", default=None)
    p.add_argument("--entry-ids", default=None)
    p.add_argument("--entry-types", default=DEFAULT_ENTRY_TYPES)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--report-path", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--audit", action="store_true")
    return p.parse_args(argv)


async def _amain(args: argparse.Namespace) -> int:
    url = os.environ.get("PERSONAL_KB_URL") or DEFAULT_URL
    key = resolve_api_key(url, os.environ)
    model = (
        args.model or os.environ.get("KB_ANTHROPIC_MODEL") or _DEFAULT_ANTHROPIC_MODEL
    )
    types = [t for t in args.entry_types.split(",") if t]
    report_path = args.report_path or f"./seed_resolutions_report-{_now()}.json"
    async with httpx.AsyncClient(
        base_url=url, headers={"Authorization": f"Bearer {key}"}, timeout=60.0
    ) as http:
        if args.audit:
            result = await audit(http, types)
            text = json.dumps(result, indent=2)
            code = audit_exit_code(result)
        else:
            pinned: dict[str, dict[str, Any]] = {}
            if args.pinned_file:
                with open(args.pinned_file) as fh:
                    pinned = json.load(fh)
            ids = [i for i in (args.entry_ids or "").split(",") if i] or None
            report = await run(
                http,
                _build_llm(model),
                apply=args.apply,
                entry_ids=ids,
                entry_types=types,
                limit=args.limit,
                pinned=pinned,
                model=model,
            )
            text = json.dumps(dataclasses.asdict(report), indent=2)
            code = exit_code(report)
    print(text)
    with open(report_path, "w") as fh:
        fh.write(text + "\n")
    return code


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    return asyncio.run(_amain(_parse_args(argv)))


if __name__ == "__main__":
    sys.exit(main())
