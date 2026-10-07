"""Hermetic tests for ``scripts/seed_resolutions.py`` (MockTransport + fake LLM)."""

import json
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import seed_resolutions as sr  # noqa: E402
from kb_core.cues import target_class  # noqa: E402

PIN_ID = "kb-00001"
PINNED = {
    PIN_ID: {
        "corrected_fact": "To create a new repo on the git server, run `bash "
        "~/scripts/add_remote.sh [remote-name]` from inside the repo directory.",
        "cue": {"tool": "Bash", "target_class": "git remote", "args_prefix": "add"},
        "scope": "global",
        "provenance": {"capture": "deliberate", "grounding": "asserted"},
    }
}


def _entry(
    eid: str,
    *,
    project: str = "proj",
    hints: dict[str, Any] | None = None,
    details: str = "Run `bash ~/scripts/x.sh` to do it.",
    superseded_by: str | None = None,
    entry_type: str = "pattern_convention",
) -> dict[str, Any]:
    return {
        "id": eid,
        "short_title": f"title {eid}",
        "project_ref": project,
        "entry_type": entry_type,
        "knowledge_details": details,
        "hints": hints or {},
        "superseded_by": superseded_by,
    }


class FakeService:
    """In-memory stand-in for the three endpoints the script uses."""

    def __init__(
        self, entries: list[dict[str, Any]], *, force_capture: str | None = None
    ):
        self.entries = {e["id"]: e for e in entries}
        self.store_posts: list[dict[str, Any]] = []
        self.force_capture = force_capture
        self.auth: list[str] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth.append(request.headers.get("authorization", ""))
        path = request.url.path
        if path == "/api/kb/graph/scope-entries":
            scope = request.url.params["scope"]
            ids = [i for i, e in self.entries.items() if e["entry_type"] == scope]
            return httpx.Response(200, json={"entry_ids": ids})
        if path == "/api/kb/get":
            ids = json.loads(request.content)["ids"]
            assert len(ids) <= 20
            results = []
            for i in ids:
                e = self.entries.get(i)
                results.append(
                    {"id": i, "found": e is not None, "entry": e, "pointer_rot": []}
                )
            return httpx.Response(200, json={"results": results})
        if path == "/api/kb/store":
            body = json.loads(request.content)
            self.store_posts.append(body)
            e = self.entries[body["update_entry_id"]]
            res = json.loads(json.dumps(body["hints"]["resolution"]))
            prov = res.setdefault("provenance", {})
            prov["capture"] = self.force_capture or prov.get("capture") or "deliberate"
            e["hints"] = {**e["hints"], "resolution": res}
            return httpx.Response(200, json={"action": "updated"})
        return httpx.Response(404)


class FakeLLM:
    def __init__(self, replies: dict[str, Any]) -> None:
        self.replies = replies
        self.calls = 0

    async def generate(self, prompt: str, *, system: str | None = None) -> str | None:
        self.calls += 1
        for eid, reply in self.replies.items():
            if f"title {eid}" in prompt:
                return reply if isinstance(reply, str) else json.dumps(reply)
        return None


def _reply(tc: str = "git remote", example: str = "git remote add origin x", **kw: Any):
    base = {
        "propose": True,
        "corrected_fact": "Use the script.",
        "cue_target_class": tc,
        "example_command": example,
        "scope": "project",
        "reason": "r",
    }
    return {**base, **kw}


async def _run(
    svc: FakeService,
    llm: FakeLLM | None,
    *,
    apply: bool = False,
    pinned: dict[str, Any] | None = None,
    **kw: Any,
) -> sr.SeedReport:
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(svc.handler), base_url="http://test"
    ) as http:
        return await sr.run(
            http,
            llm,
            apply=apply,
            entry_ids=kw.pop("entry_ids", None),
            entry_types=["pattern_convention", "factual_reference"],
            limit=kw.pop("limit", 0),
            pinned=pinned or {},
            model="m",
        )


def test_fixture_commands_hit_the_cue() -> None:
    assert (
        target_class("Bash", "git remote add origin git@example-host:repos/example")
        == "git remote"
    )
    assert target_class("Bash", "bash ~/scripts/add_remote.sh") == "bash"


async def test_dry_run_reports_and_writes_nothing() -> None:
    svc = FakeService([_entry(PIN_ID), _entry("kb-00002")])
    llm = FakeLLM({"kb-00002": _reply("docker compose", "docker compose up")})
    rep = await _run(svc, llm, pinned=PINNED)
    assert rep.proposed == 2 and rep.written == 0
    assert svc.store_posts == []
    pin = next(p for p in rep.proposals if p["entry_id"] == PIN_ID)
    assert pin["args_prefix"] == "add"
    assert llm.calls == 1


async def test_apply_writes_pinned_deliberate() -> None:
    svc = FakeService([_entry(PIN_ID)])
    llm = FakeLLM({})
    rep = await _run(svc, llm, apply=True, pinned=PINNED)
    assert rep.written == 1 and llm.calls == 0
    post = svc.store_posts[0]
    assert set(post) == {"update_entry_id", "change_reason", "hints"}
    assert post["change_reason"].startswith("seed_resolutions: add resolution hint")
    assert "pinned" in post["change_reason"]
    res = post["hints"]["resolution"]
    assert res["cue"]["target_class"] == "git remote"
    assert res["cue"]["args_prefix"] == "add"
    assert res["scope"] == "global"
    assert res["provenance"]["capture"] == "deliberate"
    assert sr.exit_code(rep) == 0


async def test_second_apply_is_idempotent() -> None:
    svc = FakeService([_entry(PIN_ID)])
    await _run(svc, None, apply=True, pinned=PINNED)
    rep = await _run(svc, None, apply=True, pinned=PINNED)
    assert rep.already_resolved == 1 and rep.written == 0
    assert len(svc.store_posts) == 1


@pytest.mark.parametrize(
    ("reply", "key"),
    [
        (_reply("git status", "git status"), "denylisted"),
        (_reply("git remote add", "git remote add x"), "not_normalized"),
        (_reply("ls", "ls -l"), "not_two_word"),
        (_reply("git remote", "ssh host create x"), "example_mismatch"),
        ("not json at all", "llm_unparseable"),
        ({"propose": True}, "llm_unparseable"),
        (_reply(propose=False), "declined"),
        (_reply(scope="team"), "bad_scope"),
        (_reply(corrected_fact="  "), "empty_fact"),
        (_reply(corrected_fact="x" * 301), "empty_fact"),
    ],
)
async def test_guard_reasons(reply: Any, key: str) -> None:
    svc = FakeService([_entry("kb-00002")])
    rep = await _run(svc, FakeLLM({"kb-00002": reply}))
    assert rep.rejected == {key: 1}
    assert rep.proposed == 0


async def test_collision_same_project_lowest_id_wins() -> None:
    svc = FakeService([_entry("kb-00002"), _entry("kb-00003")])
    llm = FakeLLM({"kb-00002": _reply(), "kb-00003": _reply()})
    rep = await _run(svc, llm)
    assert rep.proposed == 1 and rep.rejected == {"class_collision": 1}
    assert rep.proposals[0]["entry_id"] == "kb-00002"
    row = next(d for d in rep.decisions if d["entry_id"] == "kb-00003")
    assert row["reason"] == "class_collision"


async def test_collision_global_across_projects() -> None:
    svc = FakeService(
        [_entry("kb-00002", project="a"), _entry("kb-00003", project="b")]
    )
    llm = FakeLLM(
        {"kb-00002": _reply(scope="global"), "kb-00003": _reply(scope="global")}
    )
    rep = await _run(svc, llm)
    assert rep.rejected == {"class_collision": 1}


async def test_collision_seeded_by_existing_resolution() -> None:
    existing = {
        "resolution": {
            "corrected_fact": "x",
            "cue": {"tool": "Bash", "target_class": "git remote"},
            "scope": "global",
        }
    }
    svc = FakeService(
        [
            _entry("kb-00002", project="a", hints=existing),
            _entry("kb-00003", project="b"),
        ]
    )
    rep = await _run(svc, FakeLLM({"kb-00003": _reply(scope="global")}))
    assert rep.already_resolved == 1
    assert rep.rejected == {"class_collision": 1}


async def test_args_prefix_from_llm() -> None:
    svc = FakeService([_entry("kb-00002")])
    rep = await _run(svc, FakeLLM({"kb-00002": _reply(cue_args_prefix="add")}))
    assert rep.proposals[0]["args_prefix"] == "add"


async def test_superseded_never_sent_to_llm() -> None:
    svc = FakeService([_entry("kb-00002", superseded_by="kb-00009")])
    llm = FakeLLM({"kb-00002": _reply()})
    rep = await _run(svc, llm)
    assert llm.calls == 0 and rep.prefiltered_out == 1


async def test_capture_forced_detected() -> None:
    svc = FakeService([_entry(PIN_ID)], force_capture="autonomous")
    rep = await _run(svc, None, apply=True, pinned=PINNED)
    assert rep.rejected == {"capture_forced": 1}
    assert rep.written == 0
    assert sr.exit_code(rep) == 1


async def test_every_scanned_id_has_one_decision() -> None:
    svc = FakeService(
        [
            _entry(PIN_ID),
            _entry("kb-00002"),
            _entry("kb-00003", details="nothing to see"),
            _entry("kb-00004", entry_type="mental_map"),
        ]
    )
    rep = await _run(
        svc,
        FakeLLM({"kb-00002": _reply()}),
        pinned=PINNED,
        entry_ids=[PIN_ID, "kb-00002", "kb-00003", "kb-00004", "kb-99999"],
    )
    ids = [d["entry_id"] for d in rep.decisions]
    assert sorted(ids) == sorted(set(ids)) and len(ids) == rep.scanned == 5


async def test_no_llm_counts_and_pinned_still_processed() -> None:
    svc = FakeService([_entry(PIN_ID), _entry("kb-00002")])
    rep = await _run(svc, None, pinned=PINNED)
    assert rep.rejected == {"no_llm": 1}
    assert rep.proposed == 1


async def test_audit_reports_drift_and_ok() -> None:
    def res(tc: str) -> dict[str, Any]:
        return {
            "resolution": {
                "corrected_fact": "x",
                "cue": {"tool": "Bash", "target_class": tc},
            }
        }

    svc = FakeService(
        [
            _entry("kb-00002", hints=res("git remote add")),
            _entry("kb-00003", hints=res("git remote")),
            _entry("kb-00004", hints={"resolution": {"corrected_fact": ""}}),
        ]
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(svc.handler), base_url="http://test"
    ) as http:
        out = await sr.audit(http, ["pattern_convention"])
    assert out["audited"] == 3 and out["ok"] == 1
    assert out["drift"] == [
        {
            "entry_id": "kb-00002",
            "stored_target_class": "git remote add",
            "normalized_now": "git remote",
        }
    ]
    assert out["invalid"] == [{"entry_id": "kb-00004", "reason": "corrected_fact"}]
    assert sr.audit_exit_code(out) == 1


def test_key_resolution() -> None:
    assert sr.resolve_api_key("http://127.0.0.1:8765", {}) == "local-no-auth"
    assert sr.resolve_api_key("http://localhost:1", {"PERSONAL_KB_API_KEY": "k"}) == "k"
    with pytest.raises(SystemExit) as ei:
        sr.resolve_api_key("https://kb.example.com", {})
    assert ei.value.code == 2


async def test_bearer_header_sent() -> None:
    svc = FakeService([])
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(svc.handler),
        base_url="http://test",
        headers={
            "Authorization": f"Bearer {sr.resolve_api_key('http://127.0.0.1', {})}"
        },
    ) as http:
        await sr.audit(http, ["pattern_convention"])
    assert svc.auth[0] == "Bearer local-no-auth"
