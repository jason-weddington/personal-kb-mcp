"""Wire-contract fixtures under ``contracts/`` versus the live service."""

import copy
import json
import re
import socket
import sys
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from jsonschema import Draft202012Validator
from kb_core import create_sqlite

import kb_service.database as database
from kb_service.auth import get_current_user
from kb_service.main import app
from kb_service.models import PreventionResponse, TurnDigestRequest, TurnDigestResponse
from kb_service.routes import turn_routes
from tests.conftest import FakeKnowledgeBase, fake_user

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gen_contracts as gc  # noqa: E402

_REGEN = "uv run python packages/kb-service/scripts/gen_contracts.py"
_SCRUB_ENV = (
    "KB_SERVICE_DATABASE_URL",
    "KB_DATABASE_URL",
    "ANTHROPIC_API_KEY",
    "KB_AWS_PROFILE",
)
_SWITCHES = (
    "KB_SOFT_GATE_ENABLED",
    "KB_SOFT_GATE_SHADOW",
    "KB_SOFT_GATE_DISABLED_PROJECTS",
    "KB_DELIVER_OBSERVED_ONCE",
    "KB_SURPRISE_CAPTURE",
    "KB_SOFT_GATE_MAX_DENIES_PER_TURN",
    "KB_SOFT_GATE_MAX_DENIES_PER_HOUR",
    "KB_SOFT_GATE_REARM_HOURS",
)


def _load(name: str) -> dict[str, Any]:
    doc = json.loads((gc.CONTRACTS_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(doc, dict)
    return doc


def _examples(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {i["name"]: i["body"] for i in items}


def _prevention() -> dict[str, Any]:
    return _load("prevention.json")


def _turn() -> dict[str, Any]:
    return _load("turn.json")


# --- freshness and generator CLI ---------------------------------------------


@pytest.mark.parametrize("name", sorted(gc.build_contracts()))
def test_fixture_is_fresh(name: str) -> None:
    doc = gc.build_contracts()[name]
    committed = (gc.CONTRACTS_DIR / name).read_text(encoding="utf-8")
    if committed != gc.render(doc):
        where = gc.first_difference(json.loads(committed), doc) or "formatting only"
        pytest.fail(
            f"contracts/{name} is stale (first difference at {where}); run"
            f" `{_REGEN}` and see contracts/README.md for contract_version"
        )


def test_no_unexpected_fixture_files() -> None:
    found = {p.name for p in gc.CONTRACTS_DIR.glob("*.json")}
    assert found == set(gc.build_contracts())


def test_check_mode_passes() -> None:
    assert gc.main(["--check"]) == 0


def test_generator_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert gc.main(["--out-dir", str(tmp_path)]) == 0
    for name in gc.build_contracts():
        assert (tmp_path / name).read_bytes() == (gc.CONTRACTS_DIR / name).read_bytes()
    capsys.readouterr()

    prevention = tmp_path / "prevention.json"
    doc = json.loads(prevention.read_text(encoding="utf-8"))
    doc["contract_version"] = 99
    prevention.write_text(gc.render(doc), encoding="utf-8")
    assert gc.main(["--check", "--out-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "drifted: prevention.json (first difference at contract_version)" in out
    assert "breaking: prevention.json contract_version went down (99 -> 1)" in out

    assert gc.main(["--out-dir", str(tmp_path)]) == 1
    capsys.readouterr()
    prevention.unlink()
    assert gc.main(["--out-dir", str(tmp_path)]) == 0
    capsys.readouterr()

    turn = tmp_path / "turn.json"
    doc = json.loads(turn.read_text(encoding="utf-8"))
    doc["request"]["schema"]["properties"]["legacy"] = {"type": "string"}
    turn.write_text(gc.render(doc), encoding="utf-8")
    assert gc.main(["--check", "--out-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert (
        "drifted: turn.json (first difference at request.schema.properties.legacy)"
        in out
    )
    assert (
        "breaking: turn.json at request.schema.properties.legacy (property-removed);"
        ' bump CONTRACT_VERSIONS["turn"]' in out
    )
    before = {p.name: p.read_bytes() for p in tmp_path.glob("*.json")}
    assert gc.main(["--out-dir", str(tmp_path)]) == 1
    assert {p.name: p.read_bytes() for p in tmp_path.glob("*.json")} == before
    capsys.readouterr()

    (tmp_path / "x.json").write_text("{}\n", encoding="utf-8")
    assert gc.main(["--check", "--out-dir", str(tmp_path)]) == 1
    assert "unexpected: x.json" in capsys.readouterr().out


def test_check_mode_reports_missing_and_unparsable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "turn.json").write_text("not json", encoding="utf-8")
    assert gc.main(["--check", "--out-dir", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "missing: prevention.json" in out
    assert "drifted: turn.json (unparsable)" in out


@pytest.mark.parametrize(
    ("a", "b", "want"),
    [
        ({"a": {"b": 1}}, {"a": {"b": 2}}, "a.b"),
        ({"a": [1, 2]}, {"a": [1]}, "a[1]"),
        ({"a": 1}, {"a": 1, "b": 2}, "b"),
        ([], {}, "<root>"),
        ({"x": 1}, {"x": 1}, None),
        (1, 2, "<root>"),
        ({"a": 1}, {"a": "1"}, "a"),
        ([1], [1, 2], "[1]"),
    ],
)
def test_first_difference(a: Any, b: Any, want: str | None) -> None:
    assert gc.first_difference(a, b) == want


# --- breaking_changes ---------------------------------------------------------


def _turn_schema(doc: dict[str, Any], zone: str) -> dict[str, Any]:
    schema = doc[zone]["schema"]
    assert isinstance(schema, dict)
    return schema


def _mut_del_ts(d: dict[str, Any]) -> None:
    del _turn_schema(d, "request")["properties"]["ts"]


def _mut_req_add(d: dict[str, Any]) -> None:
    _turn_schema(d, "request")["required"].append("ts")


def _mut_resp_req_del(d: dict[str, Any]) -> None:
    _turn_schema(d, "response")["required"].remove("reason")


def _reason_enum(d: dict[str, Any]) -> list[str]:
    enum = _turn_schema(d, "response")["properties"]["reason"]["enum"]
    assert isinstance(enum, list)
    return enum


def _mut_enum_remove(d: dict[str, Any]) -> None:
    _reason_enum(d).remove("write-failed")


def _mut_enum_add(d: dict[str, Any]) -> None:
    _reason_enum(d).append("rate-limited")


def _mut_max_length(d: dict[str, Any]) -> None:
    defs = _turn_schema(d, "request")["$defs"]
    defs["TurnAssistantTextItem"]["properties"]["text"]["maxLength"] = 1999


def _mut_max_body(d: dict[str, Any]) -> None:
    d["request"]["max_body_bytes"] = 65535


def _mut_min_items(d: dict[str, Any]) -> None:
    _turn_schema(d, "request")["properties"]["items"]["minItems"] = 1


def _mut_kind_removed(d: dict[str, Any]) -> None:
    items = _turn_schema(d, "request")["properties"]["items"]["items"]
    del items["discriminator"]["mapping"]["tool_result"]


def _mut_def_removed(d: dict[str, Any]) -> None:
    del _turn_schema(d, "request")["$defs"]["TurnToolResultItem"]


def _mut_type(d: dict[str, Any]) -> None:
    _turn_schema(d, "request")["properties"]["turn_index"]["type"] = "string"


def _mut_path(d: dict[str, Any]) -> None:
    d["path"] = "/api/kb/turn2"


_BREAKING_ROWS = [
    (_mut_del_ts, "request.schema.properties.ts (property-removed)"),
    (_mut_req_add, "request.schema.required.ts (required-added)"),
    (_mut_resp_req_del, "response.schema.required.reason (required-removed)"),
    (
        _mut_enum_remove,
        "response.schema.properties.reason.enum.write-failed (enum-removed)",
    ),
    (
        _mut_enum_add,
        "response.schema.properties.reason.enum.rate-limited (enum-added)",
    ),
    (
        _mut_max_length,
        "request.schema.$defs.TurnAssistantTextItem.properties.text.maxLength"
        " (limit-tightened)",
    ),
    (_mut_max_body, "request.max_body_bytes (limit-tightened)"),
    (_mut_min_items, "request.schema.properties.items.minItems (limit-added)"),
    (
        _mut_kind_removed,
        "request.schema.properties.items.items.discriminator.mapping.tool_result"
        " (kind-removed)",
    ),
    (_mut_def_removed, "request.schema.$defs.TurnToolResultItem (def-removed)"),
    (_mut_type, "request.schema.properties.turn_index.type (type-changed)"),
    (_mut_path, "path (endpoint-changed)"),
]


@pytest.mark.parametrize(
    ("mutate", "want"), _BREAKING_ROWS, ids=[w for _, w in _BREAKING_ROWS]
)
def test_breaking_changes_detected(mutate: Any, want: str) -> None:
    old = gc.build_turn()
    new = copy.deepcopy(old)
    mutate(new)
    assert want in gc.breaking_changes(old, new)


def test_prevention_enum_added_is_breaking() -> None:
    old = gc.build_prevention()
    new = copy.deepcopy(old)
    new["response"]["schema"]["properties"]["surprise_capture"]["enum"].append("auto")
    assert (
        "response.schema.properties.surprise_capture.enum.auto (enum-added)"
        in gc.breaking_changes(old, new)
    )


def _add_request_prop(d: dict[str, Any]) -> None:
    _turn_schema(d, "request")["properties"]["extra"] = {"type": "string"}


def _add_request_mode(d: dict[str, Any]) -> None:
    props = _turn_schema(d, "request")["properties"]
    props["mode"]["enum"].append("batch")


def _add_kind_and_def(d: dict[str, Any]) -> None:
    schema = _turn_schema(d, "request")
    mapping = schema["properties"]["items"]["items"]["discriminator"]["mapping"]
    mapping["new_kind"] = "#/$defs/TurnNewItem"
    schema["$defs"]["TurnNewItem"] = {"type": "object"}


def _description_changed(d: dict[str, Any]) -> None:
    _turn_schema(d, "request")["description"] = "changed"


def _raise_max_length(d: dict[str, Any]) -> None:
    defs = _turn_schema(d, "request")["$defs"]
    defs["TurnAssistantTextItem"]["properties"]["text"]["maxLength"] = 5000


def _request_required_removed(d: dict[str, Any]) -> None:
    _turn_schema(d, "request")["required"].pop()


@pytest.mark.parametrize(
    "mutate",
    [
        _add_request_prop,
        _add_request_mode,
        _add_kind_and_def,
        _description_changed,
        _raise_max_length,
        _request_required_removed,
    ],
)
def test_additive_turn_changes_are_not_breaking(mutate: Any) -> None:
    old = gc.build_turn()
    new = copy.deepcopy(old)
    mutate(new)
    assert gc.breaking_changes(old, new) == []


def test_new_query_param_is_not_breaking() -> None:
    old = gc.build_prevention()
    new = copy.deepcopy(old)
    new["request"]["query_schema"]["properties"]["extra"] = {"type": "string"}
    assert gc.breaking_changes(old, new) == []


# --- env independence ---------------------------------------------------------


def test_build_is_env_independent(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline_env = {
        "KB_SOFT_GATE_ENABLED": "TRUE",
        "KB_SOFT_GATE_SHADOW": "FALSE",
        "KB_SOFT_GATE_MAX_DENIES_PER_TURN": "5",
        "KB_SURPRISE_CAPTURE": "on",
    }
    for key, value in baseline_env.items():
        monkeypatch.setenv(key, value)
    with_env = gc.build_contracts()
    for key in baseline_env:
        monkeypatch.delenv(key, raising=False)
    without_env = gc.build_contracts()
    assert with_env == without_env
    monkeypatch.setattr(socket, "gethostname", lambda: "ambient-host-xyz")
    assert gc.build_contracts() == without_env


# --- coverage -----------------------------------------------------------------


def test_every_item_kind_has_an_example() -> None:
    examples = _turn()["request"]["examples"]
    kinds = {i["kind"] for ex in examples for i in ex["body"]["items"]}
    assert kinds == set(gc.turn_item_kinds())
    claude = _examples(examples)["claude_code_turn"]
    assert {i["kind"] for i in claude["items"]} == {
        "assistant_text",
        "tool_call",
        "tool_result",
    }


def test_missing_item_kind_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    kept = [i for i in gc._TALOS_ITEMS if i["kind"] != "reasoning"]
    monkeypatch.setattr(gc, "_TALOS_ITEMS", kept)
    with pytest.raises(ValueError, match=re.escape("['reasoning']")):
        gc.build_turn()


def test_response_examples_follow_reason_enum() -> None:
    names = [e["name"] for e in _turn()["response"]["examples"]]
    live = TurnDigestResponse.model_json_schema()["properties"]["reason"]["enum"]
    assert names == live


@pytest.mark.parametrize(
    ("model", "doc_name", "zone"),
    [
        (PreventionResponse, "prevention.json", "response"),
        (TurnDigestRequest, "turn.json", "request"),
    ],
)
def test_defaulted_fields_are_exercised(model: Any, doc_name: str, zone: str) -> None:
    bodies = [e["body"] for e in _load(doc_name)[zone]["examples"]]
    for name, field in model.model_fields.items():
        if field.is_required():
            continue
        default = field.get_default(call_default_factory=True)
        assert any(b[name] != default for b in bodies), (
            f"no committed example sets {model.__name__}.{name} off its default"
        )


# --- envelopes, round trip, schema --------------------------------------------


def test_example_envelopes() -> None:
    prevention, turn = _prevention(), _turn()
    lists = [
        (prevention["request"]["examples"], {"name", "query"}),
        (prevention["response"]["examples"], {"name", "body"}),
        (turn["request"]["examples"], {"name", "body"}),
        (turn["response"]["examples"], {"name", "body"}),
    ]
    for examples, keys in lists:
        assert isinstance(examples, list)
        assert all(isinstance(e, dict) and set(e) == keys for e in examples)
        names = [e["name"] for e in examples]
        assert len(names) == len(set(names))


def test_bodies_round_trip() -> None:
    cases = [
        (TurnDigestRequest, _turn()["request"]["examples"]),
        (TurnDigestResponse, _turn()["response"]["examples"]),
        (PreventionResponse, _prevention()["response"]["examples"]),
    ]
    for model, examples in cases:
        for ex in examples:
            parsed = model.model_validate(ex["body"])
            assert parsed.model_dump(mode="json") == ex["body"], ex["name"]


def _schemas() -> list[tuple[str, dict[str, Any]]]:
    prevention, turn = _prevention(), _turn()
    return [
        ("prevention query_schema", prevention["request"]["query_schema"]),
        ("prevention response", prevention["response"]["schema"]),
        ("turn request", turn["request"]["schema"]),
        ("turn response", turn["response"]["schema"]),
    ]


@pytest.mark.parametrize(
    ("label", "schema"), _schemas(), ids=[s[0] for s in _schemas()]
)
def test_schemas_are_valid(label: str, schema: dict[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)


def test_examples_validate_against_schemas() -> None:
    prevention, turn = _prevention(), _turn()
    query_v = Draft202012Validator(prevention["request"]["query_schema"])
    for ex in prevention["request"]["examples"]:
        assert query_v.is_valid(ex["query"]), ex["name"]
    pairs = [
        (prevention["response"]["schema"], prevention["response"]["examples"]),
        (turn["request"]["schema"], turn["request"]["examples"]),
        (turn["response"]["schema"], turn["response"]["examples"]),
    ]
    for schema, examples in pairs:
        validator = Draft202012Validator(schema)
        for ex in examples:
            assert validator.is_valid(ex["body"]), ex["name"]


def test_schemas_reject_bad_input() -> None:
    prevention, turn = _prevention(), _turn()
    turn_v = Draft202012Validator(turn["request"]["schema"])
    claude = copy.deepcopy(_examples(turn["request"]["examples"])["claude_code_turn"])
    claude["items"][2]["is_error"] = "yes"
    assert not turn_v.is_valid(claude)
    claude = copy.deepcopy(_examples(turn["request"]["examples"])["claude_code_turn"])
    claude["items"].append({"kind": "bogus"})
    assert not turn_v.is_valid(claude)
    armed = copy.deepcopy(_examples(prevention["response"]["examples"])["gate_armed"])
    del armed["gate"]
    assert not Draft202012Validator(prevention["response"]["schema"]).is_valid(armed)
    query_v = Draft202012Validator(prevention["request"]["query_schema"])
    assert not query_v.is_valid({"project": "p", "unknown": "x"})


def test_no_ambient_values() -> None:
    prevention, turn = _prevention(), _turn()
    texts = [
        json.dumps(e[key])
        for doc in (prevention, turn)
        for zone in ("request", "response")
        for e in doc[zone]["examples"]
        for key in ("body", "query")
        if key in e
    ]
    for text in texts:
        assert not re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", text)
        assert not re.search(r"https?://", text)

    def hosts(node: Any) -> Iterator[Any]:
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "host":
                    yield v
                yield from hosts(v)
        elif isinstance(node, list):
            for v in node:
                yield from hosts(v)

    for doc in (prevention, turn):
        examples = [e for z in ("request", "response") for e in doc[z]["examples"]]
        # the `minimal` example leaves host at its null default
        assert all(h in (None, "contract-host") for h in hosts(examples))


def test_fixture_routes_exist() -> None:
    routes = [r for r in app.routes if isinstance(r, APIRoute)]
    for doc in (_prevention(), _turn()):
        assert any(
            r.path == doc["path"] and doc["method"] in r.methods for r in routes
        ), doc["path"]


def test_readme_documents_the_contract() -> None:
    text = (gc.CONTRACTS_DIR / "README.md").read_text(encoding="utf-8")
    for needle in (
        "prevention.json",
        "turn.json",
        "contract_version",
        "gen_contracts.py",
        "turn_event reason=invalid",
        "turn_event reason=too-large",
        "route_outcomes",
        "prevention_fetch failed",
    ):
        assert needle in text, needle


# --- POST /api/kb/turn against the committed fixtures --------------------------


@pytest.fixture
def turn_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    for var in _SCRUB_ENV:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("KB_SURPRISE_CAPTURE", raising=False)
    monkeypatch.setenv("KB_AUTH_MODE", "none")
    monkeypatch.setenv("KB_DB_PATH", str(tmp_path / "knowledge.db"))
    monkeypatch.setenv("KB_OLLAMA_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("KB_OLLAMA_TIMEOUT", "0.5")
    monkeypatch.setenv("KB_EXTRACTION_PROVIDER", "ollama")
    monkeypatch.setenv("KB_QUERY_PROVIDER", "ollama")
    monkeypatch.setattr(database, "_pool", None)
    monkeypatch.setattr(turn_routes, "_OUTCOMES", {})
    monkeypatch.setattr(turn_routes, "_UNMAPPED", {})
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


_JSON = {"Content-Type": "application/json"}


def _post(client: TestClient, body: dict[str, Any]) -> Any:
    return client.post("/api/kb/turn", content=json.dumps(body), headers=_JSON)


def test_turn_capture_off_matches_fixture(turn_client: TestClient) -> None:
    turn = _turn()
    want = _examples(turn["response"]["examples"])["capture-off"]
    for ex in turn["request"]["examples"]:
        resp = _post(turn_client, ex["body"])
        assert resp.status_code == 200, ex["name"]
        assert resp.json() == want, ex["name"]


def test_turn_shadow_records_both_shapes(
    turn_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("KB_SURPRISE_CAPTURE", "shadow")
    turn = _turn()
    requests = _examples(turn["request"]["examples"])
    duplicate = _examples(turn["response"]["examples"])["duplicate"]
    schema = turn["response"]["schema"]
    validator = Draft202012Validator(schema)
    for name in ("claude_code_turn", "talos_turn"):
        resp = _post(turn_client, requests[name])
        assert resp.status_code == 200, name
        body = resp.json()
        assert body["reason"] == "recorded", name
        assert body["redactions"] == [], name
        assert validator.is_valid(body), name
        assert set(body) == set(schema["properties"]), name
    for name in ("claude_code_turn", "talos_turn"):
        again = _post(turn_client, requests[name])
        assert again.status_code == 200
        assert again.json() == duplicate, name


def test_turn_body_limit_boundary(turn_client: TestClient) -> None:
    turn = _turn()
    limit = turn["request"]["max_body_bytes"]
    base = copy.deepcopy(_examples(turn["request"]["examples"])["claude_code_turn"])
    base["user_prompt"] = ""
    base["items"] = [{"kind": "assistant_text", "text": "x" * 2000}] * 31
    pad = limit - len(json.dumps(base).encode())
    assert 0 < pad <= 4000
    base["user_prompt"] = "p" * pad
    raw = json.dumps(base).encode()
    assert len(raw) == limit
    ok = turn_client.post("/api/kb/turn", content=raw, headers=_JSON)
    assert ok.status_code == 200
    assert ok.json()["reason"] == "capture-off"
    base["user_prompt"] = "p" * (pad + 1)
    big = turn_client.post(
        "/api/kb/turn", content=json.dumps(base).encode(), headers=_JSON
    )
    assert big.status_code == 413


# --- GET /api/kb/prevention against the committed fixtures ---------------------


@pytest.fixture(autouse=True)
def _clear_switches(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _SWITCHES:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
async def kb(tmp_path: Path) -> AsyncIterator[Any]:
    kb = await create_sqlite(
        tmp_path / "kb.db",
        embedder=None,
        extraction_llm=None,
        query_llm=None,
        synthesis_llm=None,
    )
    try:
        yield kb
    finally:
        await kb.close()


@pytest.fixture
def real_client(
    client: TestClient,
    fake_kb: FakeKnowledgeBase,
    kb: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TestClient]:
    monkeypatch.setattr(fake_kb, "db", kb.db)
    app.dependency_overrides[get_current_user] = fake_user
    yield client


async def _seed(kb: Any) -> None:
    await kb.store(
        short_title="Contract push",
        long_title="Contract push resolution",
        knowledge_details="contract",
        project_ref="personal-kb",
        hints={
            "resolution": {
                "corrected_fact": "Push to origin.",
                "wrong_belief": "Push to the mirror.",
                "cue": {"tool": "Bash", "target_class": "git push"},
                "provenance": {"capture": "deliberate", "grounding": "observed"},
            }
        },
        enrich=False,
    )


async def test_prevention_hook_query_matches_fixture(
    kb: Any, real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(kb)
    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    prevention = _prevention()
    queries = _examples(
        [
            {"name": e["name"], "body": e["query"]}
            for e in prevention["request"]["examples"]
        ]
    )
    resp = real_client.get("/api/kb/prevention", params=queries["project_and_cwd"])
    assert resp.status_code == 200
    body = resp.json()
    assert Draft202012Validator(prevention["response"]["schema"]).is_valid(body)
    assert len(body["index"]) == 1
    ex = _examples(prevention["response"]["examples"])["gate_armed"]
    assert set(body) == set(ex)
    assert set(body["gate"]) == set(ex["gate"])
    assert set(body["diagnostics"]) == set(ex["diagnostics"])
    assert set(body["index"][0]) == set(ex["index"][0])
    assert set(body["slice"][0]) == set(ex["slice"][0])
    assert body["tool_map"] == {}


async def test_prevention_talos_query_matches_fixture(
    kb: Any, real_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _seed(kb)
    monkeypatch.setenv("KB_SOFT_GATE_ENABLED", "TRUE")
    prevention = _prevention()
    query = next(
        e["query"] for e in prevention["request"]["examples"] if e["name"] == "talos"
    )
    resp = real_client.get("/api/kb/prevention", params=query)
    assert resp.status_code == 200
    body = resp.json()
    ex = _examples(prevention["response"]["examples"])["talos_gate_armed"]
    assert body["tool_map"] == ex["tool_map"]
    assert body["index"][0]["tool"] == "Bash"
