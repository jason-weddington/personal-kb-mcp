"""Hermetic tests for the surprise-detector eval harness (``scripts/surprise_eval``).

Model calls go through a StubLLM installed via the module's ``make_llm`` seam,
or through the real ``AnthropicLLMClient`` with ``_get_client`` patched to a
fake; nothing touches the network or reads a real API key.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import re
import sys
import types
import typing
from pathlib import Path
from typing import Any

import pytest
from kb_core.llm import anthropic as anthropic_llm
from kb_service import config as service_config
from kb_service import surprise, surprise_distill, surprise_worker, turn_digest
from kb_service.models import TurnItem

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO_ROOT / "scripts" / "surprise_eval"
SCRIPT_PATH = EVAL_DIR / "surprise_eval.py"
DIGESTS_FIXTURE = EVAL_DIR / "fixtures" / "synthetic_cases.jsonl"
MINED_FIXTURE = EVAL_DIR / "fixtures" / "synthetic_mined_cases.jsonl"
CONCLUSION_FIXTURE = EVAL_DIR / "fixtures" / "synthetic_shape3_conclusion_cases.jsonl"
TURNS_FIXTURE = EVAL_DIR / "fixtures" / "synthetic_turns_cases.jsonl"
HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _load(name: str, path: Path):  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


se = _load("surprise_eval_harness", SCRIPT_PATH)


def _verdict(wrong: str, fact: str, evidence: str, confidence: float) -> str:
    return json.dumps(
        {
            "surprise": True,
            "wrong_belief": wrong,
            "corrected_fact": fact,
            "evidence_excerpt": evidence,
            "confidence": confidence,
        }
    )


def _perfect(confidence: float = 0.9) -> dict[str, str]:
    return {
        "8080 is taken by caddy": _verdict(
            "Port 8080 is free",
            "8080 is taken by caddy",
            "8080 is taken by caddy",
            confidence,
        ),
        "config is at /etc/foo/main.conf": _verdict(
            "The config lives in /etc/foo.conf",
            "config is at /etc/foo/main.conf",
            "config is at /etc/foo/main.conf",
            confidence,
        ),
    }


PERFECT = _perfect()
NO_SURPRISE = '{"surprise": false}'


class StubLLM:
    # (model, prompt, system, purpose)
    calls: typing.ClassVar[list[tuple[str, str, str | None, str]]] = []
    mapping: typing.ClassVar[dict[str, str]] = {}
    default: typing.ClassVar[str | None] = NO_SURPRISE
    delay: typing.ClassVar[float] = 0.0
    raise_purposes: typing.ClassVar[set[str]] = set()

    def __init__(self, model: str) -> None:
        self._config = types.SimpleNamespace(model=model)
        self.closed = False

    async def generate(self, prompt: str, *, purpose: str, system: str | None = None) -> str | None:
        StubLLM.calls.append((self._config.model, prompt, system, purpose))
        if StubLLM.delay:
            await asyncio.sleep(StubLLM.delay)
        if purpose in StubLLM.raise_purposes:
            raise RuntimeError("stub down")
        for key, value in StubLLM.mapping.items():
            if key in prompt:
                return value
        return StubLLM.default

    async def generate_chat(
        self, messages: Any, *, purpose: str, system: str | None = None
    ) -> str | None:
        return None

    async def is_available(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "KB_SURPRISE_MIN_CONFIDENCE", "KB_SURPRISE_DETECTOR_MODEL"):
        monkeypatch.delenv(var, raising=False)
    StubLLM.calls = []
    StubLLM.mapping = dict(PERFECT)
    StubLLM.default = NO_SURPRISE
    StubLLM.delay = 0.0
    StubLLM.raise_purposes = set()


@pytest.fixture
def made(monkeypatch):
    built: list[StubLLM] = []

    def factory(
        model: str, timeout: float | None, effort: str | None = None, thinking: str | None = None
    ) -> StubLLM:
        llm = StubLLM(model)
        llm.arm = (model, effort, thinking)  # type: ignore[attr-defined]
        built.append(llm)
        return llm

    monkeypatch.setattr(se, "make_llm", factory)
    return built


def _run(capsys, out: Path, cases: Path, *extra: str) -> dict[str, Any]:
    code = se.main(["run", "--cases", str(cases), "--out", str(out), *extra])
    err = capsys.readouterr().err
    res: dict[str, Any] = {"code": code, "err": err, "rows": [], "report": None, "md": None}
    if (out / "results.jsonl").exists():
        res["rows"] = [json.loads(x) for x in (out / "results.jsonl").read_text().splitlines()]
    if (out / "report.json").exists():
        res["report"] = json.loads((out / "report.json").read_text())
    if (out / "report.md").exists():
        res["md"] = (out / "report.md").read_text()
    return res


def _row(res: dict[str, Any], case_id: str, model: str | None = None) -> dict[str, Any]:
    return next(
        r for r in res["rows"] if r["case_id"] == case_id and (model is None or r["model"] == model)
    )


def _cell(res: dict[str, Any], shape: int, model: str | None = None) -> dict[str, Any]:
    return next(
        c
        for c in res["report"]["cells"]
        if c["shape"] == shape and (model is None or c["model"] == model)
    )


def _write_cases(path: Path, cases: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(c) + "\n" for c in cases))
    return path


def _fixture_cases(path: Path = DIGESTS_FIXTURE) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in (json.loads(x) for x in path.read_text().splitlines())}


def _shape3_case(cid: str, label: bool, items: list[dict[str, Any]]) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": cid,
        "shape": 3,
        "label": label,
        "digests": [{"session_id": f"{cid}-sess", "turn_index": 0, "items": items}],
    }
    return case


def _call(i: str, target: str, cls: str) -> dict[str, Any]:
    return {
        "kind": "tool_call",
        "tool_use_id": i,
        "tool": "Bash",
        "target": target,
        "target_class": cls,
    }


def _res(i: str, err: bool, excerpt: str) -> dict[str, Any]:
    return {"kind": "tool_result", "tool_use_id": i, "is_error": err, "excerpt": excerpt}


def _text(t: str) -> dict[str, Any]:
    return {"kind": "assistant_text", "text": t}


# --- AC17: run ---------------------------------------------------------------


def test_a_digests_fixture_perfect(tmp_path, capsys, made):
    out = tmp_path / "out"
    res = _run(capsys, out, DIGESTS_FIXTURE, "--model", "stub-model")
    assert res["code"] == 0
    rows = res["rows"]
    assert [r["case_id"] for r in rows] == [
        "syn-s1-pos",
        "syn-s1-neg",
        "syn-s2-pos",
        "syn-s2-neg",
        "syn-s3-pos",
        "syn-s3-neg",
    ]
    for r in rows[:2]:
        assert r["model"] == "rule:shape1"
        assert r["llm_calls"] == 0
        assert r["prompt_sha256"] is None
    for r in rows[2:]:
        assert r["model"] == "stub-model"
        assert r["llm_calls"] == 1
        assert HEX64.match(r["prompt_sha256"])
    assert rows[0]["expected_match"] is True
    for r in rows:
        assert set(r) == set(se.RESULT_KEYS)
    assert len(made) == 1
    assert made[0].closed is True
    assert len(StubLLM.calls) == 4
    assert all(c[2] == surprise.SURPRISE_DETECTOR_SYSTEM for c in StubLLM.calls)

    report = res["report"]
    assert len(report["cells"]) == 3
    for cell in report["cells"]:
        assert set(cell) == set(se.CELL_KEYS)
        assert (cell["tp"], cell["fp"], cell["fn"], cell["tn"]) == (1, 0, 0, 1)
        assert cell["precision"] == 1.0
        assert cell["recall"] == 1.0
        assert cell["recall_applicable"] == 1.0
        assert cell["f1"] == 1.0
        assert cell["not_applicable_reasons"] == {}
        assert cell["precision_wilson95"] == [0.2065, 1.0]
        assert cell["precision_bar"] == "not-gating"
    gating = "precision is not gating (frames=synthetic, label_sources=synthetic)"
    assert report["warnings"] == [
        f"rule:shape1 shape 1: {gating}",
        f"stub-model shape 2: {gating}",
        f"stub-model shape 3: {gating}",
        f"production default detector {surprise_worker.SURPRISE_DETECTOR_DEFAULT_MODEL}"
        " was not measured",
    ]
    assert report["min_confidence"] == 0.7
    assert report["min_confidence_by_shape"] == {"2": 0.5, "3": 0.7}
    assert report["timeout_s"] == report["production_timeout_s"]
    assert report["timeout_s"] == service_config.get_anthropic_timeout()
    assert HEX64.match(report["prompt_set_sha256"])
    assert report["models"] == ["stub-model"]
    assert report["shapes"] == [1, 2, 3]

    md = res["md"]
    lines = md.splitlines()
    assert lines[0] == "# Surprise detector eval"
    assert "min_confidence=0.70 min_confidence_by_shape=s2=0.50,s3=0.70" in lines[1]
    assert se.TABLE_HEADER in lines
    assert "## Outcomes" in lines
    i = lines.index("## Disagreements")
    assert next(x for x in lines[i + 1 :] if x) == "none"
    assert lines[-5:] == [f"- {n}" for n in se.NOTES]
    assert lines[-1] == (
        '- Precision bar: precision >= 0.9 once tp+fp >= 100, our reading of "precision is at'
        ' least 0.9 on 100 labelled positives" (proposals/experience-loop-debate-synthesis.md);'
        " it reads not-gating unless every case in the cell has frame unfiltered and a"
        " label_source that is neither unspecified nor silver."
    )


def test_b_two_models_including_production_default(tmp_path, capsys, made):
    default = surprise_worker.SURPRISE_DETECTOR_DEFAULT_MODEL
    assert default == "claude-sonnet-5-5"
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-a", "--model", default)
    assert res["code"] == 0
    assert len(res["rows"]) == 10
    assert len(res["report"]["cells"]) == 5
    assert [m._config.model for m in made] == ["stub-a", default]
    assert len(StubLLM.calls) == 8
    assert not any("was not measured" in w for w in res["report"]["warnings"])


def test_c_ungrounded(tmp_path, capsys, made):
    StubLLM.mapping = {}
    StubLLM.default = _verdict("x", "y", "zzz-not-in-source", 0.95)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model")
    for shape in (2, 3):
        cell = _cell(res, shape)
        assert (cell["tp"], cell["fp"], cell["fn"], cell["tn"]) == (0, 0, 1, 1)
        assert cell["precision"] is None
        assert cell["recall"] == 0.0
        assert cell["f1"] is None
        assert cell["model_positive_precision"] == 0.5
        assert cell["model_positive_recall"] == 1.0
        assert cell["lost_to_gates"] == 1
        assert cell["outcomes"]["ungrounded"] == 2
    assert (
        "stub-model shape 2: 1 positives lost to the confidence floor or grounding check"
        in res["report"]["warnings"]
    )


def test_d_low_confidence(tmp_path, capsys, made):
    StubLLM.mapping = _perfect(0.45)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model")
    assert _row(res, "syn-s2-pos")["outcome"] == "low_confidence"
    assert _row(res, "syn-s3-pos")["outcome"] == "low_confidence"


def test_d2_min_confidence_env(tmp_path, capsys, made, monkeypatch):
    monkeypatch.setenv("KB_SURPRISE_MIN_CONFIDENCE", "0.95")
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model")
    for cid in ("syn-s2-pos", "syn-s3-pos"):
        row = _row(res, cid)
        assert row["outcome"] == "low_confidence"
        assert row["confidence"] == 0.9
    for shape in (2, 3):
        assert _cell(res, shape)["tp"] == 0
        assert _cell(res, shape)["lost_to_gates"] == 1
    assert res["report"]["min_confidence"] == 0.95
    assert res["report"]["min_confidence_by_shape"] == {"2": 0.95, "3": 0.95}
    assert "min_confidence=0.95" in res["md"].splitlines()[1]


def test_e_detector_unavailable(tmp_path, capsys, made):
    StubLLM.mapping = {}
    StubLLM.default = None
    out = tmp_path / "o"
    res = _run(capsys, out, DIGESTS_FIXTURE, "--model", "stub-model")
    for r in res["rows"][2:]:
        assert r["outcome"] == "llm_error"
        assert r["llm_calls"] == 1
    assert res["code"] == 3
    assert "run: detector unavailable for stub-model" in res["err"]
    assert (out / "report.json").exists()
    # Both shape-2 cases (positive and negative) call the model and both error.
    assert (
        "stub-model shape 2: 2 detector errors (llm_error/unparseable/invalid_fields)"
        " counted as negatives" in res["report"]["warnings"]
    )


def test_f_detect_digest_and_min_confidence_called(tmp_path, capsys, made, monkeypatch):
    real_detect = surprise_worker.detect_digest
    real_min = surprise_worker.detector_min_confidence
    seen: list[dict[str, Any]] = []
    min_calls: list[float] = []

    async def spy(*args: Any, **kwargs: Any) -> Any:
        seen.append(kwargs)
        return await real_detect(*args, **kwargs)

    def min_spy() -> float:
        value = real_min()
        min_calls.append(value)
        return value

    monkeypatch.setattr(surprise_worker, "detect_digest", spy)
    monkeypatch.setattr(surprise_worker, "detector_min_confidence", min_spy)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model")
    assert res["code"] == 0
    assert len(seen) == 6
    assert all(kw.get("min_confidence") in (None, 0.5, 0.7) for kw in seen)
    assert len(min_calls) == 1


def test_g_redaction_reaches_the_prompt(tmp_path, capsys, made):
    case = copy.deepcopy(_fixture_cases()["syn-s2-pos"])
    case["id"] = "redact-1"
    case["digests"][1]["user_prompt"] = 'no, use the vault: password = "hunter2hunter2"'
    path = _write_cases(tmp_path / "cases.jsonl", [case])
    res = _run(capsys, tmp_path / "o", path, "--model", "stub-model")
    assert len(StubLLM.calls) == 1
    prompt = StubLLM.calls[0][1]
    assert "[REDACTED:Secret Keyword]" in prompt
    assert "hunter2hunter2" not in prompt
    assert _row(res, "redact-1")["redactions"] == ["Secret Keyword"]
    assert se.main(["validate", "--cases", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["redacted_cases"] == 1


def test_h_mined_fixture(tmp_path, capsys, made):
    res = _run(capsys, tmp_path / "o", MINED_FIXTURE, "--model", "stub-model")
    assert res["code"] == 0
    c2 = _cell(res, 2)
    assert c2["tp"] == 1
    assert c2["recall_durable"] == 1.0
    c3 = _cell(res, 3)
    assert c3["tn"] == 1
    assert c3["hard_negatives"] == 1
    assert c3["fp_hard_negative"] == 0
    assert c3["label_sources"] == ["unspecified"]
    assert c3["precision_bar"] == "not-gating"


def _late_text_case() -> dict[str, Any]:
    return _shape3_case(
        "late-text-1",
        True,
        [_call("t1", "cat /x", "cat"), _res("t1", True, "missing"), _text("Oh, it is missing.")],
    )


def _early_text_case() -> dict[str, Any]:
    return _shape3_case(
        "early-text-1",
        True,
        [
            _text("The file should be there."),
            _call("t1", "cat /x", "cat"),
            _res("t1", True, "missing"),
        ],
    )


def test_i_not_applicable_and_recall_applicable(tmp_path, capsys, made):
    path = _write_cases(
        tmp_path / "cases.jsonl", [_fixture_cases()["syn-s3-pos"], _early_text_case()]
    )
    res = _run(capsys, tmp_path / "o", path, "--model", "stub-model", "--shapes", "3")
    cell = _cell(res, 3)
    assert cell["recall"] == 0.5
    assert cell["recall_applicable"] == 1.0
    assert cell["not_applicable_positives"] == 1
    assert cell["not_applicable_reasons"] == {"no_text_after_result": 1}
    assert _row(res, "early-text-1")["reason"] == "no_text_after_result"
    assert (
        "stub-model shape 3: 1 cases were not_applicable under the detector's skip rules,"
        " 1 of them labelled positive (detector scope, not model quality)"
    ) in res["report"]["warnings"]

    alone = _write_cases(tmp_path / "alone.jsonl", [_early_text_case()])
    res = _run(capsys, tmp_path / "o2", alone, "--model", "stub-model")
    assert res["code"] == 3
    assert (
        "run: no detector calls made for stub-model (every shape-2/3 case was not_applicable)"
        in res["err"]
    )

    late = _write_cases(tmp_path / "late.jsonl", [_late_text_case()])
    res = _run(capsys, tmp_path / "o3", late, "--model", "stub-model", "--shapes", "3")
    assert res["code"] == 0
    row = _row(res, "late-text-1")
    assert (row["outcome"], row["reason"], row["llm_calls"]) == ("no_surprise", "", 1)


def test_j_anomalies(tmp_path, capsys, made, monkeypatch):
    real_detect = surprise_worker.detect_digest

    async def dup(llm: Any, cur: Any, digests: Any, **kwargs: Any) -> Any:
        records = await real_detect(llm, cur, digests, **kwargs)
        if cur.session_id == "syn-sess-3":
            records.append(next(r for r in records if r.shape == 2))
        return records

    monkeypatch.setattr(surprise_worker, "detect_digest", dup)
    out = tmp_path / "o"
    res = _run(capsys, out, DIGESTS_FIXTURE, "--model", "stub-model")
    # The duplicated record is a second model-call record too, so AC8 also flags llm_calls=2.
    assert _row(res, "syn-s2-pos")["anomaly"] == "own_records=2,llm_calls=2"
    assert "anomaly stub-model syn-s2-pos own_records=2" in res["err"]
    assert res["code"] == 4
    assert (out / "report.json").exists()

    async def empty(*args: Any, **kwargs: Any) -> list[Any]:
        return []

    monkeypatch.setattr(surprise_worker, "detect_digest", empty)
    res = _run(capsys, tmp_path / "o2", DIGESTS_FIXTURE, "--model", "stub-model")
    assert all(r["anomaly"] == "own_records=0" for r in res["rows"])
    assert all(c["not_applicable_reasons"] == {"unknown": 2} for c in res["report"]["cells"])
    assert res["code"] == 4


def test_k_harness_error(tmp_path, capsys, made, monkeypatch):
    real_detect = surprise_worker.detect_digest

    async def boom(llm: Any, cur: Any, digests: Any, **kwargs: Any) -> Any:
        if cur.session_id == "syn-sess-6":
            raise RuntimeError("boom")
        return await real_detect(llm, cur, digests, **kwargs)

    monkeypatch.setattr(surprise_worker, "detect_digest", boom)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model")
    assert res["code"] == 4
    row = _row(res, "syn-s3-neg")
    assert row["harness_error"].startswith("RuntimeError: ")
    assert row["outcome"] is None
    cell = _cell(res, 3)
    assert cell["n"] == 1
    assert cell["harness_errors"] == 1
    assert "harness_error stub-model syn-s3-neg RuntimeError" in res["err"]


def _fake_client(monkeypatch, create: Any) -> None:
    fake = types.SimpleNamespace(messages=types.SimpleNamespace(create=create))
    monkeypatch.setattr(anthropic_llm.AnthropicLLMClient, "_get_client", lambda self: fake)


def test_l_provider_no_text_block(tmp_path, capsys, monkeypatch):
    async def create(**kwargs: Any) -> Any:
        if "config is at /etc/foo/main.conf" in kwargs["messages"][0]["content"]:
            return types.SimpleNamespace(
                content=[types.SimpleNamespace(type="thinking")], stop_reason="max_tokens"
            )
        return types.SimpleNamespace(
            content=[types.SimpleNamespace(type="text", text=NO_SURPRISE)],
            stop_reason="end_turn",
        )

    _fake_client(monkeypatch, create)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "m", "--concurrency", "4")
    row = _row(res, "syn-s3-pos")
    assert row["outcome"] == "llm_error"
    assert len(row["provider_warnings"]) == 1
    assert "no text block" in row["provider_warnings"][0]
    for r in res["rows"]:
        if r["case_id"] != "syn-s3-pos":
            assert r["provider_warnings"] == []
    assert _cell(res, 3)["provider_no_text"] == 1
    assert "m shape 3: 1 responses had no text block" in res["report"]["warnings"]


def test_l_provider_failure(tmp_path, capsys, monkeypatch):
    async def create(**kwargs: Any) -> Any:
        raise TimeoutError()

    _fake_client(monkeypatch, create)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "m")
    for r in res["rows"][2:]:
        assert r["outcome"] == "llm_error"
        assert len(r["provider_warnings"]) == 1
        assert "generation failed" in r["provider_warnings"][0]
        assert r["provider_warnings"][0].endswith(" exc=TimeoutError")
    # Each shape cell holds a positive and a negative case, and both calls fail.
    for shape in (2, 3):
        assert _cell(res, shape)["provider_failed"] == 2
        assert _cell(res, shape)["provider_exception_types"] == {"TimeoutError": 2}
    assert "m shape 2: 2 provider failures (TimeoutError=2)" in res["report"]["warnings"]
    assert res["code"] == 3
    assert "run: detector unavailable for m" in res["err"]


def test_m_cross_shape_candidate(tmp_path, capsys, made):
    case = _shape3_case(
        "cross-1",
        False,
        [
            _text("Pushing now."),
            _call("t1", "git push origin main", "git push"),
            _res("t1", True, "rejected"),
            _call("t2", "git push origin HEAD:main", "git push"),
            _res("t2", False, "ok"),
            _text("Pushed with HEAD:main."),
        ],
    )
    path = _write_cases(tmp_path / "cases.jsonl", [case])
    res = _run(capsys, tmp_path / "o", path, "--model", "stub-model")
    assert res["code"] == 0
    assert _row(res, "cross-1")["outcome"] == "no_surprise"
    assert _row(res, "cross-1")["llm_calls"] == 1
    assert _row(res, "cross-1")["cross_shape_candidates"] == [1]
    assert _cell(res, 3)["cross_shape_fp"] == 1
    assert (
        "stub-model shape 3: 1 negative cases produced a candidate of another shape"
        " (production would write it)" in res["report"]["warnings"]
    )


def test_s3_conclusion_fixture(tmp_path, capsys, made):
    evidence = "OnCalendar=*-*-* 02:00:00 UTC"
    StubLLM.mapping = {
        evidence: _verdict(
            "docs/export.md says the export job runs at 02:00 America/Chicago",
            "export.timer fires at 02:00 UTC",
            evidence,
            0.9,
        )
    }
    res = _run(capsys, tmp_path / "o", CONCLUSION_FIXTURE, "--model", "stub-model")
    assert res["code"] == 0
    assert _row(res, "syn-s3c-pos")["outcome"] == "candidate"
    assert _row(res, "syn-s3c-neg")["outcome"] == "no_surprise"
    for cid in ("syn-s3c-pos", "syn-s3c-neg"):
        assert _row(res, cid)["llm_calls"] == 1
        assert _row(res, cid)["cross_shape_candidates"] == []
    cell = _cell(res, 3)
    assert (cell["tp"], cell["fp"], cell["fn"], cell["tn"]) == (1, 0, 0, 1)
    assert cell["not_applicable_reasons"] == {}
    assert len(StubLLM.calls) == 2
    prompts = [c[1] for c in StubLLM.calls if "export.timer" in c[1]]
    assert len(prompts) == 1
    assert surprise.SHAPE3_INSTRUCTIONS in prompts[0]
    assert not any(x.startswith("[assistant final] ") for x in prompts[0].splitlines())


def test_n_latency_over_production_timeout(tmp_path, capsys, made, monkeypatch):
    monkeypatch.setenv("KB_ANTHROPIC_TIMEOUT", "0.001")
    StubLLM.delay = 0.01
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model")
    assert res["report"]["production_timeout_s"] == 0.001
    assert any(
        "calls took longer than the production timeout of 0.001s" in w
        for w in res["report"]["warnings"]
    )


def test_o_make_llm_parity(monkeypatch):
    llm = se.make_llm("m", None)
    assert type(llm).__name__ == "AnthropicLLMClient"
    assert llm._config.model == "m"
    assert llm._config.timeout == service_config.get_anthropic_timeout()
    assert se.make_llm("m", 7.0)._config.timeout == 7.0
    monkeypatch.setattr(surprise_worker, "_DETECTOR_LLM_CACHE", {})
    monkeypatch.setenv("KB_SURPRISE_DETECTOR_MODEL", "m")
    prod = surprise_worker.get_detector_llm()
    assert prod is not None
    assert prod._config == se.make_llm("m", None)._config  # type: ignore[attr-defined]


# --- model arms: --model MODEL[@EFFORT][+nothink] -----------------------------


@pytest.mark.parametrize(
    ("arm", "parsed"),
    [
        ("stub-a", ("stub-a", None, None)),
        ("stub-a@low", ("stub-a", "low", None)),
        ("stub-a@medium", ("stub-a", "medium", None)),
        ("stub-a@high", ("stub-a", "high", None)),
        ("stub-a@xhigh", ("stub-a", "xhigh", None)),
        ("stub-a@max", ("stub-a", "max", None)),
        ("stub-a+nothink", ("stub-a", None, "disabled")),
        ("stub-a@low+nothink", ("stub-a", "low", "disabled")),
    ],
)
def test_parse_model_arm(arm, parsed):
    assert se.parse_model_arm(arm) == parsed


@pytest.mark.parametrize(
    "arm",
    ["stub-a@bogus", "stub-a@", "stub-a+nothinkfoo", "stub-a@low+nothink@high"],
)
def test_parse_model_arm_rejects(arm):
    with pytest.raises(se.ArmError):
        se.parse_model_arm(arm)


def test_arg_error_rejects_unknown_effort(capsys):
    args = types.SimpleNamespace(
        shapes="1,2,3", concurrency=4, limit=None, timeout=None, model=["stub-a@bogus"]
    )
    reason, shapes = se._arg_error(args)
    assert reason is not None and "stub-a@bogus" in reason
    assert shapes == [1, 2, 3]
    assert "stub-a@bogus" not in capsys.readouterr().out


def test_run_rejects_unknown_effort_with_exit_2(tmp_path, capsys):
    code = se.main(
        [
            "run",
            "--cases",
            str(DIGESTS_FIXTURE),
            "--out",
            str(tmp_path / "o"),
            "--model",
            "stub-a@bogus",
        ]
    )
    assert code == 2
    assert "stub-a@bogus" in capsys.readouterr().err


def test_o_make_llm_passes_effort_and_thinking():
    llm = se.make_llm("m", None, effort="low", thinking="disabled")
    assert type(llm).__name__ == "AnthropicLLMClient"
    cfg = llm._config
    assert (cfg.effort, cfg.thinking) == ("low", "disabled")


def test_effort_arms_run_as_distinct_labels(tmp_path, capsys, made):
    res = _run(
        capsys,
        tmp_path / "o",
        DIGESTS_FIXTURE,
        "--model",
        "stub-a@low",
        "--model",
        "stub-a@high",
        "--model",
        "stub-a+nothink",
    )
    assert res["code"] == 0
    assert [m.arm for m in made] == [
        ("stub-a", "low", None),
        ("stub-a", "high", None),
        ("stub-a", None, "disabled"),
    ]
    assert res["report"]["models"] == ["stub-a@low", "stub-a@high", "stub-a+nothink"]
    models_in_rows = {r["model"] for r in res["rows"] if r["shape"] in (2, 3)}
    assert models_in_rows == {"stub-a@low", "stub-a@high", "stub-a+nothink"}


# --- AC18: validation, refusals, pure functions ------------------------------


def test_p_validate_fixtures(capsys):
    assert se.main(["validate", "--cases", str(DIGESTS_FIXTURE)]) == 0
    assert capsys.readouterr().out.strip() == (
        '{"by_label_source": {"synthetic": 6}, "by_shape": {"1": {"negatives": 1, "positives":'
        ' 1}, "2": {"negatives": 1, "positives": 1}, "3": {"negatives": 1, "positives": 1}},'
        ' "cases": 6, "redacted_cases": 0, "shape3_scope": {"final_message_only": 1,'
        ' "no_text_after_result": 0, "no_tool_result": 0, "text_after_result": 1}}'
    )
    assert se.main(["validate", "--cases", str(MINED_FIXTURE)]) == 0
    assert capsys.readouterr().out.strip() == (
        '{"by_label_source": {"unspecified": 2}, "by_shape": {"1": {"negatives": 0,'
        ' "positives": 0}, "2": {"negatives": 0, "positives": 1}, "3": {"negatives": 1,'
        ' "positives": 0}}, "cases": 2, "redacted_cases": 0, "shape3_scope":'
        ' {"final_message_only": 0, "no_text_after_result": 0, "no_tool_result": 0,'
        ' "text_after_result": 1}}'
    )
    assert se.main(["validate", "--cases", str(CONCLUSION_FIXTURE)]) == 0
    assert capsys.readouterr().out.strip() == (
        '{"by_label_source": {"synthetic": 2}, "by_shape": {"1": {"negatives": 0,'
        ' "positives": 0}, "2": {"negatives": 0, "positives": 0}, "3": {"negatives": 1,'
        ' "positives": 1}}, "cases": 2, "redacted_cases": 0, "shape3_scope":'
        ' {"final_message_only": 0, "no_text_after_result": 0, "no_tool_result": 0,'
        ' "text_after_result": 2}}'
    )


def test_q_mined_to_digests(tmp_path, capsys):
    mined = _fixture_cases(MINED_FIXTURE)
    assert se.mined_to_digests(mined["syn-m2-pos"]) == [
        {
            "session_id": "syn-m2-pos",
            "turn_index": 0,
            "project": "syn",
            "user_prompt": None,
            "items": [
                _text("Checking ports."),
                _text("Port 8080 is free; the dashboard will listen there."),
            ],
            "final_message": "Port 8080 is free; the dashboard will listen there.",
            "truncated": False,
        },
        {
            "session_id": "syn-m2-pos",
            "turn_index": 1,
            "project": "syn",
            "user_prompt": "no, 8080 is taken by caddy, use 8081",
            "items": [],
            "final_message": None,
            "truncated": False,
        },
    ]
    assert se.mined_to_digests(mined["syn-m3-neg"]) == [
        {
            "session_id": "syn-m3-neg",
            "turn_index": 0,
            "project": "syn",
            "user_prompt": "check the repo state",
            "items": [
                _text("The tree should be clean."),
                _call("t1", "git status --short", "git status"),
                {
                    "kind": "tool_call",
                    "tool_use_id": "t2",
                    "tool": "mcp__agent-gtd__get_item",
                    "target": "",
                    "target_class": "",
                },
                _res("t1", False, " M README.md"),
                _res("t2", False, "{}"),
                _res("orphan1", True, "late orphan"),
                _text("One modified file, as expected after the edit."),
            ],
            "final_message": None,
            "truncated": False,
        }
    ]
    case = {
        "id": "no-texts",
        "shape": 2,
        "label": False,
        "prev_final_message": "Done.",
        "user_prompt": "next",
    }
    path = _write_cases(tmp_path / "c.jsonl", [case])
    assert se.main(["validate", "--cases", str(path)]) == 0
    digests = se.mined_to_digests(case)
    assert digests[0]["items"] == []
    assert digests[0]["final_message"] == "Done."


def _s2_digests() -> list[dict[str, Any]]:
    return copy.deepcopy(_fixture_cases()["syn-s2-pos"]["digests"])


def _with(**changes: Any) -> dict[str, Any]:
    case = copy.deepcopy(_fixture_cases()["syn-s2-neg"])
    case["id"] = "bad-1"
    case.update(changes)
    return case


def _digest_case(digests: Any, shape: int = 2) -> dict[str, Any]:
    return {"id": "bad-1", "shape": shape, "label": False, "digests": digests}


def _mined2(**changes: Any) -> dict[str, Any]:
    case: dict[str, Any] = {
        "id": "bad-1",
        "shape": 2,
        "label": False,
        "prev_final_message": "x",
        "user_prompt": "y",
    }
    case.update(changes)
    return case


def _mined3(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {"id": "bad-1", "shape": 3, "label": False, "items": items}


def _drop(d: dict[str, Any], key: str) -> dict[str, Any]:
    out = dict(d)
    out.pop(key)
    return out


def _bad_cases() -> list[tuple[str, Any]]:
    d = _s2_digests()
    mixed = copy.deepcopy(d)
    mixed[1]["session_id"] = "other"
    desc = copy.deepcopy(d)
    desc[0]["turn_index"], desc[1]["turn_index"] = 1, 0
    gap = copy.deepcopy(d)
    gap[1]["turn_index"] = 2
    t1_result = copy.deepcopy(d)
    t1_result[1]["items"] = [_res("t9", False, "x")]
    s3_two = [
        {"session_id": "s", "turn_index": 0, "items": []},
        {"session_id": "s", "turn_index": 1, "items": []},
    ]
    big = [{"session_id": "s", "turn_index": 0, "items": [_text("x" * 2000)] * 200}]
    unknown_digest_key = copy.deepcopy(d)
    unknown_digest_key[0]["harness"] = "x"
    no_items = copy.deepcopy(d)
    del no_items[0]["items"]
    bool_turn = copy.deepcopy(d)
    bool_turn[0]["turn_index"] = True
    thinking = copy.deepcopy(d)
    thinking[0]["items"] = [{"kind": "thinking", "text": "x"}]
    extra = copy.deepcopy(d)
    extra[0]["items"] = [{"kind": "assistant_text", "text": "x", "extra": 1}]
    tool_call = {"kind": "tool_call", "tool": "Bash", "target": "ls"}
    return [
        ("non_object", "[1, 2]"),
        ("bad_id", _with(id="a b")),
        ("dup_id", [_with(), _with()]),
        ("shape_4", _with(shape=4)),
        ("shape_bool", _with(shape=True)),
        ("label_str", _with(label="yes")),
        ("hard_negative_str", _with(hard_negative="no")),
        ("hard_negative_positive", _with(label=True, hard_negative=True)),
        ("expected_on_negative", _with(expected={"wrong_belief": "a", "corrected_fact": "b"})),
        (
            "expected_extra_key",
            _with(label=True, expected={"wrong_belief": "a", "corrected_fact": "b", "why": "c"}),
        ),
        ("expected_missing_fact", _with(label=True, expected={"wrong_belief": "a"})),
        ("label_source_empty", _with(label_source="")),
        ("unknown_case_key", _with(lable=True)),
        ("mined_shape1", _mined2(shape=1)),
        ("mined2_no_user_prompt", _drop(_mined2(), "user_prompt")),
        ("mined2_no_prev_final", _drop(_mined2(), "prev_final_message")),
        ("mined2_bad_texts", _mined2(prev_assistant_texts=[1])),
        ("mined3_thinking", _mined3([{"kind": "thinking", "text": "x"}])),
        ("mined3_call_no_target", _mined3([_drop(tool_call, "target")])),
        ("empty_digests", _digest_case([])),
        ("unknown_digest_key", _digest_case(unknown_digest_key)),
        ("digest_no_items", _digest_case(no_items)),
        ("turn_index_bool", _digest_case(bool_turn)),
        ("mixed_sessions", _digest_case(mixed)),
        ("turn_index_desc", _digest_case(desc)),
        ("shape2_one_digest", _digest_case(d[:1])),
        ("shape2_gap", _digest_case(gap)),
        ("shape2_turn1_result", _digest_case(t1_result)),
        ("shape3_two_digests", _digest_case(s3_two, shape=3)),
        ("item_thinking", _digest_case(thinking)),
        ("item_extra_key", _digest_case(extra)),
        (
            "event_id_mismatch",
            _digest_case(
                [{"event_id": "s:9", "session_id": "s", "turn_index": 0, "items": []}], shape=3
            ),
        ),
        ("too_big", _digest_case(big, shape=3)),
    ]


@pytest.mark.parametrize(("name", "content"), _bad_cases(), ids=[n for n, _ in _bad_cases()])
def test_r_validate_rejects(tmp_path, capsys, name, content):
    path = tmp_path / "bad.jsonl"
    if isinstance(content, str):
        path.write_text(content + "\n")
    else:
        _write_cases(path, content if isinstance(content, list) else [content])
    assert se.main(["validate", "--cases", str(path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("validate: ")
    if name == "too_big":
        assert "exceeds 65536 bytes" in err


def test_s_validation_errors_never_echo_case_text(tmp_path, capsys):
    case = copy.deepcopy(_fixture_cases()["syn-s2-pos"])
    case["id"] = "long-1"
    case["digests"][1]["user_prompt"] = 'password = "hunter2hunter2"' + "x" * 4000
    path = _write_cases(tmp_path / "c.jsonl", [case])
    assert se.main(["validate", "--cases", str(path)]) == 2
    err = capsys.readouterr().err
    assert "string_too_long" in err
    assert "hunter2hunter2" not in err


def test_t_redaction_unavailable(capsys, monkeypatch):
    monkeypatch.setattr(turn_digest, "redact_turn_digest", lambda req: None)
    assert se.main(["validate", "--cases", str(DIGESTS_FIXTURE)]) == 2
    assert "redaction unavailable" in capsys.readouterr().err


def test_u_in_repo_cases_must_be_synthetic(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(se, "REPO_ROOT", tmp_path)
    case = copy.deepcopy(_fixture_cases()["syn-s2-neg"])
    case["id"] = "real-1"
    path = _write_cases(tmp_path / "c.jsonl", [case])
    assert se.main(["validate", "--cases", str(path)]) == 2
    assert capsys.readouterr().err == "validate: " + se.CASES_REFUSAL + "\n"


def test_v_out_inside_repo_refused(capsys, made):
    target = se.REPO_ROOT / "surprise-eval-out"
    code = se.main(["run", "--cases", str(DIGESTS_FIXTURE), "--out", str(target), "--model", "m"])
    assert code == 2
    assert se.OUT_REFUSAL in capsys.readouterr().err
    assert not target.exists()


def test_w_overwrite_refusal_and_force(tmp_path, capsys, made):
    out = tmp_path / "o"
    assert _run(capsys, out, DIGESTS_FIXTURE, "--model", "m")["code"] == 0
    res = _run(capsys, out, DIGESTS_FIXTURE, "--model", "m")
    assert res["code"] == 2
    assert "pass --force" in res["err"]

    before = (out / "results.jsonl").read_bytes()
    bad = tmp_path / "bad.jsonl"
    bad.write_text("[1]\n")
    code = se.main(["run", "--cases", str(bad), "--out", str(out), "--model", "m", "--force"])
    assert code == 2
    capsys.readouterr()
    assert (out / "results.jsonl").read_bytes() == before
    assert _run(capsys, out, DIGESTS_FIXTURE, "--model", "m", "--force")["code"] == 0


@pytest.mark.parametrize(
    "extra",
    [
        ["--shapes", "4"],
        ["--shapes", "1,1"],
        ["--concurrency", "0"],
        ["--limit", "0"],
        ["--timeout", "0"],
        ["--model", "a", "--model", "a"],
    ],
)
def test_x_argument_errors(tmp_path, capsys, made, extra):
    out = tmp_path / "o"
    code = se.main(["run", "--cases", str(DIGESTS_FIXTURE), "--out", str(out), *extra])
    assert code == 2
    assert capsys.readouterr().err.startswith("run: ")
    assert not out.exists()


def test_y_model_required_only_for_shapes_2_3(tmp_path, capsys, made):
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--shapes", "2,3")
    assert res["code"] == 2
    assert "--model is required" in res["err"]
    res = _run(capsys, tmp_path / "o1", DIGESTS_FIXTURE, "--shapes", "1")
    assert res["code"] == 0
    assert len(res["rows"]) == 2
    assert made == []


def test_z_fixture_ids_and_outcome_drift_guard():
    for path in (DIGESTS_FIXTURE, MINED_FIXTURE, CONCLUSION_FIXTURE, TURNS_FIXTURE):
        assert all(cid.startswith("syn-") for cid in _fixture_cases(path))
    assert typing.get_args(surprise.DetectionOutcome) == se.OUTCOMES
    for group in (se.ERROR_OUTCOMES, se.MODEL_POSITIVE_OUTCOMES, se.NO_CALL_OUTCOMES):
        assert set(group) <= set(se.OUTCOMES)


@pytest.mark.parametrize(
    ("k", "n", "expected"),
    [
        (1, 1, [0.2065, 1.0]),
        (0, 1, [0.0, 0.7935]),
        (1, 2, [0.0945, 0.9055]),
        (90, 100, [0.8256, 0.9448]),
        (0, 0, None),
    ],
)
def test_z_wilson(k, n, expected):
    assert se.wilson(k, n) == expected


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ((95, 5, 0.95, ["unfiltered"], ["human"]), "meets"),
        ((89, 11, 0.89, ["unfiltered"], ["human"]), "below"),
        ((50, 0, 1.0, ["unfiltered"], ["human"]), "insufficient"),
        ((95, 5, 0.95, ["marker-prefiltered"], ["human"]), "not-gating"),
        ((95, 5, 0.95, ["unfiltered"], ["silver:sonnet-judge"]), "not-gating"),
        ((95, 5, 0.95, ["unfiltered"], ["unspecified"]), "not-gating"),
    ],
)
def test_z_precision_bar_status(args, expected):
    assert se.precision_bar_status(*args) == expected


def test_z_digest_item_keys_match_server_union():
    assert {k: set(v) for k, v in se.DIGEST_ITEM_KEYS.items()} == {
        typing.get_args(c.model_fields["kind"].annotation)[0]: set(c.model_fields)
        for c in typing.get_args(typing.get_args(TurnItem)[0])
    }
    assert {k: set(v) for k, v in se.PEER_ITEM_KEYS.items()} == {
        "peer_message": {"kind", "sender", "text"}
    }
    server_kinds = {
        typing.get_args(c.model_fields["kind"].annotation)[0]
        for c in typing.get_args(typing.get_args(TurnItem)[0])
    }
    assert set(se.PEER_ITEM_KEYS).isdisjoint(server_kinds)


def test_validate_accepts_reasoning_item(tmp_path, capsys):
    case = _shape3_case(
        "syn-r1",
        False,
        [
            _call("t1", "ls", "ls"),
            _res("t1", False, "ok"),
            {"kind": "reasoning", "text": "r", "truncated": False},
        ],
    )
    path = _write_cases(tmp_path / "cases.jsonl", [case])
    assert se.main(["validate", "--cases", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["cases"] == 1


def test_validate_accepts_harness_correction_item(tmp_path, capsys):
    case = _shape3_case(
        "syn-hc1",
        False,
        [
            _call("t1", "ls", "ls"),
            _res("t1", False, "ok"),
            {
                "kind": "harness_correction",
                "trigger": "gate_red",
                "detail": "FAILED",
                "resolved_by": ["t1"],
                "resolved": True,
            },
        ],
    )
    path = _write_cases(tmp_path / "cases.jsonl", [case])
    assert se.main(["validate", "--cases", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["cases"] == 1


# --- H-024: eval-only shape-2 arms (context window, standing rule, peers) ------


def _turns_case(cid: str, label: bool, prompt_b: str | None = "use 8081") -> dict[str, Any]:
    """The pinned 4-turn sequence: two peer-started turns between the human prompts."""
    return {
        "id": cid,
        "shape": 2,
        "label": label,
        "label_source": "synthetic",
        "frame": "synthetic",
        "turns": [
            {
                "user_prompt": "start the dashboard",
                "items": [{"kind": "assistant_text", "text": "T1: Port 8080 is free"}],
                "final_message": None,
            },
            {
                "user_prompt": None,
                "items": [{"kind": "assistant_text", "text": "T2: still free"}],
                "final_message": None,
            },
            {
                "user_prompt": None,
                "items": [],
                "final_message": "F: the dashboard will bind 8080",
            },
            {"user_prompt": prompt_b, "items": [], "final_message": None},
        ],
    }


def test_arm_default_flags_reproduce_todays_prompts(tmp_path, capsys, made):
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model")
    assert res["code"] == 0
    cases = {c.id: c for c in se.load_cases(DIGESTS_FIXTURE)}
    prompts = [c[1] for c in StubLLM.calls]
    s2 = cases["syn-s2-pos"]
    assert surprise.build_shape2_prompt(s2.digests[-2], s2.digests[-1]) in prompts
    s3 = cases["syn-s3-pos"]
    assert surprise.build_shape3_prompt(s3.digests[0]) in prompts
    assert not any("[peer " in p for p in prompts)


def test_arm_explicit_default_flags_match_no_flag_rows(tmp_path, capsys, made):
    a = _run(capsys, tmp_path / "o1", DIGESTS_FIXTURE, "--model", "stub-model")
    b = _run(
        capsys,
        tmp_path / "o2",
        DIGESTS_FIXTURE,
        "--model",
        "stub-model",
        "--shape2-context",
        "previous_turn",
        "--shape2-prompt",
        "default",
        "--peer-messages",
        "off",
    )
    assert a["code"] == b["code"] == 0
    assert a["rows"] == b["rows"]
    assert b["report"]["args"]["shape2_context"] == "previous_turn"
    assert b["report"]["args"]["shape2_prompt"] == "default"
    assert b["report"]["args"]["peer_messages"] == "off"
    assert (
        "shape2_context=previous_turn shape2_prompt=default peer_messages=off"
        in b["md"].splitlines()[1]
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (
            ["--shape2-context", "bogus"],
            "run: --shape2-context must be one of previous_turn,since_human_prompt",
        ),
        (
            ["--shape2-prompt", "bogus"],
            "run: --shape2-prompt must be one of default,standing_rule",
        ),
        (
            ["--peer-messages", "bogus"],
            "run: --peer-messages must be one of off,on",
        ),
    ],
)
def test_arm_invalid_flag_values_exit_2(tmp_path, capsys, made, extra, message):
    out = tmp_path / "o"
    code = se.main(
        ["run", "--cases", str(DIGESTS_FIXTURE), "--out", str(out), "--model", "m", *extra]
    )
    assert code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()


def test_arm_validate_accepts_peer_items_and_turns_fixture(tmp_path, capsys):
    case = copy.deepcopy(_fixture_cases()["syn-s3-pos"])
    case["id"] = "syn-peer-1"
    case["digests"][0]["items"].append(
        {"kind": "peer_message", "sender": "ci-bot", "text": "main is green"}
    )
    path = _write_cases(tmp_path / "cases.jsonl", [case])
    assert se.main(["validate", "--cases", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["cases"] == 1
    assert se.main(["validate", "--cases", str(TURNS_FIXTURE)]) == 0
    assert json.loads(capsys.readouterr().out)["cases"] == 2


def test_arm_b_since_human_prompt_window(tmp_path, capsys, made):
    path = _write_cases(tmp_path / "cases.jsonl", [_turns_case("syn-b1", True)])
    res = _run(
        capsys,
        tmp_path / "o",
        path,
        "--model",
        "stub-model",
        "--shape2-context",
        "since_human_prompt",
    )
    assert res["code"] == 0
    prompt = StubLLM.calls[0][1]
    assert prompt.startswith("The assistant's previous turns:")
    assert "T1: Port 8080 is free" in prompt
    assert "T2: still free" in prompt
    assert "Final message: F: the dashboard will bind 8080" in prompt
    assert "start the dashboard" not in prompt
    row = _row(res, "syn-b1")
    assert row["shape2_context"] == "since_human_prompt"
    assert row["shape2_prompt"] == "default"
    assert row["peer_messages"] == "off"
    assert row["shape2_context_turns"] == 3
    assert row["shape2_context_anchor"] == 0


def test_arm_b_previous_turn_renders_only_prev_turn(tmp_path, capsys, made):
    path = _write_cases(tmp_path / "cases.jsonl", [_turns_case("syn-b2", True)])
    res = _run(capsys, tmp_path / "o", path, "--model", "stub-model")
    prompt = StubLLM.calls[0][1]
    assert prompt.startswith("The assistant's previous turn:")
    assert "Final message: F: the dashboard will bind 8080" in prompt
    assert "T1:" not in prompt
    assert "T2:" not in prompt
    row = _row(res, "syn-b2")
    assert row["llm_calls"] == 1
    assert row["shape2_context_turns"] is None
    assert row["shape2_context_anchor"] is None


@pytest.mark.parametrize("ctx", ["previous_turn", "since_human_prompt"])
def test_arm_b_target_without_user_prompt_skips(tmp_path, capsys, made, ctx):
    case = _turns_case("syn-b3", False)
    case["turns"][-1]["user_prompt"] = None
    path = _write_cases(tmp_path / "cases.jsonl", [case])
    res = _run(capsys, tmp_path / "o", path, "--model", "stub-model", "--shape2-context", ctx)
    row = _row(res, "syn-b3")
    assert (row["outcome"], row["reason"], row["llm_calls"]) == (
        "not_applicable",
        "no_user_prompt",
        0,
    )
    assert StubLLM.calls == []


def test_arm_b_no_prev_text_skips_even_with_window_text(tmp_path, capsys, made):
    case = {
        "id": "syn-b4",
        "shape": 2,
        "label": False,
        "label_source": "synthetic",
        "frame": "synthetic",
        "turns": [
            {
                "user_prompt": "prompt A",
                "items": [{"kind": "assistant_text", "text": "T1: earlier text"}],
                "final_message": None,
            },
            {"user_prompt": None, "items": [], "final_message": None},
            {"user_prompt": "prompt B", "items": [], "final_message": None},
        ],
    }
    path = _write_cases(tmp_path / "cases.jsonl", [case])
    res = _run(
        capsys,
        tmp_path / "o",
        path,
        "--model",
        "stub-model",
        "--shape2-context",
        "since_human_prompt",
    )
    row = _row(res, "syn-b4")
    assert (row["outcome"], row["reason"], row["llm_calls"]) == (
        "not_applicable",
        "no_prev_text",
        0,
    )
    assert StubLLM.calls == []


def test_arm_b_digests_form_differs_only_in_header(tmp_path, capsys, made):
    res = _run(
        capsys,
        tmp_path / "o",
        DIGESTS_FIXTURE,
        "--model",
        "stub-model",
        "--shape2-context",
        "since_human_prompt",
    )
    assert res["code"] == 0
    cases = {c.id: c for c in se.load_cases(DIGESTS_FIXTURE)}
    s2 = cases["syn-s2-pos"]
    prompt = StubLLM.calls[0][1]
    expected = surprise.build_shape2_context_prompt(
        surprise.shape2_context_digests(s2.digests, s2.digests[-1]), s2.digests[-1]
    )
    assert prompt == expected
    plain = surprise.build_shape2_prompt(s2.digests[-2], s2.digests[-1])
    assert [
        (x, y) for x, y in zip(prompt.splitlines(), plain.splitlines(), strict=True) if x != y
    ] == [("The assistant's previous turns:", "The assistant's previous turn:")]
    row = _row(res, "syn-s2-pos")
    assert row["shape2_context_turns"] == 1
    assert row["shape2_context_anchor"] == 0


def test_arm_c_standing_rule_prompt_and_version(tmp_path, capsys, made):
    res = _run(
        capsys,
        tmp_path / "o",
        DIGESTS_FIXTURE,
        "--model",
        "stub-model",
        "--shape2-prompt",
        "standing_rule",
    )
    assert res["code"] == 0
    cases = {c.id: c for c in se.load_cases(DIGESTS_FIXTURE)}
    s2 = cases["syn-s2-pos"]
    prompt = StubLLM.calls[0][1]
    assert prompt == surprise.build_shape2_standing_rule_prompt(s2.digests[-2], s2.digests[-1])
    assert surprise.SHAPE2_STANDING_RULE_INSTRUCTIONS in prompt.splitlines()
    assert (
        "evidence_excerpt must be copied verbatim from the human message, with no"
        " ellipses." in prompt.splitlines()
    )
    assert "Set it to false for new requests, preference or style changes" not in prompt
    row = _row(res, "syn-s2-pos")
    assert row["shape2_prompt"] == "standing_rule"
    assert row["shape2_prompt_version"] == 1
    assert row["detector_version"] == surprise.SURPRISE_DETECTOR_VERSION
    assert res["report"]["shape2_prompt_version"] == 1
    assert res["report"]["args"]["shape2_prompt"] == "standing_rule"

    default = _run(capsys, tmp_path / "o2", DIGESTS_FIXTURE, "--model", "stub-model")
    assert _row(default, "syn-s2-pos")["shape2_prompt_version"] == 4
    assert default["report"]["shape2_prompt_version"] == 4
    assert "shape2_prompt=standing_rule" not in default["md"].splitlines()[1]


def test_arm_d_peer_messages_render_only_when_on(tmp_path, capsys, made):
    case = _turns_case("syn-d1", True)
    case["turns"][1]["items"] = [
        {"kind": "peer_message", "sender": "jason-precision", "text": "GPU box is busy"},
        {"kind": "peer_message", "sender": "ci-bot", "text": "main is green"},
        {"kind": "assistant_text", "text": "T2: still free"},
    ]
    s3case = copy.deepcopy(_fixture_cases()["syn-s3-pos"])
    s3case["id"] = "syn-d3"
    s3case["digests"][0]["items"].append(
        {"kind": "peer_message", "sender": "odin", "text": "disk is 90% free"}
    )
    path = _write_cases(tmp_path / "cases.jsonl", [case, s3case])
    res = _run(
        capsys,
        tmp_path / "o",
        path,
        "--model",
        "stub-model",
        "--shape2-context",
        "since_human_prompt",
        "--peer-messages",
        "on",
    )
    assert res["code"] == 0
    s2_prompt, s3_prompt = StubLLM.calls[0][1], StubLLM.calls[1][1]
    assert "[peer jason-precision] GPU box is busy" in s2_prompt
    assert "[peer ci-bot] main is green" in s2_prompt
    assert "[peer odin] disk is 90% free" in s3_prompt
    row2 = _row(res, "syn-d1")
    assert row2["peer_items"] == 2
    assert row2["peer_items_rendered"] == 2
    row3 = _row(res, "syn-d3")
    assert row3["peer_items"] == 1
    assert row3["peer_items_rendered"] == 1

    off = _run(capsys, tmp_path / "o2", path, "--model", "stub-model", "--peer-messages", "off")
    for prompt in (StubLLM.calls[-2][1], StubLLM.calls[-1][1]):
        assert "[peer " not in prompt
    assert _row(off, "syn-d1")["peer_items"] == 2
    assert _row(off, "syn-d1")["peer_items_rendered"] == 0
    assert _row(off, "syn-d3")["peer_items"] == 1
    assert _row(off, "syn-d3")["peer_items_rendered"] == 0


@pytest.mark.parametrize("ctx", ["previous_turn", "since_human_prompt"])
@pytest.mark.parametrize("prompt", ["default", "standing_rule"])
@pytest.mark.parametrize("peers", ["off", "on"])
def test_arm_combinations_produce_zero_anomalies(tmp_path, capsys, made, ctx, prompt, peers):
    res = _run(
        capsys,
        tmp_path / "o",
        TURNS_FIXTURE,
        "--model",
        "stub-model",
        "--shape2-context",
        ctx,
        "--shape2-prompt",
        prompt,
        "--peer-messages",
        peers,
    )
    assert res["code"] == 0
    assert all(r["anomaly"] is None for r in res["rows"])
    assert all(r["harness_error"] is None for r in res["rows"])
    assert res["report"]["args"] == {
        "concurrency": 4,
        "force": False,
        "limit": None,
        "peer_messages": peers,
        "shape2_context": ctx,
        "shape2_prompt": prompt,
        "shapes": [1, 2, 3],
        "timeout": None,
    }


def test_turns_form_validation_errors(tmp_path, capsys):
    def _validate(case: dict[str, Any]) -> str:
        path = _write_cases(tmp_path / "cases.jsonl", [case])
        assert se.main(["validate", "--cases", str(path)]) == 2
        return capsys.readouterr().err.strip()

    wrong_shape = _turns_case("syn-t1", True)
    wrong_shape["shape"] = 3
    assert _validate(wrong_shape) == "validate: syn-t1: turns cases must be shape 2"

    bad_case_key = _turns_case("syn-t2", True)
    bad_case_key["bogus"] = 1
    assert _validate(bad_case_key) == "validate: syn-t2: unknown keys ['bogus']"

    bad_turn_key = _turns_case("syn-t3", True)
    bad_turn_key["turns"][0]["extra"] = 1
    assert _validate(bad_turn_key) == "validate: syn-t3: turn 0: unknown keys ['extra']"

    bad_item = _turns_case("syn-t4", True)
    bad_item["turns"][0]["items"][0]["bogus"] = 1
    assert _validate(bad_item) == "validate: syn-t4: turn 0 item 0 has unknown keys"

    bad_kind = _turns_case("syn-t5", True)
    bad_kind["turns"][0]["items"] = [{"kind": "bogus", "text": "x"}]
    assert _validate(bad_kind) == "validate: syn-t5: turn 0 item 0 has an unknown kind"

    one_turn = _turns_case("syn-t6", True)
    one_turn["turns"] = one_turn["turns"][:1]
    assert "syn-t6: turns must be a list of at least 2 turns" in _validate(one_turn)


# --- H-024 stage 2: --distil (offline distiller + critic, no writes) ----------

DISTILL_OK = json.dumps(
    {
        "durable": True,
        "short_title": "Use the mapped port",
        "long_title": "caddy owns 8080",
        "corrected_fact": "8081 is the dashboard port",
        "lesson": "Check the port map before listening",
        "lesson_class": "project_tooling",
    }
)
CRITIC_OK = json.dumps(
    {"supported": True, "scope_ok": True, "durable": True, "misleading": False, "reason": "ok"}
)
DISTIL_OUTCOME_NAMES = (
    "llm_error",
    "parse_error",
    "not_durable",
    "critic_rejected",
    "written_equivalent",
)


def _distil_mapping(distill: str = DISTILL_OK, critic: str = CRITIC_OK) -> dict[str, str]:
    """Detector answers stay PERFECT; distiller/critic anchors come first.

    Distiller prompts carry a ``Wrong belief:`` line, critic prompts a
    ``Drafted lesson:`` line; both are looked up before the PERFECT keys, so
    one stub serves detector, distiller and critic.
    """
    return {"Drafted lesson:": critic, "Wrong belief:": distill, **PERFECT}


def _calls_for(calls: list[tuple[str, str, str | None, str]], purpose: str) -> list[str]:
    return [c[1] for c in calls if c[3] == purpose]


def test_dv_off_reproduces_today(tmp_path, capsys, made):
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "off")
    plain = _run(capsys, tmp_path / "o2", DIGESTS_FIXTURE, "--model", "stub-model")
    assert res["code"] == plain["code"] == 0
    assert res["rows"] == plain["rows"]
    for r in res["rows"]:
        assert set(r) == set(se.RESULT_KEYS)
    assert res["report"]["cells"] == plain["report"]["cells"]
    assert res["md"] == plain["md"]
    report = res["report"]
    assert report["schema"] == 1
    assert report["args"] == {
        "concurrency": 4,
        "force": False,
        "limit": None,
        "peer_messages": "off",
        "shape2_context": "previous_turn",
        "shape2_prompt": "default",
        "shapes": [1, 2, 3],
        "timeout": None,
    }
    assert report.get("distil_skipped") is None
    assert report.get("distiller_version") is None
    for cell in report["cells"]:
        assert set(cell) == set(se.CELL_KEYS)
    # One client per run (the detector arm); no distiller or critic client was
    # built, and only detector calls were made.
    assert len(made) == 2
    assert all(m.closed for m in made)
    assert all(c[3].startswith("surprise_detector_") for c in StubLLM.calls)


def test_dv_on_written_equivalent(tmp_path, capsys, made):
    StubLLM.mapping = _distil_mapping()
    before = len(made)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    assert res["code"] == 0
    r2 = _row(res, "syn-s2-pos")
    assert r2["distil_outcome"] == "written_equivalent"
    assert r2["distil_capped"] is False
    assert r2["distil_corrected_fact"] == "8081 is the dashboard port"
    assert r2["distil_lesson_class"] == "project_tooling"
    assert r2["critic_verdict"] == {
        "supported": True,
        "scope_ok": True,
        "durable": True,
        "misleading": False,
        "reason": "ok",
    }
    assert r2["critic_reason"] == "ok"
    assert r2["resolution_corrected_fact"] == "8081 is the dashboard port"
    assert r2["resolution_wrong_belief"] == "Port 8080 is free"
    assert r2["resolution_cue"] is None  # shape 2 never carries a cue
    assert HEX64.match(r2["distil_prompt_sha256"])
    assert HEX64.match(r2["critic_prompt_sha256"])
    assert r2["distil_raw_excerpt"] == DISTILL_OK
    assert r2["critic_raw_excerpt"] == CRITIC_OK
    assert r2["distil_model"] == surprise_worker.SURPRISE_DISTILLER_DEFAULT_MODEL
    assert r2["critic_model"] == surprise_worker.SURPRISE_CRITIC_DEFAULT_MODEL
    # The critic prompt carried the target turn's user prompt (shape 2).
    critic_prompts = _calls_for(StubLLM.calls, "surprise_critic")
    assert any(
        "The human's correction: no, 8080 is taken by caddy, use 8081" in p for p in critic_prompts
    )
    # A no-candidate row carries null distil columns but the model arms.
    r2n = _row(res, "syn-s2-neg")
    assert r2n["outcome"] == "no_surprise"
    assert r2n["distil_outcome"] is None
    assert r2n["distil_capped"] is None
    assert r2n["distil_model"] == "claude-sonnet-5-5"
    assert r2n["critic_model"] == "claude-sonnet-5-5"
    assert r2n["distil_prompt_sha256"] is None
    # Shape 1: build_resolution's cue survives the chain.
    r1 = _row(res, "syn-s1-pos")
    assert r1["distil_outcome"] == "written_equivalent"
    assert r1["resolution_cue"] == {
        "tool": "Bash",
        "target_class": "git push",
        "args_prefix": "origin main",
    }
    report = res["report"]
    assert report["schema"] == 1
    assert report["distil_skipped"] == 0
    assert report["distil_llm_calls"] == 6  # 3 candidates x (distiller + critic)
    assert report["distiller_version"] == surprise_distill.SURPRISE_DISTILLER_VERSION == 9
    assert report["critic_version"] == surprise_distill.SURPRISE_CRITIC_VERSION == 2
    assert HEX64.match(report["distil_prompt_set_sha256"])
    assert len(made) - before == 3  # detector + distiller + critic
    assert all(m.closed for m in made)
    assert (
        report["prompt_set_sha256"]
        == _run(capsys, tmp_path / "o3", DIGESTS_FIXTURE, "--model", "stub-model")["report"][
            "prompt_set_sha256"
        ]
    )
    assert report["args"]["distil"] == "on"
    assert report["args"]["distil_model"] == "claude-sonnet-5-5"
    assert report["args"]["critic_model"] == "claude-sonnet-5-5"
    assert report["args"]["distil_max"] == 500
    for cell in report["cells"]:
        assert set(cell) == set(se.CELL_KEYS) | {"distil_outcomes", "distil_skipped_in_cell"}
        assert cell["distil_skipped_in_cell"] == 0
        assert set(cell["distil_outcomes"]) == set(DISTIL_OUTCOME_NAMES)
    for shape in (1, 2, 3):
        assert _cell(res, shape)["distil_outcomes"]["written_equivalent"] == 1
        assert sum(_cell(res, shape)["distil_outcomes"].values()) == 1
    md = res["md"]
    assert "## Distil outcomes" in md
    zero = "llm_error=0 parse_error=0 not_durable=0 critic_rejected=0"
    for shape, model in ((1, "rule:shape1"), (2, "stub-model"), (3, "stub-model")):
        assert f"{shape}/{model}: {zero} written_equivalent=1 skipped=0" in md


def test_dv_on_critic_rejected(tmp_path, capsys, made):
    critic = json.dumps(
        {
            "supported": False,
            "scope_ok": True,
            "durable": True,
            "misleading": False,
            "reason": "evidence thin",
        }
    )
    StubLLM.mapping = _distil_mapping(critic=critic)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    assert res["code"] == 0
    r2 = _row(res, "syn-s2-pos")
    assert r2["distil_outcome"] == "critic_rejected"
    assert r2["distil_capped"] is False
    assert r2["distil_corrected_fact"] == "8081 is the dashboard port"
    assert r2["distil_lesson_class"] == "project_tooling"
    assert r2["critic_verdict"]["supported"] is False
    assert r2["critic_reason"] == "evidence thin"
    assert r2["resolution_corrected_fact"] is None
    assert r2["resolution_wrong_belief"] is None
    assert r2["resolution_cue"] is None


def test_dv_on_not_durable(tmp_path, capsys, made):
    distill = json.dumps({"durable": False, "why": "one-off"})
    StubLLM.mapping = _distil_mapping(distill=distill)
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    assert res["code"] == 0
    r2 = _row(res, "syn-s2-pos")
    assert r2["distil_outcome"] == "not_durable"
    assert r2["distil_reason"] == "one-off"
    assert r2["distil_corrected_fact"] is None
    assert r2["distil_lesson_class"] is None
    assert r2["critic_verdict"] is None
    assert r2["critic_reason"] is None
    assert r2["critic_prompt_sha256"] is None
    assert r2["critic_raw_excerpt"] is None
    assert r2["distil_raw_excerpt"] == distill
    assert not _calls_for(StubLLM.calls, "surprise_critic")


def test_dv_on_cap(tmp_path, capsys, made):
    StubLLM.mapping = _distil_mapping()
    res = _run(
        capsys,
        tmp_path / "o",
        DIGESTS_FIXTURE,
        "--model",
        "stub-model",
        "--distil",
        "on",
        "--distil-max",
        "1",
    )
    assert res["code"] == 0
    assert res["err"].count("run: distil cap reached, skipped 2 candidates") == 1
    # Enumeration order: shape-1 cases first, then shape-2/3 cases in file order.
    assert _row(res, "syn-s1-pos")["distil_outcome"] == "written_equivalent"
    for cid in ("syn-s2-pos", "syn-s3-pos"):
        row = _row(res, cid)
        assert row["distil_capped"] is True
        assert row["distil_outcome"] is None
        assert row["distil_model"] == "claude-sonnet-5-5"
    assert res["report"]["distil_skipped"] == 2
    assert res["report"]["distil_llm_calls"] == 2
    assert _cell(res, 1)["distil_outcomes"]["written_equivalent"] == 1
    for shape in (2, 3):
        assert _cell(res, shape)["distil_skipped_in_cell"] == 1
        assert sum(_cell(res, shape)["distil_outcomes"].values()) == 0
    md = res["md"]
    zero = "llm_error=0 parse_error=0 not_durable=0 critic_rejected=0"
    assert f"1/rule:shape1: {zero} written_equivalent=1 skipped=0" in md
    assert f"2/stub-model: {zero} written_equivalent=0 skipped=1" in md


def test_dv_on_critic_evidence_shape3_and_shape1(tmp_path, capsys, made):
    StubLLM.mapping = _distil_mapping()
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    assert res["code"] == 0
    critic_prompts = _calls_for(StubLLM.calls, "surprise_critic")
    assert len(critic_prompts) == 3
    # Shape 3: the target turn's tool_result excerpts reach the critic prompt.
    s3 = next(p for p in critic_prompts if "Tool results from the turn:" in p)
    assert "[tool_result] No such file: /etc/foo.conf; config is at /etc/foo/main.conf" in s3
    # Shape 1: neither evidence section is rendered.
    s1 = next(p for p in critic_prompts if "git push origin main" in p)
    assert "The human's correction:" not in s1
    assert "Tool results from the turn:" not in s1


def test_dv_distil_flag_validation(tmp_path, capsys, made):
    out = tmp_path / "o"
    code = se.main(
        [
            "run",
            "--cases",
            str(DIGESTS_FIXTURE),
            "--out",
            str(out),
            "--model",
            "m",
            "--distil",
            "bogus",
        ]
    )
    assert code == 2
    assert "run: --distil must be one of off,on" in capsys.readouterr().err
    assert not out.exists()
    assert StubLLM.calls == []

    code = se.main(
        [
            "run",
            "--cases",
            str(DIGESTS_FIXTURE),
            "--out",
            str(out),
            "--model",
            "m",
            "--distil",
            "on",
            "--distil-max",
            "0",
        ]
    )
    assert code == 2
    assert "run: --distil-max must be >= 1" in capsys.readouterr().err
    assert not out.exists()

    code = se.main(
        [
            "run",
            "--cases",
            str(DIGESTS_FIXTURE),
            "--out",
            str(out),
            "--model",
            "m",
            "--distil",
            "on",
            "--distil-model",
            "bogus@x",
        ]
    )
    assert code == 2
    assert "invalid model arm 'bogus@x'" in capsys.readouterr().err
    assert not out.exists()
    assert StubLLM.calls == []


def test_dv_on_llm_error(tmp_path, capsys, made):
    StubLLM.raise_purposes = {"surprise_distiller"}
    StubLLM.mapping = _distil_mapping()
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    assert res["code"] == 0  # distil outcomes never change the exit code
    for cid in ("syn-s1-pos", "syn-s2-pos", "syn-s3-pos"):
        row = _row(res, cid)
        assert row["distil_outcome"] == "llm_error"
        assert row["distil_capped"] is False
        assert row["distil_prompt_sha256"] is not None
        assert row["critic_prompt_sha256"] is None
        assert row["distil_raw_excerpt"] is None
    assert not _calls_for(StubLLM.calls, "surprise_critic")
    for shape, model in ((1, "rule:shape1"), (2, "stub-model"), (3, "stub-model")):
        assert (
            f"{model} shape {shape}: distiller unavailable (every distil call errored)"
            in res["report"]["warnings"]
        )
    assert any("distiller unavailable" in line for line in res["md"].splitlines())


def test_dv_on_parse_error_fail_closed(tmp_path, capsys, made):
    # The distiller reply that is not a JSON object is parse_error.
    StubLLM.mapping = _distil_mapping(distill="not a json object")
    res = _run(capsys, tmp_path / "o", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    assert res["code"] == 0
    assert _row(res, "syn-s2-pos")["distil_outcome"] == "parse_error"
    assert _row(res, "syn-s2-pos")["distil_raw_excerpt"] == "not a json object"
    # A durable distiller verdict followed by an unparseable critic reply is
    # also parse_error (fail closed).
    StubLLM.mapping = _distil_mapping(critic="not a json object either")
    res2 = _run(capsys, tmp_path / "o2", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    assert res2["code"] == 0
    r2 = _row(res2, "syn-s2-pos")
    assert r2["distil_outcome"] == "parse_error"
    assert r2["distil_corrected_fact"] is None
    assert r2["critic_raw_excerpt"] == "not a json object either"
    # {"surprise": false} is a valid JSON object without durable: true, so the
    # pinned production parser reads it as not_durable with an empty why.
    StubLLM.mapping = _distil_mapping(distill=json.dumps({"surprise": False}))
    res3 = _run(capsys, tmp_path / "o3", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    assert res3["code"] == 0
    r3 = _row(res3, "syn-s2-pos")
    assert r3["distil_outcome"] == "not_durable"
    assert r3["distil_reason"] == ""


def test_dv_on_cross_shape_candidate(tmp_path, capsys, made):
    case = _shape3_case(
        "cross-1",
        False,
        [
            _text("Pushing now."),
            _call("t1", "git push origin main", "git push"),
            _res("t1", True, "rejected"),
            _call("t2", "git push origin HEAD:main", "git push"),
            _res("t2", False, "ok"),
            _text("Pushed with HEAD:main."),
        ],
    )
    path = _write_cases(tmp_path / "cases.jsonl", [case])
    StubLLM.mapping = _distil_mapping()
    res = _run(capsys, tmp_path / "o", path, "--model", "stub-model", "--distil", "on")
    assert res["code"] == 0
    assert _row(res, "cross-1")["cross_shape_candidates"] == [1]
    row = _row(res, "cross-1")
    assert row["distil_outcome"] is None
    assert row["distil_capped"] is None
    assert row["distil_model"] == "claude-sonnet-5-5"
    cell = _cell(res, 3)
    assert cell["distil_outcomes"]["written_equivalent"] == 1
    assert sum(cell["distil_outcomes"].values()) == 1
    assert cell["distil_skipped_in_cell"] == 0
    assert res["report"]["distil_llm_calls"] == 2


def test_dv_on_detector_invariants_unchanged(tmp_path, capsys, made):
    StubLLM.mapping = _distil_mapping()
    on = _run(capsys, tmp_path / "o1", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "on")
    off = _run(capsys, tmp_path / "o2", DIGESTS_FIXTURE, "--model", "stub-model", "--distil", "off")
    assert on["code"] == off["code"] == 0
    assert on["err"] == off["err"]
    detector_keys = (
        "llm_calls",
        "latency_ms",
        "prompt_chars",
        "response_chars",
        "raw_response_excerpt",
        "prompt_sha256",
        "detector_output",
        "anomaly",
        "harness_error",
        "provider_warnings",
        "outcome",
        "reason",
        "confidence",
        "predicted",
        "model_positive",
        "expected_match",
        "cross_shape_candidates",
    )
    for ron, roff in zip(on["rows"], off["rows"], strict=True):
        assert ron["case_id"] == roff["case_id"]
        for key in detector_keys:
            assert ron[key] == roff[key], (ron["case_id"], key)
    assert on["report"]["prompt_set_sha256"] == off["report"]["prompt_set_sha256"]
    assert len(on["report"]["warnings"]) == len(off["report"]["warnings"])
    for con, coff in zip(on["report"]["cells"], off["report"]["cells"], strict=True):
        assert {
            k: v for k, v in con.items() if k not in ("distil_outcomes", "distil_skipped_in_cell")
        } == coff
