"""Unit tests for the upgrade smoke's kb_search id parsing."""

import importlib.util
from pathlib import Path

_PATH = Path(__file__).parent.parent / "scripts" / "smoke_work_user_upgrade.py"
_spec = importlib.util.spec_from_file_location("smoke_work_user_upgrade", _PATH)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_result_ids = _mod._result_ids


def test_typical_output_excludes_graph_section() -> None:
    out = (
        "2 result(s)\n\n"
        "[kb-00001] factual_reference | S3 presigned URL expiry (90%)\n  #s3 | docs\n\n"
        "[kb-00007] lesson_learned | Other (80%)\n  #x\n\n"
        "Related entries via graph:\n  See also: [kb-00009] Foo (via tag:s3)"
    )
    assert _result_ids(out) == ["kb-00001", "kb-00007"]


def test_no_results() -> None:
    assert _result_ids("No results found.") == []


def test_low_relevance_filtered_header() -> None:
    out = "1 result(s) (4 low-relevance filtered)\nNote: fts\n\n[kb-00003] decision | T (70%)\n"
    assert _result_ids(out) == ["kb-00003"]
