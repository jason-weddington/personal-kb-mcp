"""configure_logging: INFO from kb_service/kb_core reaches stderr (no caplog)."""

import logging
from collections.abc import Iterator

import pytest

from kb_service.cli import _HANDLER_MARKER, configure_logging


@pytest.fixture(autouse=True)
def _restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers = list(root.handlers)
    level = root.level
    saved = {n: logging.getLogger(n).level for n in ("kb_service", "kb_core")}
    yield
    root.handlers[:] = handlers
    root.setLevel(level)
    for n, lv in saved.items():
        logging.getLogger(n).setLevel(lv)


def _emit() -> None:
    logging.getLogger("kb_service.routes.kb_write_routes").info(
        "supersession-route probe"
    )
    logging.getLogger("kb_core.supersession").info("recompute probe")
    logging.getLogger("httpx").info("httpx probe")


def test_default_info(monkeypatch, capsys) -> None:
    monkeypatch.delenv("KB_LOG_LEVEL", raising=False)
    configure_logging()
    _emit()
    err = capsys.readouterr().err
    assert "supersession-route probe" in err
    assert "recompute probe" in err
    assert "httpx probe" not in err


def test_warning_level_hides_info(monkeypatch, capsys) -> None:
    monkeypatch.setenv("KB_LOG_LEVEL", "WARNING")
    configure_logging()
    _emit()
    err = capsys.readouterr().err
    assert "probe" not in err


def test_idempotent() -> None:
    configure_logging()
    configure_logging()
    marked = [
        h for h in logging.getLogger().handlers if getattr(h, _HANDLER_MARKER, False)
    ]
    assert len(marked) == 1


def test_bad_level_falls_back_with_warning(monkeypatch, capsys) -> None:
    monkeypatch.setenv("KB_LOG_LEVEL", "LOUD")
    configure_logging()
    _emit()
    err = capsys.readouterr().err
    assert err.count("LOUD") == 1
    assert "recompute probe" in err
