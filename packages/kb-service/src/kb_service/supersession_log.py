"""Shared logging for supersession-reconcile reports (startup + admin endpoint)."""

import logging
from typing import Any


def log_reconcile_report(report: Any, log: logging.Logger) -> None:
    """Log one INFO summary line and one WARNING per drifted ``superseded_by`` row.

    Any ``supersession-reconcile drift`` line is a defect signal: some writer
    left the invariant broken.
    """
    log.info(
        "supersession-reconcile edges_added=%d set=%d cleared=%d",
        report.edges_added,
        report.set_count,
        report.cleared_count,
    )
    for target, old, new in report.changed:
        log.warning(
            "supersession-reconcile drift target=%s old=%r new=%r", target, old, new
        )
