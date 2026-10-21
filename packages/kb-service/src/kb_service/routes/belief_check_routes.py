"""``POST /api/kb/belief-check``: mid-thought correction of a stated wrong belief.

A harness posts the agent's latest text and reasoning after a model turn
that has a pending tool call; the route ranks the project's recorded wrong
beliefs against it (:func:`kb_service.belief_check.check_beliefs`) and
returns the top matches with the server's threshold. The consumer injects a
match's corrected fact only when ``matches[0].score >= threshold``.

Auth is the same as ``GET /api/kb/prevention``. The route never answers with
a 5xx: every outcome, including an exception, is HTTP 200 with a ``reason``.
Exactly one INFO ``belief_check`` line is logged per request; it carries
entry ids, scores and counters, never the request text.
"""

import logging
import os
import time
from typing import Annotated

from fastapi import APIRouter, Depends, Request

from kb_service import belief_check as bc
from kb_service.auth import get_current_user
from kb_service.config import get_embedding_model
from kb_service.models import (
    BeliefCheckMatch,
    BeliefCheckReason,
    BeliefCheckRequest,
    BeliefCheckResponse,
    User,
)
from kb_service.routes.prevention_routes import _project_disabled
from kb_service.write_policy import ENGINE_HEADER, HARNESS_HEADER, sanitize_harness

router = APIRouter(prefix="/api/kb", tags=["kb"])

logger = logging.getLogger(__name__)


def _where(exc: BaseException) -> str:
    """``<file basename>:<lineno>`` of the innermost traceback frame."""
    tb = exc.__traceback__
    if tb is None:
        return ""
    while tb.tb_next is not None:
        tb = tb.tb_next
    return f"{os.path.basename(tb.tb_frame.f_code.co_filename)}:{tb.tb_lineno}"


def _embed_model(request: Request) -> str:
    try:
        if request.app.state.kb.embedder is not None:
            return get_embedding_model()
    except Exception:  # noqa: S110 - an unknown model is reported as ""
        pass
    return ""


@router.post("/belief-check", response_model=BeliefCheckResponse)
async def post_belief_check(
    request: Request,
    body: BeliefCheckRequest,
    user: Annotated[User, Depends(get_current_user)],
) -> BeliefCheckResponse:
    """Rank the project's recorded wrong beliefs against the agent's text."""
    del user  # auth gate only
    started = time.monotonic()
    project = body.project.strip()
    outcome: bc.BeliefCheckOutcome | None = None
    reason: BeliefCheckReason
    detail = ""
    if project == "":
        reason = "no-project"
    elif _project_disabled(project):
        reason = "project-disabled"
    else:
        try:
            outcome = await bc.check_beliefs(
                request.app.state.kb, project, body.text, body.reasoning
            )
            reason = outcome.reason
            detail = outcome.detail
        except Exception as exc:
            logger.warning(
                "belief_check failed project=%s exc=%s",
                sanitize_harness(project),
                type(exc).__name__,
            )
            outcome = None
            reason = "error"
            detail = _where(exc)
    matches: list[BeliefCheckMatch] = []
    if outcome is not None and reason == "checked":
        matches = [
            m.model_copy(update={"score": float(m.score)}) for m in outcome.matches
        ]
    embed_model = _embed_model(request)
    threshold = float(bc.BELIEF_CHECK_THRESHOLD)
    _log(
        request,
        body,
        project,
        reason,
        detail,
        outcome,
        matches,
        threshold,
        embed_model,
        started,
    )
    return BeliefCheckResponse(
        matches=matches,
        threshold=threshold,
        matcher_version=bc.BELIEF_CHECK_MATCHER_VERSION,
        reason=reason,
        candidates=outcome.candidates if outcome is not None else 0,
        scored=outcome.scored if outcome is not None else 0,
        embed_model=embed_model,
    )


def _log(
    request: Request,
    body: BeliefCheckRequest,
    project: str,
    reason: BeliefCheckReason,
    detail: str,
    outcome: bc.BeliefCheckOutcome | None,
    matches: list[BeliefCheckMatch],
    threshold: float,
    embed_model: str,
    started: float,
) -> None:
    o = outcome if outcome is not None else bc.BeliefCheckOutcome(reason=reason)
    top = matches[0] if matches else None
    logger.info(
        "belief_check project=%s session_id=%s turn_index=%s harness=%s engine=%s"
        " reason=%s detail=%s warm=%s candidates=%d candidates_truncated=%d"
        " uncached=%d scored=%d chunks=%d pending_calls=%d top_id=%s top_score=%s"
        " above_threshold=%s match_ids=%s match_scores=%s top_chunk_idx=%s"
        " top_chunk_src=%s threshold=%s matcher_version=%d embed_model=%s dim=%s"
        " load_ms=%d embed_ms=%d latency_ms=%d",
        sanitize_harness(project),
        sanitize_harness(body.session_id),
        "" if body.turn_index is None else body.turn_index,
        sanitize_harness(request.headers.get(HARNESS_HEADER)),
        sanitize_harness(request.headers.get(ENGINE_HEADER)),
        reason,
        detail,
        o.warm,
        o.candidates,
        o.candidates_truncated,
        o.uncached,
        o.scored,
        o.chunks,
        len(body.pending_calls),
        top.entry_id if top else "",
        f"{top.score:.4f}" if top else "",
        "true" if reason == "checked" and top and top.score >= threshold else "false",
        ",".join(m.entry_id for m in matches),
        ",".join(f"{m.score:.4f}" for m in matches),
        "" if top is None or o.top_chunk_idx is None else o.top_chunk_idx,
        o.top_chunk_src if top else "",
        threshold,
        bc.BELIEF_CHECK_MATCHER_VERSION,
        sanitize_harness(embed_model),
        "" if o.dim is None else o.dim,
        o.load_ms,
        o.embed_ms,
        int((time.monotonic() - started) * 1000),
    )
