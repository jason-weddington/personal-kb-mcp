"""Dry-run mental_map body validation for the nightly maintenance loop.

``POST /api/kb/map-lint`` validates a candidate map body WITHOUT writing
anything: it runs kb-core's single-source-of-truth lint
(``kb_core.map_lint.lint_map_body``) and returns the structured findings
with an explicit pass/fail. The SAME findings are what the machine-principal
write gate rejects with 422 (``kb_write_routes``) and what the MCP channel
renders as advisories — one lint, three callers, no disagreement possible.

Hermetic by construction: the lint is a pure regex function, so this route
performs no KB I/O at all.
"""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from kb_core.map_lint import lint_map_body
from pydantic import BaseModel, Field

from kb_service.auth import get_current_user
from kb_service.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/kb", tags=["kb"])


class MapLintFindingModel(BaseModel):
    """One structured finding: machine-readable code plus human text."""

    code: str = Field(description="Stable machine-readable code, e.g. 'url'.")
    message: str = Field(description="Human text, channel-prefix-free.")


class MapLintValidateRequest(BaseModel):
    """A candidate mental_map body — nothing is written."""

    body: str = Field(description="The proposed knowledge_details of the map.")


class MapLintValidateResponse(BaseModel):
    """Explicit verdict plus the structured findings behind it."""

    valid: bool
    findings: list[MapLintFindingModel]


async def _read_candidate_body(request: Request) -> str:
    """Return the candidate map body from either request shape.

    TWO ACCEPTED SHAPES, and the raw one is what the gate actually uses:

    * ``application/json`` — ``{"body": "<map body>"}``, the original shape,
      kept because the dry-run debugging path and every existing test use it.
    * anything else (``text/plain``, or no content type) — the request body IS
      the map body, verbatim.

    The raw shape exists because of a real failure, not for generality. somnus
    composes a map body to a file and gates it by shelling out to
    ``curl --data-binary @<path>``; on 2026-09-21 that produced six straight
    422s, because the file held the body as prose while this endpoint demanded
    a JSON object. Requiring JSON there means something must escape a
    multi-paragraph, quote-bearing, newline-bearing document into a JSON
    string inside a shell command — which is precisely where quoting bugs
    live, and the gate is the one component whose failure silently withholds
    every map. Accepting the bytes removes the escaping step rather than
    asking the caller to get it right.

    A JSON object that is missing ``body`` or has it as a non-string is a 422
    from this function, not a silent fallback to treating the raw JSON text as
    a map body — a malformed request must not lint as though it were prose and
    pass.
    """
    raw = await request.body()
    text = raw.decode("utf-8", errors="replace")
    content_type = request.headers.get("content-type", "")
    if "application/json" not in content_type.lower():
        return text
    try:
        parsed = json.loads(text)
    except ValueError as exc:
        raise HTTPException(
            status_code=422, detail=f"content-type is JSON but body is not: {exc}"
        ) from exc
    candidate = parsed.get("body") if isinstance(parsed, dict) else None
    if not isinstance(candidate, str):
        raise HTTPException(
            status_code=422,
            detail=(
                "JSON requests must be an object with a string 'body' field;"
                " to send the map body verbatim, omit the JSON content type"
            ),
        )
    return candidate


@router.post("/map-lint", response_model=MapLintValidateResponse)
async def validate_map_body(
    request: Request,
    _user: Annotated[User, Depends(get_current_user)],
) -> JSONResponse:
    """Dry-run validation of a candidate mental_map body. Writes NOTHING.

    Accepts the body as ``{"body": "…"}`` JSON or as a raw ``text/plain``
    document — see :func:`_read_candidate_body` for why the raw shape exists.

    A failing body returns HTTP 422, NOT a 200 carrying ``valid: false`` —
    and the reason is the gate contract, not taste: the somnus nightly loop
    shells out to this endpoint as ``/bin/sh -c 'curl -sf -X POST ...'`` and
    reads the EXIT CODE as its ``run_checks`` verdict
    (docs/nightly-map-maintenance-design.md, §"THE GATE IS MANDATORY").
    ``curl -f`` treats any non-2xx status as failure; a 200 carrying
    ``{"valid": false}`` would read as a GREEN gate and silently defeat the
    whole mechanism — every guardrail in that design would reduce to a
    prompt instruction again. The failure body keeps the same JSON shape as
    the success body so a debugging caller can read the findings either way,
    but callers MUST key on the status code.

    Requires auth like every other kb route, but NOT admin: somnus runs as
    a plain non-admin user by design, and an admin-only gate would force the
    loop to hold admin credentials it should not have. The pointer count for
    the per-pointer budget is derived from the body itself
    (``kb_core.map_lint.count_map_pointers``) — a caller-supplied count could
    lie the budget into passing.
    """
    candidate = await _read_candidate_body(request)
    findings = lint_map_body(candidate)
    rendered = [
        MapLintFindingModel(code=f.code.value, message=f.message) for f in findings
    ]
    payload = MapLintValidateResponse(valid=not findings, findings=rendered)
    if findings:
        logger.info(
            "map-lint dry-run rejected a candidate body (%d findings)", len(findings)
        )
    # Same JSON payload shape either way; only the status differs, because
    # the status IS the verdict for curl -sf.
    return JSONResponse(
        status_code=200 if not findings else 422, content=payload.model_dump()
    )
