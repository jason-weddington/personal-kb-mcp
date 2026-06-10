"""KB ingest endpoints: text, URL, and file upload.

``POST /api/kb/ingest/text``   — ingest raw text content
``POST /api/kb/ingest/url``    — ingest a URL (fetch+extract or pre-fetched)
``POST /api/kb/ingest/file``   — ingest an uploaded file (multipart/form-data)

All three endpoints require authentication (any authenticated user — MCP-channel
parity; ingest is not admin-gated).

**Timeout note**: ingestion is a slow LLM pipeline (chunked extraction; per-call
provider timeouts ``KB_OLLAMA_LLM_TIMEOUT=120.0s`` / ``KB_ANTHROPIC_TIMEOUT=30.0s``
defaults, multiplied across chunks — total can be minutes for a 10 MiB file).
The service imposes no request timeout; clients must use generous read timeouts.

**Safety gate**: when ``KB_SKIP_SAFETY`` is not ``TRUE``, all three endpoints
check for the ``detect-secrets`` and ``scrubadub`` distributions at request time.
If either is absent the endpoint returns ``503`` — fail-closed, because kb-core
silently degrades when its scrubbers are missing.  Add
``kb-core[safety]`` (or ``KB_SKIP_SAFETY=TRUE``) to restore normal operation.
"""

import dataclasses
import importlib.util
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile

from kb_service import config
from kb_service.attribution import resolve_attribution
from kb_service.auth import get_current_user
from kb_service.models import (
    IngestFileResult,
    IngestTextRequest,
    IngestUrlRequest,
    User,
)

router = APIRouter(prefix="/api/kb/ingest", tags=["kb"])

_UPLOAD_CHUNK = 1_048_576  # 1 MiB streaming chunk


def _missing_safety_deps() -> list[str]:
    """Return distribution names of any missing safety dependencies.

    Checks for ``detect-secrets`` and ``scrubadub`` by module-spec lookup.
    Returns a (possibly empty) list of distribution names using hyphens
    (e.g. ``["detect-secrets", "scrubadub"]``).
    """
    missing: list[str] = []
    if importlib.util.find_spec("detect_secrets") is None:
        missing.append("detect-secrets")
    if importlib.util.find_spec("scrubadub") is None:
        missing.append("scrubadub")
    return missing


def _check_safety() -> None:
    """Raise ``HTTP 503`` when safety dependencies are missing and not bypassed.

    No-ops when ``KB_SKIP_SAFETY=TRUE``.
    """
    if config.is_safety_skip():
        return
    missing = _missing_safety_deps()
    if missing:
        names = ", ".join(missing)
        raise HTTPException(
            status_code=503,
            detail=(
                f"Safety dependencies missing: {names}. "
                "Install them or set KB_SKIP_SAFETY=TRUE to bypass."
            ),
        )


@router.post("/text", response_model=IngestFileResult)
async def ingest_text(
    body: IngestTextRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> IngestFileResult:
    """Ingest raw text content through the full kb-core pipeline.

    Calls ``kb.ingest_text(body.content, body.source_name, ...)`` on the
    singleton ``KnowledgeBase``.  kb-core enforces the extension allowlist on
    ``source_name`` and the size limit, returning ``action="skipped"`` with a
    reason — those pass through as ``HTTP 200`` with the result body, never as
    HTTP errors.

    ``dry_run=True`` runs the extraction pipeline but writes nothing to the
    database; the result carries ``action="dry_run"`` and the count of entries
    that *would* be created.
    """
    _check_safety()
    attr = await resolve_attribution(user)
    try:
        result = await request.app.state.kb.ingest_text(
            body.content,
            body.source_name,
            project_ref=body.project_ref,
            dry_run=body.dry_run,
            contributor=attr.contributor,
            team=attr.team,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return IngestFileResult.model_validate(dataclasses.asdict(result))


@router.post("/url", response_model=IngestFileResult)
async def ingest_url(
    body: IngestUrlRequest,
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
) -> IngestFileResult:
    """Ingest a URL by fetching+extracting content, or using pre-fetched content.

    When ``body.content`` is ``None``, the endpoint calls
    ``kb.ingest_url(body.url, ...)`` which performs an HTTP fetch and
    trafilatura HTML extraction internally.  This works for public sites only.

    When ``body.content`` is provided, the endpoint calls
    ``kb.ingest_url_content(body.content, body.url, ...)`` which skips the
    HTTP fetch and HTML extraction stages, feeding the pre-fetched text
    directly into the pipeline.  This is useful for authenticated/SSO/
    JavaScript-rendered pages that the service cannot fetch itself.

    kb-core does **not** raise on fetch failure — ``FileIngester.ingest_url``
    returns ``FileResult(action="error", reason="Failed to fetch: ...")`` on
    ``httpx.HTTPError`` and on empty trafilatura extraction.  Those pass
    through as ``HTTP 200`` with the result body.
    """
    _check_safety()
    attr = await resolve_attribution(user)
    try:
        if body.content is None:
            result = await request.app.state.kb.ingest_url(
                body.url,
                project_ref=body.project_ref,
                dry_run=body.dry_run,
                contributor=attr.contributor,
                team=attr.team,
            )
        else:
            result = await request.app.state.kb.ingest_url_content(
                body.content,
                body.url,
                project_ref=body.project_ref,
                dry_run=body.dry_run,
                contributor=attr.contributor,
                team=attr.team,
            )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return IngestFileResult.model_validate(dataclasses.asdict(result))


@router.post("/file", response_model=IngestFileResult)
async def ingest_file(
    request: Request,
    user: Annotated[User, Depends(get_current_user)],
    file: UploadFile,
    project_ref: str | None = Form(None),
    dry_run: bool = Form(False),
) -> IngestFileResult:
    """Ingest an uploaded file through the kb-core pipeline.

    Accepts ``multipart/form-data`` with a ``file`` part (required) and
    optional ``project_ref`` / ``dry_run`` form fields.

    **Upload size cap** (``KB_INGEST_MAX_FILE_SIZE``, default 10 MiB):
    the upload is streamed in 1 MiB chunks; if the cumulative byte count
    *exceeds* the limit the endpoint returns ``HTTP 413`` with the limit in
    the detail string.  An upload of *exactly* the limit is accepted.

    **Ingested-files identity**: kb-core keys dedup records to the original
    filename (``path.name`` when ``base_dir`` and ``display_name`` are unset —
    see ingester.py line ~217).  Re-uploading a same-named file therefore hits
    the normal ``unchanged``/``replace`` dedup path.  kb-core's deny-list,
    extension-allowlist, and size checks run on the temp path and surface as
    ``action="skipped"`` results (``HTTP 200``), not HTTP errors.
    """
    _check_safety()

    filename = file.filename or ""
    if not filename:
        raise HTTPException(status_code=422, detail="A non-empty filename is required.")
    name = Path(filename).name
    if not name:
        raise HTTPException(status_code=422, detail="A non-empty filename is required.")

    # Stream the upload, enforcing the size cap per chunk.
    max_size = config.get_ingest_max_file_size()
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(_UPLOAD_CHUNK)
        if not chunk:
            break
        total += len(chunk)
        if total > max_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Upload exceeds maximum file size of {max_size} bytes. "
                    "Raise KB_INGEST_MAX_FILE_SIZE to allow larger uploads."
                ),
            )
        chunks.append(chunk)
    data = b"".join(chunks)

    attr = await resolve_attribution(user)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / name
        tmp_path.write_bytes(data)
        try:
            result = await request.app.state.kb.ingest_file(
                tmp_path,
                project_ref=project_ref,
                dry_run=dry_run,
                contributor=attr.contributor,
                team=attr.team,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return IngestFileResult.model_validate(dataclasses.asdict(result))
