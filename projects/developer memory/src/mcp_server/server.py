"""MCP tool registrations and HTTP route handlers for Developer Memory.

Owns all @mcp.tool() callables and @mcp.custom_route() handlers.
All singletons (_sync_graph, _chroma, etc.) are imported from container.py —
never constructed here. Decorating with @mcp registers routes and tools on
the mcp instance at module import time; import order relative to container.py
does not matter because mcp itself is the singleton that accumulates handlers.

Route / tool inventory:
  HTTP routes:  /health, /quarantine, /quarantine/{doc_id}/release,
                /sync, /sync/status/{job_id}
  MCP tools:    sync_repository, query_memory, get_dev_persona,
                analyze_diff, generate_skills_pkg
"""

import threading
from typing import Any

from pydantic import BaseModel
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.mcp_server.container import (
    _GRAPH_CONFIG,
    _chroma,
    _diff_graph,
    _persona_graph,
    _query_graph,
    _skills_graph,
    _sync_graph,  # noqa: F401 — imported for patch.object interception in tests
    mcp,
)
from src.mcp_server.startup import logger
from src.mcp_server.sync_jobs import (
    _JOBS_LOCK,
    get_job,
    register_job,
)
from src.mcp_server.sync_jobs import (
    run_sync_job as _run_sync_job,
)
from src.middleware.path_guard import validate_target_path
from src.models.quarantine import QuarantineItem, QuarantineListResponse
from src.models.requests import SyncRequest

# ── Helper ────────────────────────────────────────────────────────────────────

# Alias for backwards-compatibility: tests import _register_job from sync_jobs
# via the server module. Keep underscore names visible here so that
# patch("src.mcp_server.server._register_job") works if needed.
_register_job = register_job
_get_job = get_job


def _sanitize_result(obj: Any) -> Any:
    """Recursively convert Pydantic models to dicts for JSON serialization.

    SyncState contains Pydantic model instances (SanitizedFile, UpsertResult, ParsedChunk)
    that are not JSON-serializable by stdlib json. Walk the object graph and convert any
    BaseModel instance via .model_dump(), recursing into dicts and lists.

    Args:
        obj: Any value from the SyncState result dict.

    Returns:
        A JSON-serializable equivalent of obj.
    """
    if isinstance(obj, BaseModel):
        return obj.model_dump()
    if isinstance(obj, dict):
        return {k: _sanitize_result(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_result(item) for item in obj]
    return obj


# ── Health endpoint ────────────────────────────────────────────────────────────
# FastMCP 3.x does not expose /health by default. Docker and make health both
# probe GET /health — register it as a custom route returning 200 OK.


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    """Liveness probe for Docker healthcheck and make health."""
    return JSONResponse({"status": "ok"})


# ── Quarantine REST endpoints ──────────────────────────────────────────────────
# These routes give the Streamlit UI HTTP-only access to quarantine data.
# The UI no longer imports ChromaLibrarianClient directly — all Chroma access
# goes through these endpoints so server-side middleware is never bypassed.
# Both handlers reference _chroma — the module-level ChromaLibrarianClient singleton
# constructed in container.py. Never instantiate a new client inside a handler.


@mcp.custom_route("/quarantine", methods=["GET"])
async def get_quarantine(request: Request) -> JSONResponse:
    """Return all documents currently held in the PII quarantine queue.

    Calls chroma.get_quarantined() (metadata-only get, no LLM, no embedding)
    and serialises results into QuarantineListResponse JSON.

    Response 200:
        {"items": [...], "total": N}
        Empty list when no documents are quarantined.
    """
    try:
        raw = _chroma.get_quarantined()
    except Exception as exc:  # noqa: BLE001
        logger.error("GET /quarantine failed [%s]: %s", type(exc).__name__, str(exc)[:200])
        return JSONResponse({"error": "Could not load quarantine queue."}, status_code=500)

    items = [
        QuarantineItem(
            doc_id=row["id"],
            file_path=row["metadata"].get("file_path", ""),
            quarantine_reason=row["metadata"].get("quarantine_reason", ""),
            repo_url=row["metadata"].get("repo_url", ""),
            indexed_at=row["metadata"].get("indexed_at", ""),
        )
        for row in raw
    ]
    response = QuarantineListResponse(items=items, total=len(items))
    return JSONResponse(response.model_dump())


@mcp.custom_route("/quarantine/{doc_id}/release", methods=["POST"])
async def post_quarantine_release(request: Request) -> JSONResponse:
    """Release a quarantined document for ingestion by updating its semantic_type.

    Path parameter:
        doc_id: ChromaDB document ID of the quarantined document.

    Response 200:
        {"doc_id": "<id>", "status": "released"}

    Response 404:
        {"error": "doc_id '<id>' not found in quarantine queue."}

    Note: doc_id is extracted from request.path_params, not the request body.
    """
    # path_params is populated by Starlette routing — same pattern as /sync/status/{job_id}
    doc_id: str = request.path_params["doc_id"]
    try:
        released = _chroma.release_quarantine_item(doc_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(
            "POST /quarantine/%s/release failed [%s]: %s",
            doc_id, type(exc).__name__, str(exc)[:200],
        )
        return JSONResponse({"error": "Release operation failed."}, status_code=500)

    if not released:
        return JSONResponse(
            {"error": f"doc_id {doc_id!r} not found in quarantine queue."},
            status_code=404,
        )
    return JSONResponse({"doc_id": doc_id, "status": "released"})


# ── Background sync HTTP routes ────────────────────────────────────────────────
# POST /sync   → validates, registers a job, fires a daemon thread, returns job_id immediately.
# GET  /sync/status/{job_id} → returns live job state for UI polling.
#
# These routes are NOT MCP tools — they are plain HTTP endpoints on the same server.
# The MCP tool sync_repository() below is kept for backwards-compatibility with Claude Code
# and other MCP clients that call tools by name. It blocks until the job finishes, which is
# acceptable for programmatic callers (they don't have a Streamlit session to keep alive).


@mcp.custom_route("/sync", methods=["POST"])
async def post_sync(request: Request) -> JSONResponse:
    """Start a background sync job. Returns job_id immediately.

    Request body (JSON):
        repo_url (str): Git repository URL — must be on the ALLOWED_HOSTS allowlist.
        branch   (str, optional): Branch name. Defaults to "main".

    Response (202 Accepted):
        {"job_id": "<uuid4>", "status": "queued"}

    Response (422 Unprocessable Entity):
        {"error": "<validation message>"}
    """
    try:
        body = await request.json()
    except Exception as exc:  # noqa: BLE001
        logger.debug("POST /sync: malformed JSON body [%s]", type(exc).__name__)
        return JSONResponse({"error": "Request body must be JSON."}, status_code=400)

    repo_url: str = body.get("repo_url", "")
    branch: str = body.get("branch", "main")

    try:
        SyncRequest(repo_url=repo_url, branch=branch)
    except ValueError as exc:
        logger.warning("POST /sync validation failed: %s", str(exc)[:200])
        return JSONResponse({"error": str(exc)}, status_code=422)

    job = _register_job(repo_url, branch)
    thread = threading.Thread(
        target=_run_sync_job, args=(job,), daemon=True, name=f"sync-{job.job_id[:8]}"
    )
    thread.start()
    logger.info(
        "POST /sync: job %s started (repo=%s branch=%s)", job.job_id[:8], repo_url, branch
    )
    return JSONResponse({"job_id": job.job_id, "status": "queued"}, status_code=202)


@mcp.custom_route("/sync/status/{job_id}", methods=["GET"])
async def get_sync_status(request: Request) -> JSONResponse:
    """Poll the status of a background sync job.

    Path parameter:
        job_id: UUID4 returned by POST /sync.

    Response (200 OK) — job found:
        {
          "job_id":        "<uuid4>",
          "status":        "queued" | "running" | "done" | "error",
          "stage":         "queued" | "cloning" | "pii_scanning" | "parsing" | "summarizing" | "done" | "error",
          "chunks_total":  <int>,
          "chunks_done":   <int>,
          "started_at":    "<iso8601>",
          "finished_at":   "<iso8601> | null",
          "error":         "<message> | null",
          "result":        <SyncState dict> | null
        }

    Response (404 Not Found) — unknown job_id:
        {"error": "Job not found."}
    """
    job_id: str = request.path_params.get("job_id", "")
    job = _get_job(job_id)
    if job is None:
        return JSONResponse({"error": "Job not found."}, status_code=404)

    # Snapshot job fields under lock so the response is internally consistent.
    with _JOBS_LOCK:
        raw_result = job.result if job.status == "done" else None
        payload: dict[str, Any] = {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage,
            "chunks_total": job.chunks_total,
            "chunks_done": job.chunks_done,
            "pii_files_done": job.pii_files_done,
            "pii_files_total": job.pii_files_total,
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "error": job.error,
            # Only include full result when done — avoid serialising large state on every poll.
            # _sanitize_result converts Pydantic models to plain dicts for JSONResponse.
            "result": _sanitize_result(raw_result),
        }
    return JSONResponse(payload)


# ── MCP tools ─────────────────────────────────────────────────────────────────


@mcp.tool()
def sync_repository(repo_url: str, branch: str = "main") -> dict:
    """Incremental sync of a git repository into Developer Memory.

    Validates repo_url against the ALLOWED_HOSTS allowlist before invoking the pipeline.
    Returns the final SyncState dict including upsert_results and trace.

    This MCP tool is kept for backwards-compatibility with Claude Code and other MCP
    clients. It starts a background job and blocks until completion, so the caller
    receives the full result. For the non-blocking variant, use POST /sync + GET /sync/status.

    Args:
        repo_url: Git repository URL (must be on the ALLOWED_HOSTS allowlist).
        branch: Branch to sync. Defaults to "main".

    Returns:
        SyncState dict with upsert_results, quarantined_files, trace, and any error.
    """
    try:
        SyncRequest(repo_url=repo_url, branch=branch)
    except ValueError as exc:
        logger.warning(
            "sync_repository validation failed [%s]: %s", type(exc).__name__, str(exc)[:200]
        )
        return {"error": str(exc), "upsert_results": [], "trace": []}

    job = _register_job(repo_url, branch)
    thread = threading.Thread(
        target=_run_sync_job, args=(job,), daemon=True, name=f"sync-{job.job_id[:8]}"
    )
    thread.start()
    thread.join()  # Block until complete — acceptable for MCP callers (no HTTP session)

    with _JOBS_LOCK:
        if job.error:
            return {"error": job.error, "upsert_results": [], "trace": []}
        return job.result or {"error": "sync produced no result", "upsert_results": [], "trace": []}


@mcp.tool()
def query_memory(query: str, tech_filter: list[str] | None = None) -> str:
    """Semantic search across Developer Memory with optional tech stack filter.

    Args:
        query: Natural-language query string.
        tech_filter: Optional list of tech stack labels to narrow the search.

    Returns:
        LLM-synthesized answer string, or a safe error message if the query fails.
    """
    result = _query_graph.invoke(
        {"query": query, "tech_filter": tech_filter, "trace": []},
        config=_GRAPH_CONFIG,
    )
    answer = result.get("final_answer") or result.get("error")
    if answer is None:
        logger.warning(
            "query_memory: graph returned neither final_answer nor error; state keys=%s",
            list(result.keys()),
        )
        return "No results found."
    return answer


@mcp.tool()
def get_dev_persona(scope: str = "project", recency_months: int = 6) -> dict:
    """Synthesize developer tendencies into a stylistic profile.

    Args:
        scope: Query scope — "project", "file", or "author".
        recency_months: Lookback window for temporal weighting. Docs within this window
            receive 3× weight in tendency aggregation. Default: 6.

    Returns:
        PersonaState dict with persona_profile and trace, or error if insufficient data.
    """
    return _persona_graph.invoke(
        {"scope": scope, "recency_months": recency_months, "trace": []},
        config=_GRAPH_CONFIG,
    )


@mcp.tool()
def analyze_diff(diff_text: str) -> dict:
    """Explain the 'Why' behind code changes relative to the historical persona.

    Args:
        diff_text: Unified diff text (--- / +++ header format).

    Returns:
        DiffState dict with analysis_result, coaching_alert, and trace.
    """
    return _diff_graph.invoke(
        {"diff_text": diff_text, "trace": []},
        config=_GRAPH_CONFIG,
    )


@mcp.tool()
def generate_skills_pkg(target_path: str, force: bool = False) -> dict:
    """Export PROJECT_SKILLS.md summarizing the repo's Professional DNA (FR-SKILLS-01).

    Args:
        target_path: Destination directory or .md file path for PROJECT_SKILLS.md.
            Must resolve under SKILLS_EXPORT_DIR (env var, default: cwd).
        force: When True, skip the SHA cache gate and always re-synthesize from ChromaDB.
            Defaults to False (SHA-gated — synthesis is skipped if corpus is unchanged).

    Returns:
        SkillsState dict with export_result on success, or error on failure.
    """
    try:
        safe_path = validate_target_path(target_path)
    except ValueError as exc:
        return {"error": str(exc), "export_result": None, "trace": []}

    return _skills_graph.invoke(
        {"target_path": safe_path, "trace": [], "cache_hit": False, "force": force},
        config=_GRAPH_CONFIG,
    )
