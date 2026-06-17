"""Sync job registry and background thread orchestration.

Owns the in-process job state machine for background sync operations:
  - SyncJobStatus / SyncJobStage type aliases
  - SyncJob dataclass
  - _JOBS dict + _JOBS_LOCK for thread-safe mutations
  - register_job / get_job — CRUD helpers
  - run_sync_job — background thread target (lazily imports _sync_graph from container
    to break the circular dependency: sync_jobs → container → sync_jobs)
  - _sync_progress_callback / _pii_progress_callback — called by pipeline nodes
    to push real-time progress into live SyncJob state
"""

import threading
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from src.mcp_server.startup import _log_memory, logger

# ── Type aliases ─────────────────────────────────────────────────────────────

SyncJobStatus = Literal["queued", "running", "done", "error"]

# Stage labels surfaced to the UI for stage-by-stage progress visibility.
# Ordered to match the pipeline execution sequence.
SyncJobStage = Literal[
    "queued", "cloning", "cache_filtering", "pii_scanning",
    "parsing", "summarizing", "done", "error"
]


# ── Job dataclass ─────────────────────────────────────────────────────────────

@dataclass
class SyncJob:
    """Live state for one background sync job."""

    job_id: str
    repo_url: str
    branch: str
    status: SyncJobStatus = "queued"
    stage: SyncJobStage = "queued"  # Fine-grained pipeline stage for UI visibility
    chunks_total: int = 0       # Set by summarize_and_upsert as soon as chunk count is known
    chunks_done: int = 0        # Incremented by summarize_and_upsert after each upsert
    pii_files_done: int = 0     # Incremented by pii_sanitizer per file scanned
    pii_files_total: int = 0    # Set by pii_sanitizer when scanning begins
    started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    finished_at: str | None = None
    result: dict[str, Any] | None = None   # Full SyncState dict on completion
    error: str | None = None


# ── Shared state ──────────────────────────────────────────────────────────────

_JOBS: dict[str, SyncJob] = {}
_JOBS_LOCK = threading.Lock()


# ── Registry helpers ──────────────────────────────────────────────────────────

def register_job(repo_url: str, branch: str) -> SyncJob:
    """Create a new SyncJob, register it, and return it.

    Args:
        repo_url: Validated git repository URL.
        branch: Branch name to sync.

    Returns:
        The newly created SyncJob with status='queued'.
    """
    job = SyncJob(job_id=str(uuid.uuid4()), repo_url=repo_url, branch=branch)
    with _JOBS_LOCK:
        _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> SyncJob | None:
    """Return the SyncJob for job_id, or None if not found.

    Args:
        job_id: UUID4 string identifying the sync job.

    Returns:
        The SyncJob, or None if job_id is not in the registry.
    """
    with _JOBS_LOCK:
        return _JOBS.get(job_id)


# ── Background thread target ──────────────────────────────────────────────────

def run_sync_job(job: SyncJob) -> None:
    """Background thread target: run the sync graph and update job state throughout.

    Invoked via threading.Thread(target=run_sync_job, args=(job,), daemon=True).
    Daemon=True ensures the thread does not block process exit on SIGTERM.

    _sync_graph and _GRAPH_CONFIG are lazily imported from container inside this
    function body to break the circular dependency:
      sync_jobs → container (for graph/config) AND container → sync_jobs (for callbacks).
    Lazy import is the spec-prescribed resolution — see Section 5 of ServerRefactor.md.

    Args:
        job: The SyncJob to run. Its status/stage/result/error fields are mutated in-place.
    """
    _log_memory(f"job {job.job_id[:8]} start")
    with _JOBS_LOCK:
        job.status = "running"
        job.stage = "cloning"

    try:
        # Lazy import — resolves circular dependency with container.py.
        # container imports sync_jobs (for callbacks); sync_jobs cannot import container
        # at module level without triggering that cycle. noqa: PLC0415 is intentional.
        from src.mcp_server.container import _GRAPH_CONFIG, _sync_graph  # noqa: PLC0415

        state = _sync_graph.invoke(
            {
                "repo_url": job.repo_url,
                "branch": job.branch,
                "commit_sha": "",
                "trace": [],
                # trace_events is required (operator.add) — must be present in initial state.
                "trace_events": [],
                # Job ID is threaded through state so summarize_and_upsert can report progress.
                "_job_id": job.job_id,
            },
            config=_GRAPH_CONFIG,
        )
        with _JOBS_LOCK:
            job.status = "done"
            job.stage = "done"
            job.finished_at = datetime.now(UTC).isoformat()
            job.result = state
    except Exception as exc:  # noqa: BLE001
        logger.error("sync job %s failed: %s", job.job_id[:8], exc)
        with _JOBS_LOCK:
            job.status = "error"
            job.stage = "error"
            job.finished_at = datetime.now(UTC).isoformat()
            job.error = str(exc)[:500]

    _log_memory(f"job {job.job_id[:8]} complete")


# ── Progress callbacks ─────────────────────────────────────────────────────────

def _sync_progress_callback(job_id: str, chunks_done: int, chunks_total: int) -> None:
    """Update live job state so GET /sync/status reflects real-time LLM summarization progress.

    Called by summarize_and_upsert after each chunk is processed (LLM summarized or cache-skipped).
    Advances stage to "summarizing" on first invocation so the UI can distinguish this phase
    from the earlier cloning/PII/parsing stages.

    Args:
        job_id: UUID4 of the running sync job.
        chunks_done: Number of chunks processed so far (1-based index of this chunk).
        chunks_total: Total chunks dispatched in this sync run.
    """
    job = get_job(job_id)
    if job is None:
        return
    with _JOBS_LOCK:
        job.chunks_total = chunks_total
        job.chunks_done = chunks_done
        # Advance stage to "summarizing" on first chunk — covers both LLM and cache-skip paths.
        if job.stage not in ("done", "error"):
            job.stage = "summarizing"


def _pii_progress_callback(job_id: str, pii_files_done: int, pii_files_total: int) -> None:
    """Update live job state so GET /sync/status reflects real-time PII scan progress.

    Called by pii_sanitizer after each file is scanned.
    Advances stage to "pii_scanning" on first invocation.

    Args:
        job_id: UUID4 of the running sync job.
        pii_files_done: Number of files scanned so far.
        pii_files_total: Total files entering pii_sanitizer.
    """
    job = get_job(job_id)
    if job is None:
        return
    with _JOBS_LOCK:
        job.pii_files_done = pii_files_done
        job.pii_files_total = pii_files_total
        if job.stage not in ("done", "error"):
            job.stage = "pii_scanning"
