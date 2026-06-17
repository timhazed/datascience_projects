"""summarize_and_upsert LangGraph Send-target node — Spec §12.

Collapses intent_summarizer + chroma_upsert into a single Send-target so each chunk
is persisted to ChromaDB immediately after LLM summarization, rather than after all
chunks complete (the old fan-in bottleneck described in §12.1).

Three-step per-chunk flow:
  1. Skip-before-LLM: chroma.exists(chunk_id) → skip the LLM call entirely if already indexed.
  2. LLM summarization via intent_summary_chain (with retry).
  3. Immediate upsert into ChromaDB via chroma.upsert_chunk().

Concurrency control:
  The factory accepts an ollama_concurrency parameter (int, default 1) that maps to
  OLLAMA_NUM_PARALLEL. A threading.Semaphore is acquired before each LLM call and
  released immediately after — so at most N chains run concurrently. The semaphore is
  created once at factory time; it is never rebuilt per invocation.

Progress reporting:
  If a progress_callback was injected and the sub-state contains "_job_id", the callback
  is invoked after each upsert (success or skip) so the HTTP polling endpoint reflects
  live progress. Callback errors are suppressed via contextlib.suppress(Exception).
"""

import contextlib
import hashlib
import logging
import threading
import time as _time
from collections.abc import Callable
from datetime import UTC, datetime

from langchain_ollama import ChatOllama

from src.chains.intent_summary_chain import _ChunkAnnotation, build_intent_summary_chain
from src.db.chroma_client import ChromaLibrarianClient
from src.models.chunk import ParsedChunk, SummarizedChunk, UpsertResult
from src.models.trace_event import TraceEvent, TraceEventType
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)

# Type alias for the optional progress callback injected by the server.
# Signature: (job_id: str, chunks_done: int, chunks_total: int) -> None
ProgressCallback = Callable[[str, int, int], None]


def make_summarize_and_upsert_node(
    llm: ChatOllama,
    chroma: ChromaLibrarianClient,
    ollama_concurrency: int = 1,
    progress_callback: ProgressCallback | None = None,
) -> Callable[[dict], dict]:
    """Return a Send-target node that summarizes and immediately upserts one ParsedChunk.

    The LLM chain and ChromaDB client are injected at factory time and reused across
    all invocations — never rebuilt per-call.

    Args:
        llm: ChatOllama instance constructed at server startup.
        chroma: ChromaLibrarianClient instance constructed at server startup.
        ollama_concurrency: Maximum number of simultaneous LLM calls. Must match
            OLLAMA_NUM_PARALLEL on the Ollama server. Default 1 (serial). Use 2 for
            M4 Max 128GB per §12.3.
        progress_callback: Optional callable invoked after each chunk is processed
            (upserted or skipped). Signature: (job_id, chunks_done, chunks_total) -> None.
            If None, no progress reporting is done (tests, direct invocation).

    Returns:
        LangGraph Send-target node that accepts sub-state:
            {"chunk": ParsedChunk, "repo_url": str, "branch": str, "commit_sha": str,
             "_job_id": str | None, "chunks_total": int, "chunk_index": int}
        and returns:
            {"upsert_results": [UpsertResult], "trace": [str]}
    """
    chain = build_intent_summary_chain(llm)
    _semaphore = threading.Semaphore(ollama_concurrency)

    def summarize_and_upsert(state: dict) -> dict:
        """Summarize one ParsedChunk via LLM and upsert it to ChromaDB immediately.

        Implements the skip-before-LLM fast path: if the chunk is already in ChromaDB
        (same SHA-256 content_hash), the LLM call is bypassed entirely. This enables
        free resume — a restarted sync skips all previously-indexed chunks in ~1ms each.

        Sub-state fields consumed:
            chunk: ParsedChunk to summarize and upsert.
            repo_url, branch, commit_sha: Metadata forwarded from SyncState.
            _job_id: Optional job UUID4 for progress reporting.
            chunk_index: 1-based position of this chunk in the full Send batch.
            chunks_total: Total number of summarize_and_upsert Send targets dispatched.

        Returns exactly one UpsertResult in a list for operator.add accumulation.
        """
        chunk: ParsedChunk = state["chunk"]
        repo_url: str = state.get("repo_url", "")
        branch: str = state.get("branch", "main")
        commit_sha: str = state.get("commit_sha", "")
        job_id: str | None = state.get("_job_id")
        chunk_index: int = state.get("chunk_index", 0)
        chunks_total: int = state.get("chunks_total", 0)

        indexed_at = datetime.now(UTC).isoformat()
        base_meta = {
            "repo_url": repo_url,
            "branch": branch,
            "commit_sha": commit_sha,
            "indexed_at": indexed_at,
            "source_file_sha": state.get("file_shas", {}).get(chunk.path, ""),
        }

        # Derive the content_hash the same way SummarizedChunk does — SHA-256 of content.
        # This avoids constructing a full SummarizedChunk just to get the id.
        chunk_id = hashlib.sha256(chunk.content.encode()).hexdigest()

        # ── Skip-before-LLM fast path ──────────────────────────────────────
        if chroma.exists(chunk_id):
            logger.info(
                "summarize_and_upsert: [%d/%d] skip (cached) %s",
                chunk_index, chunks_total, chunk.path,
            )
            _report_progress(progress_callback, job_id, chunk_index, chunks_total)
            return {
                "upsert_results": [
                    UpsertResult(
                        content_hash=chunk_id,
                        file_path=chunk.path,
                        action="skipped",
                    )
                ],
                "trace": [f"summarize_and_upsert: skip (cached) {chunk.path}"],
                "trace_events": [
                    TraceEvent(
                        node="summarize_and_upsert",
                        event=TraceEventType.CACHED,
                        path=chunk.path,
                    )
                ],
            }

        # ── LLM summarization (concurrency-gated) ─────────────────────────
        logger.info(
            "summarize_and_upsert: [%d/%d] LLM start  %s  (%d chars)",
            chunk_index, chunks_total, chunk.path, len(chunk.content),
        )
        _t_llm = _time.perf_counter()
        with _semaphore:
            try:
                annotation: _ChunkAnnotation = invoke_with_retry(
                    chain, {"content": chunk.content, "path": chunk.path}
                )
            except Exception as exc:
                logger.error(
                    "summarize_and_upsert: [%d/%d] LLM failed [%s] %.1fs  %s: %s",
                    chunk_index, chunks_total,
                    type(exc).__name__,
                    _time.perf_counter() - _t_llm,
                    chunk.path,
                    str(exc)[:200],
                )
                _report_progress(progress_callback, job_id, chunk_index, chunks_total)
                return {
                    "error": f"Summarization failed for {chunk.path}.",
                    "upsert_results": [
                        UpsertResult(
                            content_hash=chunk_id,
                            file_path=chunk.path,
                            action="error",
                            error=f"LLM failed: {str(exc)[:200]}",
                        )
                    ],
                    "trace": [f"summarize_and_upsert: LLM error {chunk.path}"],
                    "trace_events": [
                        TraceEvent(
                            node="summarize_and_upsert",
                            event=TraceEventType.ERROR,
                            path=chunk.path,
                            detail=type(exc).__name__,
                        )
                    ],
                }
        logger.info(
            "summarize_and_upsert: [%d/%d] LLM done   %s  (%.1fs)",
            chunk_index, chunks_total, chunk.path, _time.perf_counter() - _t_llm,
        )

        summarized = SummarizedChunk(
            content=chunk.content,
            path=chunk.path,
            intent_summary=annotation.intent_summary,
            tech_stack=annotation.tech_stack,
            semantic_type=annotation.semantic_type,
            key_identifiers=annotation.key_identifiers,
        )

        # ── Immediate upsert ───────────────────────────────────────────────
        result = chroma.upsert_chunk(summarized, base_meta)
        logger.info(
            "summarize_and_upsert: [%d/%d] %s  %s",
            chunk_index, chunks_total, result.action, chunk.path,
        )
        _report_progress(progress_callback, job_id, chunk_index, chunks_total)
        return {
            "upsert_results": [result],
            "trace": [f"summarize_and_upsert: {result.action} {chunk.path}"],
            "trace_events": [
                TraceEvent(
                    node="summarize_and_upsert",
                    event=TraceEventType.OK if result.action != "error" else TraceEventType.ERROR,
                    path=chunk.path,
                )
            ],
        }

    return summarize_and_upsert


def _report_progress(
    progress_callback: ProgressCallback | None,
    job_id: str | None,
    chunks_done: int,
    chunks_total: int,
) -> None:
    """Invoke the progress callback, suppressing any exception.

    Args:
        progress_callback: Callable or None.
        job_id: Job UUID4 string or None — if None, callback is skipped.
        chunks_done: Number of chunks processed so far (1-based, this chunk included).
        chunks_total: Total chunks in the current Send batch.
    """
    if progress_callback and job_id:
        with contextlib.suppress(Exception):
            progress_callback(job_id, chunks_done, chunks_total)
