"""chroma_upsert_quarantined LangGraph node — Spec §12.5.

Terminal node responsible solely for upserting PII-quarantined files into ChromaDB
with semantic_type="quarantine" and a per-file sentinel content value. This is a thin
wrapper around ChromaLibrarianClient.upsert_chunk(); no LLM is called.

Separate from summarize_and_upsert so the quarantine path does not acquire the Ollama
concurrency semaphore, and so progress reporting counts summarized chunks only (not
quarantine upserts, which are near-instant).

Content uniqueness: the stored content is "{QUARANTINE_SENTINEL}\\npath:{qf.path}" rather
than the bare sentinel. This ensures each quarantined file produces a distinct SHA-256
content_hash — the ChromaDB document id. Without the path suffix, all quarantined files
would share the same hash and only the last upserted file would remain in the collection
(Case 3 metadata-only overwrites would erase the earlier entries from the PII Review Queue).

ChromaDB metadata for quarantined files:
  content        = f"{QUARANTINE_SENTINEL}\\npath:{qf.path}"  (unique per file)
  semantic_type  = "quarantine"
  intent_summary = "" (no LLM ran)
  quarantine_reason = SanitizedFile.quarantine_reason (stored in meta dict)
"""

import logging
from collections.abc import Callable
from datetime import UTC, datetime

from src.db.chroma_client import ChromaLibrarianClient
from src.models.chunk import SummarizedChunk, UpsertResult
from src.models.sanitized_file import QUARANTINE_SENTINEL, SanitizedFile
from src.models.trace_event import TraceEvent, TraceEventType

logger = logging.getLogger(__name__)


def make_chroma_upsert_quarantined_node(
    chroma: ChromaLibrarianClient,
) -> Callable[[dict], dict]:
    """Return a node that upserts all quarantined files into ChromaDB.

    No LLM is called. Each quarantined file is indexed with semantic_type="quarantine"
    and QUARANTINE_SENTINEL as the content, so the document is present in ChromaDB for
    the PII Review Queue but its real content is never stored.

    Idempotency: upsert_chunk() uses SHA-256(f"{QUARANTINE_SENTINEL}\\npath:{path}") as
    the document id — unique per file path. Re-indexing the same quarantined file returns
    action="skipped" (Case 2) without re-processing.

    Args:
        chroma: ChromaLibrarianClient instance constructed at server startup.

    Returns:
        LangGraph node function that reads quarantined_files from SyncState and upserts
        all quarantined documents. Returns {"upsert_results": [...], "trace": [...]}.
    """

    def chroma_upsert_quarantined(state: dict) -> dict:
        """Upsert all quarantined files from SyncState into ChromaDB.

        Reads quarantined_files from state. Each failure produces UpsertResult(action="error")
        and is accumulated without aborting remaining upserts.
        """
        quarantined: list[SanitizedFile] = state.get("quarantined_files", [])
        repo_url: str = state.get("repo_url", "")
        branch: str = state.get("branch", "main")
        commit_sha: str = state.get("commit_sha", "")
        indexed_at: str = datetime.now(UTC).isoformat()

        base_meta = {
            "repo_url": repo_url,
            "branch": branch,
            "commit_sha": commit_sha,
            "indexed_at": indexed_at,
        }

        results: list[UpsertResult] = []
        trace: list[str] = []
        trace_events: list[TraceEvent] = []

        for qf in quarantined:
            # Include the file path in the content so each quarantined file produces a
            # distinct SHA-256 content_hash. Using bare QUARANTINE_SENTINEL for all files
            # would collapse them to one ChromaDB document (same hash → Case 3 overwrites).
            per_file_content = f"{QUARANTINE_SENTINEL}\npath:{qf.path}"
            quarantine_chunk = SummarizedChunk(
                content=per_file_content,
                path=qf.path,
                intent_summary="",
                tech_stack=[],
                semantic_type="quarantine",
                author_identity="unknown",
            )
            quarantine_meta = {
                **base_meta,
                "quarantine_reason": qf.quarantine_reason or "unknown",
            }
            result = chroma.upsert_chunk(quarantine_chunk, quarantine_meta)
            results.append(result)
            trace.append(f"chroma_upsert_quarantined: {result.action} {qf.path}")
            trace_events.append(
                TraceEvent(
                    node="chroma_upsert_quarantined",
                    event=TraceEventType.QUARANTINE if result.action != "error" else TraceEventType.ERROR,
                    path=qf.path,
                )
            )

        logger.info(
            "chroma_upsert_quarantined: %d quarantined files → %d upsert results",
            len(quarantined),
            len(results),
        )
        # When quarantined is empty, emit a SKIP event so the field is never absent.
        if not trace_events:
            trace_events.append(TraceEvent(node="chroma_upsert_quarantined", event=TraceEventType.SKIP, count=0))
        return {"upsert_results": results, "trace": trace, "trace_events": trace_events}

    return chroma_upsert_quarantined
