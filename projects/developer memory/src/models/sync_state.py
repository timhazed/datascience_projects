"""State TypedDict for the Sync Pipeline (sync_repository MCP tool).

Spec §3 + §12.5 — SyncState flows through: delta_extractor → pii_sanitizer →
multimodal_parser → chunk_dispatcher → summarize_and_upsert × N (Send fan-out) +
chroma_upsert_quarantined.

parsed_chunks is SINGLE-WRITER (only multimodal_parser writes it).
upsert_results accumulates via operator.add (one item per Send target).
"""

import operator
from typing import Annotated, NotRequired, TypedDict

from src.models.chunk import ParsedChunk, UpsertResult
from src.models.sanitized_file import SanitizedFile
from src.models.trace_event import TraceEvent


class SyncState(TypedDict):
    """Pipeline state for sync_repository.

    Fields:
        repo_url: Git repository URL (from MCP client — validated by SyncRequest).
        branch: Branch name being synced.
        commit_sha: Git commit SHA at the time of sync. Stored in ChromaDB metadata for
            idempotency (case 3: file moved — same SHA, different path). Empty string
            on first sync before any commit has been recorded.
        repo_root: Absolute path to the cloned repository on disk. Set by delta_extractor
            so that pii_sanitizer and multimodal_parser can resolve relative file paths
            to absolute paths for reading. Empty string when not yet set.
        changed_files: Relative file paths within the repo that changed (delta_extractor output).
            pii_sanitizer joins repo_root / path to read each file.
        sanitized_files: PII-masked files ready for parsing (pii_sanitizer output).
        parsed_chunks: Text chunks from all sanitized files (multimodal_parser output).
            SINGLE-WRITER — no other node writes to this field.
        quarantined_files: Files held in PII review queue (masking failed or low confidence).
            chroma_upsert_quarantined upserts these with semantic_type="quarantine".
        upsert_results: ChromaDB upsert outcomes accumulated via operator.add.
            Each summarize_and_upsert Send-target appends exactly one UpsertResult.
        error: Non-fatal error message. Pipeline continues to END with this surfaced.
        trace: Append-only execution log. Each node appends one entry.
        trace_events: Structured trace events accumulated via operator.add. Required
            field (not NotRequired) because the Send fan-out in route_after_dispatch
            dispatches sub-state dicts directly — LangGraph raises KeyError when
            attempting to reduce an absent required field. Every node must include
            "trace_events": [] on all code paths, including no-op paths.
        _job_id: Optional background job UUID4. Threaded through state so
            summarize_and_upsert can call the progress_callback with per-chunk progress.
            NotRequired — absent in direct MCP tool invocations that bypass HTTP job routes.
    """

    repo_url: str
    branch: str
    commit_sha: str
    repo_root: str
    changed_files: list[str]
    sanitized_files: list[SanitizedFile]
    parsed_chunks: list[ParsedChunk]
    quarantined_files: list[SanitizedFile]
    upsert_results: Annotated[list[UpsertResult], operator.add]
    error: str | None
    trace: Annotated[list[str], operator.add]
    trace_events: Annotated[list[TraceEvent], operator.add]
    _job_id: NotRequired[str | None]

    # ── New fields (all NotRequired for backward compatibility) ───────────────
    head_sha: NotRequired[str]             # Remote HEAD SHA from git ls-remote (pre-clone)
    cached_sha: NotRequired[str]           # Last-synced SHA from sha_store.json
    already_up_to_date: NotRequired[bool]  # True → skip entire pipeline (SHA match)
    file_shas: NotRequired[dict[str, str]] # {relative_path: sha256_of_raw_bytes}
    cache_filtered_count: NotRequired[int] # Files removed by cache_filter before PII
    pii_files_done: NotRequired[int]       # Live progress counter for PII scan
    pii_files_total: NotRequired[int]      # Total files entering pii_sanitizer
