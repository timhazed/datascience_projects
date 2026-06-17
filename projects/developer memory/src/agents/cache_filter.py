"""cache_filter LangGraph node — Spec §6B (pre-PII cache filter).

Filters out files whose raw content SHA-256 is already indexed in ChromaDB,
so they never enter pii_sanitizer, multimodal_parser, or the LLM stage.

Key design points:
  - Single bulk ChromaDB get() per invocation — 1 HTTP call regardless of repo size.
  - Applies _should_skip() and _MAX_FILE_BYTES guards *before* reading file bytes.
  - All OSError on stat() or read_bytes() → fail open (append to new_files).
  - CACHE_FILTER_ENABLED=False → pass-through (for forced full re-index).
  - Backward compatible: existing docs without source_file_sha have empty cached set
    on first post-deploy sync; all files pass through to PII (same as before).
"""

import hashlib
import logging
from collections.abc import Callable
from pathlib import Path

from src.agents.pii_sanitizer import _MAX_FILE_BYTES, _should_skip
from src.config.settings import Settings
from src.db.chroma_client import ChromaLibrarianClient
from src.models.trace_event import TraceEvent, TraceEventType

logger = logging.getLogger(__name__)


def make_cache_filter_node(
    chroma: ChromaLibrarianClient,
    settings: Settings,
) -> Callable[[dict], dict]:
    """Return a cache_filter node that removes already-indexed files pre-PII.

    Args:
        chroma: ChromaLibrarianClient singleton (single bulk get call per node run).
        settings: Settings instance for cache_filter_enabled flag.

    Returns:
        LangGraph node function that reads changed_files from state and writes
        back a filtered changed_files, file_shas dict, and cache_filtered_count.
    """

    def cache_filter(state: dict) -> dict:
        """Filter changed_files against ChromaDB cached source_file_sha values.

        On disabled: passes all files through unchanged.
        On enabled: fetches all source_file_sha values for the repo in one bulk call,
            then for each changed file:
              - skip (_should_skip) → drop silently (same as pii_sanitizer would)
              - oversized (> _MAX_FILE_BYTES) → pass through without reading bytes
              - stat() OSError → pass through (fail open)
              - sha256 in cached set → filtered out
              - sha256 not in cached set → pass through, record sha in file_shas

        Returns dict with:
            changed_files: filtered list (replaces state value — no operator.add)
            file_shas: {relative_path: sha256_hex} for new/changed files
            cache_filtered_count: number of files removed because already indexed
            trace: list of trace entries to append
        """
        if not settings.cache_filter_enabled:
            return {
                "cache_filtered_count": 0,
                "trace": ["cache_filter: disabled"],
                "trace_events": [
                    TraceEvent(node="cache_filter", event=TraceEventType.SKIP, detail="disabled")
                ],
            }

        changed_files: list[str] = state.get("changed_files", [])
        repo_root: str = state.get("repo_root", "")
        repo_url: str = state.get("repo_url", "")

        # Single bulk fetch to get all source_file_sha values for this repo.
        # 1 HTTP call regardless of how many files are in changed_files.
        try:
            raw = chroma._collection.get(
                where={"repo_url": {"$eq": repo_url}},
                include=["metadatas"],
            )
            cached_sha_set: set[str] = {
                m["source_file_sha"]
                for m in (raw.get("metadatas") or [])
                if m and m.get("source_file_sha")
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "cache_filter: bulk fetch failed [%s] — passing all %d files through",
                type(exc).__name__,
                len(changed_files),
            )
            return {
                "changed_files": changed_files,
                "file_shas": {},
                "cache_filtered_count": 0,
                "trace": [f"cache_filter: fetch error, all {len(changed_files)} files passed through"],
                "trace_events": [
                    TraceEvent(
                        node="cache_filter",
                        event=TraceEventType.ERROR,
                        detail=type(exc).__name__,
                    )
                ],
            }

        new_files: list[str] = []
        file_shas: dict[str, str] = {}
        filtered_count = 0

        for file_path in changed_files:
            # Skip excluded types (dotfiles, lock files, images, etc.) — same guard as pii_sanitizer.
            # These files never enter PII scanning anyway, so drop them here silently.
            if _should_skip(file_path):
                continue

            abs_path = Path(repo_root) / file_path if repo_root else Path(file_path)

            # Skip oversized files without reading — pass through to pii_sanitizer which
            # also handles them (will skip them with its own size guard).
            try:
                if abs_path.stat().st_size > _MAX_FILE_BYTES:
                    new_files.append(file_path)
                    continue
            except OSError:
                # stat() failed — fail open, pass file through to pii_sanitizer
                new_files.append(file_path)
                continue

            # Compute raw-file SHA and check against the cached set.
            try:
                sha = hashlib.sha256(abs_path.read_bytes()).hexdigest()
            except OSError:
                # read_bytes() failed — fail open, pass file through
                new_files.append(file_path)
                continue

            if sha in cached_sha_set:
                filtered_count += 1
            else:
                new_files.append(file_path)
                file_shas[file_path] = sha

        logger.info(
            "cache_filter: %d/%d files already indexed, %d new/changed proceed",
            filtered_count,
            len(changed_files),
            len(new_files),
        )
        return {
            "changed_files": new_files,
            "file_shas": file_shas,
            "cache_filtered_count": filtered_count,
            "trace": [f"cache_filter: {filtered_count} cached, {len(new_files)} new"],
            "trace_events": [
                TraceEvent(
                    node="cache_filter",
                    event=TraceEventType.CACHED,
                    count=filtered_count,
                )
            ],
        }

    return cache_filter
