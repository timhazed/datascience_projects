"""sha_store_update LangGraph node — Spec §6A (SHA write guard).

Writes the synced commit_sha to SHAStore only when the pipeline completed cleanly.
Skips the write if any of these are true:
  - state["error"] is set (pipeline-level error)
  - any UpsertResult has action == "error" (partial chunk failure)
  - sha_store is None (not injected — disabled path)
  - commit_sha is empty (delta_extractor did not set it)

This guard ensures the sha_store only records SHAs for fully successful syncs.
On the next run, delta_extractor will compare remote HEAD against this stored SHA;
a corrupted or partial sync will not prevent a re-sync.
"""

import logging
from collections.abc import Callable

from src.models.sync_state import SyncState
from src.models.trace_event import TraceEvent, TraceEventType

logger = logging.getLogger(__name__)


def make_sha_store_update_node(sha_store=None) -> Callable[[SyncState], dict]:
    """Return a sha_store_update node that writes commit_sha on clean sync.

    Args:
        sha_store: SHAStore instance or None. If None, the node is a no-op (pass-through).

    Returns:
        LangGraph node function that reads SyncState and conditionally writes to SHAStore.
    """

    def sha_store_update(state: SyncState) -> dict:
        """Write commit_sha to SHAStore only on a fully clean sync run.

        Skips write if:
          - state["error"] is set (pipeline-level error)
          - any UpsertResult has action == "error" (partial chunk failure)
          - sha_store is None (no store injected)
          - commit_sha is empty or missing

        Returns trace entry describing the outcome.
        """
        _skip_event = TraceEvent(node="sha_store_update", event=TraceEventType.SKIP)
        _ok_event = TraceEvent(node="sha_store_update", event=TraceEventType.OK)

        if sha_store is None:
            return {"trace": ["sha_store_update: skipped (no sha_store)"], "trace_events": [_skip_event]}

        commit_sha = state.get("commit_sha", "")
        if not commit_sha:
            return {"trace": ["sha_store_update: skipped (no commit_sha)"], "trace_events": [_skip_event]}

        has_pipeline_error = state.get("error") is not None
        has_upsert_error = any(
            r.action == "error"  # UpsertResult is a Pydantic model — use .action attribute
            for r in state.get("upsert_results", [])
        )

        if has_pipeline_error or has_upsert_error:
            logger.warning(
                "sha_store_update: skipping SHA write — pipeline_error=%s upsert_errors=%s",
                has_pipeline_error,
                has_upsert_error,
            )
            return {"trace": ["sha_store_update: skipped (partial failure)"], "trace_events": [_skip_event]}

        repo_url = state.get("repo_url", "")
        branch = state.get("branch", "main")

        try:
            sha_store.set_last_sha(repo_url, branch, commit_sha)
            logger.info(
                "sha_store_update: wrote sha=%s for %s@%s", commit_sha[:8], repo_url, branch
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "sha_store_update: write failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "trace": [f"sha_store_update: write error [{type(exc).__name__}]"],
                "trace_events": [TraceEvent(node="sha_store_update", event=TraceEventType.ERROR, detail=type(exc).__name__)],
            }

        return {"trace": [f"sha_store_update: wrote sha={commit_sha[:8]}"], "trace_events": [_ok_event]}

    return sha_store_update
