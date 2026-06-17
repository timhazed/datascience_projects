"""Tests for make_sha_store_update_node (src/agents/sha_store_update.py).

Cases:
  - Pipeline error set → SHA NOT written
  - Upsert error in results → SHA NOT written
  - Clean run → SHA written with correct key
  - No commit_sha in state → skipped
  - sha_store is None → skipped (no-op)
  - sha_store.set_last_sha raises → error trace returned, no crash
"""

from unittest.mock import MagicMock

from src.agents.sha_store_update import make_sha_store_update_node
from src.models.chunk import UpsertResult
from src.models.trace_event import TraceEventType


def _upsert_result(action: str) -> UpsertResult:
    return UpsertResult(content_hash="abc", file_path="src/f.py", action=action)


def _state(
    *,
    commit_sha: str = "deadbeef1234",
    repo_url: str = "https://github.com/u/r",
    branch: str = "main",
    error: str | None = None,
    upsert_results: list[UpsertResult] | None = None,
) -> dict:
    return {
        "commit_sha": commit_sha,
        "repo_url": repo_url,
        "branch": branch,
        "error": error,
        "upsert_results": upsert_results or [],
    }


class TestSHAStoreUpdateNoStore:
    def test_no_sha_store_returns_trace_no_error(self) -> None:
        """sha_store=None → node is a no-op, trace says skipped."""
        node = make_sha_store_update_node(sha_store=None)
        result = node(_state())
        assert result["trace"] == ["sha_store_update: skipped (no sha_store)"]

    def test_no_sha_store_no_set_called(self) -> None:
        """No sha_store → set_last_sha is never called."""
        mock_store = MagicMock()
        # Pass sha_store=None explicitly to test the None branch
        node = make_sha_store_update_node(sha_store=None)
        node(_state())
        mock_store.set_last_sha.assert_not_called()


class TestSHAStoreUpdateCleanRun:
    def test_clean_run_writes_sha(self) -> None:
        """No error, no upsert failures → SHA written with correct args."""
        mock_store = MagicMock()
        node = make_sha_store_update_node(sha_store=mock_store)

        result = node(_state(commit_sha="abc123", repo_url="https://github.com/u/r", branch="main"))

        mock_store.set_last_sha.assert_called_once_with(
            "https://github.com/u/r", "main", "abc123"
        )
        assert "sha_store_update: wrote sha=abc123" in result["trace"][0]

    def test_clean_run_with_skipped_upserts_still_writes(self) -> None:
        """skipped action is not an error — SHA should still be written."""
        mock_store = MagicMock()
        node = make_sha_store_update_node(sha_store=mock_store)

        state = _state(
            upsert_results=[_upsert_result("skipped"), _upsert_result("inserted")],
        )
        result = node(state)

        mock_store.set_last_sha.assert_called_once()
        assert "wrote sha=" in result["trace"][0]


class TestSHAStoreUpdatePipelineError:
    def test_pipeline_error_skips_write(self) -> None:
        """state['error'] set → SHA NOT written."""
        mock_store = MagicMock()
        node = make_sha_store_update_node(sha_store=mock_store)

        result = node(_state(error="Git clone failed."))

        mock_store.set_last_sha.assert_not_called()
        assert "skipped (partial failure)" in result["trace"][0]

    def test_upsert_error_skips_write(self) -> None:
        """Any UpsertResult with action='error' → SHA NOT written."""
        mock_store = MagicMock()
        node = make_sha_store_update_node(sha_store=mock_store)

        state = _state(upsert_results=[_upsert_result("inserted"), _upsert_result("error")])
        result = node(state)

        mock_store.set_last_sha.assert_not_called()
        assert "skipped (partial failure)" in result["trace"][0]

    def test_both_error_and_upsert_error_skips_write(self) -> None:
        """Both pipeline error and upsert error → SHA NOT written."""
        mock_store = MagicMock()
        node = make_sha_store_update_node(sha_store=mock_store)

        state = _state(
            error="Something failed",
            upsert_results=[_upsert_result("error")],
        )
        node(state)

        mock_store.set_last_sha.assert_not_called()


class TestSHAStoreUpdateEmptyCommitSha:
    def test_empty_commit_sha_skips_write(self) -> None:
        """commit_sha is empty string → skipped."""
        mock_store = MagicMock()
        node = make_sha_store_update_node(sha_store=mock_store)

        result = node(_state(commit_sha=""))

        mock_store.set_last_sha.assert_not_called()
        assert "no commit_sha" in result["trace"][0]

    def test_missing_commit_sha_skips_write(self) -> None:
        """commit_sha not in state (defaults to '') → skipped."""
        mock_store = MagicMock()
        node = make_sha_store_update_node(sha_store=mock_store)

        state = {
            "repo_url": "https://github.com/u/r",
            "branch": "main",
            "error": None,
            "upsert_results": [],
            # commit_sha intentionally omitted
        }
        node(state)

        mock_store.set_last_sha.assert_not_called()


class TestSHAStoreUpdateWriteError:
    def test_write_exception_returns_error_trace_no_crash(self) -> None:
        """sha_store.set_last_sha raises → error trace, node does not propagate exception."""
        mock_store = MagicMock()
        mock_store.set_last_sha.side_effect = OSError("Disk full")
        node = make_sha_store_update_node(sha_store=mock_store)

        result = node(_state(commit_sha="abc123"))

        assert "write error" in result["trace"][0]
        assert "OSError" in result["trace"][0]


class TestShaStoreUpdateTraceEvents:
    """Assert trace_events are emitted on every code path."""

    def test_no_sha_store_emits_skip(self) -> None:
        """sha_store=None → TraceEventType.SKIP."""
        node = make_sha_store_update_node(sha_store=None)
        result = node(_state(commit_sha="abc123"))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].node == "sha_store_update"
        assert events[0].event == TraceEventType.SKIP

    def test_no_commit_sha_emits_skip(self) -> None:
        """commit_sha empty → TraceEventType.SKIP."""
        store = MagicMock()
        node = make_sha_store_update_node(sha_store=store)
        result = node(_state(commit_sha=""))
        events = result["trace_events"]
        assert events[0].event == TraceEventType.SKIP

    def test_partial_failure_emits_skip(self) -> None:
        """Upsert error present → TraceEventType.SKIP."""
        store = MagicMock()
        node = make_sha_store_update_node(sha_store=store)
        result = node(_state(commit_sha="abc123", upsert_results=[
            UpsertResult(content_hash="x", file_path="f.py", action="error")
        ]))
        events = result["trace_events"]
        assert events[0].event == TraceEventType.SKIP

    def test_successful_write_emits_ok(self) -> None:
        """Clean sync → TraceEventType.OK."""
        store = MagicMock()
        node = make_sha_store_update_node(sha_store=store)
        result = node(_state(commit_sha="abc123"))
        events = result["trace_events"]
        assert events[0].event == TraceEventType.OK

    def test_write_error_emits_error(self) -> None:
        """SHAStore.set_last_sha raising → TraceEventType.ERROR."""
        store = MagicMock()
        store.set_last_sha.side_effect = OSError("disk full")
        node = make_sha_store_update_node(sha_store=store)
        result = node(_state(commit_sha="abc123"))
        events = result["trace_events"]
        assert events[0].event == TraceEventType.ERROR
