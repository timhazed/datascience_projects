"""Tests for src/mcp_server/sync_jobs.py.

Covers register_job, get_job, run_sync_job, _sync_progress_callback,
and _pii_progress_callback. _sync_graph is mocked via patch.object on
container_module so the lazy import inside run_sync_job is intercepted
correctly — patching server_module._sync_graph would NOT work because
run_sync_job does a fresh lazy import from container on each call.

Cases:
  - register_job creates SyncJob with status='queued' and stores it
  - get_job returns None for unknown job_id
  - get_job returns the job for a known job_id
  - run_sync_job sets status='running' on entry
  - run_sync_job sets status='done' + result on success
  - run_sync_job sets status='error' + error on exception
  - _sync_progress_callback updates chunks_done, chunks_total, stage='summarizing'
  - _pii_progress_callback updates pii_files_done, pii_files_total, stage='pii_scanning'
  - both callbacks are silent no-ops for unknown job_id
"""

from unittest.mock import MagicMock, patch

import src.mcp_server.container as container_module
from src.mcp_server.sync_jobs import (
    _pii_progress_callback,
    _sync_progress_callback,
    get_job,
    register_job,
    run_sync_job,
)


class TestRegisterAndGetJob:
    def test_register_job_creates_queued_job(self) -> None:
        """register_job returns a SyncJob with status='queued'."""
        job = register_job("https://github.com/u/r", "main")
        assert job.job_id is not None
        assert job.repo_url == "https://github.com/u/r"
        assert job.branch == "main"
        assert job.status == "queued"
        assert job.stage == "queued"

    def test_register_job_stores_in_registry(self) -> None:
        """register_job stores the job so get_job can retrieve it."""
        job = register_job("https://github.com/u/r2", "dev")
        retrieved = get_job(job.job_id)
        assert retrieved is job

    def test_get_job_returns_none_for_unknown_id(self) -> None:
        """get_job returns None when job_id is not in the registry."""
        result = get_job("00000000-0000-0000-0000-000000000000")
        assert result is None

    def test_get_job_returns_job_for_known_id(self) -> None:
        """get_job returns the SyncJob for a registered job_id."""
        job = register_job("https://github.com/u/r3", "feature")
        assert get_job(job.job_id) is job


class TestRunSyncJob:
    def _make_job(self, repo_url: str = "https://github.com/u/r", branch: str = "main"):
        """Helper: register a fresh job and return it."""
        return register_job(repo_url, branch)

    def test_run_sync_job_success(self) -> None:
        """run_sync_job with mocked graph → job status 'done' and result set."""
        job = self._make_job()
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {"upsert_results": [], "trace": ["ok"]}

        # Patch at container_module level — run_sync_job does a lazy import of
        # _sync_graph from container inside its body, so we must intercept it there.
        with patch.object(container_module, "_sync_graph", mock_graph):
            run_sync_job(job)

        assert job.status == "done"
        assert job.stage == "done"
        assert job.result == {"upsert_results": [], "trace": ["ok"]}
        assert job.error is None
        assert job.finished_at is not None

    def test_run_sync_job_failure(self) -> None:
        """run_sync_job with graph raising → job status 'error', error message set."""
        job = self._make_job()
        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = RuntimeError("Clone timeout")

        with patch.object(container_module, "_sync_graph", mock_graph):
            run_sync_job(job)

        assert job.status == "error"
        assert job.stage == "error"
        assert "Clone timeout" in job.error
        assert job.result is None
        assert job.finished_at is not None

    def test_run_sync_job_sets_running_on_entry(self) -> None:
        """run_sync_job sets status='running' before invoking the graph."""
        job = self._make_job()
        observed_status_on_entry = []

        def fake_invoke(state, config=None):
            observed_status_on_entry.append(job.status)
            return {}

        mock_graph = MagicMock()
        mock_graph.invoke.side_effect = fake_invoke

        with patch.object(container_module, "_sync_graph", mock_graph):
            run_sync_job(job)

        assert observed_status_on_entry == ["running"]


class TestProgressCallbacks:
    def test_sync_progress_callback_updates_job(self) -> None:
        """_sync_progress_callback updates chunks_done, chunks_total, stage."""
        job = register_job("https://github.com/u/r", "main")
        _sync_progress_callback(job.job_id, 5, 20)

        assert job.chunks_done == 5
        assert job.chunks_total == 20
        assert job.stage == "summarizing"

    def test_pii_progress_callback_updates_job(self) -> None:
        """_pii_progress_callback updates pii_files_done, pii_files_total, stage."""
        job = register_job("https://github.com/u/r", "main")
        _pii_progress_callback(job.job_id, 3, 10)

        assert job.pii_files_done == 3
        assert job.pii_files_total == 10
        assert job.stage == "pii_scanning"

    def test_sync_callback_unknown_job_id_does_not_crash(self) -> None:
        """_sync_progress_callback with unknown job_id silently no-ops."""
        unknown = "00000000-0000-0000-0000-000000000099"
        _sync_progress_callback(unknown, 1, 5)  # must not raise

    def test_pii_callback_unknown_job_id_does_not_crash(self) -> None:
        """_pii_progress_callback with unknown job_id silently no-ops."""
        unknown = "00000000-0000-0000-0000-000000000088"
        _pii_progress_callback(unknown, 1, 5)  # must not raise

    def test_sync_callback_does_not_advance_stage_when_done(self) -> None:
        """_sync_progress_callback does not overwrite stage='done' with 'summarizing'."""
        job = register_job("https://github.com/u/r", "main")
        job.stage = "done"
        _sync_progress_callback(job.job_id, 10, 10)
        # Stage must remain 'done'
        assert job.stage == "done"

    def test_pii_callback_does_not_advance_stage_when_error(self) -> None:
        """_pii_progress_callback does not overwrite stage='error' with 'pii_scanning'."""
        job = register_job("https://github.com/u/r", "main")
        job.stage = "error"
        _pii_progress_callback(job.job_id, 1, 5)
        assert job.stage == "error"
