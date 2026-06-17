"""Tests for ui.state — session state initialisation and _sync_is_active predicate.

Streamlit session state is mocked via patch.object(st, "session_state", plain_dict)
so that tests run without a live Streamlit server.
"""

from unittest.mock import patch

import streamlit as st

from ui.state import _STATE_DEFAULTS, _sync_is_active, init_session_state


class TestInitSessionState:
    """Verify init_session_state() populates defaults correctly."""

    def test_sets_all_expected_keys_from_empty(self) -> None:
        """All 11 state keys are present after init on an empty state dict."""
        mock_state: dict = {}
        with patch.object(st, "session_state", mock_state):
            init_session_state()
        expected_keys = {key for key, _ in _STATE_DEFAULTS}
        assert expected_keys.issubset(mock_state.keys()), (
            f"Missing keys: {expected_keys - mock_state.keys()}"
        )

    def test_sync_job_id_defaults_to_none(self) -> None:
        """_sync_job_id initialises to None."""
        mock_state: dict = {}
        with patch.object(st, "session_state", mock_state):
            init_session_state()
        assert mock_state["_sync_job_id"] is None

    def test_sync_chunks_done_defaults_to_zero(self) -> None:
        """_sync_chunks_done initialises to 0."""
        mock_state: dict = {}
        with patch.object(st, "session_state", mock_state):
            init_session_state()
        assert mock_state["_sync_chunks_done"] == 0

    def test_sync_stage_defaults_to_queued(self) -> None:
        """_sync_stage initialises to 'queued'."""
        mock_state: dict = {}
        with patch.object(st, "session_state", mock_state):
            init_session_state()
        assert mock_state["_sync_stage"] == "queued"

    def test_idempotent_does_not_overwrite_existing_values(self) -> None:
        """init_session_state() leaves already-set keys unchanged."""
        existing_job_id = "abc-123"
        mock_state: dict = {"_sync_job_id": existing_job_id}
        with patch.object(st, "session_state", mock_state):
            init_session_state()
        assert mock_state["_sync_job_id"] == existing_job_id

    def test_idempotent_fills_missing_keys_alongside_existing(self) -> None:
        """Only absent keys are filled; pre-existing keys survive."""
        mock_state: dict = {"_sync_job_id": "my-job", "_sync_chunks_done": 42}
        with patch.object(st, "session_state", mock_state):
            init_session_state()
        # Pre-existing values preserved
        assert mock_state["_sync_job_id"] == "my-job"
        assert mock_state["_sync_chunks_done"] == 42
        # Other keys filled in
        assert "_sync_status" in mock_state

    def test_all_eleven_keys_present(self) -> None:
        """Exactly 11 keys are defined in _STATE_DEFAULTS."""
        assert len(_STATE_DEFAULTS) == 11

    def test_init_twice_is_idempotent(self) -> None:
        """Calling init_session_state() twice produces identical state."""
        mock_state: dict = {}
        with patch.object(st, "session_state", mock_state):
            init_session_state()
            state_after_first = dict(mock_state)
            init_session_state()
        assert mock_state == state_after_first


class TestSyncIsActive:
    """Verify _sync_is_active() reflects the correct status strings."""

    def test_returns_true_when_status_queued(self) -> None:
        """Active when status is 'queued'."""
        mock_state: dict = {"_sync_status": "queued"}
        with patch.object(st, "session_state", mock_state):
            assert _sync_is_active() is True

    def test_returns_true_when_status_running(self) -> None:
        """Active when status is 'running'."""
        mock_state: dict = {"_sync_status": "running"}
        with patch.object(st, "session_state", mock_state):
            assert _sync_is_active() is True

    def test_returns_false_when_status_done(self) -> None:
        """Not active when status is 'done'."""
        mock_state: dict = {"_sync_status": "done"}
        with patch.object(st, "session_state", mock_state):
            assert _sync_is_active() is False

    def test_returns_false_when_status_error(self) -> None:
        """Not active when status is 'error'."""
        mock_state: dict = {"_sync_status": "error"}
        with patch.object(st, "session_state", mock_state):
            assert _sync_is_active() is False

    def test_returns_false_when_status_abandoned(self) -> None:
        """Not active when status is 'abandoned'."""
        mock_state: dict = {"_sync_status": "abandoned"}
        with patch.object(st, "session_state", mock_state):
            assert _sync_is_active() is False

    def test_returns_false_when_status_none(self) -> None:
        """Not active when status is None (no job started)."""
        mock_state: dict = {"_sync_status": None}
        with patch.object(st, "session_state", mock_state):
            assert _sync_is_active() is False
