"""Tests for ui.pages.sync_repository — pure display helpers.

Streamlit widget calls are mocked via patch.object. No live Streamlit server required.
The page() renderer smoke test verifies callability with all st calls mocked.
"""

from unittest.mock import MagicMock, patch

from ui.pages.sync_repository import (
    _build_stage_label,
    _format_elapsed,
    _metrics_from_trace_events,
)

# ── _format_elapsed ───────────────────────────────────────────────────────────


class TestFormatElapsed:
    """Verify _format_elapsed formats durations correctly."""

    def test_seconds_only(self) -> None:
        """38 seconds → '38s'."""
        assert _format_elapsed(38.0) == "38s"

    def test_minutes_and_seconds(self) -> None:
        """4 minutes 2 seconds → '4m 2s'."""
        assert _format_elapsed(242.0) == "4m 2s"

    def test_hours_minutes_seconds(self) -> None:
        """1 hour 23 minutes 45 seconds → '1h 23m 45s'."""
        assert _format_elapsed(3600 + 23 * 60 + 45) == "1h 23m 45s"

    def test_exactly_one_minute(self) -> None:
        """60 seconds → '1m 0s' (minutes branch)."""
        assert _format_elapsed(60.0) == "1m 0s"

    def test_zero_seconds(self) -> None:
        """0 seconds → '0s'."""
        assert _format_elapsed(0.0) == "0s"

    def test_truncates_fractional_seconds(self) -> None:
        """Fractional seconds are truncated, not rounded."""
        assert _format_elapsed(61.9) == "1m 1s"


# ── _build_stage_label ────────────────────────────────────────────────────────


class TestBuildStageLabel:
    """Verify _build_stage_label returns correct label and fraction per stage."""

    def test_pii_scanning_with_progress(self) -> None:
        """pii_scanning with pii_total > 0 → dynamic label and interpolated fraction."""
        label, frac = _build_stage_label("pii_scanning", 0, 0, pii_done=3, pii_total=10)
        assert "3 / 10" in label
        assert 0.15 <= frac <= 0.25

    def test_pii_scanning_at_full_progress(self) -> None:
        """pii_scanning at 100% → fraction capped at 0.25."""
        _, frac = _build_stage_label("pii_scanning", 0, 0, pii_done=10, pii_total=10)
        assert frac == 0.25

    def test_pii_scanning_without_total(self) -> None:
        """pii_scanning with pii_total == 0 → falls back to static _STAGE_LABELS entry."""
        label, frac = _build_stage_label("pii_scanning", 0, 0, pii_done=0, pii_total=0)
        assert "PII" in label or "Scanning" in label
        assert frac == 0.15  # static value from _STAGE_LABELS

    def test_summarizing_with_progress(self) -> None:
        """summarizing with chunks_total > 0 → dynamic label and interpolated fraction."""
        label, frac = _build_stage_label("summarizing", chunks_done=5, chunks_total=10)
        assert "5 / 10" in label
        assert 0.30 < frac < 1.0

    def test_summarizing_at_full_progress_capped(self) -> None:
        """summarizing at 100% → fraction capped at 0.99 (1.0 reserved for 'done')."""
        _, frac = _build_stage_label("summarizing", chunks_done=10, chunks_total=10)
        assert frac == 0.99

    def test_summarizing_without_total(self) -> None:
        """summarizing with chunks_total == 0 → falls back to static _STAGE_LABELS entry."""
        label, frac = _build_stage_label("summarizing", chunks_done=0, chunks_total=0)
        assert "Summarizing" in label
        assert frac == 0.30  # static value from _STAGE_LABELS

    def test_done_stage(self) -> None:
        """'done' stage → fraction 1.0."""
        _, frac = _build_stage_label("done", 0, 0)
        assert frac == 1.0

    def test_unknown_stage_fallback(self) -> None:
        """Unknown stage → fallback label 'Processing…' and fraction 0.1."""
        label, frac = _build_stage_label("unknown_future_stage", 0, 0)
        assert label == "Processing…"
        assert frac == 0.1

    def test_cloning_stage(self) -> None:
        """'cloning' stage → static label from _STAGE_LABELS."""
        label, _ = _build_stage_label("cloning", 0, 0)
        assert "Cloning" in label


# ── _metrics_from_trace_events ────────────────────────────────────────────────


class TestMetricsFromTraceEvents:
    """Verify _metrics_from_trace_events accumulates event counts correctly."""

    def test_empty_state_returns_zero_counts(self) -> None:
        """Empty state dict → all counts are 0."""
        counts = _metrics_from_trace_events({})
        assert counts["sanitized"] == 0
        assert counts["inserted"] == 0
        assert counts["errors"] == 0

    def test_empty_trace_events_list(self) -> None:
        """state with empty trace_events → all event-driven counts are 0."""
        counts = _metrics_from_trace_events({"trace_events": []})
        assert counts["sanitized"] == 0
        assert counts["chunks"] == 0

    def test_none_trace_events(self) -> None:
        """state with trace_events=None → no error, counts are 0."""
        counts = _metrics_from_trace_events({"trace_events": None})
        assert counts["sanitized"] == 0

    def test_pii_sanitizer_ok_increments_sanitized(self) -> None:
        """Each pii_sanitizer ok event increments sanitized by 1."""
        state = {
            "trace_events": [
                {"node": "pii_sanitizer", "event": "ok"},
                {"node": "pii_sanitizer", "event": "ok"},
            ]
        }
        counts = _metrics_from_trace_events(state)
        assert counts["sanitized"] == 2

    def test_summarize_and_upsert_ok_increments_inserted_and_chunks(self) -> None:
        """summarize_and_upsert ok events increment both inserted and chunks."""
        state = {
            "trace_events": [
                {"node": "summarize_and_upsert", "event": "ok"},
                {"node": "summarize_and_upsert", "event": "ok"},
            ]
        }
        counts = _metrics_from_trace_events(state)
        assert counts["inserted"] == 2
        assert counts["chunks"] == 2

    def test_summarize_and_upsert_cached_increments_skipped(self) -> None:
        """summarize_and_upsert cached events increment skipped."""
        state = {
            "trace_events": [
                {"node": "summarize_and_upsert", "event": "cached"},
            ]
        }
        counts = _metrics_from_trace_events(state)
        assert counts["skipped"] == 1
        assert counts["inserted"] == 0

    def test_summarize_and_upsert_error_increments_errors(self) -> None:
        """summarize_and_upsert error events increment errors."""
        state = {
            "trace_events": [
                {"node": "summarize_and_upsert", "event": "error"},
            ]
        }
        counts = _metrics_from_trace_events(state)
        assert counts["errors"] == 1

    def test_changed_files_count_from_state(self) -> None:
        """changed_files count derived from the state list length, not events."""
        state = {
            "changed_files": ["a.py", "b.py", "c.py"],
            "trace_events": [],
        }
        counts = _metrics_from_trace_events(state)
        assert counts["changed_files"] == 3

    def test_quarantined_count_from_state(self) -> None:
        """quarantined count derived from quarantined_files list length."""
        state = {
            "quarantined_files": [{"path": "secret.py"}],
            "trace_events": [],
        }
        counts = _metrics_from_trace_events(state)
        assert counts["quarantined"] == 1

    def test_mixed_events_accumulated_correctly(self) -> None:
        """Multiple event types in one trace_events list all accumulate independently."""
        state = {
            "trace_events": [
                {"node": "pii_sanitizer", "event": "ok"},
                {"node": "summarize_and_upsert", "event": "ok"},
                {"node": "summarize_and_upsert", "event": "cached"},
                {"node": "summarize_and_upsert", "event": "error"},
            ]
        }
        counts = _metrics_from_trace_events(state)
        assert counts["sanitized"] == 1
        assert counts["inserted"] == 1
        assert counts["skipped"] == 1
        assert counts["errors"] == 1


# ── page() smoke test ─────────────────────────────────────────────────────────


class TestPageCallable:
    """Verify page() is callable with all Streamlit calls mocked."""

    def test_page_runs_without_error_when_no_job(self) -> None:
        """page() renders without exception when session state has no active job."""
        mock_state = {
            "_sync_job_id": None,
            "_sync_status": None,
            "_sync_stage": "queued",
            "_sync_result": None,
            "_sync_error": None,
            "_sync_chunks_done": 0,
            "_sync_chunks_total": 0,
            "_sync_pii_done": 0,
            "_sync_pii_total": 0,
            "_sync_wall_start": None,
            "_sync_elapsed_secs": None,
        }
        # Mock all Streamlit widget calls that would fail outside a running server.
        st_mock = MagicMock()
        st_mock.session_state = mock_state
        with (
            patch("ui.pages.sync_repository.st", st_mock),
            patch("ui.pages.sync_repository._post_sync"),
            patch("ui.pages.sync_repository._poll_sync_status"),
        ):
            from ui.pages.sync_repository import page
            # form_submit_button returns False → no submission path triggered
            st_mock.form_submit_button.return_value = False
            st_mock.form.__enter__ = MagicMock(return_value=st_mock)
            st_mock.form.__exit__ = MagicMock(return_value=False)
            st_mock.columns.return_value = [MagicMock(), MagicMock()]
            page()  # must not raise
