"""Tests for src/db/metrics_db.py.

Covers MetricsDB.init_db(), log_entry(), get_means(), and get_operation_history().
Uses a real in-memory SQLite file via the tmp_path fixture — no mocks for DB layer.
"""

from __future__ import annotations

import pytest

from src.db.metrics_db import MetricsDB

_SESSION_A = "session-aaa"
_SESSION_B = "session-bbb"


@pytest.fixture
def db_path(tmp_path) -> str:
    """Initialised MetricsDB path, isolated per test."""
    path = str(tmp_path / "metrics.db")
    MetricsDB().init_db(path)
    return path


# ---------------------------------------------------------------------------
# init_db
# ---------------------------------------------------------------------------


class TestInitDb:
    def test_creates_session_log_table(self, tmp_path) -> None:
        """init_db creates the session_log table without raising."""
        path = str(tmp_path / "m.db")
        MetricsDB().init_db(path)
        # Verify by logging an entry without error
        MetricsDB().log_entry(path, "s1", "2026-01-01 00:00:00", "Op", True, 10.0)

    def test_idempotent_on_second_call(self, db_path) -> None:
        """Calling init_db twice does not raise or corrupt the table."""
        MetricsDB().init_db(db_path)
        MetricsDB().log_entry(db_path, "s1", "2026-01-01 00:00:00", "Op", True, 10.0)
        rows = MetricsDB().get_operation_history(db_path, "Op")
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# log_entry
# ---------------------------------------------------------------------------


class TestLogEntry:
    def test_log_entry_persists(self, db_path) -> None:
        MetricsDB().log_entry(
            db_path, _SESSION_A, "2026-04-13 10:00:00", "Patient Lookup", True, 0.4
        )
        rows = MetricsDB().get_operation_history(db_path, "Patient Lookup")
        assert len(rows) == 1
        assert rows[0]["latency_ms"] == pytest.approx(0.4)
        assert rows[0]["success"] == 1

    def test_multiple_entries_same_operation(self, db_path) -> None:
        db = MetricsDB()
        for i in range(3):
            db.log_entry(
                db_path, _SESSION_A, f"2026-04-13 10:0{i}:00",
                "Disease Search", True, float(i + 1) * 1000,
            )
        rows = db.get_operation_history(db_path, "Disease Search")
        assert len(rows) == 3

    def test_failure_entry_recorded(self, db_path) -> None:
        MetricsDB().log_entry(
            db_path, _SESSION_A, "2026-04-13 10:00:00", "Update Record", False, 0.3, "not found"
        )
        rows = MetricsDB().get_operation_history(db_path, "Update Record")
        assert rows[0]["success"] == 0
        assert rows[0]["error"] == "not found"

    def test_different_operations_stored_separately(self, db_path) -> None:
        db = MetricsDB()
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:00:00", "Op A", True, 100.0)
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:01:00", "Op B", True, 200.0)
        assert len(db.get_operation_history(db_path, "Op A")) == 1
        assert len(db.get_operation_history(db_path, "Op B")) == 1


# ---------------------------------------------------------------------------
# get_means
# ---------------------------------------------------------------------------


class TestGetMeans:
    def test_mean_calculated_correctly(self, db_path) -> None:
        db = MetricsDB()
        for ms in (1000.0, 2000.0, 3000.0):
            db.log_entry(db_path, _SESSION_A, "2026-04-13 10:00:00", "Disease Search", True, ms)
        means = db.get_means(db_path)
        assert means["Disease Search"] == pytest.approx(2000.0, abs=0.1)

    def test_sub_millisecond_entries_excluded(self, db_path) -> None:
        """Entries with latency_ms < 1.0 (direct DB ops) are excluded from means."""
        db = MetricsDB()
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:00:00", "Patient Lookup", True, 0.3)
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:01:00", "Patient Lookup", True, 0.5)
        means = db.get_means(db_path)
        assert "Patient Lookup" not in means

    def test_means_across_multiple_sessions(self, db_path) -> None:
        """Means aggregate across different session_ids."""
        db = MetricsDB()
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:00:00", "Book Appointment", True, 10.0)
        db.log_entry(db_path, _SESSION_B, "2026-04-13 11:00:00", "Book Appointment", True, 20.0)
        means = db.get_means(db_path)
        assert means["Book Appointment"] == pytest.approx(15.0, abs=0.1)

    def test_returns_empty_when_no_qualifying_rows(self, db_path) -> None:
        """No rows ≥ 1 ms → empty dict (no KeyError or None)."""
        db = MetricsDB()
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:00:00", "Patient Lookup", True, 0.1)
        assert db.get_means(db_path) == {}

    def test_returns_empty_for_missing_db(self, tmp_path) -> None:
        """Non-existent DB path returns {} without raising."""
        means = MetricsDB().get_means(str(tmp_path / "nonexistent.db"))
        assert means == {}

    def test_multiple_operations_separate_means(self, db_path) -> None:
        db = MetricsDB()
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:00:00", "Chat", True, 3000.0)
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:01:00", "Chat", True, 5000.0)
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:02:00", "Encounter Note", True, 2.0)
        means = db.get_means(db_path)
        assert means["Chat"] == pytest.approx(4000.0, abs=0.1)
        assert means["Encounter Note"] == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# get_operation_history
# ---------------------------------------------------------------------------


class TestGetOperationHistory:
    def test_returns_most_recent_first(self, db_path) -> None:
        db = MetricsDB()
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:00:00", "Chat", True, 1000.0)
        db.log_entry(db_path, _SESSION_A, "2026-04-13 10:01:00", "Chat", True, 2000.0)
        rows = db.get_operation_history(db_path, "Chat")
        # Most recent (id=2) should be first
        assert rows[0]["latency_ms"] == pytest.approx(2000.0)
        assert rows[1]["latency_ms"] == pytest.approx(1000.0)

    def test_limit_respected(self, db_path) -> None:
        db = MetricsDB()
        for i in range(10):
            db.log_entry(db_path, _SESSION_A, f"2026-04-13 10:{i:02d}:00", "Chat", True, float(i))
        rows = db.get_operation_history(db_path, "Chat", limit=5)
        assert len(rows) == 5

    def test_returns_empty_for_unknown_operation(self, db_path) -> None:
        rows = MetricsDB().get_operation_history(db_path, "Unknown Op")
        assert rows == []

    def test_returns_empty_for_missing_db(self, tmp_path) -> None:
        rows = MetricsDB().get_operation_history(str(tmp_path / "nope.db"), "Op")
        assert rows == []
