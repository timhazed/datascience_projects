"""Tests for src/db/initializer.py — _ensure_db_initialized().

Uses mocks for all DB classes and DataLoader so no real files are
touched.  The module-level singletons are patched via their names in the
initializer module's own namespace.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.db.initializer import _ensure_db_initialized

_MODULE = "src.db.initializer"


@pytest.fixture
def mock_settings(tmp_path):
    """Minimal Settings stub with sqlite_path and faiss_path."""
    s = MagicMock()
    s.db.sqlite_path = tmp_path / "test.db"
    s.db.faiss_path = tmp_path / "faiss"
    return s


@pytest.fixture
def mock_embeddings():
    """Stub embeddings model — never called directly in initializer."""
    return MagicMock()


@pytest.fixture
def patched_dbs():
    """Patch module-level DB singletons and DataLoader in initializer."""
    with (
        patch(f"{_MODULE}._appointment_db") as mock_appt,
        patch(f"{_MODULE}._patient_db") as mock_patient,
        patch(f"{_MODULE}._metrics_db") as mock_metrics,
        patch(f"{_MODULE}._data_loader") as mock_loader,
    ):
        yield {
            "appointment_db": mock_appt,
            "patient_db": mock_patient,
            "metrics_db": mock_metrics,
            "data_loader": mock_loader,
        }


# ---------------------------------------------------------------------------
# _ensure_db_initialized
# ---------------------------------------------------------------------------


class TestEnsureDbInitialized:
    def test_all_inits_called_once(self, mock_settings, mock_embeddings, patched_dbs) -> None:
        """Each DB init method is called exactly once with the correct db_path."""
        _ensure_db_initialized(mock_settings, mock_embeddings)

        db_path = str(mock_settings.db.sqlite_path)
        patched_dbs["appointment_db"].init_db.assert_called_once_with(db_path)
        patched_dbs["patient_db"].init_db.assert_called_once_with(db_path)
        patched_dbs["metrics_db"].init_db.assert_called_once_with(db_path)

    def test_data_loader_called_with_correct_args(
        self, mock_settings, mock_embeddings, patched_dbs
    ) -> None:
        """DataLoader.load_all receives db_path, faiss_path, dataset_dir, and embeddings."""
        _ensure_db_initialized(mock_settings, mock_embeddings)

        db_path = str(mock_settings.db.sqlite_path)
        faiss_path = str(mock_settings.db.faiss_path)
        patched_dbs["data_loader"].load_all.assert_called_once_with(
            db_path, faiss_path, "dataset", mock_embeddings
        )

    def test_initialization_order(self, mock_settings, mock_embeddings, patched_dbs) -> None:
        """appointment_db.init_db must be called before patient_db.init_db.

        The schema dependency (bookings references doctors) requires this order.
        Verified by checking call ordering on a shared call tracker.
        """
        call_order: list[str] = []
        patched_dbs["appointment_db"].init_db.side_effect = lambda _: call_order.append("appt")
        patched_dbs["patient_db"].init_db.side_effect = lambda _: call_order.append("patient")
        patched_dbs["metrics_db"].init_db.side_effect = lambda _: call_order.append("metrics")
        patched_dbs["data_loader"].load_all.side_effect = lambda *_: call_order.append("loader")

        _ensure_db_initialized(mock_settings, mock_embeddings)

        assert call_order == ["appt", "patient", "metrics", "loader"]

    def test_returns_none(self, mock_settings, mock_embeddings, patched_dbs) -> None:
        """_ensure_db_initialized has no return value."""
        result = _ensure_db_initialized(mock_settings, mock_embeddings)
        assert result is None
