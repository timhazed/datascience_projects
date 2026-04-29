"""Tests for history_tool — real in-memory SQLite, mocked FAISS, no live API calls."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.embeddings import Embeddings

from src.db.appointment_db import AppointmentDB
from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.models.patient import PatientRecord
from src.tools.history_tool import make_history_tool


@pytest.fixture
def seeded_db(tmp_path) -> tuple[str, PatientDB]:
    """Seeded SQLite database with one known patient."""
    db_path = str(tmp_path / "history_tool.db")
    patient_db = PatientDB()
    patient_db.init_db(db_path)
    AppointmentDB().init_db(db_path)
    patient_db.create_patient(
        db_path,
        PatientRecord(
            patient_id="P-tooltest",
            phone="+1-555-000-0099",
            name="Tool Test Patient",
            age=45,
            gender="Male",
            summary="Diabetes management.",
            conditions=["type 2 diabetes"],
            medications=["metformin 500mg"],
            allergies=[],
        ),
    )
    return db_path, patient_db


@pytest.fixture
def mock_vs(tmp_path) -> PatientVectorStore:
    """PatientVectorStore with mocked embeddings — no OpenAI API call."""
    embeddings = MagicMock(spec=Embeddings)
    embeddings.embed_documents.return_value = [[0.1] * 1536]
    embeddings.embed_query.return_value = [0.1] * 1536
    return PatientVectorStore(str(tmp_path / "faiss"), embeddings)


@pytest.fixture
def tool(seeded_db, mock_vs):
    """history_tool backed by real SQLite and mock FAISS."""
    db_path, patient_db = seeded_db
    return make_history_tool(patient_db, db_path, mock_vs)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_append_medication_succeeds(tool, seeded_db) -> None:
    """Appending a medication returns success=True and confirmation."""
    result = tool.invoke({
        "patient_id": "P-tooltest",
        "field": "medications",
        "value": "lisinopril 10mg",
        "operation": "append",
    })
    assert result.success is True
    assert result.field_updated == "medications"
    assert result.error is None


def test_replace_notes_succeeds(tool, seeded_db) -> None:
    """Replacing notes returns success=True."""
    result = tool.invoke({
        "patient_id": "P-tooltest",
        "field": "notes",
        "value": "New consultation notes.",
        "operation": "replace",
    })
    assert result.success is True


def test_confirmation_message_non_empty(tool) -> None:
    """Successful update always returns a non-empty confirmation string."""
    result = tool.invoke({
        "patient_id": "P-tooltest",
        "field": "summary",
        "value": "Updated summary.",
        "operation": "replace",
    })
    assert result.success is True
    assert len(result.confirmation) > 0


def test_append_conditions_persisted(tool, seeded_db) -> None:
    """Appended condition is readable back from the DB after the tool call."""
    db_path, patient_db = seeded_db
    tool.invoke({
        "patient_id": "P-tooltest",
        "field": "conditions",
        "value": "hypertension",
        "operation": "append",
    })
    record = patient_db.get_patient(db_path, "P-tooltest")
    assert "hypertension" in record.conditions


def test_result_patient_id_matches_input(tool) -> None:
    """HistoryUpdateResult.patient_id always matches the input."""
    result = tool.invoke({
        "patient_id": "P-tooltest",
        "field": "notes",
        "value": "some note",
    })
    assert result.patient_id == "P-tooltest"


# ---------------------------------------------------------------------------
# Patient not found
# ---------------------------------------------------------------------------


def test_unknown_patient_returns_failure(tool) -> None:
    """Updating a non-existent patient returns success=False."""
    result = tool.invoke({
        "patient_id": "P-does-not-exist",
        "field": "notes",
        "value": "irrelevant",
    })
    assert result.success is False
    assert result.error is not None


# ---------------------------------------------------------------------------
# FAISS upsert failure is non-fatal
# ---------------------------------------------------------------------------


def test_faiss_upsert_failure_does_not_propagate(seeded_db, tmp_path) -> None:
    """FAISS upsert failure is logged but does not cause the tool to raise."""
    db_path, patient_db = seeded_db

    # Vector store whose upsert always raises
    failing_vs = MagicMock(spec=PatientVectorStore)
    failing_vs.upsert.side_effect = RuntimeError("FAISS index corrupt")

    broken_tool = make_history_tool(patient_db, db_path, failing_vs)
    result = broken_tool.invoke({
        "patient_id": "P-tooltest",
        "field": "notes",
        "value": "test note",
        "operation": "replace",
    })
    # DB write succeeded even though FAISS upsert raised
    assert result.success is True
