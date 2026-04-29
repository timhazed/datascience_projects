"""Tests for history_writer_node — real SQLite, mock FAISS vector store."""

from datetime import datetime
from unittest.mock import MagicMock

from src.agents.history_writer_node import make_history_writer_node
from src.db.patient_db import PatientDB
from src.models.patient import PatientRecord
from src.models.sub_goal import SubGoal


def _state(**overrides) -> dict:
    """Build a minimal HealthcareState-compatible dict."""
    base: dict = {
        "user_query": "test",
        "patient_id": "P-001",
        "messages": [],
        "planner_output": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "intent_safe": True,
        "final_summary": None,
        "error": None,
        "trace": [],
    }
    base.update(overrides)
    return base


def _insert(db_path: str) -> None:
    """Insert a minimal test patient."""
    PatientDB().create_patient(
        db_path,
        PatientRecord(
            patient_id="P-001",
            phone="+1-555-0001",
            name="Alice Smith",
            age=45,
            gender="Female",
            conditions=["Hypertension"],
            medications=["lisinopril 10mg"],
            summary="Chronic hypertension.",
            last_updated=datetime.now(),
        ),
    )


def _sub_goal(field: str, value: str, operation: str = "append") -> SubGoal:
    """Build an update_history SubGoal."""
    return SubGoal(
        task="update_history",
        parameters={"patient_id": "P-001", "field": field, "value": value, "operation": operation},
        order=1,
    )


def _mock_vs() -> MagicMock:
    """Return a mock PatientVectorStore."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_append_condition_succeeds(in_memory_db) -> None:
    """Appending a new condition returns success TaskResult."""
    _insert(in_memory_db)
    node = make_history_writer_node(PatientDB(), in_memory_db, _mock_vs())
    result = node(_state(pending_tasks=[_sub_goal("conditions", "CKD stage 3")]))
    assert result["completed_tasks"][0].success is True
    assert result["completed_tasks"][0].result["field"] == "conditions"


def test_append_condition_persisted_in_db(in_memory_db) -> None:
    """DB reflects the appended condition after the node runs."""
    _insert(in_memory_db)
    node = make_history_writer_node(PatientDB(), in_memory_db, _mock_vs())
    node(_state(pending_tasks=[_sub_goal("conditions", "CKD stage 3")]))
    record = PatientDB().get_patient(in_memory_db, "P-001")
    assert "CKD stage 3" in record.conditions


def test_replace_notes_succeeds(in_memory_db) -> None:
    """Replace operation on notes field returns success."""
    _insert(in_memory_db)
    node = make_history_writer_node(PatientDB(), in_memory_db, _mock_vs())
    result = node(_state(pending_tasks=[_sub_goal("notes", "New clinical note.", "replace")]))
    assert result["completed_tasks"][0].success is True
    assert result["completed_tasks"][0].result["operation"] == "replace"


def test_faiss_upsert_called_on_success(in_memory_db) -> None:
    """Vector store upsert is called after a successful DB write."""
    _insert(in_memory_db)
    vs = _mock_vs()
    node = make_history_writer_node(PatientDB(), in_memory_db, vs)
    node(_state(pending_tasks=[_sub_goal("summary", "Updated summary.")]))
    vs.upsert.assert_called_once()


def test_dequeues_pending_tasks(in_memory_db) -> None:
    """Node always dequeues pending_tasks[0]."""
    _insert(in_memory_db)
    extra = SubGoal(task="search_disease", order=2)
    node = make_history_writer_node(PatientDB(), in_memory_db, _mock_vs())
    result = node(_state(pending_tasks=[_sub_goal("notes", "note"), extra]))
    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0] is extra


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_no_patient_id_returns_failure(in_memory_db) -> None:
    """Missing patient_id in state → failure TaskResult."""
    node = make_history_writer_node(PatientDB(), in_memory_db, _mock_vs())
    result = node(_state(patient_id=None, pending_tasks=[_sub_goal("notes", "x")]))
    assert result["completed_tasks"][0].success is False


def test_patient_not_in_db_returns_failure(in_memory_db) -> None:
    """Updating a non-existent patient → failure TaskResult."""
    node = make_history_writer_node(PatientDB(), in_memory_db, _mock_vs())
    result = node(_state(patient_id="P-ghost", pending_tasks=[_sub_goal("notes", "x")]))
    assert result["completed_tasks"][0].success is False


def test_faiss_upsert_failure_is_non_fatal(in_memory_db) -> None:
    """FAISS upsert error does not propagate — node still returns success."""
    _insert(in_memory_db)
    vs = _mock_vs()
    vs.upsert.side_effect = RuntimeError("FAISS unavailable")
    node = make_history_writer_node(PatientDB(), in_memory_db, vs)
    result = node(_state(pending_tasks=[_sub_goal("notes", "x")]))
    # DB write succeeded even though FAISS failed
    assert result["completed_tasks"][0].success is True


def test_invalid_params_returns_failure(in_memory_db) -> None:
    """Malformed parameters (invalid field literal) → failure TaskResult, no crash."""
    node = make_history_writer_node(PatientDB(), in_memory_db, _mock_vs())
    bad = SubGoal(
        task="update_history",
        parameters={"patient_id": "P-001", "field": "NOT_A_FIELD", "value": "x"},
        order=1,
    )
    result = node(_state(pending_tasks=[bad]))
    assert result["completed_tasks"][0].success is False
    assert len(result["pending_tasks"]) == 0


def test_empty_pending_tasks_returns_failure(in_memory_db) -> None:
    """Empty pending_tasks → failure TaskResult, no IndexError."""
    node = make_history_writer_node(PatientDB(), in_memory_db, _mock_vs())
    result = node(_state(pending_tasks=[]))
    assert result["completed_tasks"][0].success is False
    assert result["pending_tasks"] == []
