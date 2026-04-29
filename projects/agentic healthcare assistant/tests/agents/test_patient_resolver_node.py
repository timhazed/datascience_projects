"""Tests for patient_resolver_node — real SQLite, no LLM.

Covers the four required paths:
  - exact match (one result)
  - fuzzy match (partial name, one result)
  - ambiguous (multiple matches)
  - not found (zero matches)
"""

from datetime import datetime

from src.agents.patient_resolver_node import make_patient_resolver_node
from src.db.patient_db import PatientDB
from src.models.patient import PatientRecord
from src.models.sub_goal import SubGoal


def _state(**overrides) -> dict:
    """Build a minimal HealthcareState-compatible dict."""
    base: dict = {
        "user_query": "test",
        "patient_id": None,
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


def _insert(db_path: str, patient_id: str, name: str, phone: str = "+1-555-0001") -> None:
    """Insert a minimal patient row into the DB."""
    PatientDB().create_patient(
        db_path,
        PatientRecord(
            patient_id=patient_id,
            phone=phone,
            name=name,
            age=40,
            gender="Female",
            last_updated=datetime.now(),
        ),
    )


def _sub_goal(name: str | None) -> SubGoal:
    """Build a resolve_patient SubGoal with optional patient_name."""
    params: dict = {}
    if name is not None:
        params["patient_name"] = name
    return SubGoal(task="resolve_patient", parameters=params, order=1)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_exact_match_sets_patient_id(in_memory_db) -> None:
    """Exact name match resolves to patient_id and returns success TaskResult."""
    _insert(in_memory_db, "P-001", "Alice Smith")
    node = make_patient_resolver_node(PatientDB(), in_memory_db)
    result = node(
        _state(
            patient_id=None,
            pending_tasks=[_sub_goal("Alice Smith"), _sub_goal("retrieve_history")],
        )
    )
    assert result["patient_id"] == "P-001"
    assert len(result["completed_tasks"]) == 1
    assert result["completed_tasks"][0].success is True
    assert result["completed_tasks"][0].task == "resolve_patient"


def test_dequeues_pending_tasks(in_memory_db) -> None:
    """Resolver always dequeues pending_tasks[0] on success."""
    _insert(in_memory_db, "P-001", "Alice Smith")
    node = make_patient_resolver_node(PatientDB(), in_memory_db)
    extra = _sub_goal("retrieve_history")
    result = node(_state(pending_tasks=[_sub_goal("Alice Smith"), extra]))
    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0] is extra


def test_fuzzy_match_partial_name(in_memory_db) -> None:
    """Partial name 'Alice' resolves when only one Alice is in the DB."""
    _insert(in_memory_db, "P-002", "Alice Johnson")
    node = make_patient_resolver_node(PatientDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Alice")]))
    assert result["patient_id"] == "P-002"
    assert result["completed_tasks"][0].success is True


# ---------------------------------------------------------------------------
# Failure paths — all non-fatal (completed_tasks[0].success == False)
# ---------------------------------------------------------------------------


def test_ambiguous_match_returns_failure(in_memory_db) -> None:
    """Two patients with 'Alice' → failure TaskResult; other patients' names NOT in error."""
    _insert(in_memory_db, "P-003", "Alice Brown", phone="+1-555-0003")
    _insert(in_memory_db, "P-004", "Alice Green", phone="+1-555-0004")
    node = make_patient_resolver_node(PatientDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Alice")]))
    error = result["completed_tasks"][0].error
    assert result["completed_tasks"][0].success is False
    # Count is surfaced, but individual names must NOT appear (PHI leak prevention)
    assert "2" in error or "multiple" in error.lower()
    assert "Alice Brown" not in error
    assert "Alice Green" not in error
    # patient_id must NOT be set — ambiguous resolution must never write a value
    assert "patient_id" not in result


def test_empty_pending_tasks_returns_failure(in_memory_db) -> None:
    """Empty pending_tasks → failure TaskResult, no IndexError."""
    node = make_patient_resolver_node(PatientDB(), in_memory_db)
    result = node(_state(pending_tasks=[]))
    assert result["completed_tasks"][0].success is False
    assert result["pending_tasks"] == []


def test_not_found_returns_failure(in_memory_db) -> None:
    """Name with no DB match → failure TaskResult, patient_id not set."""
    node = make_patient_resolver_node(PatientDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Nonexistent Person")]))
    assert result["completed_tasks"][0].success is False
    assert "no patient found" in result["completed_tasks"][0].error.lower()
    assert "patient_id" not in result


def test_no_name_provided_returns_failure(in_memory_db) -> None:
    """Missing patient_name → failure TaskResult; no crash."""
    node = make_patient_resolver_node(PatientDB(), in_memory_db)
    result = node(_state(pending_tasks=[SubGoal(task="resolve_patient", order=1)]))
    assert result["completed_tasks"][0].success is False


def test_dequeues_on_failure(in_memory_db) -> None:
    """Resolver dequeues even when resolution fails — graph continues."""
    node = make_patient_resolver_node(PatientDB(), in_memory_db)
    extra = SubGoal(task="search_disease", order=2)
    result = node(_state(pending_tasks=[_sub_goal("Ghost"), extra]))
    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0] is extra
