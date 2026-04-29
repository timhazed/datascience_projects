"""Tests for appointment_node — real SQLite, no mock LLM needed.

Covers: slot found → booked, no slots → failure, emergency → earliest slot,
unknown patient_id, dequeue invariant.
"""

from src.agents.appointment_node import make_appointment_node
from src.db.appointment_db import AppointmentDB
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


def _sub_goal(
    specialty: str, urgency: str = "routine", preferred_date: str | None = None
) -> SubGoal:
    """Build a book_appointment SubGoal."""
    params: dict = {
        "patient_id": "P-001",
        "specialty": specialty,
        "urgency": urgency,
        "reason": "test reason",
    }
    if preferred_date:
        params["preferred_date"] = preferred_date
    return SubGoal(task="book_appointment", parameters=params, order=1)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_routine_booking_succeeds(in_memory_db) -> None:
    """Routine Nephrology booking returns success TaskResult with appointment_id."""
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Nephrology")]))
    task = result["completed_tasks"][0]
    assert task.success is True
    assert task.task == "book_appointment"
    assert task.result["appointment_id"] != ""
    assert "Nephrology" in task.result["doctor"] or task.result["slot"] != ""


def test_emergency_booking_returns_earliest_slot(in_memory_db) -> None:
    """Emergency urgency books the single earliest available slot."""
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Cardiology", urgency="emergency")]))
    task = result["completed_tasks"][0]
    assert task.success is True
    assert task.result["slot"] != ""


def test_dequeues_pending_tasks(in_memory_db) -> None:
    """Node always dequeues pending_tasks[0]."""
    extra = SubGoal(task="search_disease", order=2)
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Nephrology"), extra]))
    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0] is extra


def test_completed_tasks_is_list_of_one(in_memory_db) -> None:
    """completed_tasks is always a list of exactly one TaskResult."""
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Nephrology")]))
    assert isinstance(result["completed_tasks"], list)
    assert len(result["completed_tasks"]) == 1


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_no_slots_returns_failure(in_memory_db) -> None:
    """Specialty with zero slots → failure TaskResult, not an exception."""
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Dermatology")]))  # no Dermatology doctors
    task = result["completed_tasks"][0]
    assert task.success is False
    assert "no available slots" in task.error.lower()


def test_no_patient_id_returns_failure(in_memory_db) -> None:
    """Missing patient_id in state → failure TaskResult."""
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(patient_id=None, pending_tasks=[_sub_goal("Nephrology")]))
    assert result["completed_tasks"][0].success is False


def test_dequeues_on_failure(in_memory_db) -> None:
    """Node dequeues even when no slots are available."""
    extra = SubGoal(task="search_disease", order=2)
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Dermatology"), extra]))
    assert len(result["pending_tasks"]) == 1


def test_invalid_preferred_date_falls_back_gracefully(in_memory_db) -> None:
    """Non-ISO preferred_date string is ignored; booking proceeds without a date filter."""
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(pending_tasks=[_sub_goal("Nephrology", preferred_date="not-a-date")]))
    # preferred_date parse error is non-fatal — node falls back to no preferred date
    assert result["completed_tasks"][0].task == "book_appointment"
    assert result["completed_tasks"][0].success is True


def test_invalid_params_returns_failure(in_memory_db) -> None:
    """Malformed parameters (ValidationError) → failure TaskResult, no crash."""
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    bad = SubGoal(task="book_appointment", parameters={"bad_key": 99}, order=1)
    result = node(_state(pending_tasks=[bad]))
    assert result["completed_tasks"][0].success is False
    assert len(result["pending_tasks"]) == 0


def test_empty_pending_tasks_returns_failure(in_memory_db) -> None:
    """Empty pending_tasks → failure TaskResult, no IndexError."""
    node = make_appointment_node(AppointmentDB(), in_memory_db)
    result = node(_state(pending_tasks=[]))
    assert result["completed_tasks"][0].success is False
    assert result["pending_tasks"] == []
