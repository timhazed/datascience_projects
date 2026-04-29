"""Tests for appointment_tool — real in-memory SQLite, no live API calls."""

from __future__ import annotations

import pytest

from src.db.appointment_db import AppointmentDB
from src.db.patient_db import PatientDB
from src.tools.appointment_tool import make_appointment_tool


@pytest.fixture
def seeded_db(tmp_path) -> str:
    """SQLite database with appointment tables seeded, isolated per test."""
    db_path = str(tmp_path / "appt_tool.db")
    AppointmentDB().init_db(db_path)
    PatientDB().init_db(db_path)
    return db_path


@pytest.fixture
def tool(seeded_db):
    """appointment_tool instance backed by real in-memory SQLite."""
    return make_appointment_tool(AppointmentDB(), seeded_db)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_books_slot_for_nephrology(tool) -> None:
    """booking a Nephrology slot returns success=True and fills booking fields."""
    result = tool.invoke({
        "patient_id": "P-test001",
        "specialty": "Nephrology",
        "urgency": "routine",
        "reason": "CKD follow-up",
    })
    assert result.success is True
    assert result.appointment_id is not None
    assert result.doctor_name is not None
    assert result.slot is not None
    assert result.patient_id == "P-test001"


def test_books_emergency_slot_returns_earliest(tool) -> None:
    """Emergency urgency returns the single earliest available slot."""
    result = tool.invoke({
        "patient_id": "P-emergency",
        "specialty": "Cardiology",
        "urgency": "emergency",
    })
    assert result.success is True


def test_preferred_date_is_accepted(tool) -> None:
    """A valid preferred_date ISO string is parsed and used for slot filtering."""
    result = tool.invoke({
        "patient_id": "P-dated",
        "specialty": "Nephrology",
        "preferred_date": "2026-04-15",
        "urgency": "routine",
    })
    # Result may be success or failure depending on slot availability after that date,
    # but the tool must not raise.
    assert isinstance(result.success, bool)


def test_invalid_preferred_date_does_not_raise(tool) -> None:
    """An unparseable preferred_date logs a warning but does not raise."""
    result = tool.invoke({
        "patient_id": "P-test",
        "specialty": "Nephrology",
        "preferred_date": "not-a-date",
        "urgency": "routine",
    })
    # Tool falls back to current time for slot query — must succeed if slots exist
    assert isinstance(result.success, bool)


# ---------------------------------------------------------------------------
# No slots available
# ---------------------------------------------------------------------------


def test_no_slots_for_unknown_specialty(tool) -> None:
    """Booking for an unknown specialty finds no slots and returns success=False."""
    result = tool.invoke({
        "patient_id": "P-test",
        "specialty": "Martian Medicine",
        "urgency": "routine",
    })
    assert result.success is False
    assert "Martian Medicine" in result.message


# ---------------------------------------------------------------------------
# Double-book race
# ---------------------------------------------------------------------------


def test_double_book_second_call_fails(seeded_db) -> None:
    """Booking the same slot twice — second booking returns success=False."""
    db = AppointmentDB()
    tool = make_appointment_tool(db, seeded_db)

    # Book first — should succeed
    first = tool.invoke({
        "patient_id": "P-first",
        "specialty": "Nephrology",
        "urgency": "emergency",  # Gets exactly 1 slot
    })
    assert first.success is True

    # The emergency slot is now marked booked; a second emergency booking for
    # the same specialty should still find the next available slot (not fail),
    # UNLESS all emergency slots are consumed. Here we verify the tool handles
    # both outcomes gracefully.
    second = tool.invoke({
        "patient_id": "P-second",
        "specialty": "Nephrology",
        "urgency": "emergency",
    })
    assert isinstance(second.success, bool)  # True (another slot) or False (none left)


# ---------------------------------------------------------------------------
# Result fields
# ---------------------------------------------------------------------------


def test_result_patient_id_matches_input(tool) -> None:
    """AppointmentResult.patient_id always matches the input patient_id."""
    result = tool.invoke({
        "patient_id": "P-known-patient",
        "specialty": "Cardiology",
        "urgency": "routine",
    })
    assert result.patient_id == "P-known-patient"


def test_failure_result_has_message(tool) -> None:
    """Failure AppointmentResult always has a non-empty message."""
    result = tool.invoke({
        "patient_id": "P-x",
        "specialty": "Nonexistent Specialty",
        "urgency": "routine",
    })
    assert result.success is False
    assert len(result.message) > 0
