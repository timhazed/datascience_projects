"""Tests for AppointmentDB — real SQLite via in_memory_db fixture.

Double-book prevention is the critical path; it must return AppointmentResult(success=False),
not raise an exception.
"""

from datetime import date

import pytest

from src.db.appointment_db import AppointmentDB
from src.models.appointment_result import AppointmentResult


@pytest.fixture
def adb(in_memory_db: str) -> tuple[AppointmentDB, str]:
    """Return (AppointmentDB instance, db_path) ready for use."""
    return AppointmentDB(), in_memory_db


# ---------------------------------------------------------------------------
# init_db / slot seeding
# ---------------------------------------------------------------------------


def test_init_db_seeds_doctors(adb: tuple[AppointmentDB, str]) -> None:
    """init_db must seed exactly 5 doctors."""
    import sqlite3

    _, path = adb
    conn = sqlite3.connect(path)
    count = conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
    conn.close()
    assert count == 5


def test_init_db_seeds_slots(adb: tuple[AppointmentDB, str]) -> None:
    """init_db must seed at least 1 000 slots across all 5 specialties."""
    import sqlite3

    _, path = adb
    conn = sqlite3.connect(path)
    count = conn.execute("SELECT COUNT(*) FROM slots").fetchone()[0]
    conn.close()
    assert count >= 1000


def test_init_db_idempotent(adb: tuple[AppointmentDB, str]) -> None:
    """Calling init_db twice must not duplicate doctors or slots."""
    import sqlite3

    db_obj, path = adb
    db_obj.init_db(path)  # second call

    conn = sqlite3.connect(path)
    doc_count = conn.execute("SELECT COUNT(*) FROM doctors").fetchone()[0]
    slot_count = conn.execute("SELECT COUNT(*) FROM slots").fetchone()[0]
    conn.close()
    assert doc_count == 5
    assert slot_count >= 1000


# ---------------------------------------------------------------------------
# query_slots
# ---------------------------------------------------------------------------


def test_query_slots_routine_returns_results(adb: tuple[AppointmentDB, str]) -> None:
    """Routine query for Cardiology returns up to 50 unbooked slots."""
    db_obj, path = adb
    slots = db_obj.query_slots(path, "Cardiology", urgency="routine")
    assert len(slots) > 0
    assert len(slots) <= 50


def test_query_slots_emergency_returns_one(adb: tuple[AppointmentDB, str]) -> None:
    """Emergency query must return exactly 1 slot (the very next available)."""
    db_obj, path = adb
    slots = db_obj.query_slots(path, "Nephrology", urgency="emergency")
    assert len(slots) == 1


def test_query_slots_urgent_max_seven_days(adb: tuple[AppointmentDB, str]) -> None:
    """Urgent query must return only slots within the next 7 days."""
    from datetime import datetime, timedelta

    db_obj, path = adb
    slots = db_obj.query_slots(path, "Cardiology", urgency="urgent")
    cutoff = (datetime.now() + timedelta(days=7)).isoformat()
    for slot in slots:
        assert slot["slot_datetime"] <= cutoff


def test_query_slots_preferred_date_filters(adb: tuple[AppointmentDB, str]) -> None:
    """Slots returned must start on or after the preferred_date."""
    db_obj, path = adb
    preferred = date.today()
    slots = db_obj.query_slots(path, "Endocrinology", preferred_date=preferred, urgency="routine")
    for slot in slots:
        assert slot["slot_datetime"] >= preferred.isoformat()


def test_query_slots_unknown_specialty_returns_empty(adb: tuple[AppointmentDB, str]) -> None:
    """An unrecognised specialty returns an empty list without raising."""
    db_obj, path = adb
    slots = db_obj.query_slots(path, "Astrology", urgency="routine")
    assert slots == []


def test_query_slots_contain_required_keys(adb: tuple[AppointmentDB, str]) -> None:
    """Each returned slot dict has the required keys."""
    db_obj, path = adb
    slots = db_obj.query_slots(path, "Pulmonology", urgency="routine")
    assert len(slots) > 0
    for slot in slots:
        assert "slot_id" in slot
        assert "doctor_id" in slot
        assert "specialty" in slot
        assert "slot_datetime" in slot


# ---------------------------------------------------------------------------
# book_slot
# ---------------------------------------------------------------------------


def _first_slot(db_obj: AppointmentDB, path: str, specialty: str = "Cardiology") -> dict:
    """Helper: return the first available slot for a specialty."""
    slots = db_obj.query_slots(path, specialty, urgency="routine")
    assert slots, f"No available slots for {specialty}"
    return slots[0]


def test_book_slot_success(adb: tuple[AppointmentDB, str]) -> None:
    """book_slot returns AppointmentResult(success=True) with booking details."""
    db_obj, path = adb
    slot = _first_slot(db_obj, path)

    result = db_obj.book_slot(
        path, slot["slot_id"], "P-test01", reason="Annual check", urgency="routine"
    )

    assert isinstance(result, AppointmentResult)
    assert result.success is True
    assert result.patient_id == "P-test01"
    assert result.appointment_id is not None
    assert result.doctor_name is not None
    assert result.slot is not None


def test_book_slot_marks_slot_as_booked(adb: tuple[AppointmentDB, str]) -> None:
    """After booking, the slot must not appear in subsequent query_slots results."""
    db_obj, path = adb
    slot = _first_slot(db_obj, path)
    slot_id = slot["slot_id"]

    db_obj.book_slot(path, slot_id, "P-test02", urgency="routine")

    remaining = db_obj.query_slots(path, "Cardiology", urgency="routine")
    assert all(s["slot_id"] != slot_id for s in remaining)


def test_book_slot_double_book_returns_failure(adb: tuple[AppointmentDB, str]) -> None:
    """Booking the same slot twice must return success=False, not raise."""
    db_obj, path = adb
    slot = _first_slot(db_obj, path)

    first = db_obj.book_slot(path, slot["slot_id"], "P-first", urgency="routine")
    second = db_obj.book_slot(path, slot["slot_id"], "P-second", urgency="routine")

    assert first.success is True
    assert second.success is False
    assert "no longer available" in second.message.lower()


def test_book_slot_nonexistent_slot_returns_failure(adb: tuple[AppointmentDB, str]) -> None:
    """Booking a slot_id that does not exist returns success=False."""
    db_obj, path = adb
    result = db_obj.book_slot(
        path, "00000000-0000-0000-0000-000000000000", "P-x", urgency="routine"
    )
    assert result.success is False


# ---------------------------------------------------------------------------
# get_bookings
# ---------------------------------------------------------------------------


def test_get_bookings_returns_booking(adb: tuple[AppointmentDB, str]) -> None:
    """get_bookings returns the booking created by book_slot."""
    db_obj, path = adb
    slot = _first_slot(db_obj, path)
    db_obj.book_slot(path, slot["slot_id"], "P-lookup", reason="Checkup", urgency="routine")

    bookings = db_obj.get_bookings(path, "P-lookup")
    assert len(bookings) == 1
    assert bookings[0]["slot_id"] == slot["slot_id"]
    assert "doctor_name" in bookings[0]
    assert "slot_datetime" in bookings[0]


def test_get_bookings_no_bookings(adb: tuple[AppointmentDB, str]) -> None:
    """get_bookings returns an empty list when the patient has no bookings."""
    db_obj, path = adb
    bookings = db_obj.get_bookings(path, "P-nobody")
    assert bookings == []


def test_get_bookings_multiple(adb: tuple[AppointmentDB, str]) -> None:
    """A patient can have multiple bookings; all are returned."""
    db_obj, path = adb
    slots = db_obj.query_slots(path, "Cardiology", urgency="routine")
    assert len(slots) >= 2

    db_obj.book_slot(path, slots[0]["slot_id"], "P-multi", urgency="routine")
    db_obj.book_slot(path, slots[1]["slot_id"], "P-multi", urgency="routine")

    bookings = db_obj.get_bookings(path, "P-multi")
    assert len(bookings) == 2
