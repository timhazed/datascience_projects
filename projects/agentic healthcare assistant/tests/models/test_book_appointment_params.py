"""Tests for src/models/book_appointment_params.py — BookAppointmentParams."""

import pytest
from pydantic import ValidationError

from src.models.book_appointment_params import BookAppointmentParams


def test_valid_minimal():
    p = BookAppointmentParams(patient_id="P-abc", specialty="Nephrology")
    assert p.urgency == "routine"
    assert p.preferred_date is None
    assert p.reason == ""


def test_all_urgency_values():
    for urgency in ("routine", "urgent", "emergency"):
        p = BookAppointmentParams(patient_id="P-1", specialty="ENT", urgency=urgency)
        assert p.urgency == urgency


def test_invalid_urgency_raises():
    with pytest.raises(ValidationError):
        BookAppointmentParams(patient_id="P-1", specialty="ENT", urgency="asap")


def test_preferred_date_as_string():
    p = BookAppointmentParams(patient_id="P-1", specialty="Cardiology", preferred_date="2026-05-01")
    assert p.preferred_date == "2026-05-01"


def test_roundtrip():
    p = BookAppointmentParams(patient_id="P-xyz", specialty="Oncology", urgency="urgent")
    restored = BookAppointmentParams.model_validate(p.model_dump())
    assert restored.specialty == "Oncology"
