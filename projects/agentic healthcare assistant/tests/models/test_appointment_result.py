"""Tests for src/models/appointment_result.py — AppointmentResult."""

import pytest
from pydantic import ValidationError

from src.models.appointment_result import AppointmentResult


def test_successful_booking():
    r = AppointmentResult(
        success=True,
        appointment_id="APT-001",
        doctor_name="Dr. Patel — Nephrology",
        slot="2026-04-15T10:00:00",
        patient_id="P-abc",
        message="Appointment booked successfully.",
    )
    assert r.success is True
    assert r.appointment_id == "APT-001"


def test_failed_booking_optional_nulls():
    r = AppointmentResult(
        success=False,
        patient_id="P-abc",
        message="No available slots for the requested specialty.",
    )
    assert r.appointment_id is None
    assert r.doctor_name is None
    assert r.slot is None


def test_missing_patient_id_raises():
    with pytest.raises(ValidationError):
        AppointmentResult(success=True, message="OK")


def test_roundtrip():
    r = AppointmentResult(
        success=True,
        appointment_id="APT-002",
        doctor_name="Dr. Singh",
        slot="2026-05-01T09:00:00",
        patient_id="P-xyz",
        message="Confirmed.",
    )
    restored = AppointmentResult.model_validate(r.model_dump())
    assert restored.appointment_id == "APT-002"
