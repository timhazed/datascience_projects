"""Tests for src/models/patient.py — PatientRecord."""

import pytest
from pydantic import ValidationError

from src.models.patient import PatientRecord


def _minimal() -> dict:
    return {
        "patient_id": "P-4a2f91",
        "phone": "+91-98180-11245",
        "name": "Ravi Kumar",
        "age": 35,
        "gender": "Male",
    }


def test_valid_construction():
    p = PatientRecord(**_minimal())
    assert p.patient_id == "P-4a2f91"
    assert p.age == 35
    assert p.email is None
    assert p.conditions == []
    assert p.medications == []
    assert p.allergies == []


def test_age_boundary_zero():
    p = PatientRecord(**{**_minimal(), "age": 0})
    assert p.age == 0


def test_age_boundary_130():
    p = PatientRecord(**{**_minimal(), "age": 130})
    assert p.age == 130


def test_age_above_max_raises():
    with pytest.raises(ValidationError):
        PatientRecord(**{**_minimal(), "age": 131})


def test_age_below_min_raises():
    with pytest.raises(ValidationError):
        PatientRecord(**{**_minimal(), "age": -1})


def test_default_list_factories_independent():
    p1 = PatientRecord(**_minimal())
    p2 = PatientRecord(**_minimal())
    p1.conditions.append("hypertension")
    assert p2.conditions == []


def test_roundtrip():
    data = {**_minimal(), "conditions": ["CKD"], "email": "ravi@example.com"}
    p = PatientRecord(**data)
    restored = PatientRecord.model_validate(p.model_dump())
    assert restored.conditions == ["CKD"]
    assert restored.email == "ravi@example.com"
