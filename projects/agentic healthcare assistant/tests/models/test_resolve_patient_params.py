"""Tests for src/models/resolve_patient_params.py — ResolvePatientParams."""

from src.models.resolve_patient_params import ResolvePatientParams


def test_all_optional_defaults_none():
    p = ResolvePatientParams()
    assert p.patient_name is None
    assert p.phone is None


def test_name_only():
    p = ResolvePatientParams(patient_name="Ravi Kumar")
    assert p.patient_name == "Ravi Kumar"
    assert p.phone is None


def test_phone_only():
    p = ResolvePatientParams(phone="+91-98180-11245")
    assert p.phone == "+91-98180-11245"


def test_both_fields():
    p = ResolvePatientParams(patient_name="Ravi Kumar", phone="+91-98180-11245")
    assert p.patient_name == "Ravi Kumar"
    assert p.phone == "+91-98180-11245"


def test_roundtrip():
    p = ResolvePatientParams(patient_name="Anjali Singh")
    restored = ResolvePatientParams.model_validate(p.model_dump())
    assert restored.patient_name == "Anjali Singh"
