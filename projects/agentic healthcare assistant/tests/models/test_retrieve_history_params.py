"""Tests for src/models/retrieve_history_params.py — RetrieveHistoryParams."""

import pytest
from pydantic import ValidationError

from src.models.retrieve_history_params import RetrieveHistoryParams


def test_valid_minimal():
    p = RetrieveHistoryParams(patient_id="P-abc123")
    assert p.query is None  # default is None; node falls back to patient name


def test_with_query():
    p = RetrieveHistoryParams(patient_id="P-abc123", query="current medications")
    assert p.query == "current medications"


def test_missing_patient_id_raises():
    with pytest.raises(ValidationError):
        RetrieveHistoryParams()


def test_roundtrip():
    p = RetrieveHistoryParams(patient_id="P-xyz", query="allergies")
    restored = RetrieveHistoryParams.model_validate(p.model_dump())
    assert restored.patient_id == "P-xyz"
    assert restored.query == "allergies"
