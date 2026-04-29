"""Tests for src/models/history_update_result.py — HistoryUpdateResult."""

import pytest
from pydantic import ValidationError

from src.models.history_update_result import HistoryUpdateResult


def test_valid_success():
    r = HistoryUpdateResult(
        patient_id="P-abc",
        field_updated="conditions",
        success=True,
        confirmation="Conditions field updated with 'hypertension'.",
    )
    assert r.error is None


def test_valid_failure():
    r = HistoryUpdateResult(
        patient_id="P-abc",
        field_updated="notes",
        success=False,
        confirmation="Update failed.",
        error="Patient not found",
    )
    assert r.error == "Patient not found"


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        HistoryUpdateResult(patient_id="P-abc", field_updated="notes", success=True)


def test_roundtrip():
    r = HistoryUpdateResult(
        patient_id="P-1",
        field_updated="medications",
        success=True,
        confirmation="Medications appended.",
    )
    restored = HistoryUpdateResult.model_validate(r.model_dump())
    assert restored.field_updated == "medications"
