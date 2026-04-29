"""Tests for src/models/update_history_params.py — UpdateHistoryParams."""

import pytest
from pydantic import ValidationError

from src.models.update_history_params import UpdateHistoryParams


def test_valid_construction():
    p = UpdateHistoryParams(
        patient_id="P-abc",
        field="medications",
        value="metformin 1000mg BID",
    )
    assert p.operation == "append"


def test_all_valid_fields():
    for field in ("conditions", "medications", "allergies", "notes", "summary"):
        p = UpdateHistoryParams(patient_id="P-1", field=field, value="v")
        assert p.field == field


def test_invalid_field_raises():
    with pytest.raises(ValidationError):
        UpdateHistoryParams(patient_id="P-1", field="lab_results", value="v")


def test_replace_operation():
    p = UpdateHistoryParams(patient_id="P-1", field="notes", value="updated", operation="replace")
    assert p.operation == "replace"


def test_invalid_operation_raises():
    with pytest.raises(ValidationError):
        UpdateHistoryParams(patient_id="P-1", field="notes", value="v", operation="delete")


def test_roundtrip():
    p = UpdateHistoryParams(patient_id="P-abc", field="conditions", value="CKD stage 3")
    restored = UpdateHistoryParams.model_validate(p.model_dump())
    assert restored.value == "CKD stage 3"
