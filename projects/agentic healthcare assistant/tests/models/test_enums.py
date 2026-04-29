"""Tests for src/models/enums.py — HistoryField Literal."""

import pytest
from pydantic import BaseModel, ValidationError

from src.models.enums import HistoryField


class _FieldHolder(BaseModel):
    field: HistoryField


def test_valid_history_fields():
    for value in ("conditions", "medications", "allergies", "notes", "summary"):
        m = _FieldHolder(field=value)
        assert m.field == value


def test_invalid_history_field_raises():
    with pytest.raises(ValidationError):
        _FieldHolder(field="diagnosis")


def test_roundtrip():
    m = _FieldHolder(field="medications")
    restored = _FieldHolder.model_validate(m.model_dump())
    assert restored.field == "medications"
