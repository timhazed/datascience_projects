"""Tests for src/models/task_result.py — TaskResult."""

import pytest
from pydantic import ValidationError

from src.models.task_result import TaskResult


def test_valid_success():
    r = TaskResult(task="retrieve_history", success=True, result={"conditions": ["CKD"]})
    assert r.error is None


def test_valid_failure():
    r = TaskResult(task="book_appointment", success=False, error="No slots available")
    assert r.result == {}
    assert r.error == "No slots available"


def test_result_defaults_empty_dict():
    r = TaskResult(task="search_disease", success=True)
    assert r.result == {}


def test_default_result_factory_independent():
    r1 = TaskResult(task="retrieve_history", success=True)
    r2 = TaskResult(task="retrieve_history", success=True)
    r1.result["key"] = "val"
    assert "key" not in r2.result


def test_missing_required_raises():
    with pytest.raises(ValidationError):
        TaskResult(success=True)


def test_roundtrip():
    r = TaskResult(task="resolve_patient", success=True, result={"patient_id": "P-abc"})
    restored = TaskResult.model_validate(r.model_dump())
    assert restored.result["patient_id"] == "P-abc"
