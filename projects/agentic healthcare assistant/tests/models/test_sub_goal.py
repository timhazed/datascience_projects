"""Tests for src/models/sub_goal.py — SubGoal."""

import pytest
from pydantic import ValidationError

from src.models.sub_goal import SubGoal


def test_valid_tasks():
    for task in (
        "resolve_patient",
        "retrieve_history",
        "update_history",
        "book_appointment",
        "search_disease",
    ):
        sg = SubGoal(task=task, order=1)
        assert sg.task == task


def test_invalid_task_raises():
    with pytest.raises(ValidationError):
        SubGoal(task="send_email", order=1)


def test_order_minimum():
    SubGoal(task="resolve_patient", order=1)


def test_order_below_min_raises():
    with pytest.raises(ValidationError):
        SubGoal(task="resolve_patient", order=0)


def test_parameters_default_empty():
    sg = SubGoal(task="search_disease", order=2)
    assert sg.parameters == {}


def test_parameters_with_values():
    sg = SubGoal(
        task="retrieve_history",
        order=2,
        parameters={"patient_id": "P-abc", "query": "medications"},
    )
    assert sg.parameters["patient_id"] == "P-abc"


def test_roundtrip():
    sg = SubGoal(task="book_appointment", order=3, parameters={"specialty": "ENT"})
    restored = SubGoal.model_validate(sg.model_dump())
    assert restored.order == 3
    assert restored.parameters["specialty"] == "ENT"
