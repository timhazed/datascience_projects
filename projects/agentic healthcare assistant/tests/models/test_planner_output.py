"""Tests for src/models/planner_output.py — PlannerOutput."""

import pytest
from pydantic import ValidationError

from src.models.planner_output import PlannerOutput
from src.models.sub_goal import SubGoal


def _goal(task: str = "resolve_patient", order: int = 1) -> SubGoal:
    return SubGoal(task=task, order=order)


def test_valid_with_patient_id():
    plan = PlannerOutput(
        patient_id="P-abc",
        sub_goals=[_goal("retrieve_history")],
        reasoning="Patient already resolved; retrieve history directly.",
    )
    assert plan.patient_id == "P-abc"


def test_valid_without_patient_id():
    plan = PlannerOutput(
        sub_goals=[_goal("resolve_patient", 1), _goal("retrieve_history", 2)],
        reasoning="Patient unknown; resolve first.",
    )
    assert plan.patient_id is None
    assert len(plan.sub_goals) == 2


def test_missing_reasoning_raises():
    with pytest.raises(ValidationError):
        PlannerOutput(sub_goals=[_goal()])


def test_empty_sub_goals_allowed():
    # Pydantic allows empty list — business logic validation is graph-level
    plan = PlannerOutput(sub_goals=[], reasoning="No action needed.")
    assert plan.sub_goals == []


def test_roundtrip():
    plan = PlannerOutput(
        patient_id="P-xyz",
        sub_goals=[_goal("search_disease", 1)],
        reasoning="Search for disease information.",
    )
    restored = PlannerOutput.model_validate(plan.model_dump())
    assert restored.sub_goals[0].task == "search_disease"
