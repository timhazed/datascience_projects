"""Tests for src/data/plan.py — PlanStep and RoutingPlan."""

import pytest
from pydantic import ValidationError

from src.data.enums import AgentName
from src.data.plan import PlanStep, RoutingPlan


def _step(agent: AgentName = AgentName.BUSINESS, query: str = "test") -> PlanStep:
    return PlanStep(agent=agent, query=query)


class TestPlanStep:
    def test_basic_construction(self):
        step = _step(AgentName.SPORTS, "Premier League scores")
        assert step.agent == AgentName.SPORTS
        assert step.query == "Premier League scores"

    def test_all_agent_names_accepted(self):
        for agent in AgentName:
            step = _step(agent)
            assert step.agent == agent


class TestRoutingPlan:
    def test_single_step_plan(self):
        plan = RoutingPlan(steps=[_step()], original_query="test")
        assert len(plan.steps) == 1

    def test_three_step_plan(self):
        plan = RoutingPlan(
            steps=[_step(AgentName.BUSINESS), _step(AgentName.SPORTS), _step(AgentName.GENERAL)],
            original_query="multi",
        )
        assert len(plan.steps) == 3

    def test_zero_steps_raises(self):
        with pytest.raises(ValidationError):
            RoutingPlan(steps=[], original_query="test")

    def test_four_steps_raises(self):
        with pytest.raises(ValidationError):
            RoutingPlan(
                steps=[_step(), _step(), _step(), _step()],
                original_query="test",
            )

    def test_original_query_stored(self):
        plan = RoutingPlan(steps=[_step()], original_query="What is the S&P 500 at?")
        assert plan.original_query == "What is the S&P 500 at?"
