"""Tests for src/graph/edges.py — route_from_plan."""

from src.data import AgentName, AgentState
from src.data.plan import PlanStep, RoutingPlan
from src.data.user_query import UserQuery
from src.graph.edges import route_from_plan


def _make_state(steps: list[PlanStep] | None, session_id: str = "test-session") -> AgentState:
    plan = (
        RoutingPlan(steps=steps, original_query="test query")
        if steps is not None
        else None
    )
    return AgentState(
        query=UserQuery(text="test query", session_id=session_id),
        plan=plan,
    )


class TestRouteFromPlan:
    def test_single_step_business(self):
        state = _make_state([PlanStep(agent=AgentName.BUSINESS, query="markets")])
        result = route_from_plan(state)
        assert result == ["business_agent"]

    def test_three_step_plan_returns_three_node_names(self):
        steps = [
            PlanStep(agent=AgentName.BUSINESS, query="markets"),
            PlanStep(agent=AgentName.SPORTS, query="NFL"),
            PlanStep(agent=AgentName.GENERAL, query="world news"),
        ]
        state = _make_state(steps)
        result = route_from_plan(state)
        assert set(result) == {"business_agent", "sports_agent", "general_agent"}
        assert len(result) == 3

    def test_none_plan_falls_back_to_web_search(self):
        state = _make_state(None)
        result = route_from_plan(state)
        assert result == ["web_search_agent"]

    def test_web_search_step(self):
        state = _make_state([PlanStep(agent=AgentName.WEB_SEARCH, query="Austin festivals")])
        result = route_from_plan(state)
        assert result == ["web_search_agent"]
