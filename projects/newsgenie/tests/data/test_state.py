"""Tests for src/data/state.py — AgentState and operator.add reducer behaviour."""

import operator

from src.data.agent_result import NewsAgentResult
from src.data.enums import AgentName, NewsCategory
from src.data.plan import PlanStep, RoutingPlan
from src.data.state import AgentState
from src.data.user_query import UserQuery


def _query(text: str = "test") -> UserQuery:
    return UserQuery(text=text, session_id="s1")


def _result(agent: AgentName = AgentName.BUSINESS) -> NewsAgentResult:
    return NewsAgentResult(
        agent=agent,
        category=NewsCategory.BUSINESS,
        articles=[],
        query_used="test",
        success=True,
        latency_ms=0,
    )


class TestAgentState:
    def test_default_construction(self):
        state = AgentState(query=_query())
        assert state.plan is None
        assert state.agent_results == []
        assert state.final_response is None
        assert state.conversation_history == []
        assert state.error is None

    def test_operator_add_reducer_merges_lists(self):
        """The Annotated[..., operator.add] reducer must accumulate across branches."""
        list_a = [_result(AgentName.BUSINESS)]
        list_b = [_result(AgentName.SPORTS)]
        merged = operator.add(list_a, list_b)
        assert len(merged) == 2
        assert merged[0].agent == AgentName.BUSINESS
        assert merged[1].agent == AgentName.SPORTS

    def test_operator_add_empty_plus_list(self):
        assert operator.add([], [_result()]) == [_result()]

    def test_plan_can_be_set(self):
        state = AgentState(
            query=_query(),
            plan=RoutingPlan(
                steps=[PlanStep(agent=AgentName.BUSINESS, query="S&P 500")],
                original_query="S&P 500",
            ),
        )
        assert state.plan is not None
        assert len(state.plan.steps) == 1
