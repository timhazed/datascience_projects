"""Tests for src/graph/agent_nodes.py — business_node, sports_node, general_node."""

from unittest.mock import patch

from src.data import AgentName, AgentState, NewsAgentResult, NewsCategory
from src.data.plan import PlanStep, RoutingPlan
from src.data.user_query import UserQuery
from src.graph.agent_nodes import business_node, general_node, sports_node


def _make_state(agents: list[AgentName]) -> AgentState:
    steps = [PlanStep(agent=a, query=f"query for {a.value}") for a in agents]
    return AgentState(
        query=UserQuery(text="test", session_id="s1"),
        plan=RoutingPlan(steps=steps, original_query="test"),
    )


def _mock_result(agent: AgentName, category: NewsCategory) -> NewsAgentResult:
    return NewsAgentResult(
        agent=agent, category=category,
        articles=[], query_used="q", success=True, latency_ms=10,
    )


class TestBusinessNode:
    def test_correct_agent_invoked(self):
        state = _make_state([AgentName.BUSINESS])
        mock_result = _mock_result(AgentName.BUSINESS, NewsCategory.BUSINESS)

        with patch("src.graph.agent_nodes.BusinessNewsAgent") as mock_cls:
            mock_cls.return_value.fetch.return_value = mock_result
            patch_dict = business_node(state)

        mock_cls.assert_called_once()
        mock_cls.return_value.fetch.assert_called_once_with("query for business_agent")
        assert patch_dict["agent_results"] == [mock_result]

    def test_step_not_in_plan_returns_failure(self):
        state = _make_state([AgentName.SPORTS])  # No BUSINESS step
        patch_dict = business_node(state)
        result = patch_dict["agent_results"][0]
        assert result.success is False
        assert result.agent == AgentName.BUSINESS
        assert "not found" in result.error_message


class TestSportsNode:
    def test_correct_agent_invoked(self):
        state = _make_state([AgentName.SPORTS])
        mock_result = _mock_result(AgentName.SPORTS, NewsCategory.SPORTS)

        with patch("src.graph.agent_nodes.SportsNewsAgent") as mock_cls:
            mock_cls.return_value.fetch.return_value = mock_result
            patch_dict = sports_node(state)

        mock_cls.return_value.fetch.assert_called_once_with("query for sports_agent")
        assert patch_dict["agent_results"] == [mock_result]

    def test_step_not_in_plan_returns_failure(self):
        state = _make_state([AgentName.BUSINESS])
        patch_dict = sports_node(state)
        assert patch_dict["agent_results"][0].success is False


class TestGeneralNode:
    def test_correct_agent_invoked(self):
        state = _make_state([AgentName.GENERAL])
        mock_result = _mock_result(AgentName.GENERAL, NewsCategory.GENERAL)

        with patch("src.graph.agent_nodes.GeneralNewsAgent") as mock_cls:
            mock_cls.return_value.fetch.return_value = mock_result
            patch_dict = general_node(state)

        mock_cls.return_value.fetch.assert_called_once_with("query for general_agent")
        assert patch_dict["agent_results"] == [mock_result]

    def test_step_not_in_plan_returns_failure(self):
        state = _make_state([AgentName.WEB_SEARCH])
        patch_dict = general_node(state)
        assert patch_dict["agent_results"][0].success is False
