"""Tests for src/graph/supervisor_node.py — parsing, filtering, and supervisor_node."""

from unittest.mock import MagicMock, patch

from src.data import AgentName, AgentState, NewsCategory
from src.data.plan import PlanStep, RoutingPlan
from src.data.user_query import UserQuery
from src.graph.supervisor_node import (
    _categories_hint,
    _format_history,
    _parse_supervisor_steps,
    filter_plan_to_user_categories,
    supervisor_node,
)


class TestCategoriesHint:
    def test_empty_list_returns_all(self):
        assert _categories_hint([]) == "all"

    def test_single_category(self):
        result = _categories_hint([NewsCategory.SPORTS])
        assert result == "sports"

    def test_multiple_categories(self):
        result = _categories_hint([NewsCategory.BUSINESS, NewsCategory.SPORTS])
        assert "business" in result
        assert "sports" in result


class TestParseSupervisorSteps:
    def test_single_business_intent(self):
        raw = "INTENT: Business News | QUERY: AAPL stock price"
        steps = _parse_supervisor_steps(raw)
        assert len(steps) == 1
        assert steps[0].agent == AgentName.BUSINESS
        assert steps[0].query == "AAPL stock price"

    def test_business_and_sports_produces_correct_intent_set(self):
        raw = (
            "INTENT: Business News | QUERY: Nvidia earnings\n"
            "INTENT: Sports News | QUERY: NBA playoffs"
        )
        steps = _parse_supervisor_steps(raw)
        agents = {s.agent for s in steps}
        assert agents == {AgentName.BUSINESS, AgentName.SPORTS}

    def test_unknown_intent_label_maps_to_web_search(self):
        raw = "INTENT: Unknown Intent | QUERY: something weird"
        steps = _parse_supervisor_steps(raw)
        assert steps[0].agent == AgentName.WEB_SEARCH

    def test_zero_regex_matches_returns_empty_list(self):
        steps = _parse_supervisor_steps("This is not formatted output at all.")
        assert steps == []

    def test_three_distinct_intents(self):
        raw = (
            "INTENT: Business News | QUERY: Bitcoin price\n"
            "INTENT: Sports News | QUERY: World Cup\n"
            "INTENT: General News | QUERY: Ukraine conflict"
        )
        steps = _parse_supervisor_steps(raw)
        agents = {s.agent for s in steps}
        assert agents == {AgentName.BUSINESS, AgentName.SPORTS, AgentName.GENERAL}

    def test_web_search_intent(self):
        raw = "INTENT: Web Search | QUERY: how to bake sourdough"
        steps = _parse_supervisor_steps(raw)
        assert steps[0].agent == AgentName.WEB_SEARCH
        assert steps[0].query == "how to bake sourdough"


class TestFilterPlanToUserCategories:
    def _plan(self, agents: list[AgentName]) -> RoutingPlan:
        return RoutingPlan(
            steps=[PlanStep(agent=a, query=f"query for {a.value}") for a in agents],
            original_query="original query",
        )

    def test_empty_categories_returns_plan_unchanged(self):
        plan = self._plan([AgentName.BUSINESS, AgentName.SPORTS])
        result = filter_plan_to_user_categories(plan, [])
        assert result == plan

    def test_keeps_only_matching_agents(self):
        plan = self._plan([AgentName.BUSINESS, AgentName.SPORTS, AgentName.GENERAL])
        result = filter_plan_to_user_categories(plan, [NewsCategory.SPORTS])
        assert len(result.steps) == 1
        assert result.steps[0].agent == AgentName.SPORTS

    def test_all_filtered_falls_back_to_web_search(self):
        plan = self._plan([AgentName.BUSINESS, AgentName.SPORTS])
        result = filter_plan_to_user_categories(plan, [NewsCategory.GENERAL])
        assert len(result.steps) == 1
        assert result.steps[0].agent == AgentName.WEB_SEARCH
        assert result.steps[0].query == "original query"

    def test_web_category_allows_web_search_agent(self):
        plan = self._plan([AgentName.WEB_SEARCH])
        result = filter_plan_to_user_categories(plan, [NewsCategory.WEB])
        assert result.steps[0].agent == AgentName.WEB_SEARCH


class TestSupervisorNodeZeroMatches:
    def test_zero_regex_matches_defaults_to_web_search(self):
        """When LLM returns garbage, supervisor_node should fall back to web_search."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="This is not a valid formatted response.")
        state = AgentState(query=UserQuery(text="test query", session_id="s1"))
        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory:
            mock_factory.get.return_value = mock_llm
            result = supervisor_node(state)
        plan = result["plan"]
        assert len(plan.steps) == 1
        assert plan.steps[0].agent == AgentName.WEB_SEARCH
        assert plan.steps[0].query == "test query"


class TestFormatHistory:
    def test_empty_returns_no_prior_conversation(self):
        assert _format_history([]) == "No prior conversation."

    def test_single_user_turn(self):
        history = [{"role": "user", "content": "Tell me about markets"}]
        result = _format_history(history)
        assert "USER: Tell me about markets" in result

    def test_user_and_assistant_turns(self):
        history = [
            {"role": "user", "content": "NBA news"},
            {"role": "assistant", "content": "[Covered: NBA trade news]"},
        ]
        result = _format_history(history)
        assert "USER: NBA news" in result
        assert "ASSISTANT: [Covered: NBA trade news]" in result


class TestHistoryInjection:
    def _invoke_supervisor(self, state: AgentState) -> str:
        """Helper: run supervisor_node with mocked LLM, return the prompt string passed to invoke."""
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="INTENT: Web Search | QUERY: test")
        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory:
            mock_factory.get.return_value = mock_llm
            supervisor_node(state)
        return mock_llm.invoke.call_args[0][0]

    def test_history_injected_into_prompt(self):
        """Non-empty conversation_history appears in the formatted prompt."""
        history = [
            {"role": "user", "content": "Tell me about markets"},
            {"role": "assistant", "content": "[Covered: markets news]"},
        ]
        state = AgentState(
            query=UserQuery(text="follow up question", session_id="s1"),
            conversation_history=history,
        )
        call_arg = self._invoke_supervisor(state)
        assert "Tell me about markets" in call_arg
        assert "USER:" in call_arg

    def test_empty_history_renders_no_prior_conversation(self):
        """Empty conversation_history renders as 'No prior conversation.' in the prompt."""
        state = AgentState(query=UserQuery(text="test query", session_id="s1"))
        call_arg = self._invoke_supervisor(state)
        assert "No prior conversation." in call_arg

    def test_history_trimmed_before_injection(self):
        """trim_history() is applied; messages beyond MAX_HISTORY_TURNS cap are dropped."""
        # Zero-pad to avoid substring collisions (msg_000 is not in msg_010)
        history = [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"msg_{i:03d}"}
            for i in range(24)  # 12 turns — over the 10-turn cap
        ]
        state = AgentState(
            query=UserQuery(text="new query", session_id="s1"),
            conversation_history=history,
        )
        call_arg = self._invoke_supervisor(state)
        # After trimming 12 turns to 10, the first 2 turns (4 messages) are dropped
        assert "msg_000" not in call_arg
        assert "msg_001" not in call_arg
        assert "msg_002" not in call_arg
        assert "msg_003" not in call_arg
        # Most recent messages are retained
        assert "msg_022" in call_arg
        assert "msg_023" in call_arg
