"""Tests for src/graph/web_search_node.py — web_search_node."""

from unittest.mock import MagicMock, patch

from src.data import AgentName, AgentState
from src.data.enums import WebSearchProvider
from src.data.plan import PlanStep, RoutingPlan
from src.data.user_query import UserQuery
from src.graph.web_search_node import web_search_node


def _make_state(with_web_step: bool = True) -> AgentState:
    steps = (
        [PlanStep(agent=AgentName.WEB_SEARCH, query="Austin festivals")]
        if with_web_step
        else [PlanStep(agent=AgentName.BUSINESS, query="markets")]
    )
    return AgentState(
        query=UserQuery(text="Austin festivals", session_id="s1"),
        plan=RoutingPlan(steps=steps, original_query="Austin festivals"),
    )


def _serpapi_result() -> dict:
    return {"organic_results": [
        {"title": "Austin Event", "snippet": "Fun festival", "link": "https://example.com",
         "source": "EventSite", "date": "Apr 1, 2026"},
    ]}


def _serper_result() -> dict:
    return {"organic": [
        {"title": "Austin Fest", "snippet": "Great music", "link": "https://example.com/2",
         "source": "SerperSite"},
    ]}


class TestWebSearchNodeSerpAPI:
    def _settings(self, key: str = "fake-key"):
        s = MagicMock()
        s.web_search_provider = WebSearchProvider.SERPAPI
        s.serpapi_api_key = key
        s.serperdev_api_key = ""
        s.max_articles_per_agent = 5
        return s

    def test_missing_key_returns_failure(self):
        state = _make_state()
        with patch("src.graph.web_search_node.settings", self._settings(key="")):
            result = web_search_node(state)["agent_results"][0]
        assert result.success is False
        assert "not configured" in result.error_message

    def test_value_error_from_serpapi_returns_failure(self):
        state = _make_state()
        mock_wrapper = MagicMock()
        mock_wrapper.results.side_effect = ValueError("SerpAPI error payload")

        with patch("src.graph.web_search_node.settings", self._settings()), \
             patch("src.graph.web_search_node.SerpAPIWrapper", return_value=mock_wrapper):
            result = web_search_node(state)["agent_results"][0]

        assert result.success is False
        assert "SerpAPI error payload" in result.error_message

    def test_successful_call_returns_articles_with_serpapi_provider(self):
        state = _make_state()
        mock_wrapper = MagicMock()
        mock_wrapper.results.return_value = _serpapi_result()

        with patch("src.graph.web_search_node.settings", self._settings()), \
             patch("src.graph.web_search_node.SerpAPIWrapper", return_value=mock_wrapper):
            result = web_search_node(state)["agent_results"][0]

        assert result.success is True
        assert len(result.articles) == 1
        assert result.articles[0].provider == "serpapi"

    def test_empty_organic_results_returns_empty_articles(self):
        state = _make_state()
        mock_wrapper = MagicMock()
        mock_wrapper.results.return_value = {"organic_results": []}

        with patch("src.graph.web_search_node.settings", self._settings()), \
             patch("src.graph.web_search_node.SerpAPIWrapper", return_value=mock_wrapper):
            result = web_search_node(state)["agent_results"][0]

        assert result.success is True
        assert result.articles == []

    def test_generic_exception_returns_failure(self):
        state = _make_state()
        mock_wrapper = MagicMock()
        mock_wrapper.results.side_effect = RuntimeError("network error")

        with patch("src.graph.web_search_node.settings", self._settings()), \
             patch("src.graph.web_search_node.SerpAPIWrapper", return_value=mock_wrapper):
            result = web_search_node(state)["agent_results"][0]

        assert result.success is False

    def test_step_not_in_plan_returns_failure(self):
        state = _make_state(with_web_step=False)
        with patch("src.graph.web_search_node.settings", self._settings()):
            result = web_search_node(state)["agent_results"][0]
        assert result.success is False
        assert "not found" in result.error_message


class TestWebSearchNodeSerper:
    def _settings(self, key: str = "fake-serper-key"):
        s = MagicMock()
        s.web_search_provider = WebSearchProvider.SERPER
        s.serpapi_api_key = ""
        s.serperdev_api_key = key
        s.max_articles_per_agent = 5
        return s

    def test_missing_key_returns_failure(self):
        state = _make_state()
        with patch("src.graph.web_search_node.settings", self._settings(key="")):
            result = web_search_node(state)["agent_results"][0]
        assert result.success is False

    def test_http_200_with_message_error_body_returns_failure(self):
        state = _make_state()
        mock_wrapper = MagicMock()
        mock_wrapper.results.return_value = {"message": "Invalid API key"}

        with patch("src.graph.web_search_node.settings", self._settings()), \
             patch("src.graph.web_search_node.GoogleSerperAPIWrapper", return_value=mock_wrapper):
            result = web_search_node(state)["agent_results"][0]

        assert result.success is False
        assert "Invalid API key" in result.error_message

    def test_http_200_with_error_key_body_returns_failure(self):
        state = _make_state()
        mock_wrapper = MagicMock()
        mock_wrapper.results.return_value = {"error": "Unauthorized"}

        with patch("src.graph.web_search_node.settings", self._settings()), \
             patch("src.graph.web_search_node.GoogleSerperAPIWrapper", return_value=mock_wrapper):
            result = web_search_node(state)["agent_results"][0]

        assert result.success is False

    def test_successful_call_returns_articles_with_serper_provider(self):
        state = _make_state()
        mock_wrapper = MagicMock()
        mock_wrapper.results.return_value = _serper_result()

        with patch("src.graph.web_search_node.settings", self._settings()), \
             patch("src.graph.web_search_node.GoogleSerperAPIWrapper", return_value=mock_wrapper):
            result = web_search_node(state)["agent_results"][0]

        assert result.success is True
        assert result.articles[0].provider == "serper"

    def test_errors_list_key_returns_failure(self):
        state = _make_state()
        mock_wrapper = MagicMock()
        mock_wrapper.results.return_value = {"errors": ["Bad request"]}

        with patch("src.graph.web_search_node.settings", self._settings()), \
             patch("src.graph.web_search_node.GoogleSerperAPIWrapper", return_value=mock_wrapper):
            result = web_search_node(state)["agent_results"][0]

        assert result.success is False
