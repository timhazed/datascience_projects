"""Tests for src/agents/general_news_agent.py."""

from unittest.mock import MagicMock, patch

import httpx

from src.agents.general_news_agent import GeneralNewsAgent
from src.data.enums import AgentName, NewsCategory

_SETTINGS = MagicMock()
_SETTINGS.newsapi_api_key = "test-key"
_SETTINGS.max_articles_per_agent = 5

_RAW_ARTICLE = {
    "source": {"id": "reuters", "name": "Reuters"},
    "title": "UN Security Council meets over conflict",
    "description": "Emergency session called amid escalating tensions.",
    "url": "https://reuters.com/un-meeting",
    "urlToImage": None,
    "publishedAt": "2026-03-31T10:00:00Z",
    "content": None,
}


class TestGeneralNewsAgent:
    def test_fetch_returns_success_result_with_articles(self):
        with patch(
            "src.agents.general_news_agent.newsapi_client.fetch_articles",
            return_value=[_RAW_ARTICLE],
        ):
            agent = GeneralNewsAgent(settings=_SETTINGS)
            result = agent.fetch("UN Security Council")

        assert result.success is True
        assert result.agent == AgentName.GENERAL
        assert result.category == NewsCategory.GENERAL
        assert result.query_used == "UN Security Council"
        assert len(result.articles) == 1
        assert result.articles[0].title == "UN Security Council meets over conflict"
        assert result.articles[0].provider == "newsapi"
        assert result.articles[0].category == NewsCategory.GENERAL
        assert result.latency_ms >= 0

    def test_fetch_empty_results_returns_success_with_empty_articles(self):
        with patch(
            "src.agents.general_news_agent.newsapi_client.fetch_articles",
            return_value=[],
        ):
            agent = GeneralNewsAgent(settings=_SETTINGS)
            result = agent.fetch("noresults")

        assert result.success is True
        assert result.articles == []

    def test_fetch_http_status_error_returns_failure(self):
        fake_request = httpx.Request("GET", "https://newsapi.org/v2/everything")
        fake_response = httpx.Response(500, request=fake_request)
        exc = httpx.HTTPStatusError("Server error", request=fake_request, response=fake_response)

        with patch(
            "src.agents.general_news_agent.newsapi_client.fetch_articles",
            side_effect=exc,
        ):
            agent = GeneralNewsAgent(settings=_SETTINGS)
            result = agent.fetch("UN Security Council")

        assert result.success is False
        assert result.agent == AgentName.GENERAL
        assert result.articles == []
        assert result.error_message is not None
        assert result.latency_ms >= 0

    def test_category_is_general_not_business(self):
        """GeneralNewsAgent uses same endpoint as BusinessNewsAgent but assigns GENERAL category."""
        with patch(
            "src.agents.general_news_agent.newsapi_client.fetch_articles",
            return_value=[_RAW_ARTICLE],
        ):
            agent = GeneralNewsAgent(settings=_SETTINGS)
            result = agent.fetch("world news")

        assert result.category == NewsCategory.GENERAL
        assert result.articles[0].category == NewsCategory.GENERAL
