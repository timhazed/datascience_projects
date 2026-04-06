"""Tests for src/agents/business_agent.py."""

from unittest.mock import MagicMock, patch

import httpx

from src.agents.business_agent import BusinessNewsAgent
from src.data.enums import AgentName, NewsCategory

# ---------------------------------------------------------------------------
# Minimal settings stub — only the fields the agent reads
# ---------------------------------------------------------------------------

_SETTINGS = MagicMock()
_SETTINGS.newsapi_api_key = "test-key"
_SETTINGS.max_articles_per_agent = 5

_RAW_ARTICLE = {
    "source": {"id": "bbc-news", "name": "BBC News"},
    "title": "Nvidia earnings beat expectations",
    "description": "Chipmaker posts record Q1 revenue.",
    "url": "https://bbc.com/nvidia",
    "urlToImage": None,
    "publishedAt": "2026-03-31T12:00:00Z",
    "content": None,
}


class TestBusinessNewsAgent:
    def test_fetch_returns_success_result_with_articles(self):
        with patch(
            "src.agents.business_agent.newsapi_client.fetch_articles",
            return_value=[_RAW_ARTICLE],
        ):
            agent = BusinessNewsAgent(settings=_SETTINGS)
            result = agent.fetch("Nvidia earnings")

        assert result.success is True
        assert result.agent == AgentName.BUSINESS
        assert result.category == NewsCategory.BUSINESS
        assert result.query_used == "Nvidia earnings"
        assert len(result.articles) == 1
        assert result.articles[0].title == "Nvidia earnings beat expectations"
        assert result.articles[0].provider == "newsapi"
        assert result.articles[0].category == NewsCategory.BUSINESS
        assert result.latency_ms >= 0

    def test_fetch_empty_results_returns_success_with_empty_articles(self):
        with patch(
            "src.agents.business_agent.newsapi_client.fetch_articles",
            return_value=[],
        ):
            agent = BusinessNewsAgent(settings=_SETTINGS)
            result = agent.fetch("noresults")

        assert result.success is True
        assert result.articles == []

    def test_fetch_http_status_error_returns_failure(self):
        fake_request = httpx.Request("GET", "https://newsapi.org/v2/everything")
        fake_response = httpx.Response(500, request=fake_request)
        exc = httpx.HTTPStatusError("Server error", request=fake_request, response=fake_response)

        with patch(
            "src.agents.business_agent.newsapi_client.fetch_articles",
            side_effect=exc,
        ):
            agent = BusinessNewsAgent(settings=_SETTINGS)
            result = agent.fetch("Nvidia earnings")

        assert result.success is False
        assert result.agent == AgentName.BUSINESS
        assert result.articles == []
        assert result.error_message is not None
        assert result.latency_ms >= 0

    def test_multiple_articles_all_normalized(self):
        raw2 = {**_RAW_ARTICLE, "title": "Bitcoin hits 100k", "url": "https://bbc.com/btc"}
        with patch(
            "src.agents.business_agent.newsapi_client.fetch_articles",
            return_value=[_RAW_ARTICLE, raw2],
        ):
            agent = BusinessNewsAgent(settings=_SETTINGS)
            result = agent.fetch("crypto")

        assert len(result.articles) == 2
        assert result.articles[0].article_id == "newsapi_0"
        assert result.articles[1].article_id == "newsapi_1"
