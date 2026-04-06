"""Tests for src/data/agent_result.py — NewsAgentResult construction."""

from src.data.agent_result import NewsAgentResult
from src.data.enums import AgentName, NewsCategory


class TestNewsAgentResult:
    def test_success_result(self):
        r = NewsAgentResult(
            agent=AgentName.BUSINESS,
            category=NewsCategory.BUSINESS,
            articles=[],
            query_used="Apple earnings",
            success=True,
            latency_ms=320,
        )
        assert r.success is True
        assert r.error_message is None
        assert r.latency_ms == 320

    def test_failure_result_carries_error_message(self):
        r = NewsAgentResult(
            agent=AgentName.SPORTS,
            category=NewsCategory.SPORTS,
            articles=[],
            query_used="Premier League",
            success=False,
            error_message="HTTP 429 rate limit",
            latency_ms=105,
        )
        assert r.success is False
        assert "429" in r.error_message

    def test_articles_list_defaults_empty(self):
        r = NewsAgentResult(
            agent=AgentName.GENERAL,
            category=NewsCategory.GENERAL,
            articles=[],
            query_used="world news",
            success=True,
            latency_ms=0,
        )
        assert r.articles == []
