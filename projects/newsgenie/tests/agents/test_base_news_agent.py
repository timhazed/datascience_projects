"""Tests for src/agents/base_news_agent.py — ABC cannot be instantiated directly."""

import pytest

from src.agents.base_news_agent import BaseNewsAgent


class TestBaseNewsAgent:
    def test_cannot_instantiate_directly(self):
        """BaseNewsAgent is abstract — direct instantiation must raise TypeError."""
        with pytest.raises(TypeError):
            BaseNewsAgent()  # type: ignore[abstract]

    def test_concrete_subclass_without_fetch_raises(self):
        """A subclass that does not implement fetch() is still abstract."""
        class Incomplete(BaseNewsAgent):
            pass

        with pytest.raises(TypeError):
            Incomplete()  # type: ignore[abstract]

    def test_concrete_subclass_with_fetch_instantiates(self):
        """A subclass that implements fetch() can be instantiated."""
        from src.data.agent_result import NewsAgentResult
        from src.data.enums import AgentName, NewsCategory

        class Stub(BaseNewsAgent):
            def fetch(self, query: str) -> NewsAgentResult:
                return NewsAgentResult(
                    agent=AgentName.BUSINESS,
                    category=NewsCategory.BUSINESS,
                    articles=[],
                    query_used=query,
                    success=True,
                    latency_ms=0,
                )

        agent = Stub()
        result = agent.fetch("test")
        assert result.success is True
