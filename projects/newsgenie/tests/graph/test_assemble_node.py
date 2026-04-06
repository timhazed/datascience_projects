"""Tests for src/graph/assemble_node.py — assemble."""

from datetime import UTC, datetime

from src.data import AgentName, AgentState, NewsAgentResult, NewsCategory, NormalizedArticle
from src.data.plan import PlanStep, RoutingPlan
from src.data.user_query import UserQuery
from src.graph.assemble_node import assemble


def _article(provider: str, category: NewsCategory = NewsCategory.BUSINESS) -> NormalizedArticle:
    return NormalizedArticle(
        article_id=f"{provider}_1",
        title="Headline",
        summary="Summary",
        url="https://example.com/article",
        source_name="TestSource",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        category=category,
        provider=provider,
    )


def _make_state(steps: list[PlanStep], agent_results: list[NewsAgentResult]) -> AgentState:
    return AgentState(
        query=UserQuery(text="test", session_id="s1"),
        plan=RoutingPlan(steps=steps, original_query="test"),
        agent_results=agent_results,
    )


class TestAssemble:
    def test_articles_matched_to_correct_plan_step(self):
        steps = [PlanStep(agent=AgentName.BUSINESS, query="markets")]
        article = _article("newsapi")
        result = [NewsAgentResult(
            agent=AgentName.BUSINESS, category=NewsCategory.BUSINESS,
            articles=[article], query_used="markets", success=True, latency_ms=100,
        )]
        state = _make_state(steps, result)
        patch = assemble(state)
        response = patch["final_response"]
        assert len(response.sections) == 1
        assert response.sections[0].articles == [article]

    def test_empty_agent_results_produces_empty_section(self):
        steps = [PlanStep(agent=AgentName.SPORTS, query="NFL")]
        result = [NewsAgentResult(
            agent=AgentName.SPORTS, category=NewsCategory.SPORTS,
            articles=[], query_used="NFL", success=False,
            error_message="no key", latency_ms=0,
        )]
        state = _make_state(steps, result)
        patch = assemble(state)
        response = patch["final_response"]
        assert response.sections[0].articles == []
        assert response.fallback_used is True

    def test_sources_used_deduplicated_and_sorted(self):
        steps = [
            PlanStep(agent=AgentName.BUSINESS, query="markets"),
            PlanStep(agent=AgentName.SPORTS, query="NFL"),
        ]
        results = [
            NewsAgentResult(
                agent=AgentName.BUSINESS, category=NewsCategory.BUSINESS,
                articles=[_article("newsapi"), _article("newsapi")],
                query_used="markets", success=True, latency_ms=50,
            ),
            NewsAgentResult(
                agent=AgentName.SPORTS, category=NewsCategory.SPORTS,
                articles=[_article("guardian", NewsCategory.SPORTS)],
                query_used="NFL", success=True, latency_ms=60,
            ),
        ]
        state = _make_state(steps, results)
        patch = assemble(state)
        response = patch["final_response"]
        assert response.sources_used == sorted({"newsapi", "guardian"})

    def test_fallback_used_false_when_all_sections_have_articles(self):
        steps = [PlanStep(agent=AgentName.BUSINESS, query="markets")]
        results = [NewsAgentResult(
            agent=AgentName.BUSINESS, category=NewsCategory.BUSINESS,
            articles=[_article("newsapi")], query_used="markets", success=True, latency_ms=50,
        )]
        state = _make_state(steps, results)
        response = assemble(state)["final_response"]
        assert response.fallback_used is False

    def test_session_id_propagated(self):
        steps = [PlanStep(agent=AgentName.BUSINESS, query="x")]
        results = [NewsAgentResult(
            agent=AgentName.BUSINESS, category=NewsCategory.BUSINESS,
            articles=[], query_used="x", success=True, latency_ms=0,
        )]
        state = AgentState(
            query=UserQuery(text="x", session_id="my-session"),
            plan=RoutingPlan(steps=steps, original_query="x"),
            agent_results=results,
        )
        response = assemble(state)["final_response"]
        assert response.session_id == "my-session"
