"""Integration tests for the full LangGraph graph (Phase 3 test gate)."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from src.data import AgentName, AgentState, NewsAgentResult, NewsCategory, NormalizedArticle
from src.data.enums import LLMProvider
from src.data.user_query import UserQuery
from src.utils.llm_factory import LLMFactory


def _article(agent: AgentName, provider: str) -> NormalizedArticle:
    cat_map = {
        AgentName.BUSINESS: NewsCategory.BUSINESS,
        AgentName.SPORTS: NewsCategory.SPORTS,
        AgentName.GENERAL: NewsCategory.GENERAL,
        AgentName.WEB_SEARCH: NewsCategory.WEB,
    }
    return NormalizedArticle(
        article_id=f"{provider}_1",
        title=f"Headline from {provider}",
        summary="Summary",
        url="https://example.com",
        source_name="TestSource",
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        category=cat_map[agent],
        provider=provider,
    )


def _llm_response(text: str) -> MagicMock:
    m = MagicMock()
    m.content = text
    return m


def _make_result(agent: AgentName, category: NewsCategory, provider: str) -> NewsAgentResult:
    return NewsAgentResult(
        agent=agent, category=category,
        articles=[_article(agent, provider)],
        query_used="q", success=True, latency_ms=10,
    )


class TestFullGraphSingleIntent:
    def test_single_business_intent_produces_one_section(self):
        from src.graph.builder import build_graph

        llm_text = "INTENT: Business News | QUERY: Nvidia earnings"
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _llm_response(llm_text)

        business_result = _make_result(AgentName.BUSINESS, NewsCategory.BUSINESS, "newsapi")

        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory, \
             patch("src.graph.agent_nodes.BusinessNewsAgent"):
            mock_factory.get.return_value = mock_llm
            graph = build_graph()

        state = AgentState(query=UserQuery(text="Nvidia earnings today?", session_id="s1"))

        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory, \
             patch("src.graph.agent_nodes.BusinessNewsAgent") as mock_biz:
            mock_factory.get.return_value = mock_llm
            mock_biz.return_value.fetch.return_value = business_result
            output = graph.invoke(state)

        response = output["final_response"]
        assert len(response.sections) == 1
        assert response.sections[0].agent == AgentName.BUSINESS


class TestFullGraphMultiIntent:
    def test_business_and_sports_produces_two_sections(self):
        from src.graph.builder import build_graph

        llm_text = (
            "INTENT: Business News | QUERY: Nvidia earnings\n"
            "INTENT: Sports News | QUERY: NBA playoffs"
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _llm_response(llm_text)

        business_result = _make_result(AgentName.BUSINESS, NewsCategory.BUSINESS, "newsapi")
        sports_result = _make_result(AgentName.SPORTS, NewsCategory.SPORTS, "guardian")

        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory, \
             patch("src.graph.agent_nodes.BusinessNewsAgent"), \
             patch("src.graph.agent_nodes.SportsNewsAgent"):
            mock_factory.get.return_value = mock_llm
            graph = build_graph()

        state = AgentState(query=UserQuery(text="Nvidia and NBA update", session_id="s2"))

        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory, \
             patch("src.graph.agent_nodes.BusinessNewsAgent") as mock_biz, \
             patch("src.graph.agent_nodes.SportsNewsAgent") as mock_spt:
            mock_factory.get.return_value = mock_llm
            mock_biz.return_value.fetch.return_value = business_result
            mock_spt.return_value.fetch.return_value = sports_result
            output = graph.invoke(state)

        response = output["final_response"]
        assert len(response.sections) == 2
        agents_in_sections = {s.agent for s in response.sections}
        assert AgentName.BUSINESS in agents_in_sections
        assert AgentName.SPORTS in agents_in_sections


class TestSupervisorZeroRegexMatches:
    def test_malformed_llm_output_falls_back_to_web_search(self):
        from src.data.enums import WebSearchProvider
        from src.graph.builder import build_graph

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _llm_response("I cannot process that request.")

        mock_settings = MagicMock()
        mock_settings.web_search_provider = WebSearchProvider.SERPAPI
        mock_settings.serpapi_api_key = "fake-key"
        mock_settings.max_articles_per_agent = 5

        mock_wrapper = MagicMock()
        mock_wrapper.results.return_value = {"organic_results": []}

        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory, \
             patch("src.graph.web_search_node.settings", mock_settings), \
             patch("src.graph.web_search_node.SerpAPIWrapper", return_value=mock_wrapper):
            mock_factory.get.return_value = mock_llm
            graph = build_graph()

        state = AgentState(query=UserQuery(text="test", session_id="s3"))

        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory, \
             patch("src.graph.web_search_node.settings", mock_settings), \
             patch("src.graph.web_search_node.SerpAPIWrapper", return_value=mock_wrapper):
            mock_factory.get.return_value = mock_llm
            output = graph.invoke(state)

        response = output["final_response"]
        assert len(response.sections) == 1
        assert response.sections[0].agent == AgentName.WEB_SEARCH


class TestAgentResultsReducer:
    def test_three_parallel_agents_accumulate_all_results(self):
        from src.graph.builder import build_graph

        llm_text = (
            "INTENT: Business News | QUERY: markets\n"
            "INTENT: Sports News | QUERY: NFL\n"
            "INTENT: General News | QUERY: world news"
        )
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = _llm_response(llm_text)

        biz = _make_result(AgentName.BUSINESS, NewsCategory.BUSINESS, "newsapi_biz")
        spt = _make_result(AgentName.SPORTS, NewsCategory.SPORTS, "guardian")
        gen = _make_result(AgentName.GENERAL, NewsCategory.GENERAL, "newsapi_gen")

        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory, \
             patch("src.graph.agent_nodes.BusinessNewsAgent"), \
             patch("src.graph.agent_nodes.SportsNewsAgent"), \
             patch("src.graph.agent_nodes.GeneralNewsAgent"):
            mock_factory.get.return_value = mock_llm
            graph = build_graph()

        state = AgentState(query=UserQuery(text="everything", session_id="s4"))

        with patch("src.graph.supervisor_node.LLMFactory") as mock_factory, \
             patch("src.graph.agent_nodes.BusinessNewsAgent") as mock_biz, \
             patch("src.graph.agent_nodes.SportsNewsAgent") as mock_spt, \
             patch("src.graph.agent_nodes.GeneralNewsAgent") as mock_gen:
            mock_factory.get.return_value = mock_llm
            mock_biz.return_value.fetch.return_value = biz
            mock_spt.return_value.fetch.return_value = spt
            mock_gen.return_value.fetch.return_value = gen
            output = graph.invoke(state)

        response = output["final_response"]
        assert len(response.sections) == 3


class TestLLMFactoryGroqRegistered:
    def test_groq_builder_can_be_registered(self):
        """GroqBuilder can be registered with LLMFactory, making GROQ available."""
        from src.utils.groq_builder import GroqBuilder

        LLMFactory.register(LLMProvider.GROQ, GroqBuilder())
        assert LLMProvider.GROQ in LLMFactory.supported_providers()
