# Three graph node functions — one per domain agent.
# Each instantiates its agent, calls fetch(), and returns a state patch.
# HTTP clients (newsapi_client, guardian_client) are module-level singletons in src/api/.

from src.agents.business_agent import BusinessNewsAgent
from src.agents.general_news_agent import GeneralNewsAgent
from src.agents.sports_agent import SportsNewsAgent
from src.data import AgentName, AgentState, NewsAgentResult, NewsCategory
from src.utils.config import settings


def business_node(state: AgentState) -> dict:
    """Fetch and normalize Business News articles for the plan step assigned to this agent."""
    step = next((s for s in state.plan.steps if s.agent == AgentName.BUSINESS), None)
    if step is None:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.BUSINESS, category=NewsCategory.BUSINESS,
            articles=[], query_used="", success=False,
            error_message="Step not found in plan", latency_ms=0,
        )]}
    result = BusinessNewsAgent(settings=settings).fetch(step.query)
    return {"agent_results": [result]}


def sports_node(state: AgentState) -> dict:
    """Fetch and normalize Sports News articles for the plan step assigned to this agent."""
    step = next((s for s in state.plan.steps if s.agent == AgentName.SPORTS), None)
    if step is None:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.SPORTS, category=NewsCategory.SPORTS,
            articles=[], query_used="", success=False,
            error_message="Step not found in plan", latency_ms=0,
        )]}
    result = SportsNewsAgent(settings=settings).fetch(step.query)
    return {"agent_results": [result]}


def general_node(state: AgentState) -> dict:
    """Fetch and normalize General/World News articles for the plan step assigned to this agent."""
    step = next((s for s in state.plan.steps if s.agent == AgentName.GENERAL), None)
    if step is None:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.GENERAL, category=NewsCategory.GENERAL,
            articles=[], query_used="", success=False,
            error_message="Step not found in plan", latency_ms=0,
        )]}
    result = GeneralNewsAgent(settings=settings).fetch(step.query)
    return {"agent_results": [result]}
