from enum import Enum


class NewsCategory(str, Enum):
    """Category assigned to a normalized article. Used for filtering and display."""

    BUSINESS = "business"
    SPORTS = "sports"
    GENERAL = "general"
    WEB = "web"


class LLMProvider(str, Enum):
    """
    Registered LLM providers. Adding a new provider requires only:
    1. Add the enum value here.
    2. Implement LLMProviderBuilder (src/utils/provider_builder.py).
    3. Call LLMFactory.register() once at app startup.
    No other code changes required.
    """

    OPENAI = "openai"
    GROQ = "groq"


class WebSearchProvider(str, Enum):
    """
    Registered web search providers. Exactly one is active per run, selected by
    WEB_SEARCH_PROVIDER in .env. Only that provider's API key is required.

    SERPAPI  — LangChain SerpAPIWrapper (google-search-results SDK)
    SERPER   — LangChain GoogleSerperAPIWrapper (serper.dev API)
    """

    SERPAPI = "serpapi"
    SERPER = "serper"


class AgentName(str, Enum):
    """
    The four routing destinations produced by the supervisor node.
    Maps directly to the agent_map in supervisor_node.

    business_agent, sports_agent, general_agent: custom domain agents (src/agents/).
    web_search_agent: LangGraph tool node — invokes SerpAPI via LangChain community tool,
    not a custom agent class.
    """

    BUSINESS = "business_agent"
    SPORTS = "sports_agent"
    GENERAL = "general_agent"
    WEB_SEARCH = "web_search_agent"
