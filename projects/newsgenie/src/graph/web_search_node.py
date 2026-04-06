import time

from langchain_community.utilities import GoogleSerperAPIWrapper, SerpAPIWrapper

from src.data import AgentName, AgentState, NewsAgentResult, NewsCategory, NormalizedArticle
from src.data.enums import WebSearchProvider
from src.utils.config import settings


def web_search_node(state: AgentState) -> dict:
    """
    Execute a web search for the Web Search plan step using the configured provider.

    Provider selection (from settings.web_search_provider):
      SERPAPI → SerpAPIWrapper(serpapi_api_key=...), results key "organic_results"
      SERPER  → GoogleSerperAPIWrapper(serper_api_key=...), results key "organic"

    Returns NewsAgentResult(success=False) if the key is absent or any call fails.

    Guard order (single-exit pattern):
      1. Resolve provider + key; missing key → early return (top-of-function guard)
      2. Missing plan step → early return (defensive guard)
      3. Web search call inside try/except → single return at bottom
    """
    # Guard 1 — resolve provider and API key; return immediately if key absent.
    web_provider = settings.web_search_provider
    api_key = (
        settings.serpapi_api_key
        if web_provider == WebSearchProvider.SERPAPI
        else settings.serperdev_api_key
    )
    if not api_key:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
            articles=[], query_used="", success=False,
            error_message=f"{web_provider.value.upper()}_API_KEY not configured", latency_ms=0,
        )]}

    # Guard 2 — step not in plan (defensive; LangGraph guarantees routing, but guard regardless).
    step = next((s for s in state.plan.steps if s.agent == AgentName.WEB_SEARCH), None)
    if step is None:
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
            articles=[], query_used="", success=False,
            error_message="Step not found in plan", latency_ms=0,
        )]}

    # Construct wrapper and determine results key for the active provider.
    if web_provider == WebSearchProvider.SERPAPI:
        wrapper = SerpAPIWrapper(serpapi_api_key=api_key)
        results_key = "organic_results"
    else:
        wrapper = GoogleSerperAPIWrapper(serper_api_key=api_key)
        results_key = "organic"
    provider_label = web_provider.value

    start = time.monotonic()
    try:
        raw = wrapper.results(step.query)
    except ValueError as exc:
        # SerpAPI only: _process_response raises ValueError when response contains an "error" key.
        latency_ms = int((time.monotonic() - start) * 1000)
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
            articles=[], query_used=step.query, success=False,
            error_message=str(exc), latency_ms=latency_ms,
        )]}
    except Exception as exc:
        # Network failure, timeout, or unexpected error from either provider.
        latency_ms = int((time.monotonic() - start) * 1000)
        return {"agent_results": [NewsAgentResult(
            agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
            articles=[], query_used=step.query, success=False,
            error_message=repr(exc), latency_ms=latency_ms,
        )]}

    latency_ms = int((time.monotonic() - start) * 1000)

    # Serper guard — HTTP 200 with error in JSON body (e.g. invalid key returns
    # {"message": "Invalid API key"} at 200). SerpAPI raises ValueError instead.
    if web_provider == WebSearchProvider.SERPER:
        for key in ("message", "error", "errors"):
            val = raw.get(key) if isinstance(raw, dict) else None
            err_text = (
                val if isinstance(val, str) and val.strip()
                else val[0] if isinstance(val, list) and val and isinstance(val[0], str)
                else None
            )
            if err_text:
                return {"agent_results": [NewsAgentResult(
                    agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
                    articles=[], query_used=step.query, success=False,
                    error_message=f"Serper API error: {err_text}", latency_ms=latency_ms,
                )]}

    articles = [
        NormalizedArticle(
            article_id=f"{provider_label}_{i}",
            title=r.get("title", ""),
            summary=r.get("snippet", r.get("title", "")),
            url=r.get("link", ""),
            source_name=r.get("source", "Web"),
            # published_at: NormalizedArticle.parse_published_at handles arbitrary date strings.
            # Serper does not return a date field; epoch fallback is intentional.
            published_at=r.get("date", "1970-01-01T00:00:00Z"),
            category=NewsCategory.WEB,
            provider=provider_label,
        )
        for i, r in enumerate(raw.get(results_key, [])[:settings.max_articles_per_agent])
    ]
    return {"agent_results": [NewsAgentResult(
        agent=AgentName.WEB_SEARCH, category=NewsCategory.WEB,
        articles=articles, query_used=step.query,
        success=True, latency_ms=latency_ms,
    )]}
