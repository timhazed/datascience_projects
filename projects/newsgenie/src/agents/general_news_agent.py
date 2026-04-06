import logging
import time

import httpx

from src.agents.base_news_agent import BaseNewsAgent
from src.api import newsapi_client
from src.data.agent_result import NewsAgentResult
from src.data.enums import AgentName, NewsCategory
from src.utils.normalizer import normalize_newsapi

logger = logging.getLogger(__name__)


class GeneralNewsAgent(BaseNewsAgent):
    """Fetches general/world news articles from NewsAPI and normalizes them.

    Shares the same NewsAPI endpoint as BusinessNewsAgent; the supervisor intent
    label (General News vs Business News) is the only distinguishing factor.
    """

    def __init__(self, settings) -> None:
        """
        Args:
            settings: Application settings instance (provides API keys and limits).
        """
        self._settings = settings

    def fetch(self, query: str) -> NewsAgentResult:
        """Fetch and normalize general/world news articles for the given query.

        Returns NewsAgentResult(success=False) if the API key is missing,
        results are empty, or the API call fails after retries.
        """
        start = time.monotonic()
        try:
            raw_articles = newsapi_client.fetch_articles(
                query,
                api_key=self._settings.newsapi_api_key,
                page_size=self._settings.max_articles_per_agent,
            )
        except httpx.HTTPStatusError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            logger.error("GeneralNewsAgent: API error for %r — %s", query, exc)
            return NewsAgentResult(
                agent=AgentName.GENERAL,
                category=NewsCategory.GENERAL,
                articles=[],
                query_used=query,
                success=False,
                error_message=repr(exc),
                latency_ms=latency_ms,
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        articles = [
            normalize_newsapi(raw, category=NewsCategory.GENERAL, index=i)
            for i, raw in enumerate(raw_articles)
        ]
        return NewsAgentResult(
            agent=AgentName.GENERAL,
            category=NewsCategory.GENERAL,
            articles=articles,
            query_used=query,
            success=True,
            latency_ms=latency_ms,
        )
