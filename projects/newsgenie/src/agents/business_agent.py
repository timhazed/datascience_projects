import logging
import time

import httpx

from src.agents.base_news_agent import BaseNewsAgent
from src.api import newsapi_client
from src.data.agent_result import NewsAgentResult
from src.data.enums import AgentName, NewsCategory
from src.utils.normalizer import normalize_newsapi

logger = logging.getLogger(__name__)


class BusinessNewsAgent(BaseNewsAgent):
    """Fetches business news articles from NewsAPI and normalizes them.

    Business News covers: company earnings, stock prices, cryptocurrency,
    inflation, Federal Reserve, exchange rates, supply chain disruptions,
    and semiconductor industry news.
    """

    def __init__(self, settings) -> None:
        """
        Args:
            settings: Application settings instance (provides API keys and limits).
        """
        self._settings = settings

    def fetch(self, query: str) -> NewsAgentResult:
        """Fetch and normalize business news articles for the given query.

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
            logger.error("BusinessNewsAgent: API error for %r — %s", query, exc)
            return NewsAgentResult(
                agent=AgentName.BUSINESS,
                category=NewsCategory.BUSINESS,
                articles=[],
                query_used=query,
                success=False,
                error_message=repr(exc),
                latency_ms=latency_ms,
            )

        latency_ms = int((time.monotonic() - start) * 1000)
        articles = [
            normalize_newsapi(raw, category=NewsCategory.BUSINESS, index=i)
            for i, raw in enumerate(raw_articles)
        ]
        return NewsAgentResult(
            agent=AgentName.BUSINESS,
            category=NewsCategory.BUSINESS,
            articles=articles,
            query_used=query,
            success=True,
            latency_ms=latency_ms,
        )
