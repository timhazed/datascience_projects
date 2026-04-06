import logging
import time

import httpx

from src.agents.base_news_agent import BaseNewsAgent
from src.api import espn_client, guardian_client
from src.data.agent_result import NewsAgentResult
from src.data.enums import AgentName, NewsCategory
from src.utils.normalizer import normalize_espn, normalize_guardian

logger = logging.getLogger(__name__)


class SportsNewsAgent(BaseNewsAgent):
    """Fetches sports content from Guardian (editorial) and ESPN (scores) in parallel.

    Both sources are always queried. Guardian articles appear first in the merged
    result set; ESPN score cards follow. Guardian HTTP errors are demoted to
    warnings — partial results from ESPN are still returned with success=True.
    An empty slate from both sources is also success=True (valid off-season state).
    """

    def __init__(self, settings) -> None:
        """
        Args:
            settings: Application settings instance (provides API keys and limits).
        """
        self._settings = settings

    def fetch(self, query: str) -> NewsAgentResult:
        """Fetch and normalize sports content for the given query.

        Returns NewsAgentResult(success=True) in all cases — empty articles on
        a quiet day or when both sources are unavailable is not an error.
        Guardian HTTPStatusError is logged as a warning; ESPN results are still
        returned if available.
        """
        start = time.monotonic()
        guardian_articles = []
        espn_articles = []

        # --- Guardian: editorial sports news ---
        try:
            raw = guardian_client.fetch_articles(
                query,
                api_key=self._settings.guardian_api_key,
                page_size=self._settings.max_articles_per_agent,
            )
            guardian_articles = [
                normalize_guardian(r, category=NewsCategory.SPORTS, index=i)
                for i, r in enumerate(raw)
            ]
        except httpx.HTTPStatusError as exc:
            logger.warning("SportsNewsAgent: Guardian error for %r — %s", query, exc)

        # --- ESPN: today's scores ---
        raw_espn = espn_client.fetch_scores(
            query, limit=self._settings.max_articles_per_agent
        )
        espn_articles = [normalize_espn(e, index=i) for i, e in enumerate(raw_espn)]

        latency_ms = int((time.monotonic() - start) * 1000)
        return NewsAgentResult(
            agent=AgentName.SPORTS,
            category=NewsCategory.SPORTS,
            articles=guardian_articles + espn_articles,
            query_used=query,
            success=True,
            latency_ms=latency_ms,
        )
