"""SearchProvider — SerpAPI / Serper web search abstraction.

Provides a single `search()` interface that routes to the active provider
configured in `settings.search.provider`. A domain whitelist filter is applied
before returning results, so callers always receive results from trusted
medical sources only.

Environment variables consumed:
    SERPAPI_API_KEY  — required when provider="serpapi"
    SERPER_API_KEY   — required when provider="serper"
"""

from __future__ import annotations

import logging
import os
from urllib.parse import urlparse

import httpx
from serpapi import GoogleSearch

from src.models.search_result_item import SearchResultItem

logger = logging.getLogger(__name__)

# Medical domain whitelist — default filter applied when no sources_filter is provided.
TRUSTED_MEDICAL_DOMAINS: frozenset[str] = frozenset({
    "medlineplus.gov",
    "who.int",
    "pubmed.ncbi.nlm.nih.gov",
    "cdc.gov",
    "mayoclinic.org",
    "webmd.com",
    "nih.gov",
    "ncbi.nlm.nih.gov",
    "nhs.uk",               # Exp 2 Rec 4 — UK clinical guidelines (peer-reviewed authority)
    "uptodate.com",         # Clinical decision support used by practitioners
    "nejm.org",             # New England Journal of Medicine
})

_SERPER_URL = "https://google.serper.dev/search"
_SERPER_TIMEOUT = 10.0  # seconds


class SearchProvider:
    """Web search abstraction supporting SerpAPI and Serper providers.

    Reads the provider name at construction time and routes all `search()` calls
    to the appropriate backend. A domain whitelist filter is applied before
    returning results — callers never receive results from untrusted domains.

    Only one public method: `search()`. Backend methods are private.
    """

    @staticmethod
    def _extract_domain(url: str) -> str:
        """Extract the registrable domain from a URL, stripping the www. prefix.

        Args:
            url: Full URL string, e.g. 'https://www.medlineplus.gov/article.html'.

        Returns:
            Domain string, e.g. 'medlineplus.gov'.
        """
        netloc = urlparse(url).netloc
        return netloc.removeprefix("www.")

    def __init__(self, provider: str, api_key: str | None = None) -> None:
        """Initialise the search provider.

        Args:
            provider: Active provider name — "serpapi" or "serper".
            api_key: Optional API key override. If None, reads from the
                corresponding environment variable (SERPAPI_API_KEY or
                SERPER_API_KEY).

        Raises:
            ValueError: If provider is not "serpapi" or "serper".
        """
        if provider not in ("serpapi", "serper"):
            raise ValueError(
                f"Unsupported search provider: {provider!r}. "
                "Expected 'serpapi' or 'serper'."
            )
        self._provider = provider
        # Resolve API key: explicit override → env var → empty string (will fail at runtime)
        env_key = f"{provider.upper()}_API_KEY"
        self._api_key: str = api_key or os.getenv(env_key, "")
        if not self._api_key:
            logger.warning(
                "[SearchProvider] No API key found for %s (env: %s) — "
                "live search calls will fail.",
                provider,
                env_key,
            )

    def search(
        self,
        query: str,
        max_results: int = 5,
        sources_filter: list[str] | None = None,
    ) -> list[SearchResultItem]:
        """Run a web search and return domain-filtered results.

        Calls the active provider backend, then applies the domain whitelist
        filter before returning. Results whose `source_domain` is not in
        `sources_filter` are dropped. Returns an empty list (not an error)
        when all results are filtered out.

        Args:
            query: Clinical search query string.
            max_results: Maximum number of raw results to request from the
                provider. The returned list may be shorter after filtering.
            sources_filter: List of trusted domain strings to keep. Defaults
                to TRUSTED_MEDICAL_DOMAINS when None.

        Returns:
            List of SearchResultItem instances, filtered to trusted domains only.
        """
        domains = (
            frozenset(sources_filter) if sources_filter is not None else TRUSTED_MEDICAL_DOMAINS
        )

        if self._provider == "serpapi":
            raw = self._search_serpapi(query, max_results)
        else:
            raw = self._search_serper(query, max_results)

        # Apply domain whitelist — drop any result not in the trusted set
        filtered = [r for r in raw if r.source_domain in domains]
        logger.info(
            "[SearchProvider] %s: %d raw → %d after domain filter for query %r",
            self._provider,
            len(raw),
            len(filtered),
            query,
        )
        return filtered

    def _search_serpapi(self, query: str, max_results: int) -> list[SearchResultItem]:
        """Call SerpAPI via the google-search-results SDK.

        Args:
            query: Search query string.
            max_results: Number of results to request.

        Returns:
            Unfiltered list of SearchResultItem instances.
        """
        params = {
            "api_key": self._api_key,
            "q": query,
            "num": max_results,
            "engine": "google",
        }
        results = GoogleSearch(params).get_dict()
        items: list[SearchResultItem] = []
        for r in results.get("organic_results", [])[:max_results]:
            url = r.get("link", "")
            items.append(
                SearchResultItem(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("snippet", ""),
                    source_domain=self._extract_domain(url),
                )
            )
        return items

    def _search_serper(self, query: str, max_results: int) -> list[SearchResultItem]:
        """Call Serper via httpx POST to the JSON search endpoint.

        Args:
            query: Search query string.
            max_results: Number of results to request.

        Returns:
            Unfiltered list of SearchResultItem instances.
        """
        with httpx.Client(timeout=_SERPER_TIMEOUT) as client:
            response = client.post(
                _SERPER_URL,
                headers={
                    "X-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": max_results},
            )
        response.raise_for_status()
        data = response.json()
        items: list[SearchResultItem] = []
        for r in data.get("organic", [])[:max_results]:
            url = r.get("link", "")
            items.append(
                SearchResultItem(
                    title=r.get("title", ""),
                    url=url,
                    snippet=r.get("snippet", ""),
                    source_domain=self._extract_domain(url),
                )
            )
        return items
