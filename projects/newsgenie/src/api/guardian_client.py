"""HTTP client for The Guardian Open Platform search API.

Used by sports_agent (Sports News).

Reference: https://open-platform.theguardian.com/documentation/search
"""

import logging
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

_BASE_URL = "https://content.guardianapis.com/search"
_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


def _is_transient(exc: BaseException) -> bool:
    """Return True for errors that warrant a retry (network/rate-limit; not auth/logic)."""
    if isinstance(exc, httpx.TimeoutException):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return isinstance(exc, httpx.NetworkError)


@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def fetch_articles(
    query: str,
    api_key: str,
    page_size: int = 5,
    *,
    client: httpx.Client | None = None,
) -> list[dict[str, Any]]:
    """Fetch articles from Guardian Open Platform for a given query.

    Returns a list of raw result dicts in the Guardian API response shape.
    Returns an empty list on auth errors (401/403) or when no results are found.
    Raises httpx.HTTPStatusError for 5xx errors after retries are exhausted.

    Args:
        query: Search query string.
        api_key: Guardian API key. If empty, returns an empty list immediately.
        page_size: Max number of results to return (capped at 50 by the API).
        client: Optional httpx.Client for testing/injection. A new client is created if None.
    """
    if not api_key:
        logger.warning("guardian_client: GUARDIAN_API_KEY not set — returning empty results")
        return []

    params = {
        "api-key": api_key,
        "q": query,
        "page-size": page_size,
        "show-fields": "trailText,thumbnail,headline",
    }

    _client = client or httpx.Client(timeout=_TIMEOUT)
    try:
        response = _client.get(_BASE_URL, params=params)
    finally:
        if client is None:
            _client.close()

    if response.status_code in (401, 403):
        logger.error(
            "guardian_client: auth error %d — check GUARDIAN_API_KEY", response.status_code
        )
        return []

    if _is_transient(
        httpx.HTTPStatusError("", request=response.request, response=response)
    ) and response.is_error:
        response.raise_for_status()

    response.raise_for_status()
    data = response.json()
    return data.get("response", {}).get("results", [])
