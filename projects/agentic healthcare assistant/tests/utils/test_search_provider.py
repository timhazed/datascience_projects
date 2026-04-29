"""Tests for SearchProvider — SerpAPI and Serper paths mocked; no live HTTP calls."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.utils.search_provider import TRUSTED_MEDICAL_DOMAINS, SearchProvider

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_invalid_provider_raises_value_error() -> None:
    """Unknown provider name raises ValueError at construction time."""
    with pytest.raises(ValueError, match="Unsupported search provider"):
        SearchProvider("bing")


def test_serpapi_provider_constructs() -> None:
    """'serpapi' is a valid provider — no exception."""
    sp = SearchProvider("serpapi", api_key="test-key")
    assert sp._provider == "serpapi"


def test_serper_provider_constructs() -> None:
    """'serper' is a valid provider — no exception."""
    sp = SearchProvider("serper", api_key="test-key")
    assert sp._provider == "serper"


def test_api_key_override_used_over_env() -> None:
    """Explicit api_key kwarg takes precedence over environment variable."""
    sp = SearchProvider("serpapi", api_key="override-key")
    assert sp._api_key == "override-key"


def test_api_key_read_from_env(monkeypatch) -> None:
    """API key is read from SERPAPI_API_KEY env var when not explicitly provided."""
    monkeypatch.setenv("SERPAPI_API_KEY", "env-key-123")
    sp = SearchProvider("serpapi")
    assert sp._api_key == "env-key-123"


# ---------------------------------------------------------------------------
# SerpAPI path
# ---------------------------------------------------------------------------


def _serpapi_result(title: str, url: str, snippet: str) -> dict:
    """Helper to build a minimal SerpAPI organic result dict."""
    return {"title": title, "link": url, "snippet": snippet}


def test_serpapi_returns_filtered_results() -> None:
    """SerpAPI results with trusted domains are returned; untrusted are dropped."""
    fake_results = {
        "organic_results": [
            _serpapi_result(
                "CKD overview",
                "https://medlineplus.gov/ckd.html",
                "Chronic kidney disease...",
            ),
            _serpapi_result(
                "Untrusted site",
                "https://random-blog.example.com/article",
                "Some article...",
            ),
        ]
    }

    with patch("src.utils.search_provider.GoogleSearch") as mock_gs:
        mock_gs.return_value.get_dict.return_value = fake_results
        sp = SearchProvider("serpapi", api_key="fake")
        results = sp.search("CKD treatment", max_results=5)

    assert len(results) == 1
    assert results[0].source_domain == "medlineplus.gov"
    assert results[0].title == "CKD overview"


def test_serpapi_all_filtered_returns_empty_list() -> None:
    """When all SerpAPI results are from untrusted domains, returns empty list."""
    fake_results = {
        "organic_results": [
            _serpapi_result("Blog", "https://random-blog.com/post", "Some post..."),
        ]
    }

    with patch("src.utils.search_provider.GoogleSearch") as mock_gs:
        mock_gs.return_value.get_dict.return_value = fake_results
        sp = SearchProvider("serpapi", api_key="fake")
        results = sp.search("hypertension", max_results=5)

    assert results == []


def test_serpapi_domain_filter_applied_with_custom_sources() -> None:
    """Custom sources_filter overrides default TRUSTED_MEDICAL_DOMAINS."""
    fake_results = {
        "organic_results": [
            _serpapi_result("MedLine", "https://medlineplus.gov/a.html", "snippet"),
            _serpapi_result("WHO", "https://who.int/b.html", "snippet"),
        ]
    }

    with patch("src.utils.search_provider.GoogleSearch") as mock_gs:
        mock_gs.return_value.get_dict.return_value = fake_results
        sp = SearchProvider("serpapi", api_key="fake")
        # Only allow who.int — medlineplus.gov should be dropped
        results = sp.search("query", sources_filter=["who.int"])

    assert len(results) == 1
    assert results[0].source_domain == "who.int"


def test_serpapi_respects_max_results() -> None:
    """SerpAPI result count is capped at max_results before filtering."""
    # Build 10 trusted results
    organic = [
        _serpapi_result(f"T{i}", f"https://medlineplus.gov/a{i}.html", f"s{i}")
        for i in range(10)
    ]

    with patch("src.utils.search_provider.GoogleSearch") as mock_gs:
        mock_gs.return_value.get_dict.return_value = {"organic_results": organic}
        sp = SearchProvider("serpapi", api_key="fake")
        results = sp.search("query", max_results=3)

    # max_results=3 → only first 3 organic results examined
    assert len(results) == 3


def test_serpapi_www_prefix_stripped_from_domain() -> None:
    """www. prefix is stripped when extracting source_domain from URL."""
    fake_results = {
        "organic_results": [
            _serpapi_result("Mayo", "https://www.mayoclinic.org/article", "snippet"),
        ]
    }

    with patch("src.utils.search_provider.GoogleSearch") as mock_gs:
        mock_gs.return_value.get_dict.return_value = fake_results
        sp = SearchProvider("serpapi", api_key="fake")
        results = sp.search("query")

    assert len(results) == 1
    assert results[0].source_domain == "mayoclinic.org"


def test_serpapi_empty_organic_returns_empty_list() -> None:
    """SerpAPI response with no organic_results returns empty list."""
    with patch("src.utils.search_provider.GoogleSearch") as mock_gs:
        mock_gs.return_value.get_dict.return_value = {}
        sp = SearchProvider("serpapi", api_key="fake")
        results = sp.search("query")

    assert results == []


# ---------------------------------------------------------------------------
# Serper path
# ---------------------------------------------------------------------------


def _serper_result(title: str, url: str, snippet: str) -> dict:
    """Helper to build a minimal Serper organic result dict."""
    return {"title": title, "link": url, "snippet": snippet}


def test_serper_returns_filtered_results() -> None:
    """Serper results with trusted domains are returned; untrusted are dropped."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "organic": [
            _serper_result("NIH article", "https://nih.gov/article", "NIH snippet"),
            _serper_result("Spam", "https://spam-site.example.com/p", "spam"),
        ]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("src.utils.search_provider.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = fake_response
        sp = SearchProvider("serper", api_key="fake")
        results = sp.search("diabetes", max_results=5)

    assert len(results) == 1
    assert results[0].source_domain == "nih.gov"


def test_serper_all_filtered_returns_empty_list() -> None:
    """When all Serper results are from untrusted domains, returns empty list."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "organic": [
            _serper_result("Blog", "https://random.example.com/post", "text"),
        ]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("src.utils.search_provider.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = fake_response
        sp = SearchProvider("serper", api_key="fake")
        results = sp.search("query")

    assert results == []


def test_serper_domain_filter_applied_with_custom_sources() -> None:
    """Custom sources_filter overrides default TRUSTED_MEDICAL_DOMAINS."""
    fake_response = MagicMock()
    fake_response.json.return_value = {
        "organic": [
            _serper_result("CDC", "https://cdc.gov/article", "cdc text"),
            _serper_result("WebMD", "https://webmd.com/article", "webmd text"),
        ]
    }
    fake_response.raise_for_status = MagicMock()

    with patch("src.utils.search_provider.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = fake_response
        sp = SearchProvider("serper", api_key="fake")
        results = sp.search("query", sources_filter=["cdc.gov"])

    assert len(results) == 1
    assert results[0].source_domain == "cdc.gov"


def test_serper_empty_organic_returns_empty_list() -> None:
    """Serper response with no organic key returns empty list."""
    fake_response = MagicMock()
    fake_response.json.return_value = {}
    fake_response.raise_for_status = MagicMock()

    with patch("src.utils.search_provider.httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.return_value = fake_response
        sp = SearchProvider("serper", api_key="fake")
        results = sp.search("query")

    assert results == []


def test_serper_passes_api_key_in_header() -> None:
    """Serper POST sends X-API-KEY header with the configured key."""
    fake_response = MagicMock()
    fake_response.json.return_value = {"organic": []}
    fake_response.raise_for_status = MagicMock()

    with patch("src.utils.search_provider.httpx.Client") as mock_client:
        mock_post = mock_client.return_value.__enter__.return_value.post
        mock_post.return_value = fake_response
        sp = SearchProvider("serper", api_key="my-serper-key")
        sp.search("query")

    call_kwargs = mock_post.call_args.kwargs
    assert call_kwargs["headers"]["X-API-KEY"] == "my-serper-key"


# ---------------------------------------------------------------------------
# TRUSTED_MEDICAL_DOMAINS constant
# ---------------------------------------------------------------------------


def test_trusted_domains_is_frozenset() -> None:
    """TRUSTED_MEDICAL_DOMAINS is a frozenset (immutable, hashable)."""
    assert isinstance(TRUSTED_MEDICAL_DOMAINS, frozenset)


def test_trusted_domains_includes_expected_sources() -> None:
    """Key medical domains are present in the whitelist."""
    assert "medlineplus.gov" in TRUSTED_MEDICAL_DOMAINS
    assert "who.int" in TRUSTED_MEDICAL_DOMAINS
    assert "pubmed.ncbi.nlm.nih.gov" in TRUSTED_MEDICAL_DOMAINS
