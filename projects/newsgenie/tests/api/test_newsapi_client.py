"""Tests for src/api/newsapi_client.py — HTTP mock matrix."""

import httpx
import pytest
import respx

from src.api.newsapi_client import fetch_articles

_URL = "https://newsapi.org/v2/everything"

_SAMPLE_ARTICLE = {
    "source": {"id": "bbc-news", "name": "BBC News"},
    "author": "Test Author",
    "title": "Test Headline",
    "description": "A description",
    "url": "https://bbc.com/article",
    "urlToImage": None,
    "publishedAt": "2026-03-31T12:00:00Z",
    "content": None,
}


@respx.mock
def test_success_returns_articles():
    respx.get(_URL).mock(
        return_value=httpx.Response(200, json={"articles": [_SAMPLE_ARTICLE]})
    )
    with httpx.Client() as client:
        results = fetch_articles("Apple stock", "fake-key", client=client)
    assert len(results) == 1
    assert results[0]["title"] == "Test Headline"


@respx.mock
def test_empty_articles_returns_empty_list():
    respx.get(_URL).mock(return_value=httpx.Response(200, json={"articles": []}))
    with httpx.Client() as client:
        results = fetch_articles("noresults", "fake-key", client=client)
    assert results == []


@respx.mock
def test_missing_api_key_returns_empty_without_http_call():
    # respx will raise if any HTTP call is made — confirms no request is sent
    results = fetch_articles("test", api_key="")
    assert results == []


@respx.mock
def test_401_returns_empty_list():
    respx.get(_URL).mock(return_value=httpx.Response(401, json={"message": "Invalid API key"}))
    with httpx.Client() as client:
        results = fetch_articles("test", "bad-key", client=client)
    assert results == []


@respx.mock
def test_403_returns_empty_list():
    respx.get(_URL).mock(return_value=httpx.Response(403, json={"message": "Forbidden"}))
    with httpx.Client() as client:
        results = fetch_articles("test", "bad-key", client=client)
    assert results == []


@respx.mock
def test_500_raises_after_retry():
    """5xx errors should propagate after retries are exhausted."""
    respx.get(_URL).mock(return_value=httpx.Response(500, json={"message": "Server error"}))
    with httpx.Client() as client:
        with pytest.raises(httpx.HTTPStatusError):
            fetch_articles("test", "key", client=client)
