"""Tests for src/api/guardian_client.py — HTTP mock matrix."""

import httpx
import pytest
import respx

from src.api.guardian_client import fetch_articles

_URL = "https://content.guardianapis.com/search"

_SAMPLE_RESULT = {
    "id": "sport/2026/mar/31/champions-league",
    "webTitle": "Champions League Results",
    "webUrl": "https://theguardian.com/sport/champions-league",
    "webPublicationDate": "2026-03-31T20:00:00Z",
    "sectionName": "Sport",
    "fields": {"trailText": "Summary here", "thumbnail": None, "headline": "CL Results"},
}

_GUARDIAN_RESPONSE = {
    "response": {
        "status": "ok",
        "total": 1,
        "results": [_SAMPLE_RESULT],
    }
}


@respx.mock
def test_success_returns_results():
    respx.get(_URL).mock(return_value=httpx.Response(200, json=_GUARDIAN_RESPONSE))
    with httpx.Client() as client:
        results = fetch_articles("Champions League", "fake-key", client=client)
    assert len(results) == 1
    assert results[0]["webTitle"] == "Champions League Results"


@respx.mock
def test_empty_results_returns_empty_list():
    respx.get(_URL).mock(
        return_value=httpx.Response(200, json={"response": {"status": "ok", "results": []}})
    )
    with httpx.Client() as client:
        results = fetch_articles("noresults", "fake-key", client=client)
    assert results == []


@respx.mock
def test_missing_api_key_returns_empty_without_http_call():
    results = fetch_articles("test", api_key="")
    assert results == []


@respx.mock
def test_401_returns_empty_list():
    respx.get(_URL).mock(return_value=httpx.Response(401, json={"message": "Invalid key"}))
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
