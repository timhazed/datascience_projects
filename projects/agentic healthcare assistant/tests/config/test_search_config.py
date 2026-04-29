"""Tests for src/config/search_config.py — SearchConfig."""

import pytest
from pydantic import ValidationError

from src.config.search_config import SearchConfig


def test_serpapi_valid():
    c = SearchConfig(provider="serpapi")
    assert c.provider == "serpapi"


def test_serper_valid():
    c = SearchConfig(provider="serper")
    assert c.provider == "serper"


def test_invalid_provider_raises():
    with pytest.raises(ValidationError):
        SearchConfig(provider="bing")


def test_empty_provider_raises():
    with pytest.raises(ValidationError):
        SearchConfig(provider="")


def test_missing_provider_raises():
    with pytest.raises(ValidationError):
        SearchConfig()


def test_roundtrip():
    c = SearchConfig(provider="serper")
    restored = SearchConfig.model_validate(c.model_dump())
    assert restored.provider == "serper"
