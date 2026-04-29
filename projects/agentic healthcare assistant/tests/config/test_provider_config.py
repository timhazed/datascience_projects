"""Tests for src/config/provider_config.py — ProviderConfig."""

import pytest
from pydantic import ValidationError

from src.config.provider_config import ProviderConfig


def _minimal() -> dict:
    return {"model": "llama-3.3-70b-versatile"}


def test_valid_defaults():
    c = ProviderConfig(**_minimal())
    assert c.temperature == 0.1
    assert c.temperature_summarizer == 0.3
    assert c.temperature_search == 0.3
    assert c.max_tokens_planner == 1024
    assert c.max_tokens_history == 400
    assert c.max_tokens_search == 650
    assert c.max_tokens_appointment == 150
    assert c.max_tokens_summarizer == 700
    assert c.max_tokens_guard == 25
    assert c.max_tokens_memory_summary == 260


def test_all_token_budget_keys_present():
    c = ProviderConfig(**_minimal())
    for attr in (
        "max_tokens_planner",
        "max_tokens_history",
        "max_tokens_search",
        "max_tokens_appointment",
        "max_tokens_summarizer",
        "max_tokens_guard",
        "max_tokens_memory_summary",
    ):
        assert hasattr(c, attr), f"Missing token budget field: {attr}"


def test_temperature_upper_bound():
    ProviderConfig(model="m", temperature=1.0)


def test_temperature_above_max_raises():
    with pytest.raises(ValidationError):
        ProviderConfig(model="m", temperature=1.1)


def test_temperature_below_min_raises():
    with pytest.raises(ValidationError):
        ProviderConfig(model="m", temperature=-0.1)


def test_temperature_summarizer_bounds():
    ProviderConfig(model="m", temperature_summarizer=0.0)
    ProviderConfig(model="m", temperature_summarizer=1.0)


def test_temperature_search_bounds():
    ProviderConfig(model="m", temperature_search=0.0)
    ProviderConfig(model="m", temperature_search=1.0)


def test_custom_token_budgets():
    c = ProviderConfig(
        model="gpt-4o",
        max_tokens_planner=600,
        max_tokens_guard=30,
    )
    assert c.max_tokens_planner == 600
    assert c.max_tokens_guard == 30


def test_missing_model_raises():
    with pytest.raises(ValidationError):
        ProviderConfig()


def test_roundtrip():
    c = ProviderConfig(model="llama-3.3-70b-versatile", temperature=0.2, max_tokens_planner=550)
    restored = ProviderConfig.model_validate(c.model_dump())
    assert restored.model == "llama-3.3-70b-versatile"
    assert restored.max_tokens_planner == 550
