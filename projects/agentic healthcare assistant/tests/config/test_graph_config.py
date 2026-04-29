"""Tests for src/config/graph_config.py (GraphConfig)."""

import pytest
from pydantic import ValidationError

from src.config.graph_config import GraphConfig


def test_default_recursion_limit():
    c = GraphConfig()
    assert c.recursion_limit == 100


def test_explicit_recursion_limit():
    c = GraphConfig(recursion_limit=64)
    assert c.recursion_limit == 64


def test_recursion_limit_must_be_at_least_one():
    with pytest.raises(ValidationError) as exc_info:
        GraphConfig(recursion_limit=0)
    locs = [e["loc"] for e in exc_info.value.errors()]
    assert any(loc == ("recursion_limit",) for loc in locs)


def test_recursion_limit_negative_raises():
    with pytest.raises(ValidationError):
        GraphConfig(recursion_limit=-1)


def test_model_roundtrip():
    c = GraphConfig(recursion_limit=200)
    restored = GraphConfig.model_validate(c.model_dump())
    assert restored.recursion_limit == 200
