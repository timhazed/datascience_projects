"""Tests for src/models/search_result_item.py — SearchResultItem."""

import pytest
from pydantic import ValidationError

from src.models.search_result_item import SearchResultItem


def test_valid_construction():
    item = SearchResultItem(
        title="Chronic Kidney Disease Treatment",
        url="https://medlineplus.gov/ckd",
        snippet="CKD treatment focuses on slowing progression.",
        source_domain="medlineplus.gov",
    )
    assert item.source_domain == "medlineplus.gov"


def test_missing_required_field_raises():
    with pytest.raises(ValidationError):
        SearchResultItem(title="CKD", url="https://medlineplus.gov/ckd", snippet="...")


def test_roundtrip():
    item = SearchResultItem(
        title="Hypertension Overview",
        url="https://who.int/hypertension",
        snippet="Hypertension affects 1.28 billion adults worldwide.",
        source_domain="who.int",
    )
    restored = SearchResultItem.model_validate(item.model_dump())
    assert restored.url == "https://who.int/hypertension"
