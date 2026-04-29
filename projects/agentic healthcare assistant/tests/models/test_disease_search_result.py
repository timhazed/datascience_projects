"""Tests for src/models/disease_search_result.py — DiseaseSearchResult."""

import pytest
from pydantic import ValidationError

from src.models.disease_search_result import DiseaseSearchResult
from src.models.search_result_item import SearchResultItem


def _item(n: int = 1) -> SearchResultItem:
    return SearchResultItem(
        title=f"Article {n}",
        url=f"https://pubmed.ncbi.nlm.nih.gov/{n}/",
        snippet=f"Snippet {n}",
        source_domain="pubmed.ncbi.nlm.nih.gov",
    )


def test_valid_construction():
    result = DiseaseSearchResult(
        query="CKD treatment",
        items=[_item(1), _item(2)],
        summary="CKD treatment involves ACE inhibitors and dietary changes.",
        citations=["https://pubmed.ncbi.nlm.nih.gov/1/"],
    )
    assert len(result.items) == 2


def test_empty_items_allowed():
    result = DiseaseSearchResult(
        query="rare disease",
        items=[],
        summary="No results found.",
        citations=[],
    )
    assert result.items == []


def test_missing_required_raises():
    with pytest.raises(ValidationError):
        DiseaseSearchResult(query="q", items=[], citations=[])


def test_nested_item_validation():
    with pytest.raises(ValidationError):
        DiseaseSearchResult(
            query="q",
            items=[{"title": "X"}],  # missing required fields
            summary="s",
            citations=[],
        )


def test_roundtrip():
    result = DiseaseSearchResult(
        query="diabetes",
        items=[_item(1)],
        summary="Insulin management is key.",
        citations=["https://pubmed.ncbi.nlm.nih.gov/1/"],
    )
    restored = DiseaseSearchResult.model_validate(result.model_dump())
    assert restored.query == "diabetes"
    assert restored.items[0].title == "Article 1"
