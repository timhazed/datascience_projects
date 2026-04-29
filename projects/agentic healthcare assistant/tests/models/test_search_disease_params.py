"""Tests for src/models/search_disease_params.py — SearchDiseaseParams."""

import pytest
from pydantic import ValidationError

from src.models.search_disease_params import SearchDiseaseParams


def test_valid_minimal():
    p = SearchDiseaseParams(query="hypertension treatment guidelines")
    assert p.max_results == 5


def test_max_results_boundaries():
    SearchDiseaseParams(query="q", max_results=1)
    SearchDiseaseParams(query="q", max_results=10)


def test_max_results_above_max_raises():
    with pytest.raises(ValidationError):
        SearchDiseaseParams(query="q", max_results=11)


def test_max_results_below_min_raises():
    with pytest.raises(ValidationError):
        SearchDiseaseParams(query="q", max_results=0)


def test_missing_query_raises():
    with pytest.raises(ValidationError):
        SearchDiseaseParams()


def test_roundtrip():
    p = SearchDiseaseParams(query="diabetes type 2", max_results=3)
    restored = SearchDiseaseParams.model_validate(p.model_dump())
    assert restored.max_results == 3
