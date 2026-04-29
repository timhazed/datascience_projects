"""Tests for src/models/evaluation.py — ExperimentMetrics."""

import uuid

import pytest
from pydantic import ValidationError

from src.models.evaluation import ExperimentMetrics


def _minimal() -> dict:
    return {
        "experiment_name": "disease_search",
        "run_id": str(uuid.uuid4()),
        "latency_ms": 1234.5,
        "success": True,
        "tool_calls_made": 2,
    }


def test_valid_construction():
    m = ExperimentMetrics(**_minimal())
    assert m.quality_score is None
    assert m.error is None
    assert m.timestamp is not None


def test_all_experiment_names():
    for name in ("medical_history", "disease_search", "appointment_booking"):
        m = ExperimentMetrics(**{**_minimal(), "experiment_name": name})
        assert m.experiment_name == name


def test_invalid_experiment_name_raises():
    with pytest.raises(ValidationError):
        ExperimentMetrics(**{**_minimal(), "experiment_name": "unknown_experiment"})


def test_quality_score_bounds():
    ExperimentMetrics(**{**_minimal(), "quality_score": 0.0})
    ExperimentMetrics(**{**_minimal(), "quality_score": 1.0})
    ExperimentMetrics(**{**_minimal(), "quality_score": 0.75})


def test_quality_score_above_max_raises():
    with pytest.raises(ValidationError):
        ExperimentMetrics(**{**_minimal(), "quality_score": 1.1})


def test_quality_score_below_min_raises():
    with pytest.raises(ValidationError):
        ExperimentMetrics(**{**_minimal(), "quality_score": -0.1})


def test_with_error():
    m = ExperimentMetrics(**{**_minimal(), "success": False, "error": "LLM timeout"})
    assert m.error == "LLM timeout"


def test_roundtrip():
    m = ExperimentMetrics(**_minimal())
    restored = ExperimentMetrics.model_validate(m.model_dump())
    assert restored.run_id == m.run_id
    assert restored.latency_ms == 1234.5
