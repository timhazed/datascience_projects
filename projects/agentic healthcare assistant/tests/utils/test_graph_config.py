"""Tests for src/utils/graph_config.py — build_invoke_config."""

import uuid

from src.utils.graph_config import build_invoke_config


def test_build_invoke_config_with_patient(mock_settings):
    """Known patient_id becomes the thread_id."""
    cfg = build_invoke_config("P-001", mock_settings)
    assert cfg["configurable"]["thread_id"] == "P-001"


def test_build_invoke_config_with_patient_strips_whitespace(mock_settings):
    """Whitespace around patient_id is stripped."""
    cfg = build_invoke_config("  P-002  ", mock_settings)
    assert cfg["configurable"]["thread_id"] == "P-002"


def test_build_invoke_config_no_patient_uses_session_id(mock_settings):
    """When patient_id is None and session_id is provided, session_id is the thread_id."""
    cfg = build_invoke_config(None, mock_settings, session_id="sess-abc")
    assert cfg["configurable"]["thread_id"] == "sess-abc"


def test_build_invoke_config_no_patient_no_session_uses_uuid(mock_settings):
    """When both patient_id and session_id are absent, a fresh UUID is generated."""
    cfg = build_invoke_config(None, mock_settings)
    thread_id = cfg["configurable"]["thread_id"]
    # Must be a valid UUID string
    parsed = uuid.UUID(thread_id)
    assert str(parsed) == thread_id


def test_build_invoke_config_no_patient_no_session_unique_each_call(mock_settings):
    """Two calls without patient_id or session_id produce different thread_ids."""
    cfg1 = build_invoke_config(None, mock_settings)
    cfg2 = build_invoke_config(None, mock_settings)
    assert cfg1["configurable"]["thread_id"] != cfg2["configurable"]["thread_id"]


def test_build_invoke_config_always_has_recursion_limit(mock_settings):
    """recursion_limit is always present in the returned config."""
    cfg = build_invoke_config("P-001", mock_settings)
    assert "recursion_limit" in cfg
    assert isinstance(cfg["recursion_limit"], int)
    assert cfg["recursion_limit"] > 0


def test_build_invoke_config_recursion_limit_matches_settings(mock_settings):
    """recursion_limit in config matches settings.graph.recursion_limit."""
    cfg = build_invoke_config("P-001", mock_settings)
    # mock_settings uses the test config.yaml which may not define graph.recursion_limit;
    # verify it is present and non-zero regardless of the exact value.
    assert cfg["recursion_limit"] == mock_settings.graph.recursion_limit


def test_build_invoke_config_patient_id_empty_string_uses_session(mock_settings):
    """Empty string patient_id (falsy) falls back to session_id."""
    cfg = build_invoke_config("", mock_settings, session_id="fallback-session")
    assert cfg["configurable"]["thread_id"] == "fallback-session"
