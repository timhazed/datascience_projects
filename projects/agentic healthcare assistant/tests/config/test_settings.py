"""Tests for src/config/settings.py — Settings and load_settings()."""

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config.settings import Settings, load_settings  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_YAML = textwrap.dedent("""\
    provider:
      groq:
        model: "llama-3.3-70b-versatile"
        temperature: 0.1
        temperature_summarizer: 0.3
        temperature_search: 0.3
        max_tokens_planner: 500
        max_tokens_history: 400
        max_tokens_search: 650
        max_tokens_appointment: 150
        max_tokens_summarizer: 700
        max_tokens_guard: 25
    search:
      provider: "serper"
    embeddings:
      provider: "openai"
      model: "text-embedding-3-small"
    db:
      sqlite_path: "data/healthcare.db"
      faiss_path: "data/faiss"
""")


def _write_yaml(tmp_path: Path, content: str) -> Path:
    """Write content to a temp config file and return its path."""
    p = tmp_path / "config.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# load_settings()
# ---------------------------------------------------------------------------

def test_load_settings_valid(tmp_path):
    cfg = _write_yaml(tmp_path, _VALID_YAML)
    s = load_settings(cfg)
    assert s.provider_name == "groq"
    assert s.provider_config.model == "llama-3.3-70b-versatile"
    assert s.search.provider == "serper"


def test_load_settings_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_settings("/nonexistent/path/config.yaml")


def test_load_settings_openai_provider(tmp_path):
    yaml_content = _VALID_YAML.replace("  groq:", "  openai:")
    cfg = _write_yaml(tmp_path, yaml_content)
    s = load_settings(cfg)
    assert s.provider_name == "openai"


# ---------------------------------------------------------------------------
# Settings model validation
# ---------------------------------------------------------------------------

def _valid_data() -> dict:
    return {
        "provider": {
            "groq": {
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.1,
                "temperature_summarizer": 0.3,
                "temperature_search": 0.3,
                "max_tokens_planner": 500,
                "max_tokens_history": 400,
                "max_tokens_search": 650,
                "max_tokens_appointment": 150,
                "max_tokens_summarizer": 700,
                "max_tokens_guard": 25,
            }
        },
        "search": {"provider": "serper"},
        "db": {"sqlite_path": "data/healthcare.db", "faiss_path": "data/faiss"},
    }


def test_valid_construction():
    s = Settings(**_valid_data())
    assert s.provider_name == "groq"
    assert s.provider_config.max_tokens_planner == 500
    assert s.db.sqlite_path == Path("data/healthcare.db")


def test_provider_name_property():
    s = Settings(**_valid_data())
    assert s.provider_name == "groq"


def test_provider_config_property():
    s = Settings(**_valid_data())
    assert s.provider_config.model == "llama-3.3-70b-versatile"


def test_embeddings_defaults_when_omitted():
    s = Settings(**_valid_data())
    assert s.embeddings.provider == "openai"
    assert s.embeddings.model == "text-embedding-3-small"


def test_graph_defaults_when_omitted():
    """graph section optional — default_factory yields GraphConfig(recursion_limit=100)."""
    s = Settings(**_valid_data())
    assert s.graph.recursion_limit == 100


def test_graph_explicit_in_constructor():
    data = _valid_data()
    data["graph"] = {"recursion_limit": 64}
    s = Settings(**data)
    assert s.graph.recursion_limit == 64


def test_graph_invalid_recursion_limit_raises():
    data = _valid_data()
    data["graph"] = {"recursion_limit": 0}
    with pytest.raises(ValidationError):
        Settings(**data)


def test_load_settings_with_graph_section_in_yaml(tmp_path):
    yaml_with_graph = _VALID_YAML + textwrap.dedent("""
    graph:
      recursion_limit: 48
    """)
    cfg = _write_yaml(tmp_path, yaml_with_graph)
    s = load_settings(cfg)
    assert s.graph.recursion_limit == 48


def test_load_settings_without_graph_section_uses_default(tmp_path):
    cfg = _write_yaml(tmp_path, _VALID_YAML)
    s = load_settings(cfg)
    assert s.graph.recursion_limit == 100


def test_invalid_embeddings_provider_raises():
    data = _valid_data()
    data["embeddings"] = {"provider": "cohere", "model": "embed-v3"}
    with pytest.raises(ValidationError):
        Settings(**data)


def test_two_providers_raises():
    data = _valid_data()
    data["provider"]["openai"] = {"model": "gpt-4o"}
    with pytest.raises(ValidationError):
        Settings(**data)


def test_zero_providers_raises():
    data = _valid_data()
    data["provider"] = {}
    with pytest.raises(ValidationError):
        Settings(**data)


def test_missing_search_raises():
    data = _valid_data()
    del data["search"]
    with pytest.raises(ValidationError):
        Settings(**data)


def test_missing_db_raises():
    data = _valid_data()
    del data["db"]
    with pytest.raises(ValidationError):
        Settings(**data)


def test_invalid_search_provider_raises():
    data = _valid_data()
    data["search"] = {"provider": "google"}
    with pytest.raises(ValidationError):
        Settings(**data)


def test_unknown_provider_key_raises():
    data = _valid_data()
    data["provider"] = {"azure": {"model": "gpt-4"}}
    with pytest.raises(ValidationError, match="Unknown provider"):
        Settings(**data)


def test_nested_sections_roundtrip(tmp_path):
    cfg = _write_yaml(tmp_path, _VALID_YAML)
    s = load_settings(cfg)
    assert s.provider_config.temperature_summarizer == 0.3
    assert s.provider_config.max_tokens_guard == 25
    assert s.db.faiss_path == Path("data/faiss")
