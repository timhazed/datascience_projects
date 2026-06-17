"""Tests for Settings (src/config/settings.py) — F6 rework acceptance criteria.

Covers the new chroma_host and chroma_data_path fields and the @model_validator
that rejects malformed CHROMA_HOST values at startup.

Cases:
  - chroma_host empty string → local mode; no validator error
  - chroma_host valid HTTP URL → accepted
  - chroma_host valid HTTPS URL → accepted
  - chroma_host bare hostname (no scheme) → ValueError at construction
  - chroma_host unsupported scheme (ftp://) → ValueError at construction
  - chroma_data_path default → "chroma_data"
  - chroma_data_path custom → respects CHROMA_DATA_PATH env var
  - Error message contains the required format hint and the bad value
"""

import pytest

from src.config.settings import Settings


class TestChromaHostValidator:
    def test_empty_chroma_host_is_valid(self) -> None:
        """Empty chroma_host (local mode) passes validation without error."""
        s = Settings(chroma_host="")
        assert s.chroma_host == ""

    def test_valid_http_url_accepted(self) -> None:
        """A full HTTP URL is accepted by the validator."""
        s = Settings(chroma_host="http://chromadb:8000")
        assert s.chroma_host == "http://chromadb:8000"

    def test_valid_https_url_accepted(self) -> None:
        """A full HTTPS URL is accepted by the validator."""
        s = Settings(chroma_host="https://chromadb.example.com:8000")
        assert s.chroma_host == "https://chromadb.example.com:8000"

    def test_valid_localhost_url_accepted(self) -> None:
        """http://localhost:8000 is a valid URL."""
        s = Settings(chroma_host="http://localhost:8000")
        assert s.chroma_host == "http://localhost:8000"

    def test_bare_hostname_rejected(self) -> None:
        """Bare hostname without scheme raises ValueError with actionable message."""
        with pytest.raises(ValueError, match="must be a full HTTP URL"):
            Settings(chroma_host="chromadb:8000")

    def test_bare_hostname_no_port_rejected(self) -> None:
        """Bare hostname without port or scheme is also rejected."""
        with pytest.raises(ValueError, match="must be a full HTTP URL"):
            Settings(chroma_host="chromadb")

    def test_unsupported_scheme_rejected(self) -> None:
        """ftp:// scheme is rejected — only http/https are valid."""
        with pytest.raises(ValueError, match="must be a full HTTP URL"):
            Settings(chroma_host="ftp://chromadb:8000")

    def test_error_message_contains_bad_value(self) -> None:
        """ValueError message includes the bad value so the operator knows what to fix."""
        bad_value = "chromadb:8000"
        with pytest.raises(ValueError, match=bad_value):
            Settings(chroma_host=bad_value)

    def test_error_message_contains_format_hint(self) -> None:
        """ValueError message includes an example of the correct format."""
        with pytest.raises(ValueError, match="http://chromadb:8000"):
            Settings(chroma_host="chromadb:8000")


class TestChromaDataPath:
    def test_default_chroma_data_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """chroma_data_path defaults to 'chroma_data' when CHROMA_DATA_PATH is not set."""
        monkeypatch.delenv("CHROMA_DATA_PATH", raising=False)
        s = Settings(chroma_host="")
        assert s.chroma_data_path == "chroma_data"

    def test_custom_chroma_data_path(self) -> None:
        """chroma_data_path can be overridden by passing it directly."""
        s = Settings(chroma_host="", chroma_data_path="/custom/path")
        assert s.chroma_data_path == "/custom/path"

    def test_chroma_data_path_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHROMA_DATA_PATH env var is picked up by pydantic-settings."""
        monkeypatch.setenv("CHROMA_DATA_PATH", "/env/override")
        s = Settings()
        assert s.chroma_data_path == "/env/override"


class TestChromaHostFromEnv:
    def test_chroma_host_read_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """CHROMA_HOST env var is picked up and validated by Settings."""
        monkeypatch.setenv("CHROMA_HOST", "http://chromadb:8000")
        s = Settings()
        assert s.chroma_host == "http://chromadb:8000"

    def test_malformed_chroma_host_from_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A malformed CHROMA_HOST env var raises ValueError at Settings() construction time."""
        monkeypatch.setenv("CHROMA_HOST", "chromadb:8000")
        with pytest.raises(ValueError, match="must be a full HTTP URL"):
            Settings()
