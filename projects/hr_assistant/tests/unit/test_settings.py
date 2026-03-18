import pytest
from unittest.mock import patch, mock_open
from pydantic import ValidationError

from src.config.settings import (
    ProviderConfig,
    StoreConfig,
    Settings,
    load_settings
)


class TestProviderConfig:
    """Tests for ProviderConfig Pydantic model."""

    def test_default_values(self):
        """Default values are set correctly."""
        config = ProviderConfig()
        assert config.model == "gpt-3.5-turbo"
        assert config.temperature == 0.2
        assert config.max_tokens is None

    def test_custom_values(self):
        """Custom values are accepted."""
        config = ProviderConfig(
            model="gpt-4",
            temperature=0.5,
            max_tokens=1024
        )
        assert config.model == "gpt-4"
        assert config.temperature == 0.5
        assert config.max_tokens == 1024

    def test_temperature_validation_min(self):
        """Temperature below 0 raises error."""
        with pytest.raises(ValidationError):
            ProviderConfig(temperature=-0.1)

    def test_temperature_validation_max(self):
        """Temperature above 2 raises error."""
        with pytest.raises(ValidationError):
            ProviderConfig(temperature=2.1)

    def test_model_min_length(self):
        """Empty model string raises error."""
        with pytest.raises(ValidationError):
            ProviderConfig(model="")

    def test_temperature_at_boundaries(self):
        """Temperature at valid boundaries (0 and 2) is accepted."""
        config_zero = ProviderConfig(temperature=0)
        config_two = ProviderConfig(temperature=2)
        assert config_zero.temperature == 0
        assert config_two.temperature == 2


class TestStoreConfig:
    """Tests for StoreConfig Pydantic model."""

    def test_default_values(self):
        """Default persist_directory is set."""
        config = StoreConfig()
        assert config.persist_directory == "data/faiss"

    def test_custom_persist_directory(self):
        """Custom persist_directory is accepted."""
        config = StoreConfig(persist_directory="/custom/path")
        assert config.persist_directory == "/custom/path"

    def test_persist_directory_min_length(self):
        """Empty persist_directory raises error."""
        with pytest.raises(ValidationError):
            StoreConfig(persist_directory="")


class TestSettings:
    """Tests for Settings model."""

    def test_valid_single_provider_and_store(self, sample_config_dict):
        """Valid config with one provider and one store."""
        settings = Settings(**sample_config_dict)
        assert settings.provider_name == "openai"
        assert settings.store_name == "faiss"

    def test_provider_config_property(self, sample_config_dict):
        """provider_config property returns correct config."""
        settings = Settings(**sample_config_dict)
        assert settings.provider_config.model == "gpt-3.5-turbo"
        assert settings.provider_config.temperature == 0.2

    def test_store_config_property(self, sample_config_dict):
        """store_config property returns correct config."""
        settings = Settings(**sample_config_dict)
        assert settings.store_config.persist_directory == "data/faiss"

    def test_multiple_providers_raises_error(self, invalid_config_multiple_providers):
        """Multiple providers raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            Settings(**invalid_config_multiple_providers)
        assert "Exactly one provider must be configured" in str(exc_info.value)

    def test_no_providers_raises_error(self):
        """No providers raise ValidationError."""
        config = {
            "provider": {},
            "store": {"faiss": {"persist_directory": "data/faiss"}}
        }
        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)
        assert "Exactly one provider must be configured" in str(exc_info.value)

    def test_multiple_stores_raises_error(self):
        """Multiple stores raise ValidationError."""
        config = {
            "provider": {"openai": {"model": "gpt-3.5-turbo"}},
            "store": {
                "faiss": {"persist_directory": "data/faiss"},
                "chroma": {"persist_directory": "data/chroma"}
            }
        }
        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)
        assert "Exactly one store must be configured" in str(exc_info.value)

    def test_no_stores_raises_error(self):
        """No stores raise ValidationError."""
        config = {
            "provider": {"openai": {"model": "gpt-3.5-turbo"}},
            "store": {}
        }
        with pytest.raises(ValidationError) as exc_info:
            Settings(**config)
        assert "Exactly one store must be configured" in str(exc_info.value)

    def test_groq_provider(self, sample_config_groq):
        """Groq provider configuration works."""
        settings = Settings(**sample_config_groq)
        assert settings.provider_name == "groq"
        assert settings.store_name == "chroma"


class TestLoadSettings:
    """Tests for load_settings function."""

    def test_load_settings_from_yaml(self, sample_config_dict):
        """load_settings reads YAML and returns Settings."""
        with patch("builtins.open", mock_open(read_data="")):
            with patch("yaml.safe_load", return_value=sample_config_dict):
                settings = load_settings("config.yaml")
                assert settings.provider_name == "openai"
                assert settings.store_name == "faiss"

    def test_load_settings_custom_path(self, sample_config_dict):
        """load_settings accepts custom config path."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("yaml.safe_load", return_value=sample_config_dict):
                load_settings("/custom/config.yaml")
                mock_file.assert_called_once_with("/custom/config.yaml", "r")

    def test_load_settings_file_not_found(self):
        """load_settings raises error for missing file."""
        with patch("builtins.open", side_effect=FileNotFoundError):
            with pytest.raises(FileNotFoundError):
                load_settings("nonexistent.yaml")

    def test_load_settings_default_path(self, sample_config_dict):
        """load_settings uses default config.yaml path."""
        with patch("builtins.open", mock_open()) as mock_file:
            with patch("yaml.safe_load", return_value=sample_config_dict):
                load_settings()
                mock_file.assert_called_once_with("config.yaml", "r")
