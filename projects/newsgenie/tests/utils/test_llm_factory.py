"""Tests for src/utils/llm_factory.py — LLMFactory."""

import pytest

from src.data import LLMConfig, LLMProvider
from src.utils.groq_builder import GroqBuilder
from src.utils.llm_factory import LLMFactory
from src.utils.openai_builder import OpenAIBuilder


@pytest.fixture(autouse=True)
def reset_factory():
    """Reset LLMFactory between tests to prevent instance bleed."""
    LLMFactory.reset()
    yield
    LLMFactory.reset()


class TestLLMFactory:
    def test_same_config_returns_same_instance(self):
        """Two get() calls with identical config return the same cached instance."""
        from unittest.mock import MagicMock, patch

        mock_llm = MagicMock()
        with patch.object(OpenAIBuilder, "build", return_value=mock_llm):
            config = LLMConfig(
                provider=LLMProvider.OPENAI, model_name="gpt-4o-mini",
                temperature=0.0, max_tokens=300,
            )
            inst1 = LLMFactory.get(config)
            inst2 = LLMFactory.get(config)

        assert inst1 is inst2

    def test_different_temperature_returns_different_instance(self):
        """Configs with different temperatures must produce different cached instances."""
        from unittest.mock import MagicMock, patch

        with patch.object(OpenAIBuilder, "build", side_effect=lambda c: MagicMock()):
            config_a = LLMConfig(
                provider=LLMProvider.OPENAI, model_name="gpt-4o-mini",
                temperature=0.0, max_tokens=300,
            )
            config_b = LLMConfig(
                provider=LLMProvider.OPENAI, model_name="gpt-4o-mini",
                temperature=0.5, max_tokens=300,
            )
            inst_a = LLMFactory.get(config_a)
            inst_b = LLMFactory.get(config_b)

        assert inst_a is not inst_b

    def test_unregistered_provider_raises_value_error(self):
        """get() with a provider that has no registered builder raises ValueError."""
        # Remove OPENAI so the factory has no builders.
        LLMFactory._builders.clear()
        config = LLMConfig(
            provider=LLMProvider.OPENAI, model_name="gpt-4o-mini",
            temperature=0.0, max_tokens=300,
        )
        with pytest.raises(ValueError, match="No builder registered"):
            LLMFactory.get(config)

    def test_supported_providers_returns_only_registered(self):
        """supported_providers() returns only the builders currently registered."""
        # After reset(), only OPENAI is registered by default.
        providers = LLMFactory.supported_providers()
        assert LLMProvider.OPENAI in providers
        assert LLMProvider.GROQ not in providers

    def test_register_adds_provider(self):
        """register() makes a new provider available in supported_providers()."""
        LLMFactory.register(LLMProvider.GROQ, GroqBuilder())
        assert LLMProvider.GROQ in LLMFactory.supported_providers()

    def test_reset_clears_cache_and_restores_defaults(self):
        """reset() wipes cached instances and removes non-default registrations."""
        LLMFactory.register(LLMProvider.GROQ, GroqBuilder())
        assert LLMProvider.GROQ in LLMFactory.supported_providers()

        LLMFactory.reset()

        assert LLMProvider.GROQ not in LLMFactory.supported_providers()
        assert LLMFactory._instances == {}

    def test_groq_registered_after_register(self):
        """After registering GroqBuilder, get() with GROQ config calls GroqBuilder.build()."""
        from unittest.mock import MagicMock, patch

        mock_llm = MagicMock()
        LLMFactory.register(LLMProvider.GROQ, GroqBuilder())

        with patch.object(GroqBuilder, "build", return_value=mock_llm):
            config = LLMConfig(
                provider=LLMProvider.GROQ, model_name="llama-3.1-70b-versatile",
                temperature=0.0, max_tokens=300,
            )
            result = LLMFactory.get(config)

        assert result is mock_llm

    def test_different_max_tokens_returns_different_instance(self):
        """Configs differing only in max_tokens produce different instances."""
        from unittest.mock import MagicMock, patch

        with patch.object(OpenAIBuilder, "build", side_effect=lambda c: MagicMock()):
            config_a = LLMConfig(
                provider=LLMProvider.OPENAI, model_name="gpt-4o-mini",
                temperature=0.0, max_tokens=300,
            )
            config_b = LLMConfig(
                provider=LLMProvider.OPENAI, model_name="gpt-4o-mini",
                temperature=0.0, max_tokens=600,
            )
            inst_a = LLMFactory.get(config_a)
            inst_b = LLMFactory.get(config_b)

        assert inst_a is not inst_b
