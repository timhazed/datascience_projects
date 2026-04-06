"""Tests for src/utils/openai_builder.py — OpenAIBuilder."""

from unittest.mock import MagicMock, patch

from src.data import LLMConfig, LLMProvider
from src.utils.openai_builder import OpenAIBuilder


class TestOpenAIBuilder:
    def test_build_returns_chat_openai(self):
        """build() returns a ChatOpenAI instance with the provided config fields."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o-mini",
            temperature=0.0,
            max_tokens=300,
        )
        mock_instance = MagicMock()
        with patch("src.utils.openai_builder.ChatOpenAI", return_value=mock_instance) as mock_cls:
            builder = OpenAIBuilder()
            result = builder.build(config)

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 300
        assert "api_key" in call_kwargs
        assert result is mock_instance

    def test_build_passes_temperature(self):
        """build() passes temperature=0.5 through to ChatOpenAI."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-3.5-turbo",
            temperature=0.5,
            max_tokens=100,
        )
        with patch("src.utils.openai_builder.ChatOpenAI") as mock_cls:
            OpenAIBuilder().build(config)
        _, kwargs = mock_cls.call_args
        assert kwargs["temperature"] == 0.5

    def test_build_passes_max_tokens(self):
        """build() passes the correct max_tokens to ChatOpenAI."""
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4o",
            temperature=0.0,
            max_tokens=512,
        )
        with patch("src.utils.openai_builder.ChatOpenAI") as mock_cls:
            OpenAIBuilder().build(config)
        _, kwargs = mock_cls.call_args
        assert kwargs["max_tokens"] == 512
