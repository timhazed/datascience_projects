"""Tests for src/utils/groq_builder.py — GroqBuilder."""

from unittest.mock import MagicMock, patch

from src.data import LLMConfig, LLMProvider
from src.utils.groq_builder import GroqBuilder


class TestGroqBuilder:
    def test_build_returns_chat_groq(self):
        """build() returns a ChatGroq instance with the provided config fields."""
        config = LLMConfig(
            provider=LLMProvider.GROQ,
            model_name="llama-3.1-70b-versatile",
            temperature=0.0,
            max_tokens=300,
        )
        mock_instance = MagicMock()
        with patch("src.utils.groq_builder.ChatGroq", return_value=mock_instance) as mock_cls:
            builder = GroqBuilder()
            result = builder.build(config)

        call_kwargs = mock_cls.call_args.kwargs
        assert call_kwargs["model"] == "llama-3.1-70b-versatile"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 300
        assert "api_key" in call_kwargs
        assert result is mock_instance

    def test_build_passes_temperature(self):
        """build() passes temperature through to ChatGroq."""
        config = LLMConfig(
            provider=LLMProvider.GROQ,
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            max_tokens=200,
        )
        with patch("src.utils.groq_builder.ChatGroq") as mock_cls:
            GroqBuilder().build(config)
        _, kwargs = mock_cls.call_args
        assert kwargs["temperature"] == 0.3

    def test_build_passes_max_tokens(self):
        """build() passes the correct max_tokens to ChatGroq."""
        config = LLMConfig(
            provider=LLMProvider.GROQ,
            model_name="llama-3.1-70b-versatile",
            temperature=0.0,
            max_tokens=128,
        )
        with patch("src.utils.groq_builder.ChatGroq") as mock_cls:
            GroqBuilder().build(config)
        _, kwargs = mock_cls.call_args
        assert kwargs["max_tokens"] == 128
