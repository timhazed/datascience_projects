import pytest
from unittest.mock import patch, MagicMock

from src.llm.llm_factory import get_llm


class TestGetLLM:
    """Tests for the get_llm factory function."""

    def test_openai_provider(self):
        """OpenAI provider creates ChatOpenAI instance."""
        with patch("src.llm.llm_factory.ChatOpenAI") as mock_chat_openai:
            mock_instance = MagicMock()
            mock_chat_openai.return_value = mock_instance

            result = get_llm("openai", "gpt-3.5-turbo", temperature=0.2, max_tokens=512)

            mock_chat_openai.assert_called_once_with(
                model="gpt-3.5-turbo",
                temperature=0.2,
                max_tokens=512
            )
            assert result is mock_instance

    def test_groq_provider(self):
        """Groq provider creates ChatGroq instance."""
        with patch("src.llm.llm_factory.ChatGroq") as mock_chat_groq:
            mock_instance = MagicMock()
            mock_chat_groq.return_value = mock_instance

            result = get_llm("groq", "llama-3.3-70b-versatile", temperature=0.1, max_tokens=1024)

            mock_chat_groq.assert_called_once_with(
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1024
            )
            assert result is mock_instance

    def test_invalid_provider_raises_error(self):
        """Invalid provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            get_llm("invalid_provider", "model-name")
        assert "Invalid provider: invalid_provider" in str(exc_info.value)

    def test_default_temperature(self):
        """Default temperature is 0.2."""
        with patch("src.llm.llm_factory.ChatOpenAI") as mock:
            mock.return_value = MagicMock()
            get_llm("openai", "gpt-4")
            call_kwargs = mock.call_args[1]
            assert call_kwargs["temperature"] == 0.2

    def test_default_max_tokens(self):
        """Default max_tokens is 512."""
        with patch("src.llm.llm_factory.ChatOpenAI") as mock:
            mock.return_value = MagicMock()
            get_llm("openai", "gpt-4")
            call_kwargs = mock.call_args[1]
            assert call_kwargs["max_tokens"] == 512

    def test_case_sensitive_provider(self):
        """Provider name is case-sensitive."""
        with pytest.raises(ValueError):
            get_llm("OpenAI", "gpt-3.5-turbo")

    def test_empty_provider_raises_error(self):
        """Empty provider raises ValueError."""
        with pytest.raises(ValueError):
            get_llm("", "gpt-3.5-turbo")

    def test_custom_parameters_passed_correctly(self):
        """Custom temperature and max_tokens are passed correctly."""
        with patch("src.llm.llm_factory.ChatOpenAI") as mock:
            mock.return_value = MagicMock()
            get_llm("openai", "gpt-4", temperature=1.5, max_tokens=2048)
            call_kwargs = mock.call_args[1]
            assert call_kwargs["temperature"] == 1.5
            assert call_kwargs["max_tokens"] == 2048
