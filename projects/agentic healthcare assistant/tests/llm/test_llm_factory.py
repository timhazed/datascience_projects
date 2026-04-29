"""Tests for src/llm/llm_factory.py — get_llm()."""

from unittest.mock import MagicMock, patch

import pytest

from src.llm.llm_factory import get_llm

# Patch the classes in the factory's own namespace so no real API clients are built.
_GROQ_PATH = "src.llm.llm_factory.ChatGroq"
_OPENAI_PATH = "src.llm.llm_factory.ChatOpenAI"


def test_groq_path_returns_chat_groq():
    with patch(_GROQ_PATH) as mock_cls:
        mock_cls.return_value = MagicMock()
        result = get_llm("groq", "llama-3.3-70b-versatile")
    mock_cls.assert_called_once()
    assert result is mock_cls.return_value


def test_openai_path_returns_chat_openai():
    with patch(_OPENAI_PATH) as mock_cls:
        mock_cls.return_value = MagicMock()
        result = get_llm("openai", "gpt-4o")
    mock_cls.assert_called_once()
    assert result is mock_cls.return_value


def test_groq_called_with_correct_args():
    with patch(_GROQ_PATH) as mock_cls:
        get_llm("groq", "llama-3.3-70b-versatile", temperature=0.3, max_tokens=500)
    mock_cls.assert_called_once_with(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=500,
    )


def test_openai_called_with_correct_args():
    with patch(_OPENAI_PATH) as mock_cls:
        get_llm("openai", "gpt-4o", temperature=0.1, max_tokens=650)
    mock_cls.assert_called_once_with(
        model="gpt-4o",
        temperature=0.1,
        max_tokens=650,
    )


def test_default_temperature_is_0_1():
    with patch(_GROQ_PATH) as mock_cls:
        get_llm("groq", "llama-3.3-70b-versatile")
    _, kwargs = mock_cls.call_args
    assert kwargs["temperature"] == 0.1


def test_default_max_tokens_is_512():
    with patch(_GROQ_PATH) as mock_cls:
        get_llm("groq", "llama-3.3-70b-versatile")
    _, kwargs = mock_cls.call_args
    assert kwargs["max_tokens"] == 512


def test_invalid_provider_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported provider"):
        get_llm("anthropic", "claude-3")


def test_invalid_provider_message_contains_name():
    with pytest.raises(ValueError, match="anthropic"):
        get_llm("anthropic", "claude-3")


def test_openai_not_called_for_groq_provider():
    with patch(_GROQ_PATH), patch(_OPENAI_PATH) as mock_openai:
        get_llm("groq", "llama-3.3-70b-versatile")
    mock_openai.assert_not_called()


def test_groq_not_called_for_openai_provider():
    with patch(_OPENAI_PATH), patch(_GROQ_PATH) as mock_groq:
        get_llm("openai", "gpt-4o")
    mock_groq.assert_not_called()


def test_groq_reasoning_effort_passed_in_model_kwargs():
    """reasoning_effort is forwarded via model_kwargs when provided."""
    with patch(_GROQ_PATH) as mock_cls:
        get_llm("groq", "gpt-oss-120b", reasoning_effort="low")
    _, kwargs = mock_cls.call_args
    assert kwargs["model_kwargs"]["reasoning_effort"] == "low"


def test_groq_reasoning_format_passed_in_model_kwargs():
    """reasoning_format is forwarded via model_kwargs when provided."""
    with patch(_GROQ_PATH) as mock_cls:
        get_llm("groq", "qwen3-32b", reasoning_format="parsed")
    _, kwargs = mock_cls.call_args
    assert kwargs["model_kwargs"]["reasoning_format"] == "parsed"


def test_groq_both_reasoning_params_combined():
    """Both reasoning_effort and reasoning_format appear together in model_kwargs."""
    with patch(_GROQ_PATH) as mock_cls:
        get_llm("groq", "gpt-oss-120b", reasoning_effort="high", reasoning_format="parsed")
    _, kwargs = mock_cls.call_args
    assert kwargs["model_kwargs"] == {"reasoning_effort": "high", "reasoning_format": "parsed"}


def test_groq_no_model_kwargs_when_no_reasoning_params():
    """model_kwargs is not passed when neither reasoning param is set."""
    with patch(_GROQ_PATH) as mock_cls:
        get_llm("groq", "llama-3.3-70b-versatile")
    call_kwargs = mock_cls.call_args[1]
    assert "model_kwargs" not in call_kwargs
