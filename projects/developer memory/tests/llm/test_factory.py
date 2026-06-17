"""Tests for src/llm/factory.py.

Phase 2 exit gate — ChatOllama is never instantiated live. All tests patch
langchain_ollama.ChatOllama so no Ollama server is required.

Key cases:
  - get_llm() returns the same object on repeated calls (lru_cache singleton)
  - get_llm() with different args returns different instances (cache key includes all args)
  - ChatOllama is constructed with the correct num_predict / num_ctx
"""

from unittest.mock import MagicMock, patch

import pytest

# Clear the lru_cache before each test so tests are independent
from src.llm.factory import get_llm


@pytest.fixture(autouse=True)
def clear_llm_cache() -> None:
    """Reset the lru_cache before every test to prevent cross-test singleton leakage."""
    get_llm.cache_clear()
    yield
    get_llm.cache_clear()


class TestGetLlm:
    @patch("src.llm.factory.ChatOllama")
    def test_returns_singleton_on_repeated_calls(self, mock_cls: MagicMock) -> None:
        """Two calls with identical args must return the same object — not construct twice."""
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        first = get_llm(model="gemma4:26b", temperature=0.0, num_ctx=8192)
        second = get_llm(model="gemma4:26b", temperature=0.0, num_ctx=8192)

        assert first is second
        # ChatOllama constructor called exactly once despite two get_llm calls
        mock_cls.assert_called_once()

    @patch("src.llm.factory.ChatOllama")
    def test_different_temperature_gives_different_instance(self, mock_cls: MagicMock) -> None:
        """Cache key includes temperature — distinct temps must yield distinct instances."""
        exact = MagicMock(name="exact")
        light = MagicMock(name="light")
        mock_cls.side_effect = [exact, light]

        llm_exact = get_llm(model="gemma4:26b", temperature=0.0, num_ctx=8192)
        llm_light = get_llm(model="gemma4:26b", temperature=0.1, num_ctx=8192)

        assert llm_exact is not llm_light
        assert mock_cls.call_count == 2

    @patch("src.llm.factory.ChatOllama")
    def test_different_model_gives_different_instance(self, mock_cls: MagicMock) -> None:
        """Cache key includes model — distinct model tags yield distinct instances."""
        mock_cls.side_effect = [MagicMock(), MagicMock()]

        a = get_llm(model="gemma4:26b", temperature=0.0, num_ctx=8192)
        b = get_llm(model="gemma4:e4b", temperature=0.0, num_ctx=8192)

        assert a is not b
        assert mock_cls.call_count == 2

    @patch("src.llm.factory.ChatOllama")
    def test_different_num_ctx_gives_different_instance(self, mock_cls: MagicMock) -> None:
        """Cache key includes num_ctx — 8192 and 16384 must be separate singletons."""
        mock_cls.side_effect = [MagicMock(), MagicMock()]

        default_ctx = get_llm(model="gemma4:26b", temperature=0.2, num_ctx=8192)
        large_ctx = get_llm(model="gemma4:26b", temperature=0.2, num_ctx=16384)

        assert default_ctx is not large_ctx
        assert mock_cls.call_count == 2

    @patch("src.llm.factory.ChatOllama")
    def test_constructed_with_correct_kwargs(self, mock_cls: MagicMock, monkeypatch) -> None:
        """ChatOllama must receive num_predict=2048, the caller-supplied num_ctx, and base_url."""
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        mock_cls.return_value = MagicMock()

        get_llm(model="gemma4:26b", temperature=0.0, num_ctx=8192)

        mock_cls.assert_called_once_with(
            model="gemma4:26b",
            temperature=0.0,
            num_predict=2048,
            num_ctx=8192,
            base_url="http://localhost:11434",
            request_timeout=120.0,
        )

    @patch("src.llm.factory.ChatOllama")
    def test_skills_synthesizer_num_ctx(self, mock_cls: MagicMock) -> None:
        """skills_synthesizer requests num_ctx=16384; verify it is forwarded correctly."""
        mock_cls.return_value = MagicMock()

        get_llm(model="gemma4:26b", temperature=0.2, num_ctx=16384)

        _, kwargs = mock_cls.call_args
        assert kwargs["num_ctx"] == 16384

    @patch("src.llm.factory.ChatOllama")
    def test_defaults(self, mock_cls: MagicMock, monkeypatch) -> None:
        """Calling get_llm() with no args uses spec-defined defaults."""
        monkeypatch.setenv("OLLAMA_HOST", "http://localhost:11434")
        mock_cls.return_value = MagicMock()

        get_llm()

        mock_cls.assert_called_once_with(
            model="gemma4:26b",
            temperature=0.0,
            num_predict=2048,
            num_ctx=8192,
            base_url="http://localhost:11434",
            request_timeout=120.0,
        )
