"""Shared fixtures for tests/graph/ — inject fake env vars and reset LLMFactory."""


import pytest

from src.utils.llm_factory import LLMFactory


@pytest.fixture(autouse=True)
def reset_llm_factory():
    """Reset LLMFactory between tests to prevent instance bleed."""
    LLMFactory.reset()
    yield
    LLMFactory.reset()


@pytest.fixture(autouse=True)
def fake_llm_api_keys(monkeypatch):
    """
    Inject fake API keys so LangChain provider clients can be instantiated without
    a real .env file or OS-level env vars. Tests that invoke actual LLM calls must
    mock the client; these keys only satisfy SDK initialisation guards.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake-openai-key-for-tests")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake-groq-key-for-tests")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
