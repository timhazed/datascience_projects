"""Tests for src/utils/config.py — Settings loading and SUPERVISOR_LLM_CONFIG."""


from src.data.enums import LLMProvider
from src.utils.constants import MAX_HISTORY_TURNS


class TestSettings:
    def test_default_llm_provider_is_openai(self):
        """The Pydantic field default for llm_provider must be OPENAI (env-independent)."""
        from src.utils.config import Settings
        assert Settings.model_fields["llm_provider"].default == LLMProvider.OPENAI

    def test_api_keys_are_strings(self):
        """All API key fields must be strings (loaded or defaulting to empty)."""
        from src.utils.config import settings
        assert isinstance(settings.openai_api_key, str)
        assert isinstance(settings.guardian_api_key, str)
        assert isinstance(settings.newsapi_api_key, str)
        assert isinstance(settings.serpapi_api_key, str)

    def test_max_history_turns_default_matches_constant(self):
        """The Pydantic field default for max_history_turns must equal MAX_HISTORY_TURNS."""
        from src.utils.config import Settings
        assert Settings.model_fields["max_history_turns"].default == MAX_HISTORY_TURNS

    def test_max_articles_per_agent_is_positive(self):
        from src.utils.config import settings
        assert settings.max_articles_per_agent > 0


class TestSupervisorLLMConfig:
    def test_temperature_is_zero(self):
        from src.utils.config import SUPERVISOR_LLM_CONFIG
        assert SUPERVISOR_LLM_CONFIG.temperature == 0.0

    def test_max_tokens_is_positive(self):
        from src.utils.config import SUPERVISOR_LLM_CONFIG
        assert SUPERVISOR_LLM_CONFIG.max_tokens > 0

    def test_provider_matches_settings(self):
        from src.utils.config import SUPERVISOR_LLM_CONFIG, settings
        assert SUPERVISOR_LLM_CONFIG.provider == settings.llm_provider

    def test_model_name_matches_settings(self):
        from src.utils.config import SUPERVISOR_LLM_CONFIG, settings
        assert SUPERVISOR_LLM_CONFIG.model_name == settings.supervisor_model
