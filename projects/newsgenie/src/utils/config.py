from pydantic_settings import BaseSettings, SettingsConfigDict

from src.data.enums import LLMProvider, WebSearchProvider
from src.data.llm_config import LLMConfig
from src.utils.constants import MAX_HISTORY_TURNS


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables and .env file.
    All API key fields have empty-string defaults — a missing key disables
    that agent/provider gracefully rather than crashing at startup.
    LLM_PROVIDER must be set to a registered provider or startup fails.
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_provider: LLMProvider = LLMProvider.OPENAI
    supervisor_model: str = "gpt-4o-mini"

    # LLM API keys — only the active provider's key is required at runtime
    openai_api_key: str = ""
    groq_api_key: str = ""

    # News API keys — empty string disables that agent gracefully
    # Business + General/World share NEWSAPI_API_KEY
    guardian_api_key: str = ""
    newsapi_api_key: str = ""
    # Web search — one provider per run; only the active provider's key is required
    web_search_provider: WebSearchProvider = WebSearchProvider.SERPAPI
    serpapi_api_key: str = ""      # Required when web_search_provider=serpapi
    serperdev_api_key: str = ""    # Required when web_search_provider=serper

    max_articles_per_agent: int = 5
    max_history_turns: int = MAX_HISTORY_TURNS
    log_level: str = "INFO"


settings = Settings()  # Module-level singleton — loaded once at import time

# One LLM config in v1.0 — supervisor_node only.
# (60 avg tokens/line × 3 max lines) × 1.3 safety = 234; 300 adds headroom for long sub-queries.
SUPERVISOR_LLM_CONFIG = LLMConfig(
    provider=settings.llm_provider,
    model_name=settings.supervisor_model,
    temperature=0.0,
    max_tokens=300,
)
