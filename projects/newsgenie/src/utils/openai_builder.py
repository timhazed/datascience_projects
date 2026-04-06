from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.data import LLMConfig
from src.utils.provider_builder import LLMProviderBuilder


class OpenAIBuilder(LLMProviderBuilder):
    """Builds ChatOpenAI instances. Registered by default at app startup."""

    def build(self, config: LLMConfig) -> BaseChatModel:
        """Construct a ChatOpenAI instance from the provided config."""
        from src.utils.config import settings

        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=settings.openai_api_key or None,
        )
