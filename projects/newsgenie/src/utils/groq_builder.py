from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq

from src.data import LLMConfig
from src.utils.provider_builder import LLMProviderBuilder


class GroqBuilder(LLMProviderBuilder):
    """Builds ChatGroq instances. Supported in v1.0 — plain text output requires no JSON schema mode."""

    def build(self, config: LLMConfig) -> BaseChatModel:
        """Construct a ChatGroq instance from the provided config."""
        from src.utils.config import settings

        return ChatGroq(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            api_key=settings.groq_api_key or None,
        )
