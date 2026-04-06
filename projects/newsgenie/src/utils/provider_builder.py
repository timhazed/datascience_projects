from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel

from src.data import LLMConfig


class LLMProviderBuilder(ABC):
    """
    Abstract builder contract for LLM provider construction.
    One concrete subclass per supported LLM provider.
    Implement this class and register with LLMFactory.register() to add a new provider.
    """

    @abstractmethod
    def build(self, config: LLMConfig) -> BaseChatModel:
        """Construct and return a fully configured LLM instance for this provider."""
