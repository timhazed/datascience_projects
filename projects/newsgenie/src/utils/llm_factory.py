from langchain_core.language_models import BaseChatModel

from src.data import LLMConfig, LLMProvider
from src.utils.openai_builder import OpenAIBuilder
from src.utils.provider_builder import LLMProviderBuilder


class LLMFactory:
    """
    Singleton-per-config LLM constructor. Never instantiated per-request.

    Cache key includes provider, model_name, temperature, AND max_tokens so that
    configs with different budgets or temperatures never share an instance even
    when using the same underlying model.

    Extending to a new provider: implement LLMProviderBuilder, then call
    LLMFactory.register(LLMProvider.NEW, NewBuilder()) once at app startup.
    No changes to this class required.
    """

    _instances: dict[str, BaseChatModel] = {}
    _builders: dict[LLMProvider, LLMProviderBuilder] = {
        LLMProvider.OPENAI: OpenAIBuilder(),
    }

    @classmethod
    def register(cls, provider: LLMProvider, builder: LLMProviderBuilder) -> None:
        """Register a builder for a new provider. Call once at app startup."""
        cls._builders[provider] = builder

    @classmethod
    def reset(cls) -> None:
        """Clear all cached instances and reset builders to the built-in defaults.
        Intended for use in tests only — call in a conftest autouse fixture to prevent
        instance bleed across test cases.
        """
        cls._instances.clear()
        cls._builders = {LLMProvider.OPENAI: OpenAIBuilder()}

    @classmethod
    def supported_providers(cls) -> list[LLMProvider]:
        """Return all currently registered providers. Used by UI to populate provider selectbox."""
        return list(cls._builders.keys())

    @classmethod
    def get(cls, config: LLMConfig) -> BaseChatModel:
        """Return a cached LLM instance for this exact config. Build once, reuse forever."""
        key = f"{config.provider}:{config.model_name}:{config.temperature}:{config.max_tokens}"
        if key not in cls._instances:
            if config.provider not in cls._builders:
                raise ValueError(
                    f"No builder registered for provider '{config.provider}'. "
                    f"Call LLMFactory.register() at startup to add support."
                )
            cls._instances[key] = cls._builders[config.provider].build(config)
        return cls._instances[key]
