from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.config.settings import Settings, get_settings


def get_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Factory for creating LLM instances.

    Args:
        provider: Provider name ("openai"). Defaults to settings.default_provider.
        model: Model name. Defaults to provider's configured model.
        temperature: Temperature for generation. Defaults to provider's configured value.
        max_tokens: Max tokens for generation. Defaults to provider's configured value.
        settings: Settings instance. Defaults to global settings.
        **kwargs: Additional provider-specific arguments.

    Returns:
        Configured LLM instance.

    Raises:
        ValueError: If provider is not supported.
    """
    if settings is None:
        settings = get_settings()

    provider_name = provider or settings.default_provider
    provider_config = settings.get_provider_config(provider_name)

    final_model = model or provider_config.model
    final_temperature = temperature if temperature is not None else provider_config.temperature
    final_max_tokens = max_tokens or provider_config.max_tokens

    if provider_name == "openai":
        return ChatOpenAI(
            model=final_model,
            temperature=final_temperature,
            max_tokens=final_max_tokens,
            api_key=settings.openai_api_key or None,
            **kwargs,
        )
    raise ValueError(f"Unsupported provider: {provider_name}. Supported: openai")


def get_llm_for_agent(agent_name: str, settings: Settings | None = None) -> BaseChatModel:
    """
    Get an LLM configured for a specific agent.

    Uses the agent's configuration from settings to determine provider and parameters.

    Args:
        agent_name: Name of the agent (e.g., "kinesiologist", "gatekeeper").
        settings: Settings instance. Defaults to global settings.

    Returns:
        LLM configured with agent-specific parameters.
    """
    if settings is None:
        settings = get_settings()

    params = settings.get_llm_params(agent_name)

    return get_llm(
        provider=params["provider"],
        model=params["model"],
        temperature=params["temperature"],
        max_tokens=params["max_tokens"],
        settings=settings,
    )
