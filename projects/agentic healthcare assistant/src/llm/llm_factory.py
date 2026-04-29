"""LLM factory — constructs one BaseChatModel per application boundary.

Reused from hr_assistant with max_tokens and temperature defaults adjusted
to match the token budget spec in Architecture.md Section 4.

Usage:
    from src.llm.llm_factory import get_llm
    llm = get_llm(settings.provider_name, settings.provider_config.model,
                  temperature=settings.provider_config.temperature,
                  max_tokens=settings.provider_config.max_tokens_planner)
"""

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


def get_llm(
    provider: str,
    model: str,
    temperature: float = 0.1,
    max_tokens: int = 512,
    reasoning_effort: str | None = None,
    reasoning_format: str | None = None,
) -> BaseChatModel:
    """Construct one LLM instance per application boundary — not per request.

    Each graph node receives a pre-built instance injected at construction time.
    Never call this inside a per-request function.

    Args:
        provider: LLM provider name — "groq" or "openai".
        model: Model identifier, e.g. "llama-3.3-70b-versatile" or "gpt-4o".
            Note: "openai/gpt-oss-120b" is a Groq-hosted model — use provider="groq".
        temperature: Sampling temperature (0.0–1.0). Default 0.1 for structured
            output chains; use 0.3 for summariser and search chains.
        max_tokens: Maximum tokens in the response. Derived from the token budget
            in Architecture.md Section 4 — do not set to an arbitrary round number.
        reasoning_effort: Groq reasoning effort. gpt-oss models: "low"/"medium"/"high".
            qwen3-32b: "none"/"default". Ignored for non-reasoning models.
        reasoning_format: Groq reasoning format. qwen3-32b only: "parsed"/"raw"/"hidden".
            Do NOT use "raw" with tool-calling or structured output (causes 400 error).
            Ignored for non-reasoning models.

    Returns:
        Configured BaseChatModel instance (ChatGroq or ChatOpenAI).

    Raises:
        ValueError: If provider is not "groq" or "openai".
    """
    result = None
    if provider == "groq":
        # Build model_kwargs only when reasoning controls are requested — passing an
        # empty dict or None values to ChatGroq is harmless but keeping it clean.
        model_kwargs: dict[str, str] = {}
        if reasoning_effort is not None:
            model_kwargs["reasoning_effort"] = reasoning_effort
        if reasoning_format is not None:
            model_kwargs["reasoning_format"] = reasoning_format
        result = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **({"model_kwargs": model_kwargs} if model_kwargs else {}),
        )
    elif provider == "openai":
        result = ChatOpenAI(model=model, temperature=temperature, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unsupported provider: {provider!r}. Expected 'groq' or 'openai'.")
        
    return result
