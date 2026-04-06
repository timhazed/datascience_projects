from pydantic import BaseModel, Field

from src.data.enums import LLMProvider


class LLMConfig(BaseModel):
    """Configuration for a single LLM instantiation. One config = one cache entry in LLMFactory."""

    provider: LLMProvider = Field(description="LLM provider to use")
    model_name: str = Field(description="Model identifier string from provider")
    temperature: float = Field(
        description=(
            "Sampling temperature — must be set explicitly. "
            "Use 0.0 for routing/classification tasks; higher values only for creative generation."
        )
    )
    max_tokens: int = Field(description="Token budget for this specific use case")
