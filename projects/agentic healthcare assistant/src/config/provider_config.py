"""ProviderConfig — LLM provider settings with per-chain token budgets."""

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a single LLM provider (groq or openai).

    All temperature and max_tokens fields are per-chain to allow fine-grained
    control without rebuilding the LLM singleton. Injected at graph construction
    time via get_llm().
    """

    model: str = Field(description="Model identifier, e.g. 'llama-3.3-70b-versatile'")
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description=(
            "Default temperature for deterministic chains (guard, planner, history, appointment)"
        ),
    )
    temperature_summarizer: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for final summarizer chain — controlled paraphrase diversity",
    )
    temperature_search: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Temperature for disease search synthesis chain",
    )
    max_tokens_planner: int = Field(
        default=1024,
        ge=1,
        description=(
            "Planner structured output + reasoning-model overhead; "
            "use ≥1024 for Groq gpt-oss with reasoning_format parsed"
        ),
    )
    max_tokens_history: int = Field(
        default=400,
        ge=1,
        description="Token budget for history summariser (10 fields × 30 t/field × 1.3)",
    )
    max_tokens_search: int = Field(
        default=650,
        ge=1,
        description="Token budget for disease search summary (8 snippets × 60 t/snippet × 1.3)",
    )
    max_tokens_appointment: int = Field(
        default=150,
        ge=1,
        description="Token budget for appointment result (6 fields × 15 t/field × 1.3)",
    )
    max_tokens_summarizer: int = Field(
        default=700,
        ge=1,
        description="Token budget for final summarizer (5 sections × 100 t/section × 1.3)",
    )
    max_tokens_guard: int = Field(
        default=25,
        ge=1,
        description="Token budget for intent guard output (1 field × 10 t × 1.3)",
    )
    max_tokens_memory_summary: int = Field(
        default=260,
        ge=1,
        description=(
            "Token budget for memory summary rollup "
            "(1 summary field, ~200 t/field, 1 item, 1.3 margin ~ 260)."
        ),
    )

    # Groq reasoning controls — gpt-oss models only
    reasoning_effort: str | None = Field(
        default=None,
        description=(
            "Groq reasoning effort — gpt-oss models: 'low', 'medium', 'high'; "
            "qwen3-32b: 'none', 'default'. Leave None for non-reasoning models."
        ),
    )
    reasoning_format: str | None = Field(
        default=None,
        description=(
            "Groq reasoning format — qwen3-32b only: 'parsed', 'raw', 'hidden'. "
            "Do NOT use 'raw' with tool-calling or structured output (400 error). "
            "Leave None for non-reasoning models."
        ),
    )
