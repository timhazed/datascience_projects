from pydantic import BaseModel, Field, field_validator

from src.data.enums import NewsCategory


class UserQuery(BaseModel):
    """Validated user input. Never passed directly to an LLM — used to construct prompts."""

    text: str = Field(min_length=1, max_length=1000, description="Raw user input text")
    categories: list[NewsCategory] = Field(
        default_factory=list,
        description=(
            "Categories selected in the UI. "
            "Empty list = no restriction (all routing destinations allowed; no post-filter). "
            "Non-empty = keep only plan steps in those categories."
        ),
    )
    session_id: str = Field(description="Unique session identifier for conversation context")

    @field_validator("text")
    @classmethod
    def sanitize_text(cls, v: str) -> str:
        """Strip whitespace; reject blank strings and special-character-only input."""
        v = v.strip()
        if not any(c.isalnum() for c in v):
            raise ValueError("Query must contain at least one alphanumeric character")
        return v
