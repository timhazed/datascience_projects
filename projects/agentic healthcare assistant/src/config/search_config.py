"""SearchConfig — web search provider selection."""

from typing import Literal

from pydantic import BaseModel, Field


class SearchConfig(BaseModel):
    """Configuration for the web search provider used by the disease search node."""

    provider: Literal["serpapi", "serper"] = Field(
        description="Active search provider; switch here to change — no code change required"
    )
