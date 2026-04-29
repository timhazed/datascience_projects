"""GraphConfig — LangGraph invoke options (e.g. recursion limit)."""

from pydantic import BaseModel, Field


class GraphConfig(BaseModel):
    """Execution limits for the compiled healthcare StateGraph."""

    recursion_limit: int = Field(
        default=100,
        ge=1,
        description="Maximum graph supersteps per invoke (LangGraph recursion_limit).",
    )