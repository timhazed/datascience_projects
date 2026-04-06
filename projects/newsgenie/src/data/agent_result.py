from pydantic import BaseModel, Field

from src.data.enums import AgentName, NewsCategory
from src.data.normalized_article import NormalizedArticle


class NewsAgentResult(BaseModel):
    """Output of a single agent execution after fetching and normalizing articles."""

    agent: AgentName = Field(
        description="Which agent produced this result — used to match back to the plan step"
    )
    category: NewsCategory = Field(description="Category this agent owns")
    articles: list[NormalizedArticle] = Field(description="Normalized articles returned")
    query_used: str = Field(description="Actual query string sent to the API or search tool")
    success: bool = Field(description="Whether the API/tool call succeeded")
    error_message: str | None = Field(
        default=None, description="Error detail if success=False"
    )
    latency_ms: int = Field(description="API round-trip time in milliseconds")
