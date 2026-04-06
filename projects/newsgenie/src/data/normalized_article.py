from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from src.data.enums import NewsCategory


class NormalizedArticle(BaseModel):
    """Canonical article schema across all providers. Output of every normalizer function."""

    article_id: str = Field(
        description="Unique identifier, provider-prefixed (e.g. newsapi_abc123)"
    )
    title: str = Field(description="Article headline")
    summary: str = Field(description="Article summary or snippet, max 500 chars")
    url: str = Field(description="Canonical URL to full article")
    source_name: str = Field(description="Name of the publication or feed")
    published_at: datetime = Field(description="Publication timestamp in UTC")
    category: NewsCategory = Field(description="News category this article belongs to")
    provider: str = Field(
        description="Source id: newsapi | guardian | serpapi (or other web search adapter)"
    )
    image_url: str | None = Field(
        default=None, description="Thumbnail image URL if available"
    )
    sentiment_score: float | None = Field(
        default=None, description="Sentiment score -1.0 to 1.0 if provider supplies it"
    )
    tags: list[str] = Field(default_factory=list, description="Topic tags or entities")

    @field_validator("published_at", mode="before")
    @classmethod
    def parse_published_at(cls, v: Any) -> datetime:
        """Coerce any date string to datetime.

        Handles ISO 8601 (NewsAPI, Guardian), human-readable strings from SerpAPI
        (e.g. 'Mar 31, 2026', '3 hours ago'), and datetime objects.
        Falls back to Unix epoch (UTC) for unparseable values so that validation
        never raises on a missing or malformed date.
        """
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            from dateutil import parser as dtparser

            try:
                return dtparser.parse(v)
            except (ValueError, OverflowError):
                return datetime.fromtimestamp(0, tz=UTC)
        return datetime.fromtimestamp(0, tz=UTC)
