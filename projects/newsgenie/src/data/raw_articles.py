"""
Pre-normalization provider DTOs.

Two raw article models for the two active API backends:
- GuardianRawArticle  — Guardian Open Platform (Sports)
- NewsAPIRawArticle   — NewsAPI /v2/everything (Business + General/World)

MarketAux was evaluated and rejected as the Business lane feed (multi-second latency,
see experiments/news_api_tester/findings/News_API_Validation_Findings.pdf).
It is not implemented.

These models change only when a provider's API contract changes, and are always
imported together by normalizer.py.
"""


from pydantic import BaseModel, ConfigDict, Field


class GuardianRawArticle(BaseModel):
    """Raw response shape from Guardian Open Platform — pre-normalization."""

    model_config = ConfigDict(populate_by_name=True)

    id: str
    web_title: str = Field(alias="webTitle")
    web_url: str = Field(alias="webUrl")
    web_publication_date: str = Field(alias="webPublicationDate")
    section_name: str = Field(alias="sectionName")
    fields: dict | None = None


class NewsAPIRawArticle(BaseModel):
    """Raw response shape from NewsAPI /v2/everything — pre-normalization."""

    model_config = ConfigDict(populate_by_name=True)

    source: dict
    author: str | None = None
    title: str
    description: str | None = None
    url: str
    url_to_image: str | None = Field(default=None, alias="urlToImage")
    published_at: str = Field(alias="publishedAt")
    content: str | None = None
