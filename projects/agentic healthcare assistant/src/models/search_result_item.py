"""SearchResultItem — a single result from a web or Medline search."""

from pydantic import BaseModel, Field


class SearchResultItem(BaseModel):
    """A single result from a web or Medline search."""

    title: str = Field(description="Title of the retrieved article or page")
    url: str = Field(description="Source URL")
    snippet: str = Field(description="Relevant excerpt from the source")
    source_domain: str = Field(
        description="Domain of the source, e.g. 'medlineplus.gov'"
    )
