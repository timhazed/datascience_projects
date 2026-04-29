"""DiseaseSearchResult — aggregated output of a disease information search."""

from pydantic import BaseModel, Field

from src.models.search_result_item import SearchResultItem


class DiseaseSearchResult(BaseModel):
    """Aggregated output of a disease information search including LLM synthesis."""

    query: str = Field(description="Original search query")
    items: list[SearchResultItem] = Field(description="Ranked list of retrieved results")
    summary: str = Field(
        description=(
            "LLM-generated synthesis of the retrieved information (2-4 paragraphs)"
        )
    )
    citations: list[str] = Field(description="List of source URLs cited in the summary")
