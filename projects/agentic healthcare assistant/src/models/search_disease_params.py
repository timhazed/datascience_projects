"""SearchDiseaseParams — parameters for the disease search node."""

from pydantic import BaseModel, Field


class SearchDiseaseParams(BaseModel):
    """Parameters passed by the planner to the disease search node."""

    query: str = Field(description="Clinical search query string")
    max_results: int = Field(default=5, ge=1, le=10)
