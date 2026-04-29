"""RetrieveHistoryParams — parameters for the history retriever node."""

from pydantic import BaseModel, Field


class RetrieveHistoryParams(BaseModel):
    """Parameters passed by the planner to the history retriever node."""

    patient_id: str = Field(
        description="Resolved slug ID — populated by resolve_patient node"
    )
    query: str | None = Field(
        default=None,
        description="Optional specific query about patient history",
    )
