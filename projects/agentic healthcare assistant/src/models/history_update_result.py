"""HistoryUpdateResult — outcome of a patient history update operation."""

from pydantic import BaseModel, Field


class HistoryUpdateResult(BaseModel):
    """Result returned after attempting to update a patient history field."""

    patient_id: str = Field(description="Patient identifier that was updated")
    field_updated: str = Field(description="Field that was modified")
    success: bool = Field(description="True if the update was persisted successfully")
    confirmation: str = Field(description="One-sentence confirmation of what was changed")
    error: str | None = Field(default=None, description="Error message if success is False")
