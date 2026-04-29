"""UpdateHistoryParams — parameters for the history writer node."""

from typing import Literal

from pydantic import BaseModel, Field

from src.models.enums import HistoryField


class UpdateHistoryParams(BaseModel):
    """Parameters passed by the planner to the history writer node."""

    patient_id: str = Field(
        description="Resolved slug ID of the patient to update"
    )
    field: HistoryField = Field(
        description="Which field to update"
    )  # canonical Literal imported from enums.py — no duplication
    value: str = Field(description="Value to append or replace")
    operation: Literal["append", "replace"] = Field(default="append")
