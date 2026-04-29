"""BookAppointmentParams — parameters for the appointment booking node."""

from typing import Literal

from pydantic import BaseModel, Field


class BookAppointmentParams(BaseModel):
    """Parameters passed by the planner to the appointment booking node."""

    patient_id: str = Field(description="Resolved slug ID of the patient")
    specialty: str = Field(
        description="Required medical specialty, e.g. 'Nephrology'"
    )
    preferred_date: str | None = Field(
        default=None,
        description="Preferred date as YYYY-MM-DD string",
    )
    urgency: Literal["routine", "urgent", "emergency"] = Field(default="routine")
    reason: str = Field(default="")
