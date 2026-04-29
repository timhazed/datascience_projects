"""ResolvePatientParams — parameters for the patient resolver node."""

from pydantic import BaseModel, Field


class ResolvePatientParams(BaseModel):
    """Parameters passed by the planner to the patient resolver node."""

    patient_name: str | None = Field(
        default=None,
        description="Patient full name for fuzzy name lookup",
    )
    phone: str | None = Field(
        default=None,
        description="Phone number for exact lookup if name is ambiguous",
    )
