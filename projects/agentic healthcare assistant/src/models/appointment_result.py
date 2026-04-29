"""AppointmentResult — outcome of a booking attempt."""

from pydantic import BaseModel, Field


class AppointmentResult(BaseModel):
    """Result of an appointment booking attempt."""

    success: bool = Field(description="True if an appointment was successfully booked")
    appointment_id: str | None = Field(
        default=None,
        description="Unique booking reference if booked",
    )
    doctor_name: str | None = Field(
        default=None,
        description="Name and specialty of the assigned doctor",
    )
    slot: str | None = Field(
        default=None,
        description="Confirmed ISO 8601 datetime of the appointment",
    )
    patient_id: str = Field(description="Patient for whom the appointment was booked")
    message: str = Field(description="Human-readable confirmation or failure reason")
