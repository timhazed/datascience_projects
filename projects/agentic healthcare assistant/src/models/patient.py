"""PatientRecord — maps to records.xlsx schema + extended clinical fields."""

from datetime import datetime

from pydantic import BaseModel, Field


class PatientRecord(BaseModel):
    """Maps directly to records.xlsx schema + extended clinical fields.

    xlsx columns: Phone_number, Email, Name, Age, Gender, Address, Summary.
    patient_id is a derived slug generated at load time (P-<hash6 of phone+name>).
    """

    patient_id: str = Field(
        description="Derived slug identifier, e.g. P-4a2f91 (generated from phone+name hash)"
    )
    phone: str = Field(description="Patient phone number, e.g. '+91-98180-11245'")
    email: str | None = Field(default=None, description="Patient email address; may be null")
    name: str = Field(description="Full legal name of the patient")
    age: int = Field(ge=0, le=130, description="Age in years")
    gender: str = Field(description="Patient gender, e.g. 'Male' or 'Female'")
    address: str = Field(default="", description="Patient address from records.xlsx")
    summary: str = Field(
        default="",
        description="Pre-existing summary from records.xlsx or generated from PDF report",
    )
    conditions: list[str] = Field(
        default_factory=list,
        description=(
            "List of active chronic or acute diagnoses, "
            "e.g. ['chronic kidney disease', 'hypertension']"
        ),
    )
    medications: list[str] = Field(
        default_factory=list,
        description="List of current medications with dosage, e.g. ['metformin 1000mg BID']",
    )
    allergies: list[str] = Field(
        default_factory=list,
        description="Known drug or food allergies",
    )
    notes: str = Field(
        default="",
        description="Unstructured clinical notes extracted from PDF reports",
    )
    last_updated: datetime = Field(default_factory=datetime.now)
