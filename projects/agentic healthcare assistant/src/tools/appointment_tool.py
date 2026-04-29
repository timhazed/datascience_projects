"""appointment_tool — LangChain @tool wrapping AppointmentDB.book_slot.

This tool is used by the experiment runner and any LangChain agent that needs
to book appointments directly. The graph pipeline uses appointment_node instead
(which injects the same DB via factory); this tool is the callable API surface.

Typed input: BookAppointmentParams (validated by LangChain before invoke).
"""

from __future__ import annotations

import logging
from datetime import date

from langchain_core.tools import tool

from src.db.appointment_db import AppointmentDB
from src.models.appointment_result import AppointmentResult
from src.models.book_appointment_params import BookAppointmentParams

logger = logging.getLogger(__name__)


def make_appointment_tool(appointment_db: AppointmentDB, db_path: str):
    """Return a LangChain @tool that books the earliest available appointment slot.

    Wraps AppointmentDB.query_slots and AppointmentDB.book_slot. The tool
    is constructed once at startup with the DB dependencies injected; not
    rebuilt per invocation.

    Args:
        appointment_db: AppointmentDB instance for slot query and booking.
        db_path: SQLite database file path.

    Returns:
        A LangChain tool callable that accepts BookAppointmentParams fields
        and returns an AppointmentResult.
    """

    @tool
    def appointment_tool(
        patient_id: str,
        specialty: str,
        preferred_date: str | None = None,
        urgency: str = "routine",
        reason: str = "",
    ) -> AppointmentResult:
        """Book the earliest available appointment slot for a patient.

        Queries available slots for the requested specialty and urgency, then
        books the first result. Returns AppointmentResult with success=True
        and booking details on success, or success=False with an informative
        message on failure (no slots available, slot race condition).

        Args:
            patient_id: Resolved patient slug ID.
            specialty: Medical specialty, e.g. "Nephrology".
            preferred_date: Optional preferred date as ISO 8601 string (YYYY-MM-DD).
            urgency: Booking urgency — "routine", "urgent", or "emergency".
            reason: Brief clinical reason for the visit.

        Returns:
            AppointmentResult instance.
        """
        # Validate via the existing Pydantic params model to ensure consistency
        # with how appointment_node validates the same input.
        params = BookAppointmentParams(
            patient_id=patient_id,
            specialty=specialty,
            preferred_date=preferred_date,
            urgency=urgency,  # type: ignore[arg-type]
            reason=reason,
        )

        preferred: date | None = None
        if params.preferred_date:
            try:
                preferred = date.fromisoformat(params.preferred_date)
            except ValueError:
                logger.warning(
                    "[appointment_tool] invalid preferred_date: %s", params.preferred_date
                )

        slots = appointment_db.query_slots(
            db_path,
            params.specialty,
            preferred_date=preferred,
            urgency=params.urgency,
        )

        if not slots:
            msg = f"No available slots for {params.specialty} ({params.urgency})."
            logger.warning(
                "[appointment_tool] no slots for %s/%s", params.specialty, params.urgency
            )
            return AppointmentResult(
                success=False,
                patient_id=patient_id,
                message=msg,
            )

        return appointment_db.book_slot(
            db_path,
            slots[0]["slot_id"],
            patient_id,
            reason=params.reason,
            urgency=params.urgency,
        )

    return appointment_tool
