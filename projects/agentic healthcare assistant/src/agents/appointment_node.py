"""appointment_node — queries available slots and books the earliest one.

Pure Python + SQLite — no LLM call. Uses AppointmentDB.query_slots then
AppointmentDB.book_slot. A failed booking (double-book race) is returned as
a failure TaskResult, not an exception.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import date

from pydantic import ValidationError

from src.db.appointment_db import AppointmentDB
from src.models.book_appointment_params import BookAppointmentParams
from src.models.graph_state import HealthcareState
from src.models.task_result import TaskResult

logger = logging.getLogger(__name__)


def make_appointment_node(
    appointment_db: AppointmentDB,
    db_path: str,
) -> Callable[[HealthcareState], dict]:
    """Return an appointment_node with DB dependency injected via closure.

    Pure Python — no LLM call.

    Args:
        appointment_db: AppointmentDB instance for slot query and booking.
        db_path: SQLite database path.

    Returns:
        LangGraph node function compatible with HealthcareState.
    """

    def appointment_node(state: HealthcareState) -> dict:
        """Select the earliest available slot and book it.

        Always dequeues pending_tasks[0]. Uses state["patient_id"] as the
        authoritative patient identifier. A failed slot booking returns a
        failure TaskResult — does not raise.

        Args:
            state: Current HealthcareState.

        Returns:
            Dict with completed_tasks, pending_tasks[1:], and trace.
        """
        # Accumulate under REPLACE semantics — read prior lists and extend them.
        completed_prior = list(state.get("completed_tasks", []))
        trace_prior = list(state.get("trace", []))

        pending = state["pending_tasks"]
        if not pending:
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(
                        task="book_appointment",
                        success=False,
                        error="Node called with empty pending_tasks — graph routing error.",
                    )
                ],
                "pending_tasks": [],
                "trace": trace_prior + ["appointment: called with empty queue"],
            }
        remaining = pending[1:]

        try:
            params = BookAppointmentParams(**pending[0].parameters)
        except ValidationError as exc:
            logger.warning("[appointment] invalid params: %s", exc)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="book_appointment", success=False, error=str(exc))
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"appointment: ValidationError — {exc}"],
            }

        patient_id = state.get("patient_id")
        if not patient_id:
            msg = "patient_id not resolved — run resolve_patient first."
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="book_appointment", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + ["appointment: no patient_id in state"],
            }

        preferred: date | None = None
        if params.preferred_date:
            try:
                preferred = date.fromisoformat(params.preferred_date)
            except ValueError:
                logger.warning("[appointment] invalid preferred_date: %s", params.preferred_date)

        slots = appointment_db.query_slots(
            db_path,
            params.specialty,
            preferred_date=preferred,
            urgency=params.urgency,
        )

        if not slots:
            msg = f"No available slots for {params.specialty} ({params.urgency})."
            logger.warning("[appointment] no slots for %s/%s", params.specialty, params.urgency)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="book_appointment", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [
                    f"appointment: no slots — {params.specialty}/{params.urgency}"
                ],
            }

        booking = appointment_db.book_slot(
            db_path,
            slots[0]["slot_id"],
            patient_id,
            reason=params.reason,
            urgency=params.urgency,
        )

        if not booking.success:
            logger.warning("[appointment] booking failed: %s", booking.message)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(
                        task="book_appointment",
                        success=False,
                        error=booking.message,
                    )
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"appointment: booking failed — {booking.message}"],
            }

        logger.info("[appointment] booked %s for %s", booking.slot, patient_id)
        return {
            "completed_tasks": completed_prior + [
                TaskResult(
                    task="book_appointment",
                    success=True,
                    result={
                        "appointment_id": booking.appointment_id or "",
                        "doctor": booking.doctor_name or "",
                        "slot": booking.slot or "",
                        "patient_id": patient_id,
                    },
                )
            ],
            "pending_tasks": remaining,
            "trace": trace_prior + [
                f"appointment: booked {params.specialty} at {booking.slot} for {patient_id}"
            ],
        }

    return appointment_node
