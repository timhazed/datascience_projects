"""patient_resolver_node — resolves patient name to a slug patient_id.

Uses PatientDB.fuzzy_search. Returns a failure TaskResult (non-fatal) when
the name is ambiguous (>1 match) or not found (0 matches), so the graph
can surface the issue to the user without halting the session.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import ValidationError

from src.db.patient_db import PatientDB
from src.models.graph_state import HealthcareState
from src.models.resolve_patient_params import ResolvePatientParams
from src.models.task_result import TaskResult

logger = logging.getLogger(__name__)


def make_patient_resolver_node(
    patient_db: PatientDB,
    db_path: str,
) -> Callable[[HealthcareState], dict]:
    """Return a patient_resolver_node with DB dependency injected via closure.

    Args:
        patient_db: PatientDB instance for fuzzy name lookup.
        db_path: SQLite database path.

    Returns:
        LangGraph node function compatible with HealthcareState.
    """

    def patient_resolver_node(state: HealthcareState) -> dict:
        """Resolve patient name to slug ID via SQLite fuzzy search.

        Always dequeues pending_tasks[0]. Sets patient_id on exact match.
        Returns failure TaskResult (and sets error) on ambiguous or not-found
        — non-fatal per State Contract invariant 4.

        Args:
            state: Current HealthcareState.

        Returns:
            Dict with completed_tasks, pending_tasks[1:], and optionally patient_id / error.
        """
        # Accumulate under REPLACE semantics — read prior lists and extend them.
        completed_prior = list(state.get("completed_tasks", []))
        trace_prior = list(state.get("trace", []))

        pending = state["pending_tasks"]
        if not pending:
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(
                        task="resolve_patient",
                        success=False,
                        error="Node called with empty pending_tasks — graph routing error.",
                    )
                ],
                "pending_tasks": [],
                "trace": trace_prior + ["patient_resolver: called with empty queue"],
            }
        remaining = pending[1:]

        try:
            params = ResolvePatientParams(**pending[0].parameters)
        except ValidationError as exc:
            logger.warning("[resolver] invalid params: %s", exc)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="resolve_patient", success=False, error=str(exc))
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"patient_resolver: ValidationError — {exc}"],
            }

        search_name = params.patient_name or ""
        if not search_name:
            msg = "Patient name is required for resolution."
            logger.warning("[resolver] no patient_name provided")
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="resolve_patient", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "error": msg,
                "trace": trace_prior + ["patient_resolver: no name provided"],
            }

        matches = patient_db.fuzzy_search(db_path, search_name, limit=5)

        if len(matches) == 0:
            msg = f"No patient found matching '{search_name}'."
            logger.warning("[resolver] no match for '%s'", search_name)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="resolve_patient", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "error": msg,
                "trace": trace_prior + [f"patient_resolver: no match for '{search_name}'"],
            }

        if len(matches) > 1:
            # Log full candidate list internally; do NOT surface other patients'
            # names to the requesting user (PHI privacy constraint).
            candidate_names = [r.name for r in matches]
            logger.warning("[resolver] ambiguous '%s': %d candidates", search_name, len(matches))
            logger.debug("[resolver] candidates: %s", candidate_names)
            msg = (
                f"Multiple patients ({len(matches)}) match '{search_name}'. "
                "Please provide a more specific name or include a phone number."
            )
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="resolve_patient", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "error": msg,
                "trace": trace_prior + [
                    f"patient_resolver: ambiguous '{search_name}' — {len(matches)} matches"
                ],
            }

        resolved = matches[0]
        logger.info("[resolver] resolved '%s' → %s", search_name, resolved.patient_id)
        return {
            "patient_id": resolved.patient_id,
            "completed_tasks": completed_prior + [
                TaskResult(
                    task="resolve_patient",
                    success=True,
                    result={"patient_id": resolved.patient_id, "name": resolved.name},
                )
            ],
            "pending_tasks": remaining,
            "trace": trace_prior + [
                f"patient_resolver: resolved '{search_name}' → {resolved.patient_id}"
            ],
        }

    return patient_resolver_node
