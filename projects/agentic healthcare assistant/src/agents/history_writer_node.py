"""history_writer_node — updates a clinical field in SQLite and re-upserts FAISS.

Pure Python — no LLM call. FAISS upsert failure is non-fatal; the DB write
is the authoritative operation. Uses state["patient_id"] as authoritative.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pydantic import ValidationError

from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.models.graph_state import HealthcareState
from src.models.task_result import TaskResult
from src.models.update_history_params import UpdateHistoryParams

logger = logging.getLogger(__name__)


def make_history_writer_node(
    patient_db: PatientDB,
    db_path: str,
    vector_store: PatientVectorStore,
) -> Callable[[HealthcareState], dict]:
    """Return a history_writer_node with DB and vector store injected via closure.

    Args:
        patient_db: PatientDB instance for update_field.
        db_path: SQLite database path.
        vector_store: PatientVectorStore — re-upserted after every successful write
            so that subsequent history retrievals reflect the new value.

    Returns:
        LangGraph node function compatible with HealthcareState.
    """

    def history_writer_node(state: HealthcareState) -> dict:
        """Append or replace a clinical field, then re-upsert the FAISS vector.

        Always dequeues pending_tasks[0]. Pure Python — no LLM call.

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
                        task="update_history",
                        success=False,
                        error="Node called with empty pending_tasks — graph routing error.",
                    )
                ],
                "pending_tasks": [],
                "trace": trace_prior + ["history_writer: called with empty queue"],
            }
        remaining = pending[1:]

        try:
            params = UpdateHistoryParams(**pending[0].parameters)
        except ValidationError as exc:
            logger.warning("[history_writer] invalid params: %s", exc)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="update_history", success=False, error=str(exc))
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"history_writer: ValidationError — {exc}"],
            }

        patient_id = state.get("patient_id")
        if not patient_id:
            msg = "patient_id not resolved — run resolve_patient first."
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="update_history", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + ["history_writer: no patient_id in state"],
            }

        updated = patient_db.update_field(
            db_path, patient_id, params.field, params.value, params.operation
        )

        if not updated:
            msg = f"Patient {patient_id} not found — update failed."
            logger.warning("[history_writer] patient %s not found", patient_id)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="update_history", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"history_writer: patient {patient_id} not found"],
            }

        # Re-fetch and re-upsert FAISS — failure is non-fatal, log only
        record = patient_db.get_patient(db_path, patient_id)
        if record is not None:
            text = f"{record.summary}\n{record.notes}".strip()
            metadata = {
                "patient_id": record.patient_id,
                "name": record.name,
                "age": record.age,
                "gender": record.gender,
                "summary": record.summary,
                "conditions": ", ".join(record.conditions),
            }
            try:
                vector_store.upsert(patient_id, text or record.name, metadata)
            except Exception as exc:  # noqa: BLE001
                logger.warning("[history_writer] FAISS upsert failed (non-fatal): %s", exc)

        logger.info(
            "[history_writer] updated field '%s' for %s (%s)",
            params.field,
            patient_id,
            params.operation,
        )
        return {
            "completed_tasks": completed_prior + [
                TaskResult(
                    task="update_history",
                    success=True,
                    result={
                        "patient_id": patient_id,
                        "field": params.field,
                        "operation": params.operation,
                    },
                )
            ],
            "pending_tasks": remaining,
            "trace": trace_prior + [
                f"history_writer: updated {params.field} for {patient_id} ({params.operation})"
            ],
        }

    return history_writer_node
