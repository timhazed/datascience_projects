"""history_tool — LangChain @tool wrapping PatientDB.update_field + FAISS upsert.

This tool is used by the experiment runner and any LangChain agent that needs
to update patient history fields directly. The graph pipeline uses
history_writer_node instead (which injects the same DB via factory).

Typed input: UpdateHistoryParams (validated by LangChain before invoke).
"""

from __future__ import annotations

import logging

from langchain_core.tools import tool

from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.models.history_update_result import HistoryUpdateResult
from src.models.update_history_params import UpdateHistoryParams

logger = logging.getLogger(__name__)


def make_history_tool(
    patient_db: PatientDB,
    db_path: str,
    vector_store: PatientVectorStore,
):
    """Return a LangChain @tool that updates a patient history field and re-upserts FAISS.

    The DB write is the authoritative operation. FAISS re-upsert is best-effort
    and non-fatal — a failure is logged but does not propagate to the caller.

    Args:
        patient_db: PatientDB instance for update_field and get_patient.
        db_path: SQLite database file path.
        vector_store: PatientVectorStore — re-upserted after every successful write.

    Returns:
        A LangChain tool callable that accepts UpdateHistoryParams fields
        and returns a HistoryUpdateResult.
    """

    @tool
    def history_tool(
        patient_id: str,
        field: str,
        value: str,
        operation: str = "append",
    ) -> HistoryUpdateResult:
        """Update a clinical field on a patient record and refresh the FAISS index.

        Supported fields: conditions, medications, allergies, notes, summary.
        Supported operations: "append" (add to list or concatenate) or "replace".

        Args:
            patient_id: Resolved patient slug ID.
            field: Field to update — one of conditions, medications, allergies,
                notes, summary.
            value: Value to write.
            operation: "append" or "replace".

        Returns:
            HistoryUpdateResult with success flag and confirmation message.
        """
        # Validate via the existing Pydantic params model to ensure field/operation
        # constraints are enforced consistently with history_writer_node.
        params = UpdateHistoryParams(
            patient_id=patient_id,
            field=field,  # type: ignore[arg-type]
            value=value,
            operation=operation,  # type: ignore[arg-type]
        )

        updated = patient_db.update_field(
            db_path, params.patient_id, params.field, params.value, params.operation
        )

        if not updated:
            msg = f"Patient {patient_id} not found — update failed."
            logger.warning("[history_tool] patient %s not found", patient_id)
            return HistoryUpdateResult(
                patient_id=patient_id,
                field_updated=params.field,
                success=False,
                confirmation="",
                error=msg,
            )

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
                logger.warning("[history_tool] FAISS upsert failed (non-fatal): %s", exc)

        confirmation = (
            f"Field '{params.field}' on patient {patient_id} "
            f"{'appended' if params.operation == 'append' else 'replaced'} successfully."
        )
        logger.info("[history_tool] updated field '%s' for %s", params.field, patient_id)
        return HistoryUpdateResult(
            patient_id=patient_id,
            field_updated=params.field,
            success=True,
            confirmation=confirmation,
        )

    return history_tool
