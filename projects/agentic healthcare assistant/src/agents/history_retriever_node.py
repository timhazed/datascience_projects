"""history_retriever_node — fetches patient history from SQLite + FAISS and summarises.

Combines structured PatientRecord fields with FAISS semantic chunks into a
retrieved_chunks string, then invokes the history chain. Uses state["patient_id"]
as the authoritative identifier.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.models.graph_state import HealthcareState
from src.models.retrieve_history_params import RetrieveHistoryParams
from src.models.task_result import TaskResult
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def make_history_retriever_node(
    patient_db: PatientDB,
    db_path: str,
    vector_store: PatientVectorStore,
    chain: Any,
    faiss_k: int = 3,
) -> Callable[[HealthcareState], dict]:
    """Return a history_retriever_node with all dependencies injected via closure.

    Args:
        patient_db: PatientDB instance for structured record lookup.
        db_path: SQLite database path.
        vector_store: PatientVectorStore for semantic chunk retrieval.
        chain: Compiled history chain — history_prompt | llm | StrOutputParser.
        faiss_k: Number of nearest-neighbour chunks to retrieve from FAISS.
            Defaults to 3; increase for patients with many encounter records.

    Returns:
        LangGraph node function compatible with HealthcareState.
    """

    def history_retriever_node(state: HealthcareState) -> dict:
        """Retrieve patient history and produce a clinical summary.

        Always dequeues pending_tasks[0]. Uses state["patient_id"] for lookups —
        not params.patient_id, since the resolver may have updated it after the
        planner generated the params.

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
                        task="retrieve_history",
                        success=False,
                        error="Node called with empty pending_tasks — graph routing error.",
                    )
                ],
                "pending_tasks": [],
                "trace": trace_prior + ["history_retriever: called with empty queue"],
            }
        remaining = pending[1:]

        try:
            params = RetrieveHistoryParams(**pending[0].parameters)
        except ValidationError as exc:
            logger.warning("[history_retriever] invalid params: %s", exc)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="retrieve_history", success=False, error=str(exc))
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"history_retriever: ValidationError — {exc}"],
            }

        patient_id = state.get("patient_id")
        if not patient_id:
            msg = "patient_id not resolved — run resolve_patient first."
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="retrieve_history", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + ["history_retriever: no patient_id in state"],
            }

        record = patient_db.get_patient(db_path, patient_id)
        if record is None:
            msg = f"Patient {patient_id} not found in database."
            logger.warning("[history_retriever] patient %s not found", patient_id)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="retrieve_history", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"history_retriever: patient {patient_id} not found"],
            }

        # Build structured chunk from the patient record fields
        structured_chunk = (
            f"Patient: {record.name} | Age: {record.age} | Gender: {record.gender}\n"
            f"Conditions: {', '.join(record.conditions) or 'none recorded'}\n"
            f"Medications: {', '.join(record.medications) or 'none recorded'}\n"
            f"Allergies: {', '.join(record.allergies) or 'none recorded'}\n"
            f"Summary: {record.summary or 'none recorded'}\n"
            f"Notes: {record.notes or 'none recorded'}"
        )

        query = params.query or f"Medical history for {record.name}"
        faiss_hits = vector_store.search(query, k=faiss_k)
        # Filter to this patient's records only — FAISS searches globally and will return
        # other patients' chunks, which causes the LLM to hallucinate cross-patient data.
        faiss_text = "\n\n".join(
            hit.get("summary", "") or hit.get("conditions", "")
            for hit in faiss_hits
            if hit and hit.get("patient_id") == patient_id
        )

        retrieved_chunks = structured_chunk
        if faiss_text:
            retrieved_chunks = f"{structured_chunk}\n\nSemantic context:\n{faiss_text}"

        try:
            summary = invoke_with_retry(
                chain,
                {"retrieved_chunks": retrieved_chunks, "query": query},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[history_retriever] chain failed: %s", exc)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(
                        task="retrieve_history",
                        success=False,
                        error=f"History chain failed: {type(exc).__name__}",
                    )
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"history_retriever: chain ERROR — {exc}"],
            }

        logger.info("[history_retriever] retrieved history for %s", patient_id)
        return {
            "completed_tasks": completed_prior + [
                TaskResult(
                    task="retrieve_history",
                    success=True,
                    result={"patient_id": patient_id, "summary": summary},
                )
            ],
            "pending_tasks": remaining,
            "trace": trace_prior + [f"history_retriever: success for {patient_id}"],
        }

    return history_retriever_node
