"""Tests for history_retriever_node — real SQLite, mock FAISS, mock chain."""

from datetime import datetime
from unittest.mock import MagicMock

from langchain_core.language_models import FakeListChatModel

from src.agents.history_retriever_node import make_history_retriever_node
from src.chains.history_chain import build_history_chain
from src.db.patient_db import PatientDB
from src.models.patient import PatientRecord
from src.models.sub_goal import SubGoal


def _state(**overrides) -> dict:
    """Build a minimal HealthcareState-compatible dict."""
    base: dict = {
        "user_query": "test",
        "patient_id": "P-001",
        "messages": [],
        "planner_output": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "intent_safe": True,
        "final_summary": None,
        "error": None,
        "trace": [],
    }
    base.update(overrides)
    return base


def _insert(db_path: str) -> None:
    """Insert a minimal test patient."""
    PatientDB().create_patient(
        db_path,
        PatientRecord(
            patient_id="P-001",
            phone="+1-555-0001",
            name="Alice Smith",
            age=45,
            gender="Female",
            conditions=["Hypertension"],
            medications=["lisinopril 10mg"],
            summary="Chronic hypertension managed with ACE inhibitor.",
            last_updated=datetime.now(),
        ),
    )


def _sub_goal(query: str = "history") -> SubGoal:
    """Build a retrieve_history SubGoal."""
    return SubGoal(
        task="retrieve_history",
        parameters={"patient_id": "P-001", "query": query},
        order=1,
    )


def _mock_vector_store(hits: list | None = None) -> MagicMock:
    """Return a mock PatientVectorStore that returns given hits on search."""
    vs = MagicMock()
    vs.search.return_value = hits or []
    return vs


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_returns_summary_string(in_memory_db) -> None:
    """Node invokes chain and returns its string output in completed_tasks."""
    _insert(in_memory_db)
    chain = build_history_chain(FakeListChatModel(responses=["ACE inhibitor treatment."]))
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store(), chain
    )
    result = node(_state(pending_tasks=[_sub_goal()]))
    assert result["completed_tasks"][0].success is True
    assert result["completed_tasks"][0].result["summary"] == "ACE inhibitor treatment."


def test_dequeues_pending_tasks(in_memory_db) -> None:
    """Node always dequeues pending_tasks[0]."""
    _insert(in_memory_db)
    chain = build_history_chain(FakeListChatModel(responses=["summary"]))
    extra = SubGoal(task="search_disease", order=2)
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store(), chain
    )
    result = node(_state(pending_tasks=[_sub_goal(), extra]))
    assert len(result["pending_tasks"]) == 1
    assert result["pending_tasks"][0] is extra


def test_faiss_hits_included_in_chunks(in_memory_db) -> None:
    """Same-patient FAISS hit (patient_id matches) is included in retrieved_chunks."""
    _insert(in_memory_db)
    # patient_id key must match state["patient_id"] = "P-001" for the filter to pass.
    hits = [
        {
            "patient_id": "P-001",
            "summary": "Patient has stage 2 hypertension.",
            "conditions": "Hypertension",
        },
    ]
    captured: list[dict] = []

    def _capture(inv: dict) -> str:
        captured.append(inv)
        return "stage 2 summary"

    from langchain_core.runnables import RunnableLambda
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store(hits), RunnableLambda(_capture)
    )
    result = node(_state(pending_tasks=[_sub_goal("What stage?")]))
    assert result["completed_tasks"][0].success is True
    # FAISS text must appear in the retrieved_chunks passed to the chain.
    assert "stage 2 hypertension" in captured[0]["retrieved_chunks"]


def test_faiss_cross_patient_hit_excluded(in_memory_db) -> None:
    """FAISS hits from a different patient are excluded to prevent cross-patient hallucination."""
    _insert(in_memory_db)
    # patient_id "P-999" does NOT match state["patient_id"] = "P-001" — must be filtered out.
    hits = [{"patient_id": "P-999", "summary": "SECRET_OTHER_PATIENT_DATA", "conditions": ""}]
    captured: list[dict] = []

    def _capture(inv: dict) -> str:
        captured.append(inv)
        return "clean summary"

    from langchain_core.runnables import RunnableLambda
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store(hits), RunnableLambda(_capture)
    )
    result = node(_state(pending_tasks=[_sub_goal()]))
    assert result["completed_tasks"][0].success is True
    # Cross-patient content must not appear in the chain input.
    assert "SECRET_OTHER_PATIENT_DATA" not in captured[0]["retrieved_chunks"]


def test_empty_faiss_hits_still_succeeds(in_memory_db) -> None:
    """Zero FAISS hits is not an error — structured record is sufficient."""
    _insert(in_memory_db)
    chain = build_history_chain(FakeListChatModel(responses=["Hypertension noted."]))
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store([]), chain
    )
    result = node(_state(pending_tasks=[_sub_goal()]))
    assert result["completed_tasks"][0].success is True


# ---------------------------------------------------------------------------
# Failure paths
# ---------------------------------------------------------------------------


def test_no_patient_id_returns_failure(in_memory_db) -> None:
    """Missing patient_id in state → failure TaskResult, dequeues."""
    chain = build_history_chain(FakeListChatModel(responses=["x"]))
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store(), chain
    )
    result = node(_state(patient_id=None, pending_tasks=[_sub_goal()]))
    assert result["completed_tasks"][0].success is False
    assert "patient_id" in result["completed_tasks"][0].error.lower()
    assert len(result["pending_tasks"]) == 0


def test_patient_not_in_db_returns_failure(in_memory_db) -> None:
    """Patient ID not found in SQLite → failure TaskResult."""
    chain = build_history_chain(FakeListChatModel(responses=["x"]))
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store(), chain
    )
    result = node(_state(patient_id="P-ghost", pending_tasks=[_sub_goal()]))
    assert result["completed_tasks"][0].success is False
    assert "not found" in result["completed_tasks"][0].error.lower()


def test_invalid_params_returns_failure(in_memory_db) -> None:
    """Malformed parameters (ValidationError) → failure TaskResult, no crash."""
    chain = build_history_chain(FakeListChatModel(responses=["x"]))
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store(), chain
    )
    bad = SubGoal(task="retrieve_history", parameters={"bad_key": 999}, order=1)
    result = node(_state(pending_tasks=[bad]))
    assert result["completed_tasks"][0].success is False
    assert len(result["pending_tasks"]) == 0


def test_empty_pending_tasks_returns_failure(in_memory_db) -> None:
    """Empty pending_tasks → failure TaskResult, no IndexError."""
    chain = build_history_chain(FakeListChatModel(responses=["x"]))
    node = make_history_retriever_node(
        PatientDB(), in_memory_db, _mock_vector_store(), chain
    )
    result = node(_state(pending_tasks=[]))
    assert result["completed_tasks"][0].success is False
    assert result["pending_tasks"] == []
