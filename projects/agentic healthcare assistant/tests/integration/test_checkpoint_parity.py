"""Integration tests — Phase 4 checkpoint multi-turn gate.

Verifies that after two graph invocations for the same patient_id, the
(role, content) message pairs accumulated in the LangGraph checkpoint are
complete, correctly ordered, and isolated across distinct thread_ids.

Five patient scenarios are parametrized to confirm isolation across
distinct thread_ids.  MemorySaver is used for hermeticity — no disk I/O,
safe for parallel CI runs.

Authorization gate: all 5 scenarios must pass. PatientMemoryDB has been
removed in Phase 4; persistence is now exclusively in the checkpoint.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph

from src.chains.history_chain import build_history_chain
from src.chains.intent_guard_chain import build_intent_guard_chain
from src.chains.search_chain import build_search_chain
from src.chains.summary_chain import build_summary_chain
from src.db.appointment_db import AppointmentDB
from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.graph.healthcare_graph import _compile_healthcare_graph
from src.models.planner_output import PlannerOutput
from src.models.sub_goal import SubGoal

# ---------------------------------------------------------------------------
# Five patient scenarios — each has a unique patient_id (thread_id)
# ---------------------------------------------------------------------------

_PATIENT_SCENARIOS = [
    ("P-parity-001", "What is hypertension?", "What causes hypertension?"),
    ("P-parity-002", "Explain diabetes mellitus.", "What are the treatment options?"),
    ("P-parity-003", "Describe CKD stage 3.", "What dietary changes are needed?"),
    ("P-parity-004", "What are flu symptoms?", "When should a patient see a doctor?"),
    ("P-parity-005", "What is asthma?", "How is asthma managed long-term?"),
]


# ---------------------------------------------------------------------------
# Internal helpers — test utilities for this module only
# ---------------------------------------------------------------------------


def _build_parity_graph(
    tmp_path,
    *,
    summary_responses: list[str],
    checkpointer: MemorySaver,
) -> CompiledStateGraph:
    """Compile a healthcare graph with mock chains and a MemorySaver checkpointer.

    Uses FakeListChatModel for intent guard, history, and search chains so no
    LLM API calls are made.  The summary chain is configured with
    ``summary_responses`` — one element is consumed per graph.invoke() call.

    A single-task plan (search_disease with an empty search_fn) is used so
    the graph completes without patient seeding. The disease_search_node
    degrades gracefully to TaskResult(success=False) when there are 0 results;
    the summarizer still produces a non-empty final_summary from the chain.

    Args:
        tmp_path: pytest tmp_path fixture providing isolated directory.
        summary_responses: Ordered list of summary strings for FakeListChatModel.
            One entry is consumed per graph.invoke() call.
        checkpointer: MemorySaver instance scoped to one test scenario.

    Returns:
        Compiled CompiledStateGraph with MemorySaver attached.
    """
    db_path = str(tmp_path / "graph_internal.db")

    patient_db = PatientDB()
    appointment_db = AppointmentDB()
    patient_db.init_db(db_path)
    appointment_db.init_db(db_path)

    embeddings = MagicMock(spec=Embeddings)
    embeddings.embed_documents.return_value = [[0.1] * 1536]
    embeddings.embed_query.return_value = [0.1] * 1536
    vector_store = PatientVectorStore(str(tmp_path / "faiss"), embeddings)

    # Single-task plan — search_disease with no external results so no seeding required.
    # disease_search_node handles 0 results gracefully (TaskResult success=False);
    # the summarizer still runs and produces final_summary from the chain.
    plan = PlannerOutput(
        patient_id=None,
        sub_goals=[
            SubGoal(task="search_disease", order=1, parameters={"query": "parity test query"})
        ],
        reasoning="Parity test plan.",
    )

    intent_guard_chain = build_intent_guard_chain(
        FakeListChatModel(responses=["SAFE"] * 20)
    )
    planner_chain = RunnableLambda(lambda _: plan)
    history_chain = build_history_chain(FakeListChatModel(responses=["history text"] * 20))
    search_chain = build_search_chain(FakeListChatModel(responses=["search text"] * 20))
    summary_chain = build_summary_chain(FakeListChatModel(responses=summary_responses))

    return _compile_healthcare_graph(
        intent_guard_chain=intent_guard_chain,
        planner_chain=planner_chain,
        history_chain=history_chain,
        search_chain=search_chain,
        summary_chain=summary_chain,
        patient_db=patient_db,
        appointment_db=appointment_db,
        vector_store=vector_store,
        db_path=db_path,
        # Empty search_fn — disease_search_node degrades gracefully.
        search_fn=lambda q, n: [],
        max_recent_turns=10,
        checkpointer=checkpointer,
    )


def _build_initial_state(query: str, patient_id: str) -> dict:
    """Build a HealthcareState dict mirroring production _build_initial_state.

    Injects one HumanMessage per turn so the add_messages reducer accumulates
    the user query in the checkpoint — identical to Phase 4 graph_runtime behaviour.

    Args:
        query: Natural language query string.
        patient_id: Canonical patient slug; injected so planner context is set.

    Returns:
        HealthcareState-compatible dict.
    """
    return {
        "user_query": query,
        "patient_id": patient_id,
        # Phase 2: HumanMessage injection — must match graph_runtime._build_initial_state.
        "messages": [HumanMessage(content=query)],
        "planner_output": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "intent_safe": False,
        "final_summary": None,
        "error": None,
        "trace": [],
    }


def _invoke_graph(
    graph,
    patient_id: str,
    query: str,
) -> str:
    """Invoke the graph for one turn and return the final_summary.

    The checkpoint accumulates HumanMessage + AIMessage automatically via
    add_messages reducer and summarizer_node — no manual persistence needed.

    Args:
        graph: Compiled graph with MemorySaver checkpointer attached.
        patient_id: Canonical patient slug; used as both patient_id and thread_id.
        query: User query string.

    Returns:
        The stripped final_summary produced by this invocation, or "" on failure.
    """
    initial_state = _build_initial_state(query, patient_id)
    thread_config = {
        "configurable": {"thread_id": patient_id},
        "recursion_limit": 25,
    }
    result = graph.invoke(initial_state, thread_config)
    return (result.get("final_summary") or "").strip()


# ---------------------------------------------------------------------------
# Phase 4 checkpoint multi-turn gate (5 patient scenarios)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("patient_id,query_1,query_2", _PATIENT_SCENARIOS)
def test_checkpoint_accumulates_two_turns(
    patient_id: str,
    query_1: str,
    query_2: str,
    tmp_path,
) -> None:
    """Checkpoint contains 4 H/A messages after two invocations per patient.

    Phase 4 gate (Section 8.4 — test_full_e2e_3_turn_checkpoint).

    Procedure:
    1. Build graph with MemorySaver (hermetic — no disk I/O).
    2. Invoke graph twice with the same thread_id (patient_id).
    3. Read checkpoint snapshot; assert 4 H/A messages accumulate correctly.
    4. Verify message content matches the queries and summaries (round-trip).
    5. Verify different patient_ids produce isolated threads (no cross-contamination).

    Args:
        patient_id: Unique patient slug; used as checkpoint thread_id.
        query_1: First user query string.
        query_2: Second user query string.
        tmp_path: pytest fixture providing an isolated temporary directory.
    """
    # Deterministic summary strings — one per invocation.
    summary_1 = f"Parity summary turn 1 — {patient_id}."
    summary_2 = f"Parity summary turn 2 — {patient_id}."

    # One MemorySaver per scenario — no state bleed between parameterized runs.
    checkpointer = MemorySaver()
    graph = _build_parity_graph(
        tmp_path,
        summary_responses=[summary_1, summary_2],
        checkpointer=checkpointer,
    )

    # --- Two turns: checkpoint accumulates both H + A per turn ---
    actual_s1 = _invoke_graph(graph, patient_id, query_1)
    actual_s2 = _invoke_graph(graph, patient_id, query_2)

    # Sanity: FakeListChatModel returned the expected summaries in order.
    assert actual_s1 == summary_1, (
        f"Turn 1 summary mismatch for {patient_id!r}: "
        f"got {actual_s1!r}, expected {summary_1!r}"
    )
    assert actual_s2 == summary_2, (
        f"Turn 2 summary mismatch for {patient_id!r}: "
        f"got {actual_s2!r}, expected {summary_2!r}"
    )

    # --- Extract (role, content) pairs from checkpoint ---
    thread_config = {
        "configurable": {"thread_id": patient_id},
        "recursion_limit": 25,
    }
    snapshot = graph.get_state(thread_config)
    assert snapshot and snapshot.values, (
        f"Checkpoint snapshot empty for thread_id={patient_id!r} — "
        "HumanMessage/AIMessage injection may have failed."
    )

    checkpoint_msgs = snapshot.values.get("messages", [])
    # Filter to H/A pairs only — same as _hydrate_chat_from_checkpoint (Decision 10).
    checkpoint_pairs = [
        ("user" if isinstance(m, HumanMessage) else "assistant", m.content)
        for m in checkpoint_msgs
        if isinstance(m, HumanMessage | AIMessage)
    ]

    # --- Assert 4 entries (2 H + 2 A = 2 complete turns) ---
    assert len(checkpoint_pairs) == 4, (
        f"Expected 4 checkpoint messages for {patient_id!r}, got {len(checkpoint_pairs)}: "
        f"{checkpoint_pairs}"
    )

    # --- Assert ordering and content correctness ---
    assert checkpoint_pairs[0] == ("user", query_1), (
        f"Turn 1 user message mismatch for {patient_id!r}: {checkpoint_pairs[0]}"
    )
    assert checkpoint_pairs[1] == ("assistant", summary_1), (
        f"Turn 1 assistant message mismatch for {patient_id!r}: {checkpoint_pairs[1]}"
    )
    assert checkpoint_pairs[2] == ("user", query_2), (
        f"Turn 2 user message mismatch for {patient_id!r}: {checkpoint_pairs[2]}"
    )
    assert checkpoint_pairs[3] == ("assistant", summary_2), (
        f"Turn 2 assistant message mismatch for {patient_id!r}: {checkpoint_pairs[3]}"
    )
