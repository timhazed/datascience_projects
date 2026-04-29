"""Integration tests for healthcare_graph — mock chains, real in-memory SQLite.

All LLM chains are replaced with mock Runnables so no API calls are made.
SQLite databases are real in-memory instances seeded with known patient data.
FAISS vector store uses the tmp_faiss fixture (MagicMock(spec=Embeddings)).

Test coverage:
  - SAFE → planner → single-task path (resolve + retrieve_history)
  - SAFE → planner → multi-task path (4 tasks)
  - UNSAFE short-circuit: planner and tool nodes never run
  - Pending_tasks shrinks by exactly 1 per task node
  - completed_tasks length equals number of task nodes executed
  - Routing invariant: unknown task type falls through to summarizer
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda

from src.chains.history_chain import build_history_chain
from src.chains.intent_guard_chain import build_intent_guard_chain
from src.chains.search_chain import build_search_chain
from src.chains.summary_chain import build_summary_chain
from src.db.appointment_db import AppointmentDB
from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.graph.healthcare_graph import (
    _NODE_DRAIN_UNKNOWN,
    _compile_healthcare_graph,
    _drain_unknown_task,
    _noop_search_fn,
    _route_intent,
    _route_next_task,
)
from src.models.patient import PatientRecord
from src.models.planner_output import PlannerOutput
from src.models.search_result_item import SearchResultItem
from src.models.sub_goal import SubGoal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def seeded_db(tmp_path) -> tuple[str, PatientDB]:
    """Return a path to a seeded SQLite DB and the PatientDB instance.

    Seeds one patient "Alice Test" with patient_id "P-alice" so resolver
    tests have a known exact-match to work with.
    """
    db_path = str(tmp_path / "integration.db")
    patient_db = PatientDB()
    appointment_db = AppointmentDB()
    patient_db.init_db(db_path)
    appointment_db.init_db(db_path)

    patient_db.create_patient(
        db_path,
        PatientRecord(
            patient_id="P-alice",
            phone="+1-555-000-0001",
            email="alice@example.com",
            name="Alice Test",
            age=35,
            gender="Female",
            summary="Healthy patient with mild hypertension.",
            conditions=["hypertension"],
            medications=["lisinopril 10mg"],
            allergies=[],
        ),
    )
    return db_path, patient_db


@pytest.fixture
def mock_vector_store(tmp_path) -> PatientVectorStore:
    """PatientVectorStore with mocked embeddings — no OpenAI API call."""
    embeddings = MagicMock(spec=Embeddings)
    embeddings.embed_documents.return_value = [[0.1] * 1536]
    embeddings.embed_query.return_value = [0.1] * 1536
    return PatientVectorStore(str(tmp_path / "faiss"), embeddings)


def _build_graph(
    seeded_db,
    mock_vector_store,
    *,
    guard_response: str = "SAFE",
    planner_output: PlannerOutput,
    history_response: str = "Patient history: hypertension, lisinopril.",
    summary_response: str = "Final summary of completed tasks.",
    search_fn=None,
):
    """Helper to assemble the graph with mock chains and injected real DBs."""
    db_path, patient_db = seeded_db

    # Intent guard chain — FakeListChatModel returns guard_response
    intent_guard_chain = build_intent_guard_chain(
        FakeListChatModel(responses=[guard_response])
    )

    # Planner chain — RunnableLambda returns a fixed PlannerOutput (structured output)
    planner_chain = RunnableLambda(lambda _: planner_output)

    # History chain — FakeListChatModel returns history_response
    history_chain = build_history_chain(
        FakeListChatModel(responses=[history_response] * 10)
    )

    # Search chain — FakeListChatModel returns a fixed synthesis string
    search_chain = build_search_chain(
        FakeListChatModel(responses=["Disease search summary."] * 10)
    )

    # Summary chain — FakeListChatModel returns summary_response
    summary_chain = build_summary_chain(
        FakeListChatModel(responses=[summary_response] * 10)
    )

    appointment_db = AppointmentDB()

    return _compile_healthcare_graph(
        intent_guard_chain=intent_guard_chain,
        planner_chain=planner_chain,
        history_chain=history_chain,
        search_chain=search_chain,
        summary_chain=summary_chain,
        patient_db=patient_db,
        appointment_db=appointment_db,
        vector_store=mock_vector_store,
        db_path=db_path,
        search_fn=search_fn or (lambda q, n: []),
        max_recent_turns=10,
    )


def _initial_state(user_query: str = "Get history for Alice Test.") -> dict:
    """Build a minimal initial HealthcareState for graph.invoke.

    Mirrors production ``_build_initial_state`` by injecting one HumanMessage so
    integration tests exercise the same Phase 2 message-accumulation path.
    """
    from langchain_core.messages import HumanMessage

    return {
        "user_query": user_query,
        "patient_id": None,
        "messages": [HumanMessage(content=user_query)],
        "planner_output": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "intent_safe": False,
        "final_summary": None,
        "error": None,
        "trace": [],
    }


def _plan(*tasks: str) -> PlannerOutput:
    """Build a PlannerOutput with the given task types in order."""
    return PlannerOutput(
        patient_id=None,
        sub_goals=[SubGoal(task=t, order=i + 1) for i, t in enumerate(tasks)],
        reasoning="Test plan.",
    )


# ---------------------------------------------------------------------------
# Test 1 — UNSAFE short-circuit
# ---------------------------------------------------------------------------


def test_unsafe_query_exits_before_planner(seeded_db, mock_vector_store) -> None:
    """UNSAFE classification routes directly to END — planner and tools never run."""
    graph = _build_graph(
        seeded_db,
        mock_vector_store,
        guard_response="UNSAFE",
        planner_output=_plan("resolve_patient"),  # never reached
    )
    result = graph.invoke(_initial_state("How do I make explosives?"))

    assert result["intent_safe"] is False
    assert result["planner_output"] is None
    assert result["pending_tasks"] == []
    assert result["completed_tasks"] == []
    assert result["final_summary"] is None


# ---------------------------------------------------------------------------
# Test 2 — Single-tool path: resolve_patient + retrieve_history
# ---------------------------------------------------------------------------


def test_single_tool_history_retrieval(seeded_db, mock_vector_store) -> None:
    """SAFE → planner → resolve_patient → retrieve_history → summarizer.

    Asserts final_summary is non-empty and exactly 2 completed_tasks are recorded.
    """
    plan = _plan("resolve_patient", "retrieve_history")
    # resolve_patient params must match the seeded patient name
    plan.sub_goals[0] = SubGoal(
        task="resolve_patient",
        order=1,
        parameters={"patient_name": "Alice Test"},
    )
    plan.sub_goals[1] = SubGoal(
        task="retrieve_history",
        order=2,
        parameters={"query": "What are the current medications?"},
    )

    graph = _build_graph(seeded_db, mock_vector_store, planner_output=plan)
    result = graph.invoke(_initial_state())

    assert result["final_summary"] is not None
    assert len(result["final_summary"]) > 0
    assert len(result["completed_tasks"]) == 2
    assert result["pending_tasks"] == []


def test_single_tool_completed_tasks_has_correct_task_names(
    seeded_db, mock_vector_store
) -> None:
    """completed_tasks contains exactly resolve_patient and retrieve_history results."""
    plan = _plan("resolve_patient", "retrieve_history")
    plan.sub_goals[0] = SubGoal(
        task="resolve_patient", order=1, parameters={"patient_name": "Alice Test"}
    )
    plan.sub_goals[1] = SubGoal(
        task="retrieve_history", order=2, parameters={"query": "medications"}
    )

    graph = _build_graph(seeded_db, mock_vector_store, planner_output=plan)
    result = graph.invoke(_initial_state())

    task_names = [t.task for t in result["completed_tasks"]]
    assert "resolve_patient" in task_names
    assert "retrieve_history" in task_names


# ---------------------------------------------------------------------------
# Test 3 — Multi-step path: 4 tasks
# ---------------------------------------------------------------------------


def test_multi_step_four_tasks_all_complete(seeded_db, mock_vector_store) -> None:
    """SAFE → 4-task plan → all 4 TaskResults in completed_tasks; pending_tasks == [].

    Tasks: resolve_patient, retrieve_history, update_history, search_disease.
    """
    plan = PlannerOutput(
        patient_id=None,
        reasoning="Test 4-task plan.",
        sub_goals=[
            SubGoal(
                task="resolve_patient",
                order=1,
                parameters={"patient_name": "Alice Test"},
            ),
            SubGoal(
                task="retrieve_history",
                order=2,
                parameters={"query": "conditions"},
            ),
            SubGoal(
                task="update_history",
                order=3,
                parameters={
                    "field": "notes",
                    "value": "Follow-up in 3 months.",
                    "operation": "append",
                },
            ),
            SubGoal(
                task="search_disease",
                order=4,
                parameters={"query": "hypertension treatment", "max_results": 3},
            ),
        ],
    )

    # Provide 3 search results so disease_search_node doesn't short-circuit
    def _mock_search(query: str, max_results: int) -> list[SearchResultItem]:
        return [
            SearchResultItem(
                title="Hypertension overview",
                url="https://example.com/1",
                snippet="Hypertension is a common condition.",
                source_domain="example.com",
            )
        ]

    graph = _build_graph(
        seeded_db,
        mock_vector_store,
        planner_output=plan,
        search_fn=_mock_search,
    )
    result = graph.invoke(_initial_state())

    assert result["pending_tasks"] == []
    assert len(result["completed_tasks"]) == 4
    task_names = {t.task for t in result["completed_tasks"]}
    assert "resolve_patient" in task_names
    assert "retrieve_history" in task_names
    assert "update_history" in task_names
    assert "search_disease" in task_names


# ---------------------------------------------------------------------------
# Test 4 — State contract: pending_tasks shrinks by 1 per node
# ---------------------------------------------------------------------------


def test_pending_tasks_shrinks_by_one_per_node(seeded_db, mock_vector_store) -> None:
    """Each task node dequeues exactly one task from pending_tasks.

    Verified by checking final state: all tasks consumed → pending_tasks == [].
    """
    plan = _plan("resolve_patient", "retrieve_history")
    plan.sub_goals[0] = SubGoal(
        task="resolve_patient", order=1, parameters={"patient_name": "Alice Test"}
    )
    plan.sub_goals[1] = SubGoal(
        task="retrieve_history", order=2, parameters={"query": "conditions"}
    )

    graph = _build_graph(seeded_db, mock_vector_store, planner_output=plan)
    result = graph.invoke(_initial_state())

    # All 2 tasks consumed — dequeue invariant held
    assert result["pending_tasks"] == []


# ---------------------------------------------------------------------------
# Test 5 — State contract: completed_tasks accumulates (REPLACE semantics)
# ---------------------------------------------------------------------------


def test_completed_tasks_length_equals_nodes_executed(seeded_db, mock_vector_store) -> None:
    """Accumulator pattern accumulates one TaskResult per task node execution."""
    plan = _plan("resolve_patient", "retrieve_history")
    plan.sub_goals[0] = SubGoal(
        task="resolve_patient", order=1, parameters={"patient_name": "Alice Test"}
    )
    plan.sub_goals[1] = SubGoal(
        task="retrieve_history", order=2, parameters={"query": "medications"}
    )

    graph = _build_graph(seeded_db, mock_vector_store, planner_output=plan)
    result = graph.invoke(_initial_state())

    # 2 task nodes ran → 2 TaskResults accumulated
    assert len(result["completed_tasks"]) == 2


# ---------------------------------------------------------------------------
# Test 6 — Trace accumulates across nodes
# ---------------------------------------------------------------------------


def test_trace_accumulates_entries_from_all_nodes(seeded_db, mock_vector_store) -> None:
    """trace list grows with one entry per node via the accumulator pattern (REPLACE semantics)."""
    plan = _plan("resolve_patient", "retrieve_history")
    plan.sub_goals[0] = SubGoal(
        task="resolve_patient", order=1, parameters={"patient_name": "Alice Test"}
    )
    plan.sub_goals[1] = SubGoal(
        task="retrieve_history", order=2, parameters={"query": "conditions"}
    )

    graph = _build_graph(seeded_db, mock_vector_store, planner_output=plan)
    result = graph.invoke(_initial_state())

    # At minimum: intent_guard, planner, resolve_patient, retrieve_history, summarizer
    assert len(result["trace"]) >= 5


# ---------------------------------------------------------------------------
# Test 7 — Routing functions (unit tests for the conditional edge logic)
# ---------------------------------------------------------------------------


def test_route_intent_safe_returns_planner() -> None:
    """_route_intent returns 'planner' when intent_safe is True."""
    state = {"intent_safe": True}
    assert _route_intent(state) == "planner"  # type: ignore[arg-type]


def test_route_intent_unsafe_returns_end() -> None:
    """_route_intent returns END sentinel when intent_safe is False."""
    from langgraph.constants import END

    state = {"intent_safe": False}
    assert _route_intent(state) == END  # type: ignore[arg-type]


def test_route_next_task_empty_pending_returns_summarizer() -> None:
    """_route_next_task returns 'summarizer' when pending_tasks is empty."""
    state = {"pending_tasks": []}
    assert _route_next_task(state) == "summarizer"  # type: ignore[arg-type]


def test_route_next_task_returns_correct_node_for_each_task_type() -> None:
    """_route_next_task maps each known task type to the correct node name."""
    expected = {
        "resolve_patient": "resolve_patient",
        "retrieve_history": "retrieve_history",
        "update_history": "update_history",
        "book_appointment": "book_appointment",
        "search_disease": "search_disease",
    }
    for task, node_name in expected.items():
        state = {"pending_tasks": [SubGoal(task=task, order=1)]}
        assert _route_next_task(state) == node_name  # type: ignore[arg-type]


def test_route_next_task_unknown_task_routes_to_drain_node() -> None:
    """_route_next_task routes unrecognised task types to the drain node.

    Uses a MagicMock to simulate a SubGoal-like object with an unregistered task
    value — bypassing Pydantic's Literal validation on SubGoal.task which would
    reject the value before the routing function is ever called.
    """
    fake_task = MagicMock()
    fake_task.task = "unknown_future_task"
    state = {"pending_tasks": [fake_task]}
    assert _route_next_task(state) == _NODE_DRAIN_UNKNOWN  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Test 8 — Patient not found leaves error set but graph still reaches summarizer
# ---------------------------------------------------------------------------


def test_unresolved_patient_graph_still_reaches_summarizer(
    seeded_db, mock_vector_store
) -> None:
    """When resolver returns failure, graph continues to summarizer (non-fatal error)."""
    plan = _plan("resolve_patient")
    plan.sub_goals[0] = SubGoal(
        task="resolve_patient",
        order=1,
        parameters={"patient_name": "Nonexistent Person XYZ"},
    )

    graph = _build_graph(seeded_db, mock_vector_store, planner_output=plan)
    result = graph.invoke(_initial_state())

    # Graph reaches summarizer even though resolver failed
    assert result["final_summary"] is not None
    # error is set (non-fatal per State Contract invariant 4)
    assert result["error"] is not None
    # resolver recorded a failure TaskResult
    assert len(result["completed_tasks"]) == 1
    assert result["completed_tasks"][0].success is False


def test_noop_search_fn_returns_empty_list() -> None:
    """_noop_search_fn always returns an empty list regardless of arguments."""
    assert _noop_search_fn("any query", 10) == []
    assert _noop_search_fn("", 0) == []


# ---------------------------------------------------------------------------
# _drain_unknown_task unit tests
# ---------------------------------------------------------------------------


def test_drain_unknown_task_pops_task_and_records_failure() -> None:
    """_drain_unknown_task removes the unknown task and appends a failure TaskResult."""
    fake_task = MagicMock()
    fake_task.task = "unknown_future_task"
    state: dict = {"pending_tasks": [fake_task], "completed_tasks": [], "error": None}

    delta = _drain_unknown_task(state)  # type: ignore[arg-type]

    assert delta["pending_tasks"] == []
    assert len(delta["completed_tasks"]) == 1
    result = delta["completed_tasks"][0]
    assert result.success is False
    assert result.task == "unknown_future_task"
    assert "unknown_future_task" in result.error
    assert delta["error"] is not None


def test_drain_unknown_task_leaves_remaining_tasks_intact() -> None:
    """_drain_unknown_task only removes the head task; remaining tasks are preserved."""
    fake_bad = MagicMock()
    fake_bad.task = "bogus_task"
    fake_good = MagicMock()
    fake_good.task = "search_disease"
    state: dict = {"pending_tasks": [fake_bad, fake_good], "completed_tasks": [], "error": None}

    delta = _drain_unknown_task(state)  # type: ignore[arg-type]

    assert len(delta["pending_tasks"]) == 1
    assert delta["pending_tasks"][0].task == "search_disease"


def test_drain_unknown_task_empty_queue_returns_empty_dict() -> None:
    """_drain_unknown_task with empty pending_tasks is a no-op (defensive guard)."""
    delta = _drain_unknown_task(  # type: ignore[arg-type]
        {"pending_tasks": [], "completed_tasks": [], "error": None}
    )
    assert delta == {}


def test_drain_unknown_task_preserves_existing_completed_tasks() -> None:
    """_drain_unknown_task appends to completed_tasks without discarding prior results."""
    from src.models.task_result import TaskResult

    prior = TaskResult(task="resolve_patient", success=True)
    fake_task = MagicMock()
    fake_task.task = "bogus"
    state: dict = {"pending_tasks": [fake_task], "completed_tasks": [prior], "error": None}

    delta = _drain_unknown_task(state)  # type: ignore[arg-type]

    assert len(delta["completed_tasks"]) == 2
    assert delta["completed_tasks"][0].task == "resolve_patient"
    assert delta["completed_tasks"][1].success is False


# ---------------------------------------------------------------------------
# Integration — unknown task in plan sets error, graph still reaches summarizer
# ---------------------------------------------------------------------------


def test_unknown_task_in_plan_error_set_graph_reaches_summarizer(
    seeded_db, mock_vector_store
) -> None:
    """Unknown task type is drained: error set, completed_tasks records failure,
    graph still reaches summarizer.

    SubGoal.model_construct() bypasses Pydantic Literal validation so we can
    inject an invalid task name into the plan without a ValidationError.
    """
    from src.models.planner_output import PlannerOutput
    from src.models.sub_goal import SubGoal

    fake_subgoal = SubGoal.model_construct(task="lookup_patient", order=1, parameters={})
    plan = PlannerOutput.model_construct(
        patient_id=None, sub_goals=[fake_subgoal], reasoning="test plan"
    )

    graph = _build_graph(seeded_db, mock_vector_store, planner_output=plan)
    result = graph.invoke(_initial_state("Look up a patient."))

    assert result["final_summary"] is not None, "Graph must reach summarizer"
    assert result["error"] is not None, "Error must be set for the unknown task"
    assert "lookup_patient" in result["error"]
    assert len(result["completed_tasks"]) == 1
    assert result["completed_tasks"][0].success is False
    assert result["completed_tasks"][0].task == "lookup_patient"


# ---------------------------------------------------------------------------
# Phase 2 — Messages channel accumulation (Decision 4 exit gate tests)
# ---------------------------------------------------------------------------


def test_messages_channel_populated_after_two_turns(seeded_db, mock_vector_store) -> None:
    """After two invocations with MemorySaver, checkpoint has 2 HumanMessage + 2 AIMessage.

    This is the Phase 2 exit gate test (Section 8.2). Verifies that:
    - _build_initial_state HumanMessage injection arrives in the checkpoint.
    - summarizer_node AIMessage injection closes each turn.
    - add_messages reducer accumulates across invocations under the same thread_id.
    """
    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    # Build graph with MemorySaver so checkpoint accumulates across invocations.
    plan = _plan("search_disease")
    plan.sub_goals[0] = SubGoal(task="search_disease", order=1, parameters={"query": "diabetes"})

    db_path, patient_db = seeded_db
    intent_guard_chain = build_intent_guard_chain(FakeListChatModel(responses=["SAFE"] * 10))
    planner_chain = RunnableLambda(lambda _: plan)
    history_chain = build_history_chain(FakeListChatModel(responses=["history"] * 10))
    search_chain = build_search_chain(FakeListChatModel(responses=["search result"] * 10))
    summary_chain = build_summary_chain(FakeListChatModel(responses=["summary A", "summary B"]))
    appointment_db = AppointmentDB()
    checkpointer = MemorySaver()

    graph = _compile_healthcare_graph(
        intent_guard_chain=intent_guard_chain,
        planner_chain=planner_chain,
        history_chain=history_chain,
        search_chain=search_chain,
        summary_chain=summary_chain,
        patient_db=patient_db,
        appointment_db=appointment_db,
        vector_store=mock_vector_store,
        db_path=db_path,
        search_fn=lambda q, n: [],
        max_recent_turns=10,
        checkpointer=checkpointer,
    )

    thread_config = {"configurable": {"thread_id": "test-p001"}, "recursion_limit": 25}

    # Turn 1
    graph.invoke(_initial_state("What is diabetes?"), thread_config)
    # Turn 2 — same thread_id so messages accumulate in checkpoint
    graph.invoke(_initial_state("Any treatment options?"), thread_config)

    snapshot = graph.get_state(thread_config)
    msgs = snapshot.values.get("messages", [])
    human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]
    ai_msgs = [m for m in msgs if isinstance(m, AIMessage)]

    assert len(human_msgs) == 2, f"Expected 2 HumanMessages, got {len(human_msgs)}: {msgs}"
    assert len(ai_msgs) == 2, f"Expected 2 AIMessages, got {len(ai_msgs)}: {msgs}"
    assert human_msgs[0].content == "What is diabetes?"
    assert human_msgs[1].content == "Any treatment options?"
    assert ai_msgs[0].content == "summary A"
    assert ai_msgs[1].content == "summary B"


def test_no_node_adds_human_message(seeded_db, mock_vector_store) -> None:
    """No node may inject a HumanMessage into the messages channel (Decision 4).

    Only _build_initial_state may introduce HumanMessages. This test inspects the
    checkpoint state delta after a full invocation and asserts that all HumanMessages
    in the checkpoint were already present in the initial state (i.e. no node added new ones).
    """
    from langchain_core.messages import HumanMessage
    from langgraph.checkpoint.memory import MemorySaver

    plan = _plan("search_disease")
    plan.sub_goals[0] = SubGoal(task="search_disease", order=1, parameters={"query": "flu"})

    db_path, patient_db = seeded_db
    intent_guard_chain = build_intent_guard_chain(FakeListChatModel(responses=["SAFE"] * 5))
    planner_chain = RunnableLambda(lambda _: plan)
    history_chain = build_history_chain(FakeListChatModel(responses=["history"] * 5))
    search_chain = build_search_chain(FakeListChatModel(responses=["results"] * 5))
    summary_chain = build_summary_chain(FakeListChatModel(responses=["final summary"] * 5))
    appointment_db = AppointmentDB()
    checkpointer = MemorySaver()

    graph = _compile_healthcare_graph(
        intent_guard_chain=intent_guard_chain,
        planner_chain=planner_chain,
        history_chain=history_chain,
        search_chain=search_chain,
        summary_chain=summary_chain,
        patient_db=patient_db,
        appointment_db=appointment_db,
        vector_store=mock_vector_store,
        db_path=db_path,
        search_fn=lambda q, n: [],
        max_recent_turns=10,
        checkpointer=checkpointer,
    )

    query = "Tell me about flu."
    thread_config = {"configurable": {"thread_id": "test-no-hm"}, "recursion_limit": 25}

    # The initial state contains exactly one HumanMessage with the query content.
    initial = _initial_state(query)
    initial_human_contents = {
        m.content for m in initial["messages"] if isinstance(m, HumanMessage)
    }

    graph.invoke(initial, thread_config)

    snapshot = graph.get_state(thread_config)
    msgs = snapshot.values.get("messages", [])
    human_msgs = [m for m in msgs if isinstance(m, HumanMessage)]

    # All HumanMessages in checkpoint must have content from the initial state —
    # no node is permitted to inject additional HumanMessages.
    for hm in human_msgs:
        assert hm.content in initial_human_contents, (
            f"Node injected unexpected HumanMessage: {hm.content!r}"
        )
