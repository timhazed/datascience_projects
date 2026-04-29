"""Tests for planner_node state-based memory_context injection (Phase 4 cutover)."""

from __future__ import annotations

from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.agents.planner_node import build_planner_memory_inputs, make_planner_node
from src.models.planner_output import PlannerOutput
from src.models.sub_goal import SubGoal


def _state(**overrides) -> dict:
    base: dict = {
        "user_query": "Get history for Alice.",
        "patient_id": None,
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


def _plan(*tasks: str) -> PlannerOutput:
    return PlannerOutput(
        patient_id=None,
        sub_goals=[SubGoal(task=t, order=i + 1) for i, t in enumerate(tasks)],
        reasoning="Test plan.",
    )


# ---------------------------------------------------------------------------
# make_planner_node factory
# ---------------------------------------------------------------------------


def test_planner_node_factory_is_callable() -> None:
    """make_planner_node(chain, max_recent_turns) returns a callable node."""
    chain = RunnableLambda(lambda _: _plan("retrieve_history"))
    node = make_planner_node(chain, 10)
    assert callable(node)


# ---------------------------------------------------------------------------
# build_planner_memory_inputs — state-based reads
# ---------------------------------------------------------------------------


def test_memory_context_empty_when_no_messages() -> None:
    """No messages in state → empty memory_context."""
    state = _state(patient_id="P-1", messages=[])
    _, mc = build_planner_memory_inputs(patient_id="P-1", state=state, max_recent_turns=10)
    assert mc == ""


def test_memory_context_includes_recent_messages() -> None:
    """H/A messages in state are formatted into memory_context."""
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ]
    state = _state(patient_id="P-1", messages=msgs)
    _, mc = build_planner_memory_inputs(patient_id="P-1", state=state, max_recent_turns=10)
    assert "User: hi" in mc
    assert "Assistant: hello" in mc


def test_memory_context_includes_conversation_summary() -> None:
    """conversation_summary from checkpoint is prepended to memory_context."""
    state = _state(
        patient_id="P-1",
        messages=[HumanMessage(content="q")],
        conversation_summary="Prior rolling summary.",
    )
    _, mc = build_planner_memory_inputs(patient_id="P-1", state=state, max_recent_turns=10)
    assert "Summary: Prior rolling summary." in mc
    assert "User: q" in mc


def test_memory_context_empty_when_no_patient() -> None:
    """patient_id=None → patient identity unknown; memory still read from state."""
    msgs = [
        HumanMessage(content="What is CKD?"),
        AIMessage(content="CKD is ..."),
    ]
    state = _state(patient_id=None, messages=msgs)
    pc, mc = build_planner_memory_inputs(patient_id=None, state=state, max_recent_turns=10)
    assert pc == "Patient identity unknown."
    # memory_context still built from messages even for anonymous queries
    assert "User: What is CKD?" in mc


def test_memory_context_capped_at_max_turns() -> None:
    """Only the most recent max_recent_turns * 2 messages are included."""
    # 20 H/A pairs = 40 messages; max_recent_turns=3 → keep last 6
    msgs = []
    for i in range(20):
        msgs.append(HumanMessage(content=f"q{i}"))
        msgs.append(AIMessage(content=f"a{i}"))

    state = _state(patient_id="P-x", messages=msgs)
    _, mc = build_planner_memory_inputs(patient_id="P-x", state=state, max_recent_turns=3)
    # Last 6 messages: q17, a17, q18, a18, q19, a19
    assert "q19" in mc
    assert "q0" not in mc


# ---------------------------------------------------------------------------
# planner_node invocation — memory_context forwarded to chain
# ---------------------------------------------------------------------------


def test_memory_context_injected_into_chain_from_state() -> None:
    """planner_node passes memory_context built from checkpoint messages to the chain."""
    plan = _plan("retrieve_history")
    payloads: list[dict] = []

    def fake_retry(_chain: object, inputs: dict) -> PlannerOutput:
        payloads.append(inputs)
        return plan

    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),
    ]

    with patch("src.agents.planner_node.invoke_with_retry", side_effect=fake_retry):
        node = make_planner_node(RunnableLambda(lambda _: plan), 10)
        node(_state(patient_id="P-1", messages=msgs))

    assert len(payloads) == 1
    assert "User: hi" in payloads[0]["memory_context"]
    assert "Assistant: hello" in payloads[0]["memory_context"]
