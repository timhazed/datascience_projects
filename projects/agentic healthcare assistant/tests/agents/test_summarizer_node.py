"""Tests for summarizer_node — FakeListChatModel chain, no API calls."""

from langchain_core.language_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda

from src.agents.summarizer_node import make_summarizer_node
from src.chains.summary_chain import build_summary_chain
from src.models.task_result import TaskResult


def _state(**overrides) -> dict:
    """Build a minimal HealthcareState-compatible dict."""
    base: dict = {
        "user_query": "Summarise for Alice.",
        "patient_id": "P-001",
        "messages": [HumanMessage(content="Summarise for Alice.")],
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


def _tasks(*pairs) -> list[TaskResult]:
    """Build TaskResult list from (task, success) pairs."""
    return [TaskResult(task=t, success=s) for t, s in pairs]


def _chain(response: str):
    """Return a build_summary_chain backed by FakeListChatModel."""
    return build_summary_chain(FakeListChatModel(responses=[response]))


def _node(chain):
    return make_summarizer_node(chain, memory_summary_chain=None)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_sets_final_summary() -> None:
    """Summarizer writes the chain output to final_summary."""
    expected = "Alice has CKD. Appointment booked with Dr. Patel."
    node = _node(_chain(expected))
    result = node(_state(completed_tasks=_tasks(("retrieve_history", True))))
    assert result["final_summary"] == expected


def test_trace_entry_added() -> None:
    """Summarizer appends its trace entry (accumulates under REPLACE semantics)."""
    node = _node(_chain("summary"))
    result = node(_state(completed_tasks=_tasks(("retrieve_history", True))))
    assert len(result["trace"]) >= 1
    assert any("summarizer" in entry for entry in result["trace"])


def test_single_task_result() -> None:
    """Summarizer handles a single completed task."""
    node = _node(_chain("One task done."))
    result = node(_state(completed_tasks=_tasks(("resolve_patient", True))))
    assert result["final_summary"] == "One task done."


def test_mixed_success_failure_tasks() -> None:
    """Summarizer handles mix of successful and failed tasks."""
    tasks = _tasks(("retrieve_history", True), ("book_appointment", False))
    tasks[1].error = "No slots available."
    node = _node(_chain("History retrieved; booking failed."))
    result = node(_state(completed_tasks=tasks))
    assert result["final_summary"] == "History retrieved; booking failed."


def test_does_not_dequeue_pending_tasks() -> None:
    """Terminal node — does not modify pending_tasks (key absent from return dict)."""
    from src.models.sub_goal import SubGoal

    pending = [SubGoal(task="search_disease", order=1)]
    node = _node(_chain("summary"))
    result = node(_state(completed_tasks=_tasks(("retrieve_history", True)), pending_tasks=pending))
    # summarizer must NOT include pending_tasks in its return dict
    assert "pending_tasks" not in result


# ---------------------------------------------------------------------------
# Phase 2: AIMessage injection into messages channel
# ---------------------------------------------------------------------------


def test_ai_message_injected_into_messages() -> None:
    """Summarizer injects AIMessage into messages for add_messages accumulation."""
    expected = "Clinical summary text."
    node = _node(_chain(expected))
    result = node(_state(completed_tasks=_tasks(("retrieve_history", True))))
    assert "messages" in result
    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == expected


def test_ai_message_content_matches_final_summary() -> None:
    """AIMessage content always mirrors final_summary."""
    expected = "Patient history retrieved successfully."
    node = _node(_chain(expected))
    result = node(_state(completed_tasks=_tasks(("retrieve_history", True))))
    assert result["final_summary"] == result["messages"][0].content


def test_ai_message_produced_on_chain_failure() -> None:
    """Fallback summary path also injects AIMessage into messages."""

    def _fail(_):
        raise RuntimeError("LLM down")

    node = _node(RunnableLambda(_fail))
    result = node(_state(completed_tasks=_tasks(("retrieve_history", True))))
    assert "messages" in result
    assert isinstance(result["messages"][0], AIMessage)
    # Content should be the fallback bullet list, not None
    assert result["messages"][0].content is not None


# ---------------------------------------------------------------------------
# Phase 2: conversation_summary from checkpoint state used as context
# ---------------------------------------------------------------------------


def test_conversation_memory_empty_when_no_messages_or_summary() -> None:
    """When state has no messages and no conversation_summary, memory is empty."""
    captured: list[dict] = []

    def _capture(inv: dict) -> str:
        captured.append(inv)
        return "ok"

    node = make_summarizer_node(RunnableLambda(_capture), memory_summary_chain=None)
    # Pass empty messages list so build_planner_memory_inputs produces no context.
    node(_state(completed_tasks=_tasks(("retrieve_history", True)), messages=[]))
    assert len(captured) == 1
    assert captured[0]["conversation_memory"].strip() == ""


def test_conversation_summary_from_state_included_in_context() -> None:
    """When state has conversation_summary, it is included in conversation_memory."""
    captured: list[dict] = []
    prior_summary = "Alice is a 45-year-old with CKD stage 3."

    def _capture(inv: dict) -> str:
        captured.append(inv)
        return "ok"

    node = make_summarizer_node(RunnableLambda(_capture), memory_summary_chain=None)
    node(_state(
        completed_tasks=_tasks(("retrieve_history", True)),
        conversation_summary=prior_summary,
        messages=[],  # empty to isolate the summary contribution
    ))
    # Phase 4: memory is built from both conversation_summary and messages channel.
    assert prior_summary in captured[0]["conversation_memory"]


# ---------------------------------------------------------------------------
# Phase 2: Rolling-summary rollup cadence logic
# ---------------------------------------------------------------------------


def _make_prior_messages(n_turns: int) -> list:
    """Build n_turns H/A pairs plus the current HumanMessage for cadence testing."""
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"Query {i}"))
        msgs.append(AIMessage(content=f"Response {i}"))
    # Current turn's HumanMessage (injected by _build_initial_state before invoke)
    msgs.append(HumanMessage(content="Current query"))
    return msgs


def test_rolling_summary_fires_at_cadence() -> None:
    """Rollup fires when total complete turns is a multiple of cadence."""
    rollup_called: list[dict] = []

    def _rollup_chain(inv: dict) -> str:
        rollup_called.append(inv)
        return "Rolled-up summary text that is long enough."

    # 2 prior H/A pairs + current HumanMessage = 5 messages → total_turns = 3
    # cadence=3: 3 % 3 == 0 → rollup fires
    node = make_summarizer_node(
        _chain("summary"),
        memory_summary_chain=RunnableLambda(_rollup_chain),
        cadence=3,
    )
    result = node(_state(
        completed_tasks=_tasks(("retrieve_history", True)),
        messages=_make_prior_messages(2),
    ))
    assert len(rollup_called) == 1
    assert "conversation_summary" in result
    assert result["conversation_summary"] == "Rolled-up summary text that is long enough."


def test_rolling_summary_skipped_between_cadence() -> None:
    """Rollup is skipped when total turns is not a multiple of cadence."""
    rollup_called: list[dict] = []

    def _rollup_chain(inv: dict) -> str:
        rollup_called.append(inv)
        return "Should not be called."

    # 0 prior H/A pairs + current HumanMessage = 1 message → total_turns = 1
    # cadence=3: 1 % 3 != 0 → rollup skipped
    node = make_summarizer_node(
        _chain("summary"),
        memory_summary_chain=RunnableLambda(_rollup_chain),
        cadence=3,
    )
    result = node(_state(
        completed_tasks=_tasks(("retrieve_history", True)),
        messages=_make_prior_messages(0),
    ))
    assert len(rollup_called) == 0
    assert "conversation_summary" not in result


def test_rolling_summary_not_fired_when_chain_is_none() -> None:
    """No rollup attempt when memory_summary_chain is None."""
    # Build state that would fire cadence if chain were present (total_turns=3, cadence=3)
    node = make_summarizer_node(
        _chain("summary"),
        memory_summary_chain=None,
        cadence=3,
    )
    result = node(_state(
        completed_tasks=_tasks(("retrieve_history", True)),
        messages=_make_prior_messages(2),
    ))
    assert "conversation_summary" not in result


def test_rolling_summary_caps_at_max_summary_chars() -> None:
    """Rolled-up summary is capped to max_summary_chars before storing."""

    def _rollup_chain(_inv: dict) -> str:
        return "X" * 200  # longer than max_summary_chars=100

    node = make_summarizer_node(
        _chain("summary"),
        memory_summary_chain=RunnableLambda(_rollup_chain),
        max_summary_chars=100,
        cadence=3,
    )
    result = node(_state(
        completed_tasks=_tasks(("retrieve_history", True)),
        messages=_make_prior_messages(2),
    ))
    assert "conversation_summary" in result
    assert len(result["conversation_summary"]) == 100


def test_rolling_summary_short_output_kept_prior() -> None:
    """Rollup that returns too-short text is discarded; prior summary kept."""

    def _rollup_chain(_inv: dict) -> str:
        return "Too short"  # < 20 chars

    node = make_summarizer_node(
        _chain("summary"),
        memory_summary_chain=RunnableLambda(_rollup_chain),
        cadence=3,
    )
    result = node(_state(
        completed_tasks=_tasks(("retrieve_history", True)),
        messages=_make_prior_messages(2),
        conversation_summary="Prior summary that should be kept.",
    ))
    # conversation_summary NOT in result → LangGraph preserves the prior state value
    assert "conversation_summary" not in result


def test_rolling_summary_chain_failure_does_not_break_node() -> None:
    """Rollup chain failure is logged and swallowed; final_summary still produced."""

    def _fail_rollup(_inv: dict) -> str:
        raise RuntimeError("Rollup chain timed out")

    node = make_summarizer_node(
        _chain("main summary"),
        memory_summary_chain=RunnableLambda(_fail_rollup),
        cadence=3,
    )
    result = node(_state(
        completed_tasks=_tasks(("retrieve_history", True)),
        messages=_make_prior_messages(2),
    ))
    assert result["final_summary"] == "main summary"
    assert "conversation_summary" not in result


def test_memory_summary_chain_kwarg_accepted() -> None:
    """make_summarizer_node accepts all Phase 2 kwargs without error."""
    node = make_summarizer_node(
        _chain("summary"),
        memory_summary_chain=None,
        max_recent_turns=5,
        max_summary_chars=500,
        cadence=2,
    )
    result = node(_state(completed_tasks=_tasks(("retrieve_history", True))))
    assert result["final_summary"] == "summary"


# ---------------------------------------------------------------------------
# Degraded mode — chain failure
# ---------------------------------------------------------------------------


def test_chain_failure_produces_fallback_summary() -> None:
    """If chain raises, summarizer degrades to a plain bullet list, not None."""

    def _failing(inputs):
        raise RuntimeError("LLM unavailable")

    node = _node(RunnableLambda(_failing))
    tasks = _tasks(("retrieve_history", True), ("book_appointment", False))
    result = node(_state(completed_tasks=tasks))
    assert result["final_summary"] is not None
    assert "retrieve_history" in result["final_summary"]
    assert "book_appointment" in result["final_summary"]
