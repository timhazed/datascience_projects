"""Tests for planner_node — RunnableLambda mock chain, no API calls."""

from langchain_core.runnables import RunnableLambda

from src.agents.planner_node import build_planner_memory_inputs, make_planner_node
from src.models.planner_output import PlannerOutput
from src.models.sub_goal import SubGoal


def _state(**overrides) -> dict:
    """Build a minimal HealthcareState-compatible dict."""
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
    """Build a PlannerOutput with the given task types in order."""
    return PlannerOutput(
        patient_id=None,
        sub_goals=[SubGoal(task=t, order=i + 1) for i, t in enumerate(tasks)],
        reasoning="Test plan.",
    )


def _chain(output: PlannerOutput):
    """Return a RunnableLambda that ignores its input and returns output."""
    return RunnableLambda(lambda _: output)


def _make_planner_node(chain):
    """Factory — state-based memory, no DB needed."""
    return make_planner_node(chain, 10)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_sets_pending_tasks_from_sub_goals() -> None:
    """planner_node populates pending_tasks from PlannerOutput.sub_goals."""
    plan = _plan("resolve_patient", "retrieve_history")
    node = _make_planner_node(_chain(plan))
    result = node(_state())
    assert len(result["pending_tasks"]) == 2
    assert result["pending_tasks"][0].task == "resolve_patient"


def test_sets_planner_output() -> None:
    """planner_node stores the PlannerOutput in planner_output."""
    plan = _plan("search_disease")
    node = _make_planner_node(_chain(plan))
    # Use a known patient_id so the ordering-fix branch does not trigger and
    # return a model_copy (which would break the `is` check).
    result = node(_state(patient_id="P-known"))
    assert result["planner_output"] is plan


def test_trace_entry_added() -> None:
    """planner_node appends a trace string."""
    plan = _plan("retrieve_history")
    node = _make_planner_node(_chain(plan))
    result = node(_state())
    assert len(result["trace"]) == 1
    assert "planner" in result["trace"][0]


def test_patient_context_known_when_patient_id_set() -> None:
    """planner_node builds known patient_context when patient_id is in state."""
    plan = _plan("retrieve_history")
    node = _make_planner_node(_chain(plan))
    # No assertion on chain input — just verify node doesn't crash with patient_id set
    result = node(_state(patient_id="P-abc123"))
    assert result["planner_output"] is plan


def test_multi_task_plan() -> None:
    """planner_node handles 4-task plan correctly."""
    plan = _plan("resolve_patient", "retrieve_history", "book_appointment", "search_disease")
    node = _make_planner_node(_chain(plan))
    result = node(_state())
    assert len(result["pending_tasks"]) == 4


# ---------------------------------------------------------------------------
# Failure path
# ---------------------------------------------------------------------------


def test_chain_failure_returns_error_and_empty_pending_tasks() -> None:
    """If the chain raises, planner_node returns error and empty pending_tasks."""

    def _failing_chain(inputs):
        raise ValueError("LLM unavailable")

    node = _make_planner_node(RunnableLambda(_failing_chain))
    result = node(_state())
    assert result["pending_tasks"] == []
    assert result["planner_output"] is None
    assert result["error"] is not None
    assert "Planning failed" in result["error"]


# ---------------------------------------------------------------------------
# Ordering invariant enforcement
# ---------------------------------------------------------------------------


def test_resolve_patient_prepended_when_missing_and_patient_id_unknown() -> None:
    """When patient_id is None and LLM omits resolve_patient, node prepends it."""
    # LLM hallucination: plan starts with retrieve_history, no resolve_patient
    plan = _plan("retrieve_history", "search_disease")
    node = _make_planner_node(_chain(plan))
    result = node(_state(patient_id=None))
    tasks = [sg.task for sg in result["pending_tasks"]]
    assert tasks[0] == "resolve_patient"
    assert "retrieve_history" in tasks
    assert "search_disease" in tasks
    assert len(tasks) == 3  # prepended + original 2


def test_no_prepend_when_resolve_patient_already_first() -> None:
    """When plan already starts with resolve_patient, it is not duplicated."""
    plan = _plan("resolve_patient", "retrieve_history")
    node = _make_planner_node(_chain(plan))
    result = node(_state(patient_id=None))
    tasks = [sg.task for sg in result["pending_tasks"]]
    assert tasks.count("resolve_patient") == 1
    assert len(tasks) == 2


def test_no_prepend_when_patient_id_already_known() -> None:
    """When patient_id is already resolved, the ordering fix does not trigger."""
    plan = _plan("retrieve_history", "search_disease")
    node = _make_planner_node(_chain(plan))
    result = node(_state(patient_id="P-known"))
    # No prepend — patient already resolved
    tasks = [sg.task for sg in result["pending_tasks"]]
    assert tasks[0] == "retrieve_history"
    assert "resolve_patient" not in tasks


def test_build_planner_memory_inputs_unknown_patient() -> None:
    """Unknown patient produces identity-unknown context with empty memory."""
    state = _state(patient_id=None, messages=[])
    pc, mc = build_planner_memory_inputs(
        patient_id=None,
        state=state,
        max_recent_turns=10,
    )
    assert pc == "Patient identity unknown."
    assert mc == ""


def test_build_planner_memory_inputs_resolved_empty_messages() -> None:
    """Known patient with no prior messages produces correct context strings."""
    state = _state(patient_id="P-known", messages=[])
    pc, mc = build_planner_memory_inputs(
        patient_id="P-known",
        state=state,
        max_recent_turns=10,
    )
    assert pc == "Known patient ID: P-known"
    assert mc == ""
