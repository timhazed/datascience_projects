"""Tests for planner_chain — MagicMock LLM, no API calls.

with_structured_output does not work with FakeListChatModel (no tool-calling support).
Instead we mock the LLM's with_structured_output method to return a FakeRunnable
that yields the desired PlannerOutput directly. This isolates chain wiring
from LLM schema binding.
"""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from src.chains.planner_chain import build_planner_chain
from src.models.planner_output import PlannerOutput
from src.models.sub_goal import SubGoal


def _make_planner_output(*tasks: str) -> PlannerOutput:
    """Build a PlannerOutput with the given task types in order."""
    return PlannerOutput(
        patient_id=None,
        sub_goals=[SubGoal(task=task, order=i + 1) for i, task in enumerate(tasks)],
        reasoning="Test plan.",
    )


def _mock_llm(output: PlannerOutput) -> MagicMock:
    """Return a mock LLM whose with_structured_output returns a RunnableLambda.

    Using RunnableLambda ensures the chain pipeline (prompt | structured_llm)
    executes correctly — the lambda receives the formatted prompt messages and
    returns the PlannerOutput directly, bypassing JSON parsing.
    """
    llm = MagicMock()
    # RunnableLambda ignores its input and returns output — simulates a perfect LLM
    llm.with_structured_output.return_value = RunnableLambda(lambda _: output)
    return llm


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_single_task_plan() -> None:
    """Planner returns a single retrieve_history sub-goal."""
    expected = _make_planner_output("retrieve_history")
    chain = build_planner_chain(_mock_llm(expected))
    result = chain.invoke({
        "user_query": "Get my history.",
        "patient_context": "Known: P-abc",
        "memory_context": "",
    })
    assert isinstance(result, PlannerOutput)
    assert len(result.sub_goals) == 1
    assert result.sub_goals[0].task == "retrieve_history"


def test_multi_task_plan() -> None:
    """Planner returns multiple sub-goals in order."""
    expected = _make_planner_output(
        "resolve_patient", "retrieve_history", "book_appointment", "search_disease"
    )
    chain = build_planner_chain(_mock_llm(expected))
    result = chain.invoke({
        "user_query": "Book and summarize for my father.",
        "patient_context": "Unknown",
        "memory_context": "",
    })
    assert len(result.sub_goals) == 4
    assert result.sub_goals[0].task == "resolve_patient"


def test_resolve_patient_first_when_patient_unknown() -> None:
    """When patient_id is None, resolve_patient must be the first sub-goal."""
    expected = _make_planner_output("resolve_patient", "retrieve_history")
    chain = build_planner_chain(_mock_llm(expected))
    result = chain.invoke({
        "user_query": "Show history for John.",
        "patient_context": "",
        "memory_context": "",
    })
    assert result.patient_id is None
    assert result.sub_goals[0].task == "resolve_patient"


def test_with_structured_output_called_with_planner_output_schema() -> None:
    """build_planner_chain must call llm.with_structured_output(PlannerOutput)."""
    llm = _mock_llm(_make_planner_output("retrieve_history"))
    build_planner_chain(llm)
    llm.with_structured_output.assert_called_once_with(PlannerOutput)


def test_returns_planner_output_instance() -> None:
    """Chain result is a PlannerOutput instance, not a dict or string."""
    expected = _make_planner_output("search_disease")
    chain = build_planner_chain(_mock_llm(expected))
    result = chain.invoke({
        "user_query": "Latest CKD treatments?",
        "patient_context": "",
        "memory_context": "",
    })
    assert isinstance(result, PlannerOutput)


def test_reasoning_field_present() -> None:
    """PlannerOutput.reasoning is populated (not empty)."""
    expected = PlannerOutput(
        patient_id=None,
        sub_goals=[SubGoal(task="search_disease", order=1)],
        reasoning="User asked about disease treatment — search_disease is appropriate.",
    )
    chain = build_planner_chain(_mock_llm(expected))
    result = chain.invoke({
        "user_query": "Hypertension treatments?",
        "patient_context": "",
        "memory_context": "",
    })
    assert result.reasoning != ""
