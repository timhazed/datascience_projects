"""Tests for src/models/graph_state.py — HealthcareState TypedDict."""

from langgraph.graph.message import add_messages

from src.models.graph_state import HealthcareState


def test_can_import():
    """HealthcareState is importable and is a TypedDict."""
    assert isinstance(HealthcareState.__annotations__, dict)


def test_required_keys_present():
    """All required state keys are declared on HealthcareState."""
    required = {
        "user_query",
        "patient_id",
        "messages",
        "planner_output",
        "pending_tasks",
        "completed_tasks",
        "intent_safe",
        "final_summary",
        "error",
        "trace",
    }
    assert required.issubset(set(HealthcareState.__annotations__))


def test_conversation_summary_key_present():
    """conversation_summary NotRequired field is declared on HealthcareState."""
    assert "conversation_summary" in HealthcareState.__annotations__


def test_completed_tasks_uses_replace_semantics():
    """completed_tasks uses REPLACE semantics — no operator.add annotation."""
    import operator

    annotation = HealthcareState.__annotations__["completed_tasks"]
    # Should not be Annotated with operator.add (REPLACE semantics)
    if hasattr(annotation, "__metadata__"):
        assert operator.add not in annotation.__metadata__, (
            "completed_tasks must NOT use operator.add reducer (REPLACE semantics)"
        )


def test_trace_uses_replace_semantics():
    """trace uses REPLACE semantics — no operator.add annotation."""
    import operator

    annotation = HealthcareState.__annotations__["trace"]
    if hasattr(annotation, "__metadata__"):
        assert operator.add not in annotation.__metadata__, (
            "trace must NOT use operator.add reducer (REPLACE semantics)"
        )


def test_messages_has_add_messages_reducer():
    """messages must carry the LangGraph add_messages reducer."""
    annotation = HealthcareState.__annotations__["messages"]
    assert hasattr(annotation, "__metadata__"), "messages should be Annotated"
    assert add_messages in annotation.__metadata__


def test_pending_tasks_not_annotated():
    """pending_tasks uses REPLACE semantics — no reducer annotation."""
    annotation = HealthcareState.__annotations__["pending_tasks"]
    assert not hasattr(annotation, "__metadata__"), (
        "pending_tasks should NOT be Annotated (REPLACE semantics)"
    )
