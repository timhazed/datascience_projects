"""Tests for intent_guard_node — FakeListChatModel mock chain, no API calls."""

from langchain_community.chat_models.fake import FakeListChatModel

from src.agents.intent_guard_node import make_intent_guard_node
from src.chains.intent_guard_chain import build_intent_guard_chain


def _state(**overrides) -> dict:
    """Build a minimal HealthcareState-compatible dict."""
    base: dict = {
        "user_query": "What are my medications?",
        "patient_id": None,
        "messages": [],
        "planner_output": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "intent_safe": False,
        "final_summary": None,
        "error": None,
        "trace": [],
    }
    base.update(overrides)
    return base


def _guard_chain(response: str):
    """Build an intent guard chain backed by a FakeListChatModel."""
    llm = FakeListChatModel(responses=[response])
    return build_intent_guard_chain(llm)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_safe_query_sets_intent_safe_true() -> None:
    """A SAFE response from the chain sets intent_safe=True."""
    node = make_intent_guard_node(_guard_chain("SAFE"))
    result = node(_state())
    assert result["intent_safe"] is True


def test_unsafe_query_sets_intent_safe_false() -> None:
    """An UNSAFE response from the chain sets intent_safe=False."""
    node = make_intent_guard_node(_guard_chain("UNSAFE"))
    result = node(_state())
    assert result["intent_safe"] is False


def test_trace_entry_added() -> None:
    """intent_guard_node always appends a trace entry."""
    node = make_intent_guard_node(_guard_chain("SAFE"))
    result = node(_state())
    assert len(result["trace"]) == 1
    assert "intent_guard" in result["trace"][0]


def test_trace_records_classification() -> None:
    """Trace entry includes the raw classification string."""
    node = make_intent_guard_node(_guard_chain("SAFE"))
    result = node(_state(user_query="Show my history"))
    assert "SAFE" in result["trace"][0]


def test_lowercase_response_normalised_to_unsafe() -> None:
    """The chain normalises output; 'safe' (lowercase) is stripped+uppercased to SAFE."""
    # The _normalizer inside the chain converts 'safe' → 'SAFE'
    node = make_intent_guard_node(_guard_chain("safe"))
    result = node(_state())
    assert result["intent_safe"] is True


def test_off_topic_query_returns_false() -> None:
    """An off-topic query produces intent_safe=False."""
    node = make_intent_guard_node(_guard_chain("UNSAFE"))
    result = node(_state(user_query="What is the weather today?"))
    assert result["intent_safe"] is False


# ---------------------------------------------------------------------------
# Failure path (fail-closed behaviour)
# ---------------------------------------------------------------------------


def test_chain_failure_sets_intent_safe_false() -> None:
    """When the chain raises, intent_guard_node defaults to UNSAFE (fail-closed)."""
    from langchain_core.runnables import RunnableLambda

    def _failing_chain(inputs):
        raise RuntimeError("API timeout")

    node = make_intent_guard_node(RunnableLambda(_failing_chain))
    result = node(_state())
    assert result["intent_safe"] is False


def test_chain_failure_sets_error_field() -> None:
    """When the chain raises, error is set on the returned dict."""
    from langchain_core.runnables import RunnableLambda

    def _failing_chain(inputs):
        raise RuntimeError("API timeout")

    node = make_intent_guard_node(RunnableLambda(_failing_chain))
    result = node(_state())
    assert result.get("error") is not None
    assert "Intent guard failed" in result["error"]


def test_chain_failure_adds_trace() -> None:
    """When the chain raises, a trace entry is still appended."""
    from langchain_core.runnables import RunnableLambda

    def _failing_chain(inputs):
        raise RuntimeError("API timeout")

    node = make_intent_guard_node(RunnableLambda(_failing_chain))
    result = node(_state())
    assert len(result["trace"]) == 1
    assert "ERROR" in result["trace"][0]
