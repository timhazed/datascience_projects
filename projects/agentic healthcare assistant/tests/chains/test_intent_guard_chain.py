"""Tests for intent_guard_chain — FakeListChatModel, no API calls."""

from langchain_core.language_models import FakeListChatModel

from src.chains.intent_guard_chain import build_intent_guard_chain


def _fake_llm(response: str) -> FakeListChatModel:
    """Return a FakeListChatModel that always returns the given string."""
    return FakeListChatModel(responses=[response])


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_safe_verdict_for_medical_query() -> None:
    """Medical query returns exactly 'SAFE' — normalised inside the chain."""
    chain = build_intent_guard_chain(_fake_llm("SAFE"))
    result = chain.invoke({"user_query": "What is Ramesh Kulkarni's diagnosis?"})
    assert result == "SAFE"


def test_unsafe_verdict_for_off_topic_query() -> None:
    """Off-topic query returns exactly 'UNSAFE' — normalised inside the chain."""
    chain = build_intent_guard_chain(_fake_llm("UNSAFE"))
    result = chain.invoke({"user_query": "What is the capital of France?"})
    assert result == "UNSAFE"


def test_safe_for_appointment_query() -> None:
    """Appointment scheduling is an allowed topic — returns 'SAFE'."""
    chain = build_intent_guard_chain(_fake_llm("SAFE"))
    result = chain.invoke({"user_query": "Book a cardiologist for my father."})
    assert result == "SAFE"


def test_safe_for_treatment_summary() -> None:
    """Treatment summary request is an allowed topic — returns 'SAFE'."""
    chain = build_intent_guard_chain(_fake_llm("SAFE"))
    result = chain.invoke({
        "user_query": "Summarize latest treatments for type 2 diabetes."
    })
    assert result == "SAFE"


# ---------------------------------------------------------------------------
# Edge cases — normalisation now happens inside the chain
# ---------------------------------------------------------------------------


def test_llm_response_with_trailing_whitespace_normalised() -> None:
    """Trailing whitespace from the LLM is stripped inside the chain."""
    chain = build_intent_guard_chain(_fake_llm("SAFE  \n"))
    result = chain.invoke({"user_query": "What medications is David Thompson on?"})
    assert result == "SAFE"


def test_llm_response_lowercase_normalised() -> None:
    """Lowercase LLM output is uppercased inside the chain."""
    chain = build_intent_guard_chain(_fake_llm("safe"))
    result = chain.invoke({"user_query": "Book an appointment."})
    assert result == "SAFE"


def test_returns_string_not_message() -> None:
    """StrOutputParser must unwrap AIMessage — result is a plain string."""
    chain = build_intent_guard_chain(_fake_llm("UNSAFE"))
    result = chain.invoke({"user_query": "Tell me a joke."})
    assert isinstance(result, str)


def test_whitespace_only_input_still_invokes_chain() -> None:
    """Whitespace-only input is passed to the chain; guard decides UNSAFE."""
    chain = build_intent_guard_chain(_fake_llm("UNSAFE"))
    result = chain.invoke({"user_query": "   "})
    assert result == "UNSAFE"
