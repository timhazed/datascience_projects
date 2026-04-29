"""Tests for history_chain — FakeListChatModel, no API calls."""

from langchain_core.language_models import FakeListChatModel

from src.chains.history_chain import build_history_chain


def _fake_llm(response: str) -> FakeListChatModel:
    """Return a FakeListChatModel that always returns the given string."""
    return FakeListChatModel(responses=[response])


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_returns_clinical_answer() -> None:
    """Chain returns the LLM string output for valid retrieved chunks."""
    chain = build_history_chain(_fake_llm("Essential Hypertension, treated with Telmisartan."))
    result = chain.invoke({
        "retrieved_chunks": "Patient: Ramesh Kulkarni | Conditions: Essential Hypertension",
        "query": "What is Ramesh's diagnosis?",
    })
    assert "Hypertension" in result


def test_returns_string_not_message() -> None:
    """StrOutputParser must unwrap AIMessage — result is a plain string."""
    chain = build_history_chain(_fake_llm("metformin 1000mg BID"))
    result = chain.invoke({"retrieved_chunks": "...", "query": "medications?"})
    assert isinstance(result, str)


def test_output_matches_llm_response() -> None:
    """Chain output matches the response the fake LLM was configured with."""
    expected = "Patient has CKD stage 3."
    chain = build_history_chain(_fake_llm(expected))
    result = chain.invoke({
        "retrieved_chunks": "Patient: Alice | Conditions: CKD stage 3",
        "query": "What stage of CKD?",
    })
    assert result == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_retrieved_chunks() -> None:
    """Chain handles empty retrieved_chunks gracefully — LLM decides output."""
    chain = build_history_chain(_fake_llm("not recorded"))
    result = chain.invoke({"retrieved_chunks": "", "query": "Any allergies?"})
    assert result == "not recorded"


def test_multiline_chunks() -> None:
    """Chain accepts multi-line chunk strings without error."""
    chunks = "Patient: Alice\nConditions: CKD\nMedications: lisinopril 10mg\n"
    chain = build_history_chain(_fake_llm("CKD, treated with lisinopril."))
    result = chain.invoke({"retrieved_chunks": chunks, "query": "What conditions?"})
    assert isinstance(result, str)
    assert len(result) > 0
