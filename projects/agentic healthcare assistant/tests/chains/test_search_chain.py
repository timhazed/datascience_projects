"""Tests for search_chain — FakeListChatModel, no API calls."""

from langchain_core.language_models import FakeListChatModel

from src.chains.search_chain import build_search_chain


def _fake_llm(response: str) -> FakeListChatModel:
    """Return a FakeListChatModel that always returns the given string."""
    return FakeListChatModel(responses=[response])


_SAMPLE_RESULTS = (
    "[1] (SERPER) ACE inhibitors in CKD: Used to slow progression...\n"
    "[2] (MEDLINE) Dietary protein restriction in CKD: 0.6-0.8g/kg/day..."
)

# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_returns_synthesis_string() -> None:
    """Chain returns the LLM string output for valid search results."""
    summary = "ACE inhibitors and dietary restriction are first-line for CKD [1][2]."
    chain = build_search_chain(_fake_llm(summary))
    result = chain.invoke({
        "search_results": _SAMPLE_RESULTS,
        "query": "latest treatment for chronic kidney disease stage 3",
    })
    assert "ACE" in result


def test_returns_string_not_message() -> None:
    """StrOutputParser must unwrap AIMessage — result is a plain string."""
    chain = build_search_chain(_fake_llm("Clinical summary here."))
    result = chain.invoke({"search_results": _SAMPLE_RESULTS, "query": "CKD treatment"})
    assert isinstance(result, str)


def test_output_matches_llm_response() -> None:
    """Chain output matches the response the fake LLM was configured with."""
    expected = "Treatment involves ACE inhibitors [1] and low protein diet [2]."
    chain = build_search_chain(_fake_llm(expected))
    result = chain.invoke({"search_results": _SAMPLE_RESULTS, "query": "CKD"})
    assert result == expected


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_search_results() -> None:
    """Chain handles empty search results gracefully — LLM decides output."""
    chain = build_search_chain(_fake_llm("No relevant results found."))
    result = chain.invoke({"search_results": "", "query": "rare disease XYZ"})
    assert isinstance(result, str)


def test_citation_markers_in_output() -> None:
    """Output with citation markers [1][2] is returned as-is."""
    summary = "Treatment: [1] ACE inhibitors and [2] low protein diet."
    chain = build_search_chain(_fake_llm(summary))
    result = chain.invoke({"search_results": _SAMPLE_RESULTS, "query": "CKD"})
    assert "[1]" in result
