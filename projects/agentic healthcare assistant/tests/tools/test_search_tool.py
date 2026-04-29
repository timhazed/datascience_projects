"""Tests for search_tool — mock search_fn and FakeListChatModel chain; no live calls."""

from __future__ import annotations

from langchain_community.chat_models.fake import FakeListChatModel

from src.chains.search_chain import build_search_chain
from src.models.search_result_item import SearchResultItem
from src.tools.search_tool import _NO_RESULTS_SUMMARY, make_search_tool


def _chain(response: str = "Clinical synthesis here."):
    """Build a search chain backed by a FakeListChatModel."""
    return build_search_chain(FakeListChatModel(responses=[response] * 20))


def _item(
    title: str = "CKD Article",
    url: str = "https://medlineplus.gov/ckd.html",
    snippet: str = "Chronic kidney disease overview.",
    source_domain: str = "medlineplus.gov",
) -> SearchResultItem:
    """Build a minimal SearchResultItem."""
    return SearchResultItem(title=title, url=url, snippet=snippet, source_domain=source_domain)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_disease_search_result_with_items() -> None:
    """When search_fn returns results, DiseaseSearchResult has non-empty items."""
    items = [_item()]
    tool = make_search_tool(lambda q, n: items, _chain())
    result = tool.invoke({"query": "CKD treatment", "max_results": 5})

    assert len(result.items) == 1
    assert result.query == "CKD treatment"


def test_summary_non_empty_when_results_available() -> None:
    """summary field is non-empty when results are returned."""
    tool = make_search_tool(lambda q, n: [_item()], _chain("Detailed clinical summary."))
    result = tool.invoke({"query": "hypertension", "max_results": 3})

    assert len(result.summary) > 0


def test_citations_populated_from_item_urls() -> None:
    """citations list contains URLs from all returned SearchResultItems."""
    items = [
        _item(url="https://medlineplus.gov/a.html"),
        _item(url="https://who.int/b.html", source_domain="who.int"),
    ]
    tool = make_search_tool(lambda q, n: items, _chain())
    result = tool.invoke({"query": "diabetes", "max_results": 5})

    assert "https://medlineplus.gov/a.html" in result.citations
    assert "https://who.int/b.html" in result.citations


def test_multiple_results_all_included() -> None:
    """All items returned by search_fn appear in DiseaseSearchResult.items."""
    items = [_item(title=f"Article {i}") for i in range(4)]
    tool = make_search_tool(lambda q, n: items, _chain())
    result = tool.invoke({"query": "COPD", "max_results": 5})

    assert len(result.items) == 4


# ---------------------------------------------------------------------------
# Zero results after domain filter
# ---------------------------------------------------------------------------


def test_zero_results_returns_empty_items_and_no_results_summary() -> None:
    """When search_fn returns empty list, items=[] and summary is the no-results constant."""
    tool = make_search_tool(lambda q, n: [], _chain())
    result = tool.invoke({"query": "unknown disease XYZ", "max_results": 5})

    assert result.items == []
    assert result.citations == []
    assert result.summary == _NO_RESULTS_SUMMARY


def test_zero_results_query_field_preserved() -> None:
    """query field is preserved even when no results are returned."""
    tool = make_search_tool(lambda q, n: [], _chain())
    result = tool.invoke({"query": "rare tropical disease", "max_results": 5})

    assert result.query == "rare tropical disease"


# ---------------------------------------------------------------------------
# Search function failure
# ---------------------------------------------------------------------------


def test_search_fn_raises_returns_failure_result() -> None:
    """When search_fn raises an exception, the tool returns a failure DiseaseSearchResult."""

    def _failing_fn(q, n):
        raise ConnectionError("API timeout")

    tool = make_search_tool(_failing_fn, _chain())
    result = tool.invoke({"query": "test query", "max_results": 3})

    assert result.items == []
    assert "ConnectionError" in result.summary


# ---------------------------------------------------------------------------
# Chain failure (synthesis degrades gracefully)
# ---------------------------------------------------------------------------


def test_chain_failure_returns_result_with_error_summary() -> None:
    """When synthesis chain fails, tool returns items but summary describes the failure."""
    from langchain_core.runnables import RunnableLambda

    def _failing_chain_fn(inputs):
        raise ValueError("LLM unavailable")

    failing_chain = RunnableLambda(_failing_chain_fn)
    tool = make_search_tool(lambda q, n: [_item()], failing_chain)
    result = tool.invoke({"query": "test query"})

    # Items were retrieved successfully
    assert len(result.items) == 1
    # Summary describes the failure instead of raising
    assert "ValueError" in result.summary or "Synthesis failed" in result.summary


# ---------------------------------------------------------------------------
# max_results constraint
# ---------------------------------------------------------------------------


def test_max_results_validated_by_pydantic() -> None:
    """max_results=0 raises ValidationError (ge=1 constraint in SearchDiseaseParams)."""
    import pytest
    from pydantic import ValidationError

    tool = make_search_tool(lambda q, n: [], _chain())
    with pytest.raises((ValidationError, Exception)):
        tool.invoke({"query": "test", "max_results": 0})
