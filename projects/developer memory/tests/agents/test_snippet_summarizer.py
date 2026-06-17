"""Tests for make_snippet_summarizer_node (src/agents/snippet_summarizer.py).

Uses FakeListChatModel — no live Ollama calls.

Cases:
  - Happy path: final_answer populated from LLM output
  - Empty raw_results → uses "(No results found.)" placeholder, still succeeds
  - LLM failure → final_answer=None, error set, no crash
  - Trace entry added on success and failure
  - _format_results: path + intent_summary + excerpt appear in formatted text
"""

from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.snippet_summarizer import _format_results, make_snippet_summarizer_node


def _make_llm(response: str = "Here is what I found.") -> FakeListChatModel:
    return FakeListChatModel(responses=[response])


def _result(path: str = "src/x.py", summary: str = "Does X.", doc: str = "x = 1") -> dict:
    return {
        "metadata": {"file_path": path, "intent_summary": summary, "tech_stack": ["Python"]},
        "document": doc,
    }


def _state(query: str = "how are routes structured", results: list[dict] | None = None) -> dict:
    return {"query": query, "raw_results": results or [], "trace": []}


class TestSnippetSummarizerNode:
    def test_happy_path_populates_final_answer(self) -> None:
        """LLM output lands in state['final_answer']."""
        node = make_snippet_summarizer_node(_make_llm("The routes are structured as follows."))
        result = node(_state(results=[_result()]))

        assert result["final_answer"] == "The routes are structured as follows."

    def test_empty_results_still_succeeds(self) -> None:
        """Empty raw_results → LLM still called with placeholder text, final_answer set."""
        node = make_snippet_summarizer_node(_make_llm("No results."))
        result = node(_state(results=[]))

        assert result["final_answer"] == "No results."
        assert "error" not in result or result.get("error") is None

    def test_llm_failure_returns_none_and_error(self) -> None:
        """LLM error → final_answer=None, error set, no exception raised."""
        with patch("src.agents.snippet_summarizer.invoke_with_retry", side_effect=RuntimeError("Ollama down")):
            node = make_snippet_summarizer_node(_make_llm())
            result = node(_state())

        assert result["final_answer"] is None
        assert result.get("error")

    def test_trace_added_on_success(self) -> None:
        """Trace gains an entry after successful summarization."""
        node = make_snippet_summarizer_node(_make_llm("answer"))
        result = node(_state())

        assert any("snippet_summarizer" in t for t in result["trace"])

    def test_trace_added_on_failure(self) -> None:
        """Trace gains an error entry when LLM fails."""
        with patch("src.agents.snippet_summarizer.invoke_with_retry", side_effect=RuntimeError("boom")):
            node = make_snippet_summarizer_node(_make_llm())
            result = node(_state())

        assert any("error" in t for t in result["trace"])


class TestFormatResults:
    def test_empty_returns_placeholder(self) -> None:
        """Empty list → '(No results found.)' placeholder."""
        assert _format_results([]) == "(No results found.)"

    def test_result_contains_path_and_summary(self) -> None:
        """Formatted text includes file path and intent_summary."""
        text = _format_results([_result(path="src/api.py", summary="Registers routes.")])

        assert "src/api.py" in text
        assert "Registers routes." in text

    def test_document_truncated_to_500_chars(self) -> None:
        """Document content is truncated to 500 characters in the formatted output."""
        long_doc = "x" * 700
        text = _format_results([_result(doc=long_doc)])

        assert "x" * 500 in text
        assert "x" * 501 not in text
