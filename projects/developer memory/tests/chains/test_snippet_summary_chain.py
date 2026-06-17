"""Tests for build_snippet_summary_chain (src/chains/snippet_summary_chain.py).

Uses FakeListChatModel for StrOutputParser chains — no live Ollama calls.

Cases:
  - Chain returns a string answer
  - Chain is composable (Runnable)
  - Empty results string still produces a response
  - System prompt contains code-output instruction (regression guard)
  - Human template contains code-block instruction (regression guard)
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.chains.snippet_summary_chain import _HUMAN, _SYSTEM, build_snippet_summary_chain

_FAKE_ANSWER = "The codebase primarily uses FastAPI for route handling, as seen in src/api.py."


class TestBuildSnippetSummaryChain:
    def test_chain_returns_string(self) -> None:
        """Chain returns a plain text answer string."""
        llm = FakeListChatModel(responses=[_FAKE_ANSWER])
        chain = build_snippet_summary_chain(llm)

        result = chain.invoke({
            "query": "How are HTTP routes handled?",
            "results": "Result 1: src/api.py — FastAPI route handler\n",
        })

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_is_runnable(self) -> None:
        """build_snippet_summary_chain returns a Runnable."""
        llm = FakeListChatModel(responses=[_FAKE_ANSWER])
        chain = build_snippet_summary_chain(llm)
        assert callable(getattr(chain, "invoke", None))

    def test_chain_answer_content(self) -> None:
        """Chain returns the fake LLM response as-is through StrOutputParser."""
        llm = FakeListChatModel(responses=[_FAKE_ANSWER])
        chain = build_snippet_summary_chain(llm)

        result = chain.invoke({"query": "test", "results": "result context"})

        assert result == _FAKE_ANSWER

    def test_chain_with_empty_results(self) -> None:
        """Chain handles empty results string without crashing."""
        llm = FakeListChatModel(responses=["No relevant results found."])
        chain = build_snippet_summary_chain(llm)

        result = chain.invoke({"query": "unknown topic", "results": ""})

        assert isinstance(result, str)

    def test_system_prompt_contains_code_output_rule(self) -> None:
        """System prompt must instruct the LLM to reproduce code verbatim when asked.

        Regression guard: if this instruction is accidentally removed, code-request
        queries will produce prose descriptions instead of fenced code blocks.
        """
        assert "fenced code block" in _SYSTEM
        assert "verbatim" in _SYSTEM

    def test_human_template_contains_code_block_instruction(self) -> None:
        """Human turn must remind the LLM to include a code block for code queries."""
        assert "fenced code block" in _HUMAN
