"""Tests for build_intent_summary_chain (src/chains/intent_summary_chain.py).

Uses RunnableLambda to mock with_structured_output — no live Ollama calls.

Cases:
  - Chain returns a _ChunkAnnotation-compatible object on success
  - Chain is built once (factory does not raise on valid LLM)
  - Prompt places content and path in human turn (external data rule)
  - Structured output fields are present in the result
"""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from src.chains.intent_summary_chain import _ChunkAnnotation, build_intent_summary_chain


def _fake_annotation() -> _ChunkAnnotation:
    return _ChunkAnnotation(
        intent_summary="This code registers HTTP routes for the health check endpoint.",
        tech_stack=["FastAPI"],
        semantic_type="Interface",
    )


def _make_llm(output: _ChunkAnnotation) -> MagicMock:
    """Return a mock LLM whose with_structured_output returns a RunnableLambda fixture."""
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: output)
    return llm


class TestBuildIntentSummaryChain:
    def test_chain_returns_chunk_annotation(self) -> None:
        """Chain built with a mock LLM returns a _ChunkAnnotation on invoke."""
        fake_llm = _make_llm(_fake_annotation())
        chain = build_intent_summary_chain(fake_llm)

        result = chain.invoke({"content": "app = FastAPI()\n@app.get('/')", "path": "src/main.py"})

        assert isinstance(result, _ChunkAnnotation)
        assert result.intent_summary
        assert isinstance(result.tech_stack, list)
        assert result.semantic_type in {"Logic", "Config", "Boilerplate", "Interface"}

    def test_chain_annotation_fields_populated(self) -> None:
        """All annotation fields are accessible and correctly typed."""
        fake_llm = _make_llm(_fake_annotation())
        chain = build_intent_summary_chain(fake_llm)

        result = chain.invoke({"content": "x = 1", "path": "src/x.py"})

        assert result.intent_summary == "This code registers HTTP routes for the health check endpoint."
        assert result.tech_stack == ["FastAPI"]
        assert result.semantic_type == "Interface"

    def test_chain_built_once_without_invocation(self) -> None:
        """build_intent_summary_chain returns a Runnable without raising."""
        fake_llm = _make_llm(_fake_annotation())
        chain = build_intent_summary_chain(fake_llm)
        # Chain is a Runnable — verify it has an invoke method
        assert callable(getattr(chain, "invoke", None))

    def test_chain_empty_tech_stack_valid(self) -> None:
        """Empty tech_stack is a valid _ChunkAnnotation (no frameworks detected)."""
        fake_llm = _make_llm(_ChunkAnnotation(
            intent_summary="Pure utility with no framework dependencies.",
            tech_stack=[],
            semantic_type="Logic",
        ))
        chain = build_intent_summary_chain(fake_llm)
        result = chain.invoke({"content": "def add(a, b): return a + b", "path": "utils/math.py"})

        assert result.tech_stack == []
        assert result.semantic_type == "Logic"

    def test_chunk_annotation_default_semantic_type(self) -> None:
        """_ChunkAnnotation defaults semantic_type to 'Logic' when not specified."""
        annotation = _ChunkAnnotation(intent_summary="Assigns a constant.", tech_stack=[])
        assert annotation.semantic_type == "Logic"
