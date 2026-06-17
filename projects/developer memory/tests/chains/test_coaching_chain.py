"""Tests for build_coaching_chain (src/chains/coaching_chain.py).

Uses RunnableLambda to mock with_structured_output — no live Ollama calls.

Cases:
  - Chain returns an AnalysisResult on success
  - deviation_score is within [0.0, 1.0]
  - Chain handles absent persona context (empty string)
"""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from src.chains.coaching_chain import build_coaching_chain
from src.models.analysis import AnalysisResult


def _make_llm(output: AnalysisResult) -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: output)
    return llm


def _low_deviation_result() -> AnalysisResult:
    return AnalysisResult(
        deviation_score=0.15,
        rationale="The diff adds type annotations consistent with the developer's established style.",
        patterns_matched=["uses_type_annotations"],
    )


def _high_deviation_result() -> AnalysisResult:
    return AnalysisResult(
        deviation_score=0.8,
        rationale="The diff introduces global mutable state, contradicting the developer's pattern.",
        patterns_matched=["avoids_global_state"],
    )


class TestBuildCoachingChain:
    def test_chain_returns_analysis_result(self) -> None:
        """Chain returns an AnalysisResult with all required fields."""
        fake_llm = _make_llm(_low_deviation_result())
        chain = build_coaching_chain(fake_llm)

        result = chain.invoke({
            "diff_text": "--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-x = 1\n+x: int = 1",
            "persona_context": "Developer uses type annotations consistently.",
        })

        assert isinstance(result, AnalysisResult)
        assert 0.0 <= result.deviation_score <= 1.0
        assert result.rationale
        assert isinstance(result.patterns_matched, list)

    def test_low_deviation_score(self) -> None:
        """Low deviation score (< 0.65) indicates good alignment."""
        fake_llm = _make_llm(_low_deviation_result())
        chain = build_coaching_chain(fake_llm)

        result = chain.invoke({"diff_text": "trivial", "persona_context": "context"})

        assert result.deviation_score < 0.65

    def test_high_deviation_score(self) -> None:
        """High deviation score (> 0.65) indicates deviation from persona patterns."""
        fake_llm = _make_llm(_high_deviation_result())
        chain = build_coaching_chain(fake_llm)

        result = chain.invoke({"diff_text": "bad code", "persona_context": ""})

        assert result.deviation_score > 0.65

    def test_chain_handles_empty_persona_context(self) -> None:
        """Chain does not crash when persona_context is an empty string."""
        fake_llm = _make_llm(_low_deviation_result())
        chain = build_coaching_chain(fake_llm)

        result = chain.invoke({"diff_text": "some diff", "persona_context": ""})

        assert isinstance(result, AnalysisResult)

    def test_chain_is_runnable(self) -> None:
        """build_coaching_chain returns a Runnable."""
        fake_llm = _make_llm(_low_deviation_result())
        chain = build_coaching_chain(fake_llm)
        assert callable(getattr(chain, "invoke", None))
