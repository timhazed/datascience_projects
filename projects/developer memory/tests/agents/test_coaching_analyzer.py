"""Tests for make_coaching_analyzer_node (src/agents/coaching_analyzer.py).

Uses RunnableLambda chain mock — no live Ollama calls.

Cases:
  - Happy path: diff + persona → AnalysisResult with deviation_score
  - persona_context=None → still succeeds (uses baseline description)
  - LLM failure → analysis_result=None, error set, no crash
  - Trace entry added on success and failure
  - _format_persona: None → baseline text; populated profile → fields in output
"""

from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableLambda

from src.agents.coaching_analyzer import _format_persona, make_coaching_analyzer_node
from src.models.analysis import AnalysisResult
from src.models.persona import PersonaProfile

_VALID_DIFF = "--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-old\n+new\n"


def _make_result(score: float = 0.3) -> AnalysisResult:
    return AnalysisResult(
        deviation_score=score,
        rationale="Change aligns with established patterns.",
        patterns_matched=["uses_type_annotations"],
    )


def _make_mock_llm(result: AnalysisResult) -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: result)
    return llm


def _persona() -> PersonaProfile:
    return PersonaProfile(
        style_summary="Prefers declarative patterns.",
        dominant_patterns=["uses_type_annotations"],
        tech_preferences=["FastAPI"],
        coaching_notes=["Avoid mutable defaults."],
    )


def _state(persona: PersonaProfile | None = None, diff: str = _VALID_DIFF) -> dict:
    return {"diff_text": diff, "persona_context": persona, "trace": []}


class TestCoachingAnalyzerNode:
    def test_happy_path_returns_analysis_result(self) -> None:
        """diff + persona → AnalysisResult written to state."""
        node = make_coaching_analyzer_node(_make_mock_llm(_make_result()))
        result = node(_state(_persona()))

        assert "analysis_result" in result
        assert isinstance(result["analysis_result"], AnalysisResult)

    def test_deviation_score_preserved(self) -> None:
        """deviation_score from LLM output is preserved in state."""
        node = make_coaching_analyzer_node(_make_mock_llm(_make_result(score=0.72)))
        result = node(_state(_persona()))

        assert abs(result["analysis_result"].deviation_score - 0.72) < 0.001

    def test_none_persona_still_succeeds(self) -> None:
        """persona_context=None → analysis_result populated using baseline description."""
        node = make_coaching_analyzer_node(_make_mock_llm(_make_result()))
        result = node(_state(None))

        assert result["analysis_result"] is not None
        assert "error" not in result or result.get("error") is None

    def test_llm_failure_returns_none_and_error(self) -> None:
        """LLM error → analysis_result=None, error set, no exception raised."""
        with patch("src.agents.coaching_analyzer.invoke_with_retry", side_effect=RuntimeError("Ollama down")):
            node = make_coaching_analyzer_node(_make_mock_llm(_make_result()))
            result = node(_state())

        assert result["analysis_result"] is None
        assert result.get("error")

    def test_trace_added_on_success(self) -> None:
        """Trace gains an entry on successful analysis."""
        node = make_coaching_analyzer_node(_make_mock_llm(_make_result()))
        result = node(_state())

        assert any("coaching_analyzer" in t for t in result["trace"])

    def test_trace_added_on_failure(self) -> None:
        """Trace gains an error entry when LLM fails."""
        with patch("src.agents.coaching_analyzer.invoke_with_retry", side_effect=RuntimeError("boom")):
            node = make_coaching_analyzer_node(_make_mock_llm(_make_result()))
            result = node(_state())

        assert any("error" in t for t in result["trace"])


class TestFormatPersona:
    def test_none_returns_baseline(self) -> None:
        """None persona → generic baseline text returned."""
        text = _format_persona(None)
        assert "best practices" in text.lower()

    def test_profile_style_summary_in_output(self) -> None:
        """Populated profile → style_summary included in formatted text."""
        text = _format_persona(_persona())
        assert "Prefers declarative patterns." in text

    def test_profile_tech_preferences_in_output(self) -> None:
        """Populated profile → tech_preferences included in formatted text."""
        text = _format_persona(_persona())
        assert "FastAPI" in text

    def test_coaching_notes_included_when_present(self) -> None:
        """coaching_notes appended to formatted text when non-empty."""
        text = _format_persona(_persona())
        assert "Avoid mutable defaults." in text
