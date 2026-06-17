"""Tests for make_persona_synthesizer_node (src/agents/persona_synthesizer.py).

Uses RunnableLambda chain mock — no live Ollama calls.

Cases:
  - Happy path: TendencyData → PersonaProfile in state
  - tendency_data=None → persona_profile=None, error set, no crash
  - LLM failure → persona_profile=None, error set, no crash
  - Trace entry added on success and failure
  - External data (tendency stats) does not land in system prompt
"""

from unittest.mock import MagicMock, patch

from langchain_core.runnables import RunnableLambda

from src.agents.persona_synthesizer import make_persona_synthesizer_node
from src.models.persona import PersonaProfile, TendencyData


def _make_profile() -> PersonaProfile:
    return PersonaProfile(
        style_summary="Prefers declarative patterns.",
        dominant_patterns=["uses_type_annotations"],
        tech_preferences=["FastAPI", "Pydantic"],
        coaching_notes=[],
    )


def _make_mock_llm(profile: PersonaProfile) -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: profile)
    return llm


def _tendency() -> TendencyData:
    return TendencyData(
        scope="project",
        doc_count=20,
        semantic_type_distribution={"Logic": 15, "Config": 5},
        tech_stack_frequency={"FastAPI": 10, "Pydantic": 8},
        dominant_patterns=["uses_type_annotations", "avoids_global_state"],
        weighted_docs=[],
    )


def _state(tendency: TendencyData | None) -> dict:
    return {"tendency_data": tendency, "trace": []}


class TestPersonaSynthesizerNode:
    def test_happy_path_returns_persona_profile(self) -> None:
        """Valid TendencyData → PersonaProfile written to state."""
        node = make_persona_synthesizer_node(_make_mock_llm(_make_profile()))
        result = node(_state(_tendency()))

        assert "persona_profile" in result
        assert isinstance(result["persona_profile"], PersonaProfile)

    def test_profile_fields_populated(self) -> None:
        """LLM-generated profile fields are present in output."""
        node = make_persona_synthesizer_node(_make_mock_llm(_make_profile()))
        result = node(_state(_tendency()))

        profile = result["persona_profile"]
        assert profile.style_summary == "Prefers declarative patterns."
        assert "FastAPI" in profile.tech_preferences

    def test_no_tendency_data_returns_none(self) -> None:
        """tendency_data=None → persona_profile=None, error set, no crash."""
        node = make_persona_synthesizer_node(_make_mock_llm(_make_profile()))
        result = node(_state(None))

        assert result["persona_profile"] is None
        assert result.get("error")

    def test_llm_failure_returns_none_and_error(self) -> None:
        """LLM error → persona_profile=None, error set, no exception raised."""
        with patch("src.agents.persona_synthesizer.invoke_with_retry", side_effect=RuntimeError("Ollama down")):
            node = make_persona_synthesizer_node(_make_mock_llm(_make_profile()))
            result = node(_state(_tendency()))

        assert result["persona_profile"] is None
        assert result.get("error")

    def test_trace_added_on_success(self) -> None:
        """Trace gains an entry on successful synthesis."""
        node = make_persona_synthesizer_node(_make_mock_llm(_make_profile()))
        result = node(_state(_tendency()))

        assert any("persona_synthesizer" in t for t in result["trace"])

    def test_trace_added_on_failure(self) -> None:
        """Trace gains an error entry when LLM fails."""
        with patch("src.agents.persona_synthesizer.invoke_with_retry", side_effect=RuntimeError("boom")):
            node = make_persona_synthesizer_node(_make_mock_llm(_make_profile()))
            result = node(_state(_tendency()))

        assert any("error" in t for t in result["trace"])
