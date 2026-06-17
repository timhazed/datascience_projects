"""Tests for build_persona_synthesis_chain (src/chains/persona_synthesis_chain.py).

Uses RunnableLambda to mock with_structured_output — no live Ollama calls.

Cases:
  - Chain returns a PersonaProfile on success
  - All PersonaProfile fields are present and correctly typed
  - Chain is built once (factory pattern)
"""

from unittest.mock import MagicMock

from langchain_core.runnables import RunnableLambda

from src.chains.persona_synthesis_chain import build_persona_synthesis_chain
from src.models.persona import PersonaProfile


def _fake_profile() -> PersonaProfile:
    return PersonaProfile(
        style_summary="This developer favours type-annotated, functional Python.",
        dominant_patterns=["uses_type_annotations", "avoids_global_state"],
        tech_preferences=["FastAPI", "Pydantic"],
        coaching_notes=["Inconsistent error handling in async paths."],
    )


def _make_llm(output: PersonaProfile) -> MagicMock:
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: output)
    return llm


class TestBuildPersonaSynthesisChain:
    def test_chain_returns_persona_profile(self) -> None:
        """Chain built with a mock LLM returns a PersonaProfile on invoke."""
        fake_llm = _make_llm(_fake_profile())
        chain = build_persona_synthesis_chain(fake_llm)

        result = chain.invoke({
            "doc_count": 42,
            "semantic_type_distribution": "{'Logic': 30, 'Config': 12}",
            "tech_stack_frequency": "{'FastAPI': 20, 'Pydantic': 18}",
            "dominant_patterns": "- Uses type annotations consistently\n- Avoids global state",
        })

        assert isinstance(result, PersonaProfile)
        assert result.style_summary
        assert isinstance(result.dominant_patterns, list)
        assert isinstance(result.tech_preferences, list)
        assert isinstance(result.coaching_notes, list)

    def test_chain_profile_fields_match_fixture(self) -> None:
        """Chain passes through the fake profile fields unchanged."""
        fake_llm = _make_llm(_fake_profile())
        chain = build_persona_synthesis_chain(fake_llm)

        result = chain.invoke({
            "doc_count": 10,
            "semantic_type_distribution": "{}",
            "tech_stack_frequency": "{}",
            "dominant_patterns": "",
        })

        assert result.tech_preferences == ["FastAPI", "Pydantic"]
        assert "uses_type_annotations" in result.dominant_patterns

    def test_chain_is_runnable(self) -> None:
        """build_persona_synthesis_chain returns a Runnable without raising."""
        fake_llm = _make_llm(_fake_profile())
        chain = build_persona_synthesis_chain(fake_llm)
        assert callable(getattr(chain, "invoke", None))
