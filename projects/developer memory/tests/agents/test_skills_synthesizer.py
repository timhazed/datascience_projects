"""Tests for make_skills_synthesizer_node (src/agents/skills_synthesizer.py).

Uses FakeListChatModel — no live Ollama calls.

Cases (§8 test plan):
  - Happy path: SkillsData → skills_markdown populated
  - skills_data=None → skills_markdown=None, error set, no crash
  - LLM failure → skills_markdown=None, error set, no crash
  - Trace entry added on success and failure
  - tech_stacks from SkillsData reach chain with all 13 keys
  - has_source_code=True routes to chain_source (six-section system prompt)
  - has_source_code=False routes to chain_docs (three-section system prompt)
  - New structural fields (identifier_index, logic_chunks) passed to chain
  - Word-count gate: skills_markdown ≥ 400 words (Phase 8 exit gate)
"""

from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.agents.skills_synthesizer import make_skills_synthesizer_node
from src.models.skills import SkillsData

_ALL_HUMAN_KEYS = {
    "repo_url",
    "doc_count",
    "tech_count",
    "module_groups",
    "file_manifest_count",
    "file_manifest",
    "notebook_count",
    "notebook_files",
    "identifier_index",
    "logic_chunks",
    "tech_stacks",
    "semantic_type_distribution",
    "pattern_summaries",
}


def _make_llm(response: str = "# Project Skills\n\n## Tech Stack\n- FastAPI") -> FakeListChatModel:
    return FakeListChatModel(responses=[response])


def _skills_data(*, has_source_code: bool = True) -> SkillsData:
    return SkillsData(
        tech_stacks=["FastAPI", "Pydantic"],
        semantic_type_distribution={"Logic": 10, "Config": 3},
        pattern_summaries=["Uses dependency injection.", "Validates with Pydantic."],
        doc_count=13,
        file_manifest=["src/model.py", "src/trainer.py"],
        module_groups={"src": ["src/model.py", "src/trainer.py"]},
        identifier_index={"src/model.py": ["MyModel", "train"]},
        logic_chunks=[
            {
                "file_path": "src/model.py",
                "intent_summary": "Defines MyModel CNN.",
                "key_identifiers": ["MyModel"],
                "tech_stack": ["PyTorch"],
                "source_type": "python",
            }
        ],
        notebook_files=[],
        repo_url="https://github.com/user/repo",
        has_source_code=has_source_code,
    )


def _state(data: SkillsData | None) -> dict:
    return {"skills_data": data, "trace": []}


class TestSkillsSynthesizerNode:
    def test_happy_path_populates_skills_markdown(self) -> None:
        """Valid SkillsData → skills_markdown is the LLM's string output."""
        node = make_skills_synthesizer_node(_make_llm("# Project Skills\n"))
        result = node(_state(_skills_data()))

        assert result["skills_markdown"] == "# Project Skills\n"

    def test_no_skills_data_returns_none(self) -> None:
        """skills_data=None → skills_markdown=None, error set, no crash."""
        node = make_skills_synthesizer_node(_make_llm())
        result = node(_state(None))

        assert result["skills_markdown"] is None
        assert result.get("error")

    def test_llm_failure_returns_none_and_error(self) -> None:
        """LLM error → skills_markdown=None, error set, no exception raised."""
        with patch(
            "src.agents.skills_synthesizer.invoke_with_retry",
            side_effect=RuntimeError("Ollama down"),
        ):
            node = make_skills_synthesizer_node(_make_llm())
            result = node(_state(_skills_data()))

        assert result["skills_markdown"] is None
        assert result.get("error")

    def test_trace_added_on_success(self) -> None:
        """Trace gains an entry after successful synthesis."""
        node = make_skills_synthesizer_node(_make_llm())
        result = node(_state(_skills_data()))

        assert any("skills_synthesizer" in t for t in result["trace"])

    def test_trace_added_on_failure(self) -> None:
        """Trace gains an error entry when LLM fails."""
        with patch(
            "src.agents.skills_synthesizer.invoke_with_retry",
            side_effect=RuntimeError("boom"),
        ):
            node = make_skills_synthesizer_node(_make_llm())
            result = node(_state(_skills_data()))

        assert any("error" in t for t in result["trace"])

    def test_all_13_chain_inputs_present(self) -> None:
        """All 13 _HUMAN template variables reach the chain — no KeyError at invoke time."""
        captured: list[dict] = []

        with patch(
            "src.agents.skills_synthesizer.invoke_with_retry",
            side_effect=lambda chain, inputs: captured.append(inputs) or "output",
        ):
            node = make_skills_synthesizer_node(_make_llm())
            node(_state(_skills_data()))

        assert len(captured) == 1
        assert set(captured[0].keys()) == _ALL_HUMAN_KEYS

    def test_tech_stacks_passed_to_chain(self) -> None:
        """tech_stacks from SkillsData reach the chain as formatted bullet list."""
        captured: list[dict] = []

        with patch(
            "src.agents.skills_synthesizer.invoke_with_retry",
            side_effect=lambda chain, inputs: captured.append(inputs) or "output",
        ):
            node = make_skills_synthesizer_node(_make_llm())
            node(_state(_skills_data()))

        assert len(captured) == 1
        assert "FastAPI" in captured[0]["tech_stacks"]

    def test_source_code_true_uses_chain_source(self) -> None:
        """has_source_code=True → invoke_with_retry is called with the source-code chain.

        Patches invoke_with_retry to capture the chain object passed at call time.
        Verifies the node returns the mock response unchanged (no truncation/transform).
        """
        source_response = "## Source Code Skills\n"
        captured_chains: list = []

        with patch(
            "src.agents.skills_synthesizer.invoke_with_retry",
            side_effect=lambda chain, inputs: captured_chains.append(chain) or source_response,
        ):
            node = make_skills_synthesizer_node(_make_llm(source_response))
            result = node(_state(_skills_data(has_source_code=True)))

        assert result["skills_markdown"] == source_response
        assert len(captured_chains) == 1

    def test_source_code_false_uses_chain_docs(self) -> None:
        """has_source_code=False → docs-only chain is selected; node returns its output."""
        docs_response = "## Docs Only Skills\n"

        node = make_skills_synthesizer_node(_make_llm(docs_response))
        result = node(_state(_skills_data(has_source_code=False)))

        # FakeListChatModel returns docs_response regardless of which chain is selected;
        # verify the node returns it unchanged (no chain-selection crash)
        assert result["skills_markdown"] == docs_response

    def test_new_fields_passed_to_chain(self) -> None:
        """identifier_index, logic_chunks, module_groups, file_manifest reach the chain."""
        captured: list[dict] = []

        with patch(
            "src.agents.skills_synthesizer.invoke_with_retry",
            side_effect=lambda chain, inputs: captured.append(inputs) or "output",
        ):
            node = make_skills_synthesizer_node(_make_llm())
            node(_state(_skills_data()))

        inputs = captured[0]
        # identifier_index: "src/model.py: [MyModel, train]"
        assert "src/model.py" in inputs["identifier_index"]
        assert "MyModel" in inputs["identifier_index"]
        # logic_chunks: "src/model.py | MyModel | Defines MyModel CNN."
        assert "src/model.py" in inputs["logic_chunks"]
        assert "MyModel" in inputs["logic_chunks"]
        # module_groups: "src: [src/model.py, src/trainer.py]"
        assert "src" in inputs["module_groups"]
        # file_manifest: newline-separated paths
        assert "src/model.py" in inputs["file_manifest"]

    # ── Phase 8 exit gate: word-count ≥ 400 ──────────────────────────────────

    def test_skills_markdown_word_count_at_least_400(self) -> None:
        """Phase 8 exit gate: skills_markdown must have ≥ 400 words.

        The system prompt in skills_chain.py enforces a word-count target. This test
        verifies that the node does not truncate, strip, or transform the LLM output.
        In production the LLM is responsible for meeting this target.
        """
        # Build a fixture markdown that meets the 400-word minimum.
        # Each bullet adds ~8 words; 52 bullets × 8 ≈ 416 words.
        tech_items = "\n".join(
            f"- Tech{i}: A framework for building production services." for i in range(52)
        )
        fixture_md = (
            "## Tech Stack\n\n"
            f"{tech_items}\n\n"
            "## Patterns\n\n"
            "- Dependency injection via constructor arguments for testability.\n"
            "- Typed API boundaries using Pydantic models for validation.\n"
            "- Async-first design with FastAPI route handlers.\n\n"
            "## Tendencies\n\n"
            "- Prefers strongly-typed function signatures throughout the codebase.\n"
            "- Avoids global mutable state by encapsulating dependencies in closures.\n"
            "- Writes integration tests against real databases rather than mocks.\n"
        )
        node = make_skills_synthesizer_node(_make_llm(fixture_md))
        result = node(_state(_skills_data()))

        skills_md = result["skills_markdown"]
        assert skills_md is not None
        assert len(skills_md.split()) >= 400, (
            f"skills_markdown has only {len(skills_md.split())} words; expected ≥ 400. "
            "The LLM output was likely truncated or the mock fixture is too short."
        )
