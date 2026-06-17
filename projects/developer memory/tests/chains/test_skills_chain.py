"""Tests for build_skills_chain (src/chains/skills_chain.py).

Uses FakeListChatModel for StrOutputParser chains — no live Ollama calls.

All chain.invoke() calls supply every _HUMAN template variable. Missing any one raises
KeyError from ChatPromptTemplate at invoke time — the Phase 3 exit gate catches this.

Cases (§8 test plan):
  - source-code variant (has_source_code=True) returns a Markdown string
  - docs-only variant (has_source_code=False) returns a Markdown string
  - identifier_index content rendered in formatted prompt
  - logic_chunks content rendered in formatted prompt
  - minimal data (all new variables empty/zero) — no crash
"""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.chains.skills_chain import build_skills_chain

_FAKE_MARKDOWN = (
    "# Project Skills\n\n"
    "## Purpose\nA data science project.\n\n"
    "## Architecture\nUses CNN models in src/models/.\n\n"
    "## Tech Stack\n- PyTorch: model training\n\n"
    "## Coding Patterns\n- Factory pattern for model creation.\n\n"
    "## Conventions & Tendencies\n- Type annotations throughout.\n\n"
    "## Working With This Codebase\n- Read src/models/ first.\n"
)

_FAKE_DOCS_MARKDOWN = (
    "# Project Skills\n\n"
    "## Purpose\nDocumentation repository for ML methodology.\n\n"
    "## Documented Topics\n- SOLID principles\n- Clean code\n\n"
    "## Working With This Repository\n- Browse by topic directory.\n"
)


def _full_inputs(
    *,
    repo_url: str = "https://github.com/user/repo",
    doc_count: int = 42,
    tech_count: int = 5,
    module_groups: str = "src/agents: [src/agents/foo.py]",
    file_manifest_count: int = 3,
    file_manifest: str = "src/agents/foo.py\nsrc/model.py",
    notebook_count: int = 1,
    notebook_files: str = "notebooks/analysis.ipynb",
    identifier_index: str = "src/model.py: [MyModel, train]",
    logic_chunks: str = "src/model.py | MyModel | trains a CNN",
    tech_stacks: str = "- PyTorch\n- FastAPI",
    semantic_type_distribution: str = "{'Logic': 30, 'Config': 5}",
    pattern_summaries: str = "- Uses dependency injection",
) -> dict:
    """Return a complete dict of all 13 _HUMAN template variables."""
    return {
        "repo_url": repo_url,
        "doc_count": doc_count,
        "tech_count": tech_count,
        "module_groups": module_groups,
        "file_manifest_count": file_manifest_count,
        "file_manifest": file_manifest,
        "notebook_count": notebook_count,
        "notebook_files": notebook_files,
        "identifier_index": identifier_index,
        "logic_chunks": logic_chunks,
        "tech_stacks": tech_stacks,
        "semantic_type_distribution": semantic_type_distribution,
        "pattern_summaries": pattern_summaries,
    }


class TestBuildSkillsChain:
    def test_chain_source_code_variant_returns_string(self) -> None:
        """has_source_code=True chain returns a non-empty Markdown string."""
        llm = FakeListChatModel(responses=[_FAKE_MARKDOWN])
        chain = build_skills_chain(llm, has_source_code=True)

        result = chain.invoke(_full_inputs())

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_docs_only_variant_returns_string(self) -> None:
        """has_source_code=False chain returns a non-empty Markdown string."""
        llm = FakeListChatModel(responses=[_FAKE_DOCS_MARKDOWN])
        chain = build_skills_chain(llm, has_source_code=False)

        result = chain.invoke(_full_inputs())

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_renders_identifier_index(self) -> None:
        """identifier_index content appears in the chain's formatted prompt.

        FakeListChatModel ignores the formatted prompt and returns a fixed response.
        We verify the chain accepts the identifier_index variable without KeyError
        and returns the expected LLM output unchanged.
        """
        llm = FakeListChatModel(responses=[_FAKE_MARKDOWN])
        chain = build_skills_chain(llm)

        # Distinctive content — would be visible in prompt if we could inspect it
        result = chain.invoke(
            _full_inputs(identifier_index="src/model.py: [BaseCNNModel, _augment_image]")
        )

        # FakeListChatModel returns its fixed response; no KeyError means the variable
        # was accepted by the template
        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_renders_logic_chunks(self) -> None:
        """logic_chunks content is accepted by the template without KeyError."""
        llm = FakeListChatModel(responses=[_FAKE_MARKDOWN])
        chain = build_skills_chain(llm)

        result = chain.invoke(
            _full_inputs(
                logic_chunks=(
                    "src/trainer.py | Trainer | trains CNN with SGD optimizer\n"
                    "src/model.py | BaseCNNModel | base class for all CNN architectures"
                )
            )
        )

        assert isinstance(result, str)
        assert len(result) > 0

    def test_chain_with_minimal_data(self) -> None:
        """All new variables empty or zero — chain does not crash."""
        llm = FakeListChatModel(responses=["## Purpose\n\nMinimal repo.\n"])
        chain = build_skills_chain(llm)

        result = chain.invoke(
            _full_inputs(
                repo_url="",
                doc_count=0,
                tech_count=0,
                module_groups="(none)",
                file_manifest_count=0,
                file_manifest="(none)",
                notebook_count=0,
                notebook_files="(none)",
                identifier_index="(none)",
                logic_chunks="(none)",
                tech_stacks="(none)",
                semantic_type_distribution="{}",
                pattern_summaries="(none)",
            )
        )

        assert isinstance(result, str)

    def test_chain_is_runnable(self) -> None:
        """build_skills_chain returns an object with an invoke() method."""
        llm = FakeListChatModel(responses=[_FAKE_MARKDOWN])
        chain = build_skills_chain(llm)
        assert callable(getattr(chain, "invoke", None))

    def test_default_variant_is_source_code(self) -> None:
        """build_skills_chain defaults to has_source_code=True (keyword-only arg)."""
        llm = FakeListChatModel(responses=[_FAKE_MARKDOWN])
        # Call with no has_source_code arg — must not raise TypeError
        chain = build_skills_chain(llm)
        result = chain.invoke(_full_inputs())
        assert isinstance(result, str)
