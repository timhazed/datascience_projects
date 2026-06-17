"""Integration tests for build_skills_graph (src/graphs/skills_graph.py).

Uses real ChromaDB EphemeralClient + FakeListChatModel — no live Ollama calls.

Cases:
  - Empty ChromaDB → routes to END (no skills data), skills_markdown=None
  - Seeded ChromaDB → skills_aggregator produces SkillsData → LLM generates markdown
  - File written to tmp_path → export_result populated
  - ChromaDB error → skills_data=None, error set, no crash
  - Graph compiles without error
"""

import hashlib
import uuid
from unittest.mock import patch

import chromadb
import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.db.chroma_client import ChromaLibrarianClient
from src.db.sha_store import SHAStore
from src.graphs.skills_graph import build_skills_graph
from src.models.chunk import SummarizedChunk


def _fake_embed(texts: list[str]) -> list[list[float]]:
    result = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        result.append([float(digest[i] - 128) / 128 for i in range(4)])
    return result


def _make_chroma() -> ChromaLibrarianClient:
    return ChromaLibrarianClient(
        client=chromadb.EphemeralClient(),
        embedding_fn=_fake_embed,
        collection_name=f"test_{uuid.uuid4().hex[:12]}",
    )


_META = {
    "repo_url": "https://github.com/u/r",
    "branch": "main",
    "commit_sha": "abc",
    "indexed_at": "2025-01-01",
}

_SKILLS_MARKDOWN = (
    "## Tech Stack\n- FastAPI\n- Pydantic\n\n## Patterns\n- Dependency injection\n\n## Tendencies\n- Typed APIs\n"
)


def _seed(chroma: ChromaLibrarianClient, n: int = 3) -> None:
    for i in range(n):
        chunk = SummarizedChunk(
            content=f"class Service{i}: ...",
            path=f"src/svc/s{i}.py",
            intent_summary=f"Service layer {i} with dependency injection.",
            tech_stack=["FastAPI", "Pydantic"],
            semantic_type="Logic",
            author_identity="alice",
        )
        chroma.upsert_chunk(chunk, _META)


def _state(target_path: str) -> dict:
    return {"target_path": target_path, "trace": []}


class TestSkillsGraphCompilation:
    def test_builds_without_error(self) -> None:
        """build_skills_graph returns a compiled graph without raising."""
        chroma = _make_chroma()
        llm = FakeListChatModel(responses=[_SKILLS_MARKDOWN])
        graph = build_skills_graph(llm=llm, chroma=chroma)
        assert graph is not None


class TestSkillsGraphEmptyPath:
    def test_empty_collection_routes_to_end(self, tmp_path: pytest.TempPathFactory) -> None:
        """Empty ChromaDB → skills_data=None, pipeline ends, no markdown generated."""
        chroma = _make_chroma()
        llm = FakeListChatModel(responses=["should not be called"])
        graph = build_skills_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state(str(tmp_path / "skills.md")))

        assert result.get("skills_data") is None
        assert result.get("skills_markdown") is None
        assert result.get("error")


class TestSkillsGraphHappyPath:
    def test_seeded_collection_writes_file(self, tmp_path: pytest.TempPathFactory) -> None:
        """Seeded ChromaDB → LLM generates markdown → file written to target_path."""
        chroma = _make_chroma()
        _seed(chroma)
        llm = FakeListChatModel(responses=[_SKILLS_MARKDOWN])
        graph = build_skills_graph(llm=llm, chroma=chroma)

        target = str(tmp_path / "PROJECT_SKILLS.md")
        result = graph.invoke(_state(target))

        assert result.get("skills_data") is not None
        assert result.get("skills_markdown") == _SKILLS_MARKDOWN
        assert result.get("export_result") is not None
        assert result["export_result"]["path"] == target

    def test_file_contents_match_llm_output(self, tmp_path: pytest.TempPathFactory) -> None:
        """The written file contains the exact markdown returned by the LLM."""
        chroma = _make_chroma()
        _seed(chroma)
        llm = FakeListChatModel(responses=[_SKILLS_MARKDOWN])
        graph = build_skills_graph(llm=llm, chroma=chroma)

        target = str(tmp_path / "PROJECT_SKILLS.md")
        graph.invoke(_state(target))

        with open(target) as f:
            assert f.read() == _SKILLS_MARKDOWN

    def test_trace_records_all_nodes(self, tmp_path: pytest.TempPathFactory) -> None:
        """Trace has entries from skills_aggregator, skills_synthesizer, file_exporter."""
        chroma = _make_chroma()
        _seed(chroma)
        llm = FakeListChatModel(responses=[_SKILLS_MARKDOWN])
        graph = build_skills_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state(str(tmp_path / "s.md")))

        trace_str = " ".join(result["trace"])
        assert "skills_aggregator" in trace_str
        assert "skills_synthesizer" in trace_str
        assert "file_exporter" in trace_str


class TestSkillsGraphErrorPath:
    def test_chroma_error_returns_error_state(self, tmp_path: pytest.TempPathFactory) -> None:
        """ChromaDB error → skills_data=None, error set, no crash."""
        chroma = _make_chroma()
        llm = FakeListChatModel(responses=["should not be called"])
        graph = build_skills_graph(llm=llm, chroma=chroma)

        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB down")):
            result = graph.invoke(_state(str(tmp_path / "s.md")))

        assert result.get("skills_data") is None
        assert result.get("error")


_SHA = "abc123def456" * 4


def _sha_state(target_path: str, force: bool = False) -> dict:
    return {"target_path": target_path, "trace": [], "cache_hit": False, "force": force}


class TestSkillsGraphSHAGate:
    def test_cache_hit_skips_aggregator(self, tmp_path: pytest.TempPathFactory) -> None:
        """SHA match + file on disk → graph exits after skills_cache_guard, no LLM call."""
        chroma = _make_chroma()
        _seed(chroma)  # so get_indexed_repo_metadata returns non-empty metadata
        store = SHAStore(data_path=str(tmp_path))

        cached = tmp_path / "cached.md"
        cached.write_text(_SKILLS_MARKDOWN, encoding="utf-8")
        # The SHA in chroma metadata is "abc" (set in _META above); match it in the store
        store.set_skills_sha(_META["repo_url"], _META["branch"], _META["commit_sha"], str(cached))

        llm = FakeListChatModel(responses=["should not be called"])
        graph = build_skills_graph(llm=llm, chroma=chroma, sha_store=store)

        result = graph.invoke(_sha_state(str(tmp_path / "out")))
        assert result.get("cache_hit") is True
        assert result.get("export_result") is not None

    def test_cache_miss_runs_full_pipeline(self, tmp_path: pytest.TempPathFactory) -> None:
        """Empty sha_store → cache miss → full aggregator→synthesizer→exporter run."""
        chroma = _make_chroma()
        _seed(chroma)
        store = SHAStore(data_path=str(tmp_path))  # no stored skills SHA

        llm = FakeListChatModel(responses=[_SKILLS_MARKDOWN])
        graph = build_skills_graph(llm=llm, chroma=chroma, sha_store=store)

        result = graph.invoke(_sha_state(str(tmp_path / "out")))
        assert result.get("cache_hit") is False
        assert result.get("export_result") is not None

    def test_force_bypasses_cache(self, tmp_path: pytest.TempPathFactory) -> None:
        """force=True + matching SHA → cache_hit=False, full pipeline runs."""
        chroma = _make_chroma()
        _seed(chroma)
        store = SHAStore(data_path=str(tmp_path))

        cached = tmp_path / "cached.md"
        cached.write_text(_SKILLS_MARKDOWN, encoding="utf-8")
        store.set_skills_sha(_META["repo_url"], _META["branch"], _META["commit_sha"], str(cached))

        llm = FakeListChatModel(responses=[_SKILLS_MARKDOWN])
        graph = build_skills_graph(llm=llm, chroma=chroma, sha_store=store)

        result = graph.invoke(_sha_state(str(tmp_path / "out"), force=True))
        assert result.get("cache_hit") is False

    def test_no_sha_store_uses_legacy_topology(self, tmp_path: pytest.TempPathFactory) -> None:
        """sha_store=None → no skills_cache_guard node, pipeline starts at aggregator."""
        chroma = _make_chroma()
        _seed(chroma)
        llm = FakeListChatModel(responses=[_SKILLS_MARKDOWN])
        graph = build_skills_graph(llm=llm, chroma=chroma, sha_store=None)

        result = graph.invoke(_state(str(tmp_path / "out")))
        assert result.get("export_result") is not None
        # cache_hit was never set — must not be True
        assert result.get("cache_hit") is not True
