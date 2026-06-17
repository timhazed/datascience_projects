"""Integration tests for build_persona_graph (src/graphs/persona_graph.py).

Uses real ChromaDB EphemeralClient + RunnableLambda LLM mock — no live Ollama calls.

Cases:
  - Empty ChromaDB → routes to END (insufficient data), persona_profile=None
  - Seeded ChromaDB (≥5 docs) → tendency_data populated, persona_synthesizer called
  - RunnableLambda mock returns PersonaProfile → persona_profile in final state
  - ChromaDB error → tendency_data=None, error set, no crash
  - Graph compiles without error
"""

import hashlib
import uuid
from unittest.mock import MagicMock, patch

import chromadb
from langchain_core.runnables import RunnableLambda

from src.db.chroma_client import ChromaLibrarianClient
from src.graphs.persona_graph import build_persona_graph
from src.models.chunk import SummarizedChunk
from src.models.persona import PersonaProfile


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

_PROFILE = PersonaProfile(
    style_summary="Test developer prefers FastAPI and Pydantic.",
    dominant_patterns=["Uses dependency injection.", "Writes typed APIs."],
    tech_preferences=["FastAPI", "Pydantic"],
    coaching_notes=[],
)


def _make_llm_mock() -> MagicMock:
    """Build a mock LLM whose with_structured_output returns a RunnableLambda."""
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: _PROFILE)
    return llm


def _seed(chroma: ChromaLibrarianClient, n: int = 6) -> None:
    for i in range(n):
        chunk = SummarizedChunk(
            content=f"async def handler_{i}(): ...",
            path=f"src/api/h{i}.py",
            intent_summary=f"HTTP handler {i}.",
            tech_stack=["FastAPI", "Pydantic"],
            semantic_type="Interface",
            author_identity="alice",
        )
        chroma.upsert_chunk(chunk, _META)


def _state(scope: str = "project", recency_months: int = 6) -> dict:
    return {"scope": scope, "recency_months": recency_months, "trace": []}


class TestPersonaGraphCompilation:
    def test_builds_without_error(self) -> None:
        """build_persona_graph returns a compiled graph without raising."""
        chroma = _make_chroma()
        llm = _make_llm_mock()
        graph = build_persona_graph(llm=llm, chroma=chroma)
        assert graph is not None


class TestPersonaGraphInsufficientData:
    def test_empty_collection_routes_to_end(self) -> None:
        """Empty ChromaDB → tendency_data=None, pipeline ends before LLM call."""
        chroma = _make_chroma()
        llm = _make_llm_mock()
        graph = build_persona_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state())

        assert result.get("tendency_data") is None
        assert result.get("persona_profile") is None
        assert result.get("error")

    def test_fewer_than_min_docs_routes_to_end(self) -> None:
        """Fewer than 5 docs → tendency_scanner returns None, persona skipped."""
        chroma = _make_chroma()
        # Seed only 3 docs (below _MIN_DOCS_FOR_PERSONA=5)
        _seed(chroma, n=3)
        llm = _make_llm_mock()
        graph = build_persona_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state())

        assert result.get("tendency_data") is None


class TestPersonaGraphHappyPath:
    def test_seeded_collection_produces_profile(self) -> None:
        """≥5 docs → tendency_data present, persona_synthesizer produces PersonaProfile."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        llm = _make_llm_mock()
        graph = build_persona_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state())

        assert result.get("tendency_data") is not None
        assert result.get("persona_profile") == _PROFILE

    def test_trace_records_both_nodes(self) -> None:
        """Trace has entries from tendency_scanner and persona_synthesizer."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        llm = _make_llm_mock()
        graph = build_persona_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state())

        trace_str = " ".join(result["trace"])
        assert "tendency_scanner" in trace_str
        assert "persona_synthesizer" in trace_str

    def test_recency_months_forwarded_to_scanner(self) -> None:
        """recency_months from state is consumed by tendency_scanner (no crash with custom value)."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        llm = _make_llm_mock()
        graph = build_persona_graph(llm=llm, chroma=chroma)

        result = graph.invoke(_state(recency_months=3))

        # Pipeline should still complete successfully with a custom recency window
        assert result.get("tendency_data") is not None


class TestPersonaGraphErrorPath:
    def test_chroma_error_returns_error_state(self) -> None:
        """ChromaDB failure → tendency_data=None, error set, no crash."""
        chroma = _make_chroma()
        llm = _make_llm_mock()
        graph = build_persona_graph(llm=llm, chroma=chroma)

        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB down")):
            result = graph.invoke(_state())

        assert result.get("tendency_data") is None
        assert result.get("error")
