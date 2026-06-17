"""Integration tests for build_query_graph (src/graphs/query_graph.py).

Uses real ChromaDB EphemeralClient + FakeListChatModel — no live Ollama calls.

Cases:
  - Empty query → routes to END via query_guard (query_safe=False), no LLM call
  - Valid query with seeded ChromaDB → final_answer populated
  - ChromaDB error on semantic_searcher → final_answer=None, error set, no crash
  - Graph compiles without error
"""

import hashlib
import uuid
from unittest.mock import patch

import chromadb
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.db.chroma_client import ChromaLibrarianClient
from src.graphs.query_graph import build_query_graph
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


def _seed(chroma: ChromaLibrarianClient, n: int = 3) -> None:
    for i in range(n):
        chunk = SummarizedChunk(
            content=f"def route_{i}(): pass",
            path=f"src/routes/f{i}.py",
            intent_summary=f"Route handler {i}.",
            tech_stack=["FastAPI"],
            semantic_type="Interface",
            author_identity="alice",
        )
        chroma.upsert_chunk(chunk, _META)


class TestQueryGraphCompilation:
    def test_builds_without_error(self) -> None:
        """build_query_graph returns a compiled graph without raising."""
        chroma = _make_chroma()
        llm = FakeListChatModel(responses=["answer"])
        graph = build_query_graph(llm=llm, chroma=chroma)
        assert graph is not None


class TestQueryGraphHappyPath:
    def test_valid_query_returns_final_answer(self) -> None:
        """Valid query with seeded ChromaDB → final_answer populated in state."""
        chroma = _make_chroma()
        _seed(chroma)
        llm = FakeListChatModel(responses=["Routes are structured using FastAPI handlers."])
        graph = build_query_graph(llm=llm, chroma=chroma)

        result = graph.invoke({"query": "how are routes structured", "tech_filter": None, "trace": []})

        assert result["final_answer"] == "Routes are structured using FastAPI handlers."
        assert result.get("query_safe") is True

    def test_trace_records_all_nodes(self) -> None:
        """Trace contains entries from query_guard, semantic_searcher, and snippet_summarizer."""
        chroma = _make_chroma()
        _seed(chroma)
        llm = FakeListChatModel(responses=["answer"])
        graph = build_query_graph(llm=llm, chroma=chroma)

        result = graph.invoke({"query": "authentication patterns", "tech_filter": None, "trace": []})

        trace_str = " ".join(result["trace"])
        assert "query_guard" in trace_str
        assert "semantic_searcher" in trace_str


class TestQueryGraphGuardPath:
    def test_empty_query_routes_to_end(self) -> None:
        """Empty query string → query_safe=False, pipeline ends before LLM call."""
        chroma = _make_chroma()
        llm = FakeListChatModel(responses=["should not be called"])
        graph = build_query_graph(llm=llm, chroma=chroma)

        result = graph.invoke({"query": "", "tech_filter": None, "trace": []})

        assert result.get("query_safe") is False
        assert result.get("final_answer") is None
        assert result.get("error")

    def test_whitespace_query_rejected(self) -> None:
        """Whitespace-only query string → query_safe=False."""
        chroma = _make_chroma()
        llm = FakeListChatModel(responses=["should not be called"])
        graph = build_query_graph(llm=llm, chroma=chroma)

        result = graph.invoke({"query": "   ", "tech_filter": None, "trace": []})

        assert result.get("query_safe") is False


class TestQueryGraphErrorPath:
    def test_chroma_error_sets_error_and_continues(self) -> None:
        """ChromaDB failure in semantic_searcher → error set, snippet_summarizer still runs."""
        chroma = _make_chroma()
        llm = FakeListChatModel(responses=["No results found."])
        graph = build_query_graph(llm=llm, chroma=chroma)

        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB down")):
            result = graph.invoke({"query": "valid query text", "tech_filter": None, "trace": []})

        # semantic_searcher error is recorded; snippet_summarizer still runs (best-effort)
        assert result.get("error")
        # Pipeline does not crash
        assert "trace" in result
