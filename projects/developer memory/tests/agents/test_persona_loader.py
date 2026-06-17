"""Tests for make_persona_loader_node (src/agents/persona_loader.py).

Uses real ChromaDB EphemeralClient with fake embeddings — no live Ollama calls.

Cases:
  - Fewer than 5 docs → persona_context=None (insufficient data)
  - 5+ docs → persona_context is a PersonaProfile
  - PersonaProfile has style_summary and tech_preferences populated
  - ChromaDB error → persona_context=None (best-effort, pipeline continues)
  - Trace entry added in all paths
"""

import hashlib
import uuid
from unittest.mock import patch

import chromadb

from src.agents.persona_loader import make_persona_loader_node
from src.db.chroma_client import ChromaLibrarianClient
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


def _chunk(i: int) -> SummarizedChunk:
    return SummarizedChunk(
        content=f"def func_{i}(): pass",
        path=f"src/f{i}.py",
        intent_summary=f"Pattern {i}.",
        tech_stack=["FastAPI", "Pydantic"],
        semantic_type="Logic",
        author_identity="alice",
    )


_META = {"repo_url": "https://github.com/u/r", "branch": "main", "commit_sha": "abc123", "indexed_at": "2025-01-01"}


def _seed(chroma: ChromaLibrarianClient, n: int) -> None:
    for i in range(n):
        chroma.upsert_chunk(_chunk(i), _META)


def _state() -> dict:
    return {"trace": []}


class TestPersonaLoaderNode:
    def test_insufficient_docs_returns_none(self) -> None:
        """Fewer than 5 docs → persona_context=None."""
        chroma = _make_chroma()
        _seed(chroma, n=3)
        node = make_persona_loader_node(chroma)
        result = node(_state())

        assert result["persona_context"] is None

    def test_sufficient_docs_returns_profile(self) -> None:
        """5+ docs → persona_context is a PersonaProfile."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_persona_loader_node(chroma)
        result = node(_state())

        assert isinstance(result["persona_context"], PersonaProfile)

    def test_profile_has_style_summary(self) -> None:
        """PersonaProfile.style_summary is non-empty when built from seeded data."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_persona_loader_node(chroma)
        result = node(_state())

        assert result["persona_context"].style_summary

    def test_profile_tech_preferences_populated(self) -> None:
        """PersonaProfile.tech_preferences contains 'FastAPI' from seeded chunks."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_persona_loader_node(chroma)
        result = node(_state())

        assert "FastAPI" in result["persona_context"].tech_preferences

    def test_chroma_error_returns_none_gracefully(self) -> None:
        """ChromaDB error → persona_context=None, no error key, pipeline continues."""
        chroma = _make_chroma()
        node = make_persona_loader_node(chroma)
        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB unavailable")):
            result = node(_state())

        assert result["persona_context"] is None
        # persona_loader is best-effort: it does not set error
        assert "error" not in result

    def test_trace_added_on_success(self) -> None:
        """Trace gains an entry when a profile is loaded."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_persona_loader_node(chroma)
        result = node(_state())

        assert len(result["trace"]) >= 1

    def test_trace_added_on_insufficient_data(self) -> None:
        """Trace gains an entry when doc count is below threshold."""
        chroma = _make_chroma()
        _seed(chroma, n=2)
        node = make_persona_loader_node(chroma)
        result = node(_state())

        assert len(result["trace"]) >= 1
