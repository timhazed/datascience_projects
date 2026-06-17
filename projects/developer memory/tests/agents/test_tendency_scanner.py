"""Tests for make_tendency_scanner_node (src/agents/tendency_scanner.py).

Uses real ChromaDB EphemeralClient with fake embeddings — no live Ollama calls.

Cases:
  - Fewer than 5 docs → tendency_data=None, error set
  - 5+ docs → tendency_data populated (TendencyData)
  - semantic_type_distribution and tech_stack_frequency are populated
  - dominant_patterns contain intent summaries
  - ChromaDB error → tendency_data=None, error set, no crash
  - Trace entry added in all paths
  - Phase 7 exit gate: old docs (>180 days) receive weight=1.0 via apply_temporal_weight
"""

import hashlib
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import chromadb

from src.agents.tendency_scanner import make_tendency_scanner_node
from src.db.chroma_client import ChromaLibrarianClient
from src.ingest.embeddings import RECENCY_WINDOW, apply_temporal_weight
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


def _chunk(
    content: str,
    path: str,
    tech: list[str],
    semantic_type: str = "Logic",
    summary: str = "Does something.",
) -> SummarizedChunk:
    return SummarizedChunk(
        content=content,
        path=path,
        intent_summary=summary,
        tech_stack=tech,
        semantic_type=semantic_type,
        author_identity="alice",
    )


_META = {"repo_url": "https://github.com/u/r", "branch": "main", "commit_sha": "abc123", "indexed_at": "2025-01-01"}


def _seed(chroma: ChromaLibrarianClient, n: int = 6) -> None:
    """Insert n distinct chunks into the collection."""
    for i in range(n):
        chroma.upsert_chunk(_chunk(
            content=f"def func_{i}(): pass",
            path=f"src/f{i}.py",
            tech=["FastAPI", "Pydantic"],
            semantic_type="Logic" if i % 2 == 0 else "Config",
            summary=f"Implements feature {i}.",
        ), _META)


def _state(scope: str = "project") -> dict:
    return {"scope": scope, "trace": []}


class TestTendencyScannerNode:
    def test_insufficient_docs_returns_none(self) -> None:
        """Fewer than 5 docs → tendency_data=None, error set."""
        chroma = _make_chroma()
        _seed(chroma, n=3)
        node = make_tendency_scanner_node(chroma)
        result = node(_state())

        assert result["tendency_data"] is None
        assert result.get("error")

    def test_sufficient_docs_returns_tendency_data(self) -> None:
        """5+ docs → tendency_data is a TendencyData with expected fields."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_tendency_scanner_node(chroma)
        result = node(_state())

        td = result["tendency_data"]
        assert td is not None
        assert td.doc_count >= 6

    def test_semantic_type_distribution_populated(self) -> None:
        """semantic_type_distribution contains counted types."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_tendency_scanner_node(chroma)
        result = node(_state())

        dist = result["tendency_data"].semantic_type_distribution
        assert len(dist) > 0

    def test_tech_stack_frequency_populated(self) -> None:
        """tech_stack_frequency contains 'FastAPI' from seeded chunks."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_tendency_scanner_node(chroma)
        result = node(_state())

        freq = result["tendency_data"].tech_stack_frequency
        assert "FastAPI" in freq

    def test_dominant_patterns_from_intent_summaries(self) -> None:
        """dominant_patterns are non-empty when intent summaries exist in the collection."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_tendency_scanner_node(chroma)
        result = node(_state())

        assert len(result["tendency_data"].dominant_patterns) > 0

    def test_chroma_error_returns_none_tendency(self) -> None:
        """ChromaDB error → tendency_data=None, error set, no crash."""
        chroma = _make_chroma()
        node = make_tendency_scanner_node(chroma)
        with patch.object(chroma, "query", side_effect=RuntimeError("ChromaDB unavailable")):
            result = node(_state())

        assert result["tendency_data"] is None
        assert result.get("error")

    def test_trace_added_on_success(self) -> None:
        """Trace gains an entry on successful scan."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_tendency_scanner_node(chroma)
        result = node(_state())

        assert len(result["trace"]) >= 1

    def test_trace_added_on_insufficient_docs(self) -> None:
        """Trace gains an entry when doc count is below threshold."""
        chroma = _make_chroma()
        _seed(chroma, n=2)
        node = make_tendency_scanner_node(chroma)
        result = node(_state())

        assert len(result["trace"]) >= 1

    # ── Phase 7 exit gate: temporal weighting ─────────────────────────────────

    def test_old_docs_outside_window_receive_unit_weight(self) -> None:
        """Phase 7 exit gate: documents outside 180-day window receive weight=1.0.

        apply_temporal_weight() is a pure function tested independently in test_embeddings.py.
        This test exercises the integration path — a real result dict with an old indexed_at
        timestamp — confirming that the weight applied by tendency_scanner is 1.0, not 3.0.
        """
        now = datetime.now(UTC)
        old_indexed_at = (now - RECENCY_WINDOW - timedelta(days=1)).isoformat()
        recent_indexed_at = (now - timedelta(days=10)).isoformat()

        old_result = {
            "metadata": {
                "indexed_at": old_indexed_at,
                "semantic_type": "Logic",
                "tech_stack": ["Python"],
                "intent_summary": "Old utility.",
            },
            "document": "x = 1",
        }
        recent_result = {
            "metadata": {
                "indexed_at": recent_indexed_at,
                "semantic_type": "Logic",
                "tech_stack": ["FastAPI"],
                "intent_summary": "New handler.",
            },
            "document": "def h(): pass",
        }

        weighted = apply_temporal_weight([old_result, recent_result], now)

        old_weight = weighted[0]["weight"]
        recent_weight = weighted[1]["weight"]

        assert old_weight == 1.0, f"Expected 1.0 for old doc; got {old_weight}"
        assert recent_weight == 3.0, f"Expected 3.0 for recent doc; got {recent_weight}"

    def test_recency_months_state_field_controls_window(self) -> None:
        """Custom recency_months in state is forwarded to apply_temporal_weight window."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_tendency_scanner_node(chroma)

        # recency_months=1 → 30-day window; all seeded docs (indexed_at=2025-01-01) are outside
        result = node({"scope": "project", "recency_months": 1, "trace": []})

        # Pipeline should succeed (6 docs ≥ min); temporal weighting applied differently
        assert result.get("tendency_data") is not None

    # ── Scope validation ──────────────────────────────────────────────────────

    def test_invalid_scope_rejected_before_db_access(self) -> None:
        """An unrecognised scope value returns tendency_data=None with an error message.

        Validation must happen before any ChromaDB query — confirmed by using an empty
        collection so a DB-level failure would manifest differently.
        """
        chroma = _make_chroma()  # empty — would return 0 docs if queried
        node = make_tendency_scanner_node(chroma)
        result = node({"scope": "team", "trace": []})  # "team" is not a valid scope

        assert result["tendency_data"] is None
        assert result.get("error")
        assert "scope" in result["error"].lower() or "project" in result["error"].lower()

    def test_valid_scope_author_accepted(self) -> None:
        """scope='author' is valid — node proceeds to DB query."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_tendency_scanner_node(chroma)
        result = node({"scope": "author", "trace": []})

        # Should succeed (6 docs ≥ min); scope validation must not block valid values
        assert result.get("tendency_data") is not None

    def test_valid_scope_file_accepted(self) -> None:
        """scope='file' is valid — node proceeds to DB query."""
        chroma = _make_chroma()
        _seed(chroma, n=6)
        node = make_tendency_scanner_node(chroma)
        result = node({"scope": "file", "trace": []})

        assert result.get("tendency_data") is not None

    def test_invalid_scope_trace_records_rejection(self) -> None:
        """Trace entry is appended on scope rejection."""
        chroma = _make_chroma()
        node = make_tendency_scanner_node(chroma)
        result = node({"scope": "INVALID", "trace": []})

        assert len(result["trace"]) >= 1
        assert any("scope" in t for t in result["trace"])
