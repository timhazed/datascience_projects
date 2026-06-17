"""Tests for make_chroma_upsert_quarantined_node (src/agents/chroma_upsert_quarantined.py).

Uses real chromadb.EphemeralClient() with fake embeddings — no live Ollama calls.

Cases:
  - One quarantined file → UpsertResult(action='inserted') + semantic_type='quarantine' in ChromaDB
  - Multiple quarantined files → all appear in upsert_results
  - Empty quarantined_files → empty upsert_results, no crash
  - ChromaDB error on individual file → UpsertResult(action='error'); remaining files still processed
  - quarantine_reason stored in ChromaDB metadata
  - Sentinel content (QUARANTINE_SENTINEL) is stored, not original file content
  - Idempotency: same quarantined file twice → first 'inserted', second 'skipped'
  - Trace entries added per file
"""

import hashlib
import uuid

import chromadb

from src.agents.chroma_upsert_quarantined import make_chroma_upsert_quarantined_node
from src.db.chroma_client import ChromaLibrarianClient
from src.models.sanitized_file import QUARANTINE_SENTINEL, SanitizedFile
from src.models.trace_event import TraceEventType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _quarantined(path: str = "src/secret.py", reason: str = "API key detected") -> SanitizedFile:
    return SanitizedFile(path=path, content=QUARANTINE_SENTINEL, quarantine_reason=reason)


def _base_state(**overrides) -> dict:
    return {
        "repo_url": "https://github.com/user/repo",
        "branch": "main",
        "commit_sha": "abc123",
        "quarantined_files": [],
        "trace": [],
        **overrides,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestChromaUpsertQuarantinedNode:
    def test_one_quarantined_file_inserted(self) -> None:
        """One quarantined file → UpsertResult(action='inserted')."""
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)
        result = node(_base_state(quarantined_files=[_quarantined()]))

        assert len(result["upsert_results"]) == 1
        assert result["upsert_results"][0].action == "inserted"

    def test_semantic_type_stored_as_quarantine(self) -> None:
        """ChromaDB document has semantic_type='quarantine' after upsert."""
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)
        node(_base_state(quarantined_files=[_quarantined(path="src/key.py")]))

        docs = chroma.get_quarantined()
        assert len(docs) == 1
        assert docs[0]["metadata"]["semantic_type"] == "quarantine"

    def test_quarantine_reason_stored_in_metadata(self) -> None:
        """quarantine_reason from SanitizedFile is stored in ChromaDB metadata."""
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)
        reason = "SSN pattern detected"
        node(_base_state(quarantined_files=[_quarantined(reason=reason)]))

        docs = chroma.get_quarantined()
        assert docs[0]["metadata"]["quarantine_reason"] == reason

    def test_sentinel_content_stored_not_original(self) -> None:
        """ChromaDB stores per-file sentinel content, not the original file content.

        The stored content is f"{QUARANTINE_SENTINEL}\\npath:{path}" — it starts with
        QUARANTINE_SENTINEL and embeds the path to ensure a unique SHA-256 content_hash
        per quarantined file (bare QUARANTINE_SENTINEL would collide across all files).
        """
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)
        path = "src/secret.py"
        node(_base_state(quarantined_files=[_quarantined(path=path)]))

        docs = chroma.get_quarantined()
        expected_content = f"{QUARANTINE_SENTINEL}\npath:{path}"
        assert docs[0]["document"] == expected_content
        assert docs[0]["document"].startswith(QUARANTINE_SENTINEL)

    def test_multiple_quarantined_files(self) -> None:
        """Multiple quarantined files → all appear in upsert_results AND in ChromaDB.

        Verifies that distinct file paths produce distinct content_hashes — each file
        gets its own ChromaDB document rather than colliding on the shared sentinel value.
        """
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)
        files = [
            _quarantined("src/secret_a.py", "PII"),
            _quarantined("src/secret_b.py", "API key"),
            _quarantined("src/secret_c.py", "credentials"),
        ]
        result = node(_base_state(quarantined_files=files))

        assert len(result["upsert_results"]) == 3
        # Critical: one ChromaDB document per quarantined file, not one shared document
        assert chroma._collection.count() == 3

    def test_empty_quarantined_files_no_crash(self) -> None:
        """No quarantined files → empty upsert_results, no exception."""
        node = make_chroma_upsert_quarantined_node(_make_chroma())
        result = node(_base_state(quarantined_files=[]))

        assert result["upsert_results"] == []

    def test_idempotency_same_file_twice(self) -> None:
        """Same quarantined file twice → first 'inserted', second 'skipped'; one doc in ChromaDB."""
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)
        qf = _quarantined()

        r1 = node(_base_state(quarantined_files=[qf]))
        r2 = node(_base_state(quarantined_files=[qf]))

        assert r1["upsert_results"][0].action == "inserted"
        assert r2["upsert_results"][0].action == "skipped"
        assert chroma._collection.count() == 1

    def test_trace_entries_added_per_file(self) -> None:
        """Trace gains one entry per quarantined file processed."""
        node = make_chroma_upsert_quarantined_node(_make_chroma())
        files = [_quarantined("src/a.py"), _quarantined("src/b.py")]
        result = node(_base_state(quarantined_files=files))

        assert len(result["trace"]) == 2
        paths_in_trace = " ".join(result["trace"])
        assert "src/a.py" in paths_in_trace
        assert "src/b.py" in paths_in_trace

    def test_chroma_error_on_one_file_continues_processing(self) -> None:
        """ChromaDB error on one file → UpsertResult(action='error'); remaining files processed.

        Verifies non-fatal error handling: a failure on file A does not abort file B.
        """
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)

        call_count = 0
        original_upsert = chroma.upsert_chunk

        def failing_first_then_ok(chunk, meta):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                from src.models.chunk import UpsertResult
                return UpsertResult(
                    content_hash=chunk.content_hash,
                    file_path=chunk.path,
                    action="error",
                    error="simulated ChromaDB failure",
                )
            return original_upsert(chunk, meta)

        chroma.upsert_chunk = failing_first_then_ok  # type: ignore[method-assign]
        files = [_quarantined("src/fail.py"), _quarantined("src/ok.py")]
        result = node(_base_state(quarantined_files=files))

        assert len(result["upsert_results"]) == 2
        assert result["upsert_results"][0].action == "error"
        assert result["upsert_results"][1].action == "inserted"


class TestChromaUpsertQuarantinedTraceEvents:
    """Assert trace_events are emitted on every code path."""

    def test_empty_quarantine_emits_skip(self) -> None:
        """No quarantined files → TraceEventType.SKIP with count=0."""
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)
        result = node(_base_state(quarantined_files=[]))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].node == "chroma_upsert_quarantined"
        assert events[0].event == TraceEventType.SKIP
        assert events[0].count == 0

    def test_quarantined_file_emits_quarantine_event(self) -> None:
        """One quarantined file → TraceEventType.QUARANTINE with correct path."""
        chroma = _make_chroma()
        node = make_chroma_upsert_quarantined_node(chroma)
        qf = SanitizedFile(path="src/secret.py", content=QUARANTINE_SENTINEL, quarantine_reason="email")
        result = node(_base_state(quarantined_files=[qf]))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].event == TraceEventType.QUARANTINE
        assert events[0].path == "src/secret.py"
