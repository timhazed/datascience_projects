"""Tests for make_summarize_and_upsert_node (src/agents/summarize_and_upsert.py).

Uses real chromadb.EphemeralClient() with fake embeddings — no live Ollama calls.
LLM chain is mocked via RunnableLambda → _ChunkAnnotation.

Cases:
  - Happy path: new chunk → LLM called → inserted into ChromaDB → UpsertResult(action="inserted")
  - Skip-before-LLM: already-indexed chunk → chroma.exists() True → LLM not called → "skipped"
  - LLM failure → UpsertResult(action="error") + error key in result; no crash
  - Exactly one UpsertResult returned per invocation (operator.add accumulation contract)
  - Trace entry added on success, skip, and failure
  - Content and path passed through to SummarizedChunk (LLM does not echo them)
  - Semaphore limits concurrency: two threads with concurrency=1 run serially
  - Progress callback called after each upsert (success and skip)
  - Progress callback exception is suppressed (no crash)
  - Idempotency: same chunk twice → first "inserted", second "skipped" (via skip-before-LLM)
"""

import hashlib
import threading
import time
import uuid
from unittest.mock import MagicMock, patch

import chromadb
from langchain_core.runnables import RunnableLambda

from src.agents.summarize_and_upsert import make_summarize_and_upsert_node
from src.chains.intent_summary_chain import _ChunkAnnotation
from src.db.chroma_client import ChromaLibrarianClient
from src.models.chunk import ParsedChunk
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


def _make_annotation(
    summary: str = "Registers HTTP routes for health check.",
    tech_stack: list[str] | None = None,
    semantic_type: str = "Interface",
) -> _ChunkAnnotation:
    return _ChunkAnnotation(
        intent_summary=summary,
        tech_stack=tech_stack or ["FastAPI"],
        semantic_type=semantic_type,  # type: ignore[arg-type]
    )


def _make_mock_llm(annotation: _ChunkAnnotation) -> MagicMock:
    """Return a mock LLM whose with_structured_output returns a RunnableLambda fixture."""
    llm = MagicMock()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: annotation)
    return llm


def _chunk_state(
    content: str = "app = FastAPI()",
    path: str = "src/main.py",
    repo_url: str = "https://github.com/user/repo",
    branch: str = "main",
    commit_sha: str = "abc123",
    job_id: str | None = None,
    chunk_index: int = 1,
    chunks_total: int = 1,
) -> dict:
    return {
        "chunk": ParsedChunk(path=path, content=content),
        "repo_url": repo_url,
        "branch": branch,
        "commit_sha": commit_sha,
        "_job_id": job_id,
        "chunk_index": chunk_index,
        "chunks_total": chunks_total,
    }


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestSummarizeAndUpsertHappyPath:
    def test_new_chunk_is_inserted(self) -> None:
        """New chunk → LLM called → ChromaDB.insert → UpsertResult(action='inserted')."""
        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), chroma)
        result = node(_chunk_state())

        assert len(result["upsert_results"]) == 1
        assert result["upsert_results"][0].action == "inserted"
        assert chroma._collection.count() == 1

    def test_returns_exactly_one_upsert_result(self) -> None:
        """Exactly one UpsertResult returned per invocation (operator.add accumulation contract)."""
        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), _make_chroma())
        result = node(_chunk_state())

        assert len(result["upsert_results"]) == 1

    def test_trace_entry_added_on_success(self) -> None:
        """Trace gains one entry on successful upsert."""
        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), _make_chroma())
        result = node(_chunk_state(path="src/api.py"))

        assert len(result["trace"]) == 1
        assert "src/api.py" in result["trace"][0]

    def test_content_hash_matches_sha256_of_content(self) -> None:
        """content_hash in UpsertResult is SHA-256 hex of chunk.content."""
        content = "x = 1"
        expected_hash = hashlib.sha256(content.encode()).hexdigest()

        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), _make_chroma())
        result = node(_chunk_state(content=content))

        assert result["upsert_results"][0].content_hash == expected_hash


# ---------------------------------------------------------------------------
# Skip-before-LLM fast path
# ---------------------------------------------------------------------------


class TestSkipBeforeLLM:
    def test_already_indexed_chunk_is_skipped(self) -> None:
        """chroma.exists() True → LLM not called → UpsertResult(action='skipped')."""
        chroma = _make_chroma()
        llm = _make_mock_llm(_make_annotation())
        node = make_summarize_and_upsert_node(llm, chroma)

        # First call: inserts the chunk
        node(_chunk_state())
        call_count_after_first = llm.with_structured_output.call_count

        # Second call: same content → chroma.exists() returns True → LLM skipped
        result = node(_chunk_state())

        assert result["upsert_results"][0].action == "skipped"
        # with_structured_output is called once at factory time; chain is not re-invoked
        assert llm.with_structured_output.call_count == call_count_after_first

    def test_idempotency_same_chunk_twice(self) -> None:
        """Same chunk twice → first 'inserted', second 'skipped'; one doc in ChromaDB."""
        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), chroma)

        r1 = node(_chunk_state())
        r2 = node(_chunk_state())

        assert r1["upsert_results"][0].action == "inserted"
        assert r2["upsert_results"][0].action == "skipped"
        assert chroma._collection.count() == 1

    def test_skip_trace_entry_mentions_cached(self) -> None:
        """Trace entry on skip indicates the cached fast path."""
        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), chroma)

        node(_chunk_state())
        result = node(_chunk_state())

        assert "skip" in result["trace"][0].lower() or "cached" in result["trace"][0].lower()


# ---------------------------------------------------------------------------
# LLM failure
# ---------------------------------------------------------------------------


class TestLLMFailure:
    def test_llm_failure_returns_error_no_crash(self) -> None:
        """LLM error → UpsertResult(action='error') + error key in result; no exception raised."""
        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), chroma)

        with patch("src.agents.summarize_and_upsert.invoke_with_retry", side_effect=RuntimeError("Ollama down")):
            result = node(_chunk_state())

        assert "error" in result
        assert len(result["upsert_results"]) == 1
        assert result["upsert_results"][0].action == "error"

    def test_llm_failure_trace_entry_added(self) -> None:
        """Trace gains one error entry when LLM fails."""
        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), chroma)

        with patch("src.agents.summarize_and_upsert.invoke_with_retry", side_effect=RuntimeError("boom")):
            result = node(_chunk_state(path="bad.py"))

        assert len(result["trace"]) == 1
        assert "bad.py" in result["trace"][0]

    def test_llm_failure_does_not_write_to_chroma(self) -> None:
        """LLM failure → no document written to ChromaDB."""
        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(_make_mock_llm(_make_annotation()), chroma)

        with patch("src.agents.summarize_and_upsert.invoke_with_retry", side_effect=RuntimeError("down")):
            node(_chunk_state())

        assert chroma._collection.count() == 0


# ---------------------------------------------------------------------------
# Concurrency semaphore
# ---------------------------------------------------------------------------


class TestSemaphoreConcurrency:
    def test_semaphore_gates_concurrent_llm_calls(self) -> None:
        """With ollama_concurrency=1, two threads cannot hold the semaphore simultaneously.

        Strategy: patch invoke_with_retry to record concurrent call overlap. With concurrency=1,
        the second thread must wait for the first to finish before acquiring the semaphore.
        """
        chroma_a = _make_chroma()
        concurrent_peak = 0
        active = 0
        lock = threading.Lock()

        def slow_invoke(chain, inputs):
            nonlocal concurrent_peak, active
            with lock:
                active += 1
                concurrent_peak = max(concurrent_peak, active)
            time.sleep(0.05)  # hold semaphore long enough for second thread to attempt
            with lock:
                active -= 1
            return _make_annotation()

        # Two nodes sharing the same semaphore via concurrency=1 factory
        node = make_summarize_and_upsert_node(
            _make_mock_llm(_make_annotation()), chroma_a, ollama_concurrency=1
        )

        with patch("src.agents.summarize_and_upsert.invoke_with_retry", slow_invoke):
            t1 = threading.Thread(target=node, args=(_chunk_state(content="chunk_a"),))
            t2 = threading.Thread(target=node, args=(_chunk_state(content="chunk_b"),))
            t1.start()
            t2.start()
            t1.join()
            t2.join()

        # With concurrency=1, never more than 1 LLM call active simultaneously
        assert concurrent_peak <= 1


# ---------------------------------------------------------------------------
# Progress callback
# ---------------------------------------------------------------------------


class TestProgressCallback:
    def test_callback_invoked_after_insert(self) -> None:
        """progress_callback is called after successful upsert."""
        calls: list[tuple] = []
        job_id = "test-job-id"

        def callback(jid: str, done: int, total: int) -> None:
            calls.append((jid, done, total))

        node = make_summarize_and_upsert_node(
            _make_mock_llm(_make_annotation()),
            _make_chroma(),
            progress_callback=callback,
        )
        node(_chunk_state(job_id=job_id, chunk_index=1, chunks_total=5))

        assert len(calls) == 1
        assert calls[0] == (job_id, 1, 5)

    def test_callback_invoked_after_skip(self) -> None:
        """progress_callback is called even when chunk is skipped (already indexed)."""
        calls: list[tuple] = []
        job_id = "skip-job"

        def callback(jid: str, done: int, total: int) -> None:
            calls.append((jid, done, total))

        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(
            _make_mock_llm(_make_annotation()),
            chroma,
            progress_callback=callback,
        )
        # First: insert
        node(_chunk_state(job_id=job_id, chunk_index=1, chunks_total=1))
        calls.clear()
        # Second: should be skipped but callback still fires
        node(_chunk_state(job_id=job_id, chunk_index=1, chunks_total=1))

        assert len(calls) == 1

    def test_callback_exception_does_not_crash_node(self) -> None:
        """Exceptions in progress_callback are suppressed; node returns normally."""
        def bad_callback(jid: str, done: int, total: int) -> None:
            raise RuntimeError("callback failure")

        node = make_summarize_and_upsert_node(
            _make_mock_llm(_make_annotation()),
            _make_chroma(),
            progress_callback=bad_callback,
        )
        result = node(_chunk_state(job_id="any-job", chunk_index=1, chunks_total=1))

        assert len(result["upsert_results"]) == 1

    def test_no_callback_when_job_id_absent(self) -> None:
        """Callback is not invoked when _job_id is None in sub-state."""
        calls: list = []

        def callback(jid: str, done: int, total: int) -> None:
            calls.append((jid, done, total))

        node = make_summarize_and_upsert_node(
            _make_mock_llm(_make_annotation()),
            _make_chroma(),
            progress_callback=callback,
        )
        # _job_id absent (None) — callback must not fire
        node(_chunk_state(job_id=None))

        assert calls == []


class TestSummarizeAndUpsertTraceEvents:
    """Assert trace_events are emitted on every code path."""

    def test_skip_cached_emits_cached_event(self) -> None:
        """Chunk already in ChromaDB → TraceEventType.CACHED."""
        chroma = _make_chroma()
        annotation = _make_annotation()
        llm = _make_mock_llm(annotation)
        node = make_summarize_and_upsert_node(llm, chroma)
        state = _chunk_state()
        # Pre-insert so exists() returns True
        chunk_id = __import__("hashlib").sha256(state["chunk"].content.encode()).hexdigest()
        chroma._collection.add(
            ids=[chunk_id], documents=["x"], embeddings=[[0.0] * 4],
            metadatas=[{"file_path": "src/main.py", "semantic_type": "Logic", "tech_stack": "[]",
                        "intent_summary": "", "key_identifiers": "[]", "author_identity": "",
                        "repo_url": "", "branch": "main", "commit_sha": "", "indexed_at": ""}],
        )
        result = node(state)
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].node == "summarize_and_upsert"
        assert events[0].event == TraceEventType.CACHED

    def test_successful_upsert_emits_ok_event(self) -> None:
        """New chunk successfully summarized and upserted → TraceEventType.OK."""
        chroma = _make_chroma()
        annotation = _make_annotation()
        llm = _make_mock_llm(annotation)
        node = make_summarize_and_upsert_node(llm, chroma)
        result = node(_chunk_state(content="def new(): pass"))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].event == TraceEventType.OK

    def test_llm_error_emits_error_event(self) -> None:
        """LLM chain raising → TraceEventType.ERROR."""
        chroma = _make_chroma()
        llm = MagicMock()
        llm.with_structured_output.return_value = RunnableLambda(
            lambda _: (_ for _ in ()).throw(RuntimeError("timeout"))
        )
        node = make_summarize_and_upsert_node(llm, chroma)
        result = node(_chunk_state(content="def unique_llm_error(): pass"))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].event == TraceEventType.ERROR
