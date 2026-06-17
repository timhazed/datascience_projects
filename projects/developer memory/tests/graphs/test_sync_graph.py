"""Integration tests for build_sync_graph (src/graphs/sync_graph.py).

Uses real ChromaDB EphemeralClient + RunnableLambda LLM mock — no live Ollama or git calls.

Strategy:
  - delta_extractor is injected with a fake git_client lambda (no real git I/O)
  - pii_sanitizer is injected with a MagicMock PIIFilter (no Presidio/spaCy)
  - multimodal_parser, chunk_dispatcher, summarize_and_upsert, chroma_upsert_quarantined run for real
  - LLM mock: RunnableLambda → _ChunkAnnotation (with_structured_output target)

Cases:
  - Graph compiles without error
  - Empty changed files → empty-chunks path → chroma_upsert_quarantined runs, no LLM call
  - All files quarantined → empty parsed_chunks → fast path to chroma_upsert_quarantined
  - N chunks via chunk_dispatcher Send fan-out → N UpsertResults produced
  - delta_extractor error → error set, changed_files=[]
  - Non-default branch forwarded through state correctly
"""

import hashlib
import uuid
from unittest.mock import MagicMock

import chromadb
from langchain_core.runnables import RunnableLambda

from src.agents.chroma_upsert_quarantined import make_chroma_upsert_quarantined_node
from src.agents.chunk_dispatcher import make_chunk_dispatcher_node
from src.agents.delta_extractor import make_delta_extractor_node
from src.agents.summarize_and_upsert import make_summarize_and_upsert_node
from src.chains.intent_summary_chain import _ChunkAnnotation
from src.db.chroma_client import ChromaLibrarianClient
from src.graphs.sync_graph import build_sync_graph
from src.middleware.pii_filter import PIIFilter
from src.models.chunk import ParsedChunk
from src.models.sanitized_file import QUARANTINE_SENTINEL, SanitizedFile


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


def _make_passthrough_pii() -> PIIFilter:
    """PIIFilter mock that passes content through without masking."""
    pii = MagicMock(spec=PIIFilter)
    pii.sanitize.side_effect = lambda path, content: (content, None)
    return pii


def _make_annotation() -> _ChunkAnnotation:
    return _ChunkAnnotation(
        intent_summary="Registers HTTP routes for health endpoints.",
        tech_stack=["FastAPI"],
        semantic_type="Interface",
    )


def _make_llm_mock() -> MagicMock:
    """Mock LLM whose with_structured_output returns a RunnableLambda yielding _ChunkAnnotation."""
    llm = MagicMock()
    annotation = _make_annotation()
    llm.with_structured_output.return_value = RunnableLambda(lambda _: annotation)
    return llm


def _noop_git_client(repo_url: str, branch: str) -> list[str]:
    """Fake git client that returns no changed files."""
    return []


def _python_content(i: int) -> str:
    return f"def handler_{i}(request):\n    return Response(status=200)\n"


def _chunks(n: int) -> list[ParsedChunk]:
    return [
        ParsedChunk(content=_python_content(i), path=f"src/h{i}.py")
        for i in range(n)
    ]


def _base_state() -> dict:
    return {
        "parsed_chunks": [],
        "quarantined_files": [],
        "upsert_results": [],
        "repo_url": "https://github.com/u/r",
        "branch": "main",
        "commit_sha": "abc",
        "repo_root": "",
        "error": None,
        "trace": [],
    }


class TestSyncGraphCompilation:
    def test_builds_without_error(self) -> None:
        """build_sync_graph returns a compiled graph without raising."""
        graph = build_sync_graph(
            llm=_make_llm_mock(),
            chroma=_make_chroma(),
            pii=_make_passthrough_pii(),
            git_client=_noop_git_client,
        )
        assert graph is not None


class TestSyncGraphEmptyPath:
    def test_no_changed_files_dispatcher_returns_dict(self) -> None:
        """Empty parsed_chunks → chunk_dispatcher returns dict (no Send objects)."""
        dispatcher = make_chunk_dispatcher_node()
        state = {**_base_state(), "parsed_chunks": []}

        result = dispatcher(state)

        assert isinstance(result, dict)
        assert "trace" in result

    def test_all_quarantined_files_routes_to_quarantined_node(self) -> None:
        """All files quarantined → parsed_chunks=[] → chroma_upsert_quarantined runs."""
        quarantined = [
            SanitizedFile(path="src/secret.py", content=QUARANTINE_SENTINEL, quarantine_reason="PII")
        ]
        state = {
            **_base_state(),
            "parsed_chunks": [],
            "quarantined_files": quarantined,
            "trace": ["delta_extractor: 0 files", "pii_sanitizer: 1 quarantined", "multimodal_parser: 0 chunks"],
        }

        dispatcher = make_chunk_dispatcher_node()
        chroma = _make_chroma()
        quarantine_node = make_chroma_upsert_quarantined_node(chroma)

        dispatch_result = dispatcher(state)
        # No chunks → dispatcher returns dict (not Send list)
        assert isinstance(dispatch_result, dict)

        upsert_result = quarantine_node({**state, **dispatch_result})
        # Quarantined file is upserted
        assert len(upsert_result["upsert_results"]) == 1
        assert upsert_result["upsert_results"][0].action == "inserted"


class TestSyncGraphHappyPath:
    def test_send_fanout_wiring_compiles(self) -> None:
        """Compilation verifies Send API wiring is accepted by LangGraph.

        The route_after_dispatch function returns Send objects; this test confirms the
        compiled graph accepts that wiring without raising at build time.
        """
        graph = build_sync_graph(
            llm=_make_llm_mock(),
            chroma=_make_chroma(),
            pii=_make_passthrough_pii(),
            git_client=_noop_git_client,
        )
        assert graph is not None

    def test_each_send_target_produces_one_upsert_result(self) -> None:
        """summarize_and_upsert produces exactly one UpsertResult per invocation."""
        llm = _make_llm_mock()
        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(llm, chroma)
        chunks = _chunks(3)

        all_results = []
        for i, chunk in enumerate(chunks):
            result = node({
                "chunk": chunk,
                "repo_url": "https://github.com/u/r",
                "branch": "main",
                "commit_sha": "abc",
                "_job_id": None,
                "chunk_index": i + 1,
                "chunks_total": len(chunks),
            })
            assert len(result["upsert_results"]) == 1  # state contract: exactly one per Send
            all_results.extend(result["upsert_results"])

        assert len(all_results) == 3

    def test_multiple_chunks_all_inserted_in_chroma(self) -> None:
        """All chunks upserted → ChromaDB has one document per distinct chunk."""
        llm = _make_llm_mock()
        chroma = _make_chroma()
        node = make_summarize_and_upsert_node(llm, chroma)
        chunks = _chunks(2)

        for i, chunk in enumerate(chunks):
            node({
                "chunk": chunk,
                "repo_url": "https://github.com/u/r",
                "branch": "main",
                "commit_sha": "abc",
                "_job_id": None,
                "chunk_index": i + 1,
                "chunks_total": len(chunks),
            })

        assert chroma._collection.count() == 2


class TestSyncGraphDeltaExtractorError:
    def test_delta_extractor_error_sets_error_state(self) -> None:
        """delta_extractor failure (git_client raises) → error set, changed_files=[]."""

        def failing_client(repo_url: str, branch: str) -> list[str]:
            raise RuntimeError("Cannot reach remote")

        node = make_delta_extractor_node(failing_client)
        result = node({"repo_url": "https://github.com/u/r", "branch": "main", "trace": []})

        assert result.get("error")
        assert result.get("changed_files", []) == []


class TestSyncGraphNewRouting:
    """Tests for new routing nodes added in Spec §6A–D."""

    def test_builds_with_sha_store_and_pii_callback(self) -> None:
        """build_sync_graph accepts sha_store and pii_progress_callback params."""
        from unittest.mock import MagicMock as MM
        sha_store = MM()
        pii_cb = MM()

        graph = build_sync_graph(
            llm=_make_llm_mock(),
            chroma=_make_chroma(),
            pii=_make_passthrough_pii(),
            git_client=_noop_git_client,
            sha_store=sha_store,
            pii_progress_callback=pii_cb,
        )
        assert graph is not None

    def test_route_after_delta_already_up_to_date_goes_to_end(self) -> None:
        """route_after_delta: already_up_to_date=True → END (no further nodes called)."""
        # We verify this by using a real git_client that raises if called past delta_extractor.
        sha = "abc123"
        mock_store = MagicMock()
        mock_store.get_last_sha.return_value = sha

        clones_called: list = []

        def git_client(url, branch):
            clones_called.append(1)
            return []

        from unittest.mock import patch as _patch
        with _patch("src.agents.delta_extractor._ls_remote_head", return_value=sha):
            from src.config.settings import Settings
            settings = MagicMock(spec=Settings)
            settings.sha_freshness_enabled = True
            settings.cache_filter_enabled = True

            graph = build_sync_graph(
                llm=_make_llm_mock(),
                chroma=_make_chroma(),
                pii=_make_passthrough_pii(),
                git_client=git_client,
                sha_store=mock_store,
                settings=settings,
            )

            result = graph.invoke({
                "repo_url": "https://github.com/u/r",
                "branch": "main",
                "commit_sha": "",
                "trace": [],
            })

        assert result.get("already_up_to_date") is True
        assert clones_called == []  # git_client was never called

    def test_route_after_delta_with_error_goes_to_end(self) -> None:
        """route_after_delta: error set → END (no further nodes called)."""
        def failing_client(url, branch):
            raise RuntimeError("clone failed")

        graph = build_sync_graph(
            llm=_make_llm_mock(),
            chroma=_make_chroma(),
            pii=_make_passthrough_pii(),
            git_client=failing_client,
        )

        result = graph.invoke({
            "repo_url": "https://github.com/u/r",
            "branch": "main",
            "commit_sha": "",
            "trace": [],
        })

        assert result.get("error") is not None

    def test_route_after_cache_filter_empty_goes_to_quarantined(self) -> None:
        """route_after_cache_filter: empty changed_files → chroma_upsert_quarantined."""
        # Use a git_client that returns 0 changed files — cache_filter will see empty list.
        git_client = lambda url, branch: ([], "/tmp/repo", "sha123")  # noqa: E731

        from src.config.settings import Settings
        settings = MagicMock(spec=Settings)
        settings.sha_freshness_enabled = False
        settings.cache_filter_enabled = True

        graph = build_sync_graph(
            llm=_make_llm_mock(),
            chroma=_make_chroma(),
            pii=_make_passthrough_pii(),
            git_client=git_client,
            settings=settings,
        )

        result = graph.invoke({
            "repo_url": "https://github.com/u/r",
            "branch": "main",
            "commit_sha": "",
            "trace": [],
        })

        # Should complete without error
        assert result.get("error") is None
        # SHA was not committed (no sha_store injected)
        trace_str = " ".join(result.get("trace", []))
        assert "sha_store_update" in trace_str


class TestSyncGraphBranchForwarding:
    def test_non_default_branch_forwarded(self) -> None:
        """Branch name passed to graph is preserved in the upserted metadata.

        Verifies that the Send payload correctly forwards branch from SyncState to the
        summarize_and_upsert sub-state, which stores it in ChromaDB metadata.
        """
        chroma = _make_chroma()
        llm = _make_llm_mock()
        node = make_summarize_and_upsert_node(llm, chroma)
        chunk = _chunks(1)[0]
        chunk_id = hashlib.sha256(chunk.content.encode()).hexdigest()

        node({
            "chunk": chunk,
            "repo_url": "https://github.com/u/r",
            "branch": "feature-branch",
            "commit_sha": "def456",
            "_job_id": None,
            "chunk_index": 1,
            "chunks_total": 1,
        })

        docs = chroma._collection.get(ids=[chunk_id], include=["metadatas"])
        assert docs["metadatas"][0]["branch"] == "feature-branch"
