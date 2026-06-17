"""Tests for ChromaLibrarianClient (src/db/chroma_client.py).

Phase 3 exit gate — all tests use chromadb.EphemeralClient(); no test touches
the persistent chroma_data/ volume. A deterministic fake embedding function is
injected to avoid live Ollama calls.

Coverage target: ≥ 90% branch coverage on chroma_client.py.

Cases covered:
  - Case 1 (new): upsert inserts document; collection count increases
  - Case 2 (exact duplicate): second upsert with same hash + same path/repo → skipped; no new doc
  - Case 3 (file moved): same hash, different file_path → metadata-only update; still 1 doc
  - Multi-repo idempotency: identical content from repo A then repo B → Case 3; 1 doc; repo B wins
  - query(): empty collection returns []
  - query(): tech_filter returns only matching docs
  - query(): where= filter works
  - get_by_author(): returns docs for matching author
  - get_quarantined(): returns only docs with semantic_type == "quarantine"
  - upsert_chunk(): content_hash is a valid 64-char hex string (SHA-256)
  - upsert_chunk() exception → UpsertResult(action="error") returned, no crash
"""

import hashlib
import json
import uuid
from unittest.mock import MagicMock, patch

import chromadb

from src.config.settings import Settings
from src.db.chroma_client import ChromaLibrarianClient
from src.models.chunk import SummarizedChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_embed(texts: list[str]) -> list[list[float]]:
    """Deterministic 4-dim embedding: SHA-256 bytes → 4 floats. Fast; never calls Ollama."""
    result = []
    for text in texts:
        digest = hashlib.sha256(text.encode()).digest()
        vec = [float(digest[i] - 128) / 128 for i in range(4)]
        result.append(vec)
    return result


def _make_client() -> ChromaLibrarianClient:
    """Return a ChromaLibrarianClient backed by an in-memory EphemeralClient.

    A unique collection_name is passed per call because chromadb.EphemeralClient()
    shares in-memory state across instantiations in the same process — without unique
    names, tests pollute each other's collections. collection_name is a legitimate
    config parameter (multi-tenant / migration use cases), not a test-only concern.
    """
    return ChromaLibrarianClient(
        client=chromadb.EphemeralClient(),
        embedding_fn=_fake_embed,
        collection_name=f"test_{uuid.uuid4().hex[:12]}",
    )


def _make_chunk(
    content: str = "x = 1",
    path: str = "src/main.py",
    intent_summary: str = "A simple assignment.",
    tech_stack: list[str] | None = None,
    semantic_type: str = "Logic",
    author_identity: str = "alice",
) -> SummarizedChunk:
    return SummarizedChunk(
        content=content,
        path=path,
        intent_summary=intent_summary,
        tech_stack=tech_stack or [],
        semantic_type=semantic_type,  # type: ignore[arg-type]
        author_identity=author_identity,
    )


_BASE_META = {
    "repo_url": "https://github.com/user/repo-a",
    "branch": "main",
    "commit_sha": "abc123",
    "indexed_at": "2026-05-01T00:00:00+00:00",
}


# ---------------------------------------------------------------------------
# exists()
# ---------------------------------------------------------------------------


class TestExists:
    def test_returns_false_for_unknown_id(self) -> None:
        """exists() returns False when the chunk_id is not in the collection."""
        client = _make_client()
        assert client.exists("nonexistent" * 4) is False

    def test_returns_true_after_insert(self) -> None:
        """exists() returns True for a chunk_id that was already upserted."""
        client = _make_client()
        chunk = _make_chunk()
        client.upsert_chunk(chunk, _BASE_META)
        assert client.exists(chunk.content_hash) is True

    def test_returns_false_before_insert(self) -> None:
        """exists() returns False for a valid SHA-256 id before any upsert."""
        client = _make_client()
        chunk = _make_chunk()
        assert client.exists(chunk.content_hash) is False

    def test_exception_returns_false(self) -> None:
        """exists() returns False (safe default) when ChromaDB raises an exception."""
        bad_collection = MagicMock()
        bad_collection.get.side_effect = RuntimeError("connection refused")
        client = ChromaLibrarianClient.__new__(ChromaLibrarianClient)
        client._collection = bad_collection
        client._embed = _fake_embed

        assert client.exists("any_id") is False


# ---------------------------------------------------------------------------
# Upsert contract (3-case)
# ---------------------------------------------------------------------------


class TestUpsertChunk:
    def test_case1_new_document_inserted(self) -> None:
        """Case 1: new content_hash → document added; action == 'inserted'."""
        client = _make_client()
        chunk = _make_chunk()
        result = client.upsert_chunk(chunk, _BASE_META)

        assert result.action == "inserted"
        assert client._collection.count() == 1

    def test_case2_exact_duplicate_skipped(self) -> None:
        """Case 2: same hash + same file_path + same repo_url → skip; still 1 doc."""
        client = _make_client()
        chunk = _make_chunk()

        first = client.upsert_chunk(chunk, _BASE_META)
        second = client.upsert_chunk(chunk, _BASE_META)

        assert first.action == "inserted"
        assert second.action == "skipped"
        assert client._collection.count() == 1

    def test_case3_file_moved_metadata_updated(self) -> None:
        """Case 3: same content_hash, different file_path → metadata-only update; 1 doc."""
        client = _make_client()
        chunk_a = _make_chunk(path="src/old.py")
        chunk_b = _make_chunk(path="src/new.py")  # same content → same hash

        client.upsert_chunk(chunk_a, _BASE_META)
        result = client.upsert_chunk(chunk_b, _BASE_META)

        assert result.action == "updated"
        assert client._collection.count() == 1
        # Verify metadata was updated to the new path
        stored = client._collection.get(ids=[chunk_b.content_hash])
        assert stored["metadatas"][0]["file_path"] == "src/new.py"

    def test_case3_different_repo_url_triggers_update(self) -> None:
        """Case 3 variant: same file_path but different repo_url → update (not skip)."""
        client = _make_client()
        chunk = _make_chunk()
        meta_a = {**_BASE_META, "repo_url": "https://github.com/user/repo-a"}
        meta_b = {**_BASE_META, "repo_url": "https://github.com/user/repo-b"}

        client.upsert_chunk(chunk, meta_a)
        result = client.upsert_chunk(chunk, meta_b)

        assert result.action == "updated"
        assert client._collection.count() == 1

    def test_multi_repo_idempotency(self) -> None:
        """Phase 3 exit gate: identical content from repo A then repo B → Case 3.

        Two repos sharing identical boilerplate produce one ChromaDB document, not two.
        The second upsert (repo B) wins — repo_url metadata is updated to repo B.
        """
        client = _make_client()
        shared_content = "# shared boilerplate\n__version__ = '1.0.0'"
        meta_a = {**_BASE_META, "repo_url": "https://github.com/user/repo-a"}
        meta_b = {**_BASE_META, "repo_url": "https://github.com/user/repo-b"}

        chunk_a = _make_chunk(content=shared_content, path="src/main.py")
        chunk_b = _make_chunk(content=shared_content, path="src/main.py")

        client.upsert_chunk(chunk_a, meta_a)
        result = client.upsert_chunk(chunk_b, meta_b)

        assert result.action == "updated"
        assert client._collection.count() == 1
        stored = client._collection.get(ids=[chunk_a.content_hash])
        assert stored["metadatas"][0]["repo_url"] == "https://github.com/user/repo-b"

    def test_content_hash_is_valid_sha256_hex(self) -> None:
        """content_hash must be a 64-character lowercase hex string (SHA-256)."""
        client = _make_client()
        chunk = _make_chunk()
        result = client.upsert_chunk(chunk, _BASE_META)

        assert len(result.content_hash) == 64
        assert all(c in "0123456789abcdef" for c in result.content_hash)

    def test_upsert_exception_returns_error_result(self) -> None:
        """If ChromaDB raises an exception, UpsertResult(action='error') is returned; no crash."""
        chunk = _make_chunk()

        # Inject a broken client that always raises on get()
        bad_collection = MagicMock()
        bad_collection.get.side_effect = RuntimeError("connection refused")
        client = ChromaLibrarianClient.__new__(ChromaLibrarianClient)
        client._collection = bad_collection
        client._embed = _fake_embed

        result = client.upsert_chunk(chunk, _BASE_META)

        assert result.action == "error"
        assert result.error is not None
        assert "connection refused" in result.error

    def test_batch_upsert_multiple_chunks(self) -> None:
        """10 distinct chunks → 10 documents in collection."""
        client = _make_client()
        for i in range(10):
            chunk = _make_chunk(content=f"content_{i}", path=f"src/file_{i}.py")
            result = client.upsert_chunk(chunk, _BASE_META)
            assert result.action == "inserted"
        assert client._collection.count() == 10

    def test_tech_stack_serialized_as_json(self) -> None:
        """tech_stack is stored as a JSON string in ChromaDB metadata."""
        client = _make_client()
        chunk = _make_chunk(tech_stack=["FastAPI", "Pydantic"])
        client.upsert_chunk(chunk, _BASE_META)

        stored = client._collection.get(ids=[chunk.content_hash])
        raw_tech_stack = stored["metadatas"][0]["tech_stack"]
        # ChromaDB must store it as a scalar string
        assert isinstance(raw_tech_stack, str)
        assert json.loads(raw_tech_stack) == ["FastAPI", "Pydantic"]


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


class TestQuery:
    def test_empty_collection_returns_empty_list(self) -> None:
        """query() on an empty collection returns [] without raising."""
        client = _make_client()
        results = client.query("find something")
        assert results == []

    def test_query_returns_matching_documents(self) -> None:
        """query() returns results from a non-empty collection."""
        client = _make_client()
        chunk = _make_chunk(content="FastAPI route handler for health check")
        client.upsert_chunk(chunk, _BASE_META)

        results = client.query("FastAPI route handler for health check")
        assert len(results) == 1
        assert results[0]["id"] == chunk.content_hash

    def test_tech_filter_restricts_results(self) -> None:
        """query() with tech_filter=['FastAPI'] returns only docs with FastAPI in tech_stack."""
        client = _make_client()
        fastapi_chunk = _make_chunk(
            content="FastAPI endpoint", path="src/api.py", tech_stack=["FastAPI", "Pydantic"]
        )
        django_chunk = _make_chunk(
            content="Django view class", path="src/views.py", tech_stack=["Django"]
        )
        client.upsert_chunk(fastapi_chunk, _BASE_META)
        client.upsert_chunk(django_chunk, _BASE_META)

        results = client.query("web framework", tech_filter=["FastAPI"])
        result_ids = {r["id"] for r in results}

        assert fastapi_chunk.content_hash in result_ids
        assert django_chunk.content_hash not in result_ids

    def test_tech_filter_and_semantics(self) -> None:
        """tech_filter with two terms requires both to be present (AND semantics)."""
        client = _make_client()
        both = _make_chunk(
            content="uses FastAPI and Pydantic", path="src/a.py",
            tech_stack=["FastAPI", "Pydantic"]
        )
        only_fastapi = _make_chunk(
            content="only FastAPI", path="src/b.py", tech_stack=["FastAPI"]
        )
        client.upsert_chunk(both, _BASE_META)
        client.upsert_chunk(only_fastapi, _BASE_META)

        results = client.query("framework", tech_filter=["FastAPI", "Pydantic"])
        result_ids = {r["id"] for r in results}

        assert both.content_hash in result_ids
        assert only_fastapi.content_hash not in result_ids

    def test_tech_stack_deserialized_in_results(self) -> None:
        """tech_stack in result metadata is a list[str], not a raw JSON string."""
        client = _make_client()
        chunk = _make_chunk(tech_stack=["LangChain", "ChromaDB"])
        client.upsert_chunk(chunk, _BASE_META)

        results = client.query("vector database")
        assert isinstance(results[0]["metadata"]["tech_stack"], list)
        assert "LangChain" in results[0]["metadata"]["tech_stack"]

    def test_where_filter_applied(self) -> None:
        """where= filter restricts results by scalar metadata field."""
        client = _make_client()
        logic_chunk = _make_chunk(content="business logic code", path="src/l.py", semantic_type="Logic")
        config_chunk = _make_chunk(content="configuration settings", path="src/c.py", semantic_type="Config")
        client.upsert_chunk(logic_chunk, _BASE_META)
        client.upsert_chunk(config_chunk, _BASE_META)

        results = client.query("code", where={"semantic_type": {"$eq": "Config"}})
        result_ids = {r["id"] for r in results}

        assert config_chunk.content_hash in result_ids
        assert logic_chunk.content_hash not in result_ids


# ---------------------------------------------------------------------------
# get_by_author / get_quarantined
# ---------------------------------------------------------------------------


class TestGetByAuthor:
    def test_returns_docs_for_matching_author(self) -> None:
        """get_by_author() returns documents with the given author_identity."""
        client = _make_client()
        alice_chunk = _make_chunk(content="alice writes this", path="src/a.py", author_identity="alice")
        bob_chunk = _make_chunk(content="bob writes that", path="src/b.py", author_identity="bob")
        client.upsert_chunk(alice_chunk, _BASE_META)
        client.upsert_chunk(bob_chunk, _BASE_META)

        results = client.get_by_author("alice")
        assert len(results) == 1
        assert results[0]["id"] == alice_chunk.content_hash

    def test_returns_empty_for_unknown_author(self) -> None:
        """get_by_author() returns [] when no documents match author_identity."""
        client = _make_client()
        chunk = _make_chunk(author_identity="alice")
        client.upsert_chunk(chunk, _BASE_META)

        results = client.get_by_author("nobody")
        assert results == []

    def test_empty_collection_returns_empty(self) -> None:
        """get_by_author() returns [] from an empty collection without raising."""
        client = _make_client()
        results = client.get_by_author("alice")
        assert results == []


class TestGetQuarantined:
    def test_returns_quarantined_documents(self) -> None:
        """get_quarantined() returns only docs with semantic_type == 'quarantine'."""
        client = _make_client()
        quarantine_chunk = _make_chunk(
            content="[CONTENT_QUARANTINED — PII review required]",
            path="src/secret.py",
            semantic_type="quarantine",
        )
        normal_chunk = _make_chunk(
            content="normal business logic",
            path="src/logic.py",
            semantic_type="Logic",
        )
        client.upsert_chunk(quarantine_chunk, _BASE_META)
        client.upsert_chunk(normal_chunk, _BASE_META)

        results = client.get_quarantined()
        result_ids = {r["id"] for r in results}

        assert quarantine_chunk.content_hash in result_ids
        assert normal_chunk.content_hash not in result_ids

    def test_returns_empty_when_no_quarantined_docs(self) -> None:
        """get_quarantined() returns [] when no documents have semantic_type == 'quarantine'."""
        client = _make_client()
        client.upsert_chunk(_make_chunk(semantic_type="Logic"), _BASE_META)

        results = client.get_quarantined()
        assert results == []

    def test_empty_collection_returns_empty(self) -> None:
        """get_quarantined() returns [] from an empty collection without raising."""
        client = _make_client()
        results = client.get_quarantined()
        assert results == []


# ---------------------------------------------------------------------------
# CHROMA_HOST URL parsing — default client selection (F6: settings= injection)
# ---------------------------------------------------------------------------


class TestDefaultClientSelection:
    def test_http_client_used_when_chroma_host_set(self) -> None:
        """When chroma_host is set in Settings, HttpClient is used — no env mutation needed.

        The HttpClient constructor is patched so no real network connection is made.
        We verify the correct hostname and port are parsed from the full URL.
        """
        settings = Settings(chroma_host="http://chromadb:8000")
        mock_http_client = MagicMock(spec=chromadb.ClientAPI)
        mock_collection = MagicMock()
        mock_http_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.HttpClient", return_value=mock_http_client) as mock_cls:
            ChromaLibrarianClient(
                embedding_fn=_fake_embed,
                collection_name="test_host_parse",
                settings=settings,
            )
            mock_cls.assert_called_once_with(host="chromadb", port=8000)

    def test_persistent_client_used_when_chroma_host_empty(self) -> None:
        """When chroma_host is empty, PersistentClient is used at chroma_data_path."""
        settings = Settings(chroma_host="", chroma_data_path="chroma_data")
        mock_persistent = MagicMock(spec=chromadb.ClientAPI)
        mock_collection = MagicMock()
        mock_persistent.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_persistent) as mock_cls:
            ChromaLibrarianClient(
                embedding_fn=_fake_embed,
                collection_name="test_persistent",
                settings=settings,
            )
            mock_cls.assert_called_once_with(path="chroma_data")

    def test_persistent_client_uses_custom_data_path(self) -> None:
        """chroma_data_path from Settings is passed to PersistentClient."""
        settings = Settings(chroma_host="", chroma_data_path="/custom/chroma")
        mock_persistent = MagicMock(spec=chromadb.ClientAPI)
        mock_collection = MagicMock()
        mock_persistent.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.PersistentClient", return_value=mock_persistent) as mock_cls:
            ChromaLibrarianClient(
                embedding_fn=_fake_embed,
                collection_name="test_custom_path",
                settings=settings,
            )
            mock_cls.assert_called_once_with(path="/custom/chroma")

    def test_http_client_port_defaults_to_8000_when_absent(self) -> None:
        """CHROMA_HOST without explicit port uses 8000 as the default."""
        settings = Settings(chroma_host="http://chromadb")
        mock_http_client = MagicMock(spec=chromadb.ClientAPI)
        mock_collection = MagicMock()
        mock_http_client.get_or_create_collection.return_value = mock_collection

        with patch("chromadb.HttpClient", return_value=mock_http_client) as mock_cls:
            ChromaLibrarianClient(
                embedding_fn=_fake_embed,
                collection_name="test_default_port",
                settings=settings,
            )
            _, kwargs = mock_cls.call_args
            assert kwargs["port"] == 8000

    def test_injected_client_bypasses_settings(self) -> None:
        """When client= is injected directly, Settings chroma_host is not used."""
        settings = Settings(chroma_host="http://should-not-be-used:9999")
        ephemeral = chromadb.EphemeralClient()

        # No patch needed — injected client is used as-is
        client = ChromaLibrarianClient(
            client=ephemeral,
            embedding_fn=_fake_embed,
            collection_name=f"test_bypass_{uuid.uuid4().hex[:8]}",
            settings=settings,
        )
        # Verify we got a working client (not connected to the Settings URL)
        assert client._collection.count() == 0


# ---------------------------------------------------------------------------
# release_quarantine_item (F2)
# ---------------------------------------------------------------------------


class TestReleaseQuarantineItem:
    def test_release_updates_semantic_type(self) -> None:
        """release_quarantine_item sets semantic_type to 'reviewed' and returns True."""
        client = _make_client()
        chunk = _make_chunk(
            content="[CONTENT_QUARANTINED — PII review required]\npath:src/secret.py",
            path="src/secret.py",
            semantic_type="quarantine",
        )
        meta = {**_BASE_META, "quarantine_reason": "email detected"}
        client.upsert_chunk(chunk, meta)
        doc_id = chunk.content_hash

        result = client.release_quarantine_item(doc_id)

        assert result is True
        stored = client._collection.get(ids=[doc_id], include=["metadatas"])
        assert stored["metadatas"][0]["semantic_type"] == "reviewed"

    def test_release_preserves_existing_metadata(self) -> None:
        """release_quarantine_item preserves all metadata fields — get-merge-update contract.

        Specifically verifies file_path is intact after release, proving the
        implementation used get-merge-update and not a bare update() call.
        """
        client = _make_client()
        original_path = "src/sensitive.py"
        chunk = _make_chunk(
            content="[CONTENT_QUARANTINED — PII review required]\npath:src/sensitive.py",
            path=original_path,
            semantic_type="quarantine",
        )
        meta = {**_BASE_META, "quarantine_reason": "phone number detected", "indexed_at": "2026-05-01T00:00:00+00:00"}
        client.upsert_chunk(chunk, meta)
        doc_id = chunk.content_hash

        client.release_quarantine_item(doc_id)

        stored = client._collection.get(ids=[doc_id], include=["metadatas"])
        stored_meta = stored["metadatas"][0]
        # Both assertions required — semantic_type changed AND file_path preserved
        assert stored_meta["semantic_type"] == "reviewed"
        assert stored_meta["file_path"] == original_path

    def test_release_unknown_doc_id_returns_false(self) -> None:
        """release_quarantine_item returns False (not raises) for an unknown doc_id."""
        client = _make_client()
        result = client.release_quarantine_item("nonexistent_id_" * 3)
        assert result is False

    def test_release_removes_from_quarantine_queue(self) -> None:
        """After release, get_quarantined() no longer returns the document."""
        client = _make_client()
        chunk = _make_chunk(
            content="[CONTENT_QUARANTINED — PII review required]\npath:src/pii.py",
            path="src/pii.py",
            semantic_type="quarantine",
        )
        client.upsert_chunk(chunk, _BASE_META)

        assert len(client.get_quarantined()) == 1
        client.release_quarantine_item(chunk.content_hash)
        # Released doc has semantic_type="reviewed" — no longer returned by get_quarantined()
        assert client.get_quarantined() == []


class TestGetIndexedRepoMetadata:
    """Tests for ChromaLibrarianClient.get_indexed_repo_metadata() — spec §2d."""

    def test_empty_collection_returns_empty_dict(self) -> None:
        """Returns {} when collection has no documents."""
        client = _make_client()
        assert client.get_indexed_repo_metadata() == {}

    def test_returns_repo_url_branch_commit_sha(self) -> None:
        """Returns repo_url, branch, commit_sha from an indexed document."""
        client = _make_client()
        meta = {
            "repo_url": "https://github.com/u/r",
            "branch": "main",
            "commit_sha": "abc123def456",
            "indexed_at": "2026-05-13",
        }
        chunk = _make_chunk(content="class Foo: pass", path="src/foo.py")
        client.upsert_chunk(chunk, meta)

        result = client.get_indexed_repo_metadata()
        assert result["repo_url"] == "https://github.com/u/r"
        assert result["branch"] == "main"
        assert result["commit_sha"] == "abc123def456"

    def test_returns_only_three_keys(self) -> None:
        """Result dict has exactly the three documented keys."""
        client = _make_client()
        chunk = _make_chunk(content="class Bar: pass", path="src/bar.py")
        client.upsert_chunk(chunk, _BASE_META)
        result = client.get_indexed_repo_metadata()
        assert set(result.keys()) == {"repo_url", "branch", "commit_sha"}

    def test_chroma_error_returns_empty_dict(self) -> None:
        """Returns {} on ChromaDB exception — no raise."""
        client = _make_client()
        # Seed one document so count() > 0, then patch get() to raise
        chunk = _make_chunk(content="class Baz: pass", path="src/baz.py")
        client.upsert_chunk(chunk, _BASE_META)

        with patch.object(client._collection, "get", side_effect=RuntimeError("chroma down")):
            result = client.get_indexed_repo_metadata()

        assert result == {}
