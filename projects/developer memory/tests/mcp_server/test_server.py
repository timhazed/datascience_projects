"""Tests for src/mcp_server/server.py — MCP tools + HTTP routes.

Migrated from tests/test_server.py per Phase 6 of ServerRefactor.md.
All import paths and patch targets updated per the complete mapping table
in docs/ServerRefactor.md §Phase 6.

Strategy:
  - All five MCP tools tested with mocked graphs — no live Ollama, no live ChromaDB.
  - Graph singletons patched at container_module level (where they actually live after
    Phase 5). _query_graph, _persona_graph, _diff_graph, _skills_graph are also bound
    as module-level names in src.mcp_server.server — patch.object on server_module works
    for those (they are looked up from the server module's __dict__ at call time).
  - _chroma patched at container_module level (consumed directly from container import
    inside each handler via the module-level binding).
  - _run_sync_job patched at src.mcp_server.sync_jobs.run_sync_job — the canonical
    location — so the patch survives future module moves.
  - HTTP routes tested via Starlette TestClient with mcp.http_app().

Cases: see class docstrings.
"""

from unittest.mock import MagicMock, patch

import src.mcp_server.container as container_module
import src.mcp_server.server as server_module
from src.mcp_server.server import (
    analyze_diff,
    generate_skills_pkg,
    get_dev_persona,
    query_memory,
    sync_repository,
)
from src.mcp_server.startup import _log_memory  # noqa: F401 — verifies importable
from src.mcp_server.sync_jobs import (  # noqa: F401
    _JOBS_LOCK,
    _pii_progress_callback,
    _sync_progress_callback,
)
from src.mcp_server.sync_jobs import (
    register_job as _register_job,
)

# ── Helper ────────────────────────────────────────────────────────────────────


def _mock_graph(return_value: dict) -> MagicMock:
    """Return a MagicMock graph whose .invoke() returns return_value."""
    g = MagicMock()
    g.invoke.return_value = return_value
    return g


# ── sync_repository() ─────────────────────────────────────────────────────────


class TestSyncRepository:
    """sync_repository() validates URLs, blocks on a daemon thread, and returns SyncState."""

    def test_valid_url_returns_dict_with_upsert_results(self) -> None:
        """Mocked sync graph returns upsert_results → tool returns that dict."""
        expected = {"upsert_results": [{"id": "abc"}], "trace": ["delta_extractor: 1 file"]}
        with patch.object(container_module, "_sync_graph", _mock_graph(expected)):
            result = sync_repository("https://github.com/user/repo")

        assert isinstance(result, dict)
        assert "upsert_results" in result

    def test_invalid_host_returns_error_dict(self) -> None:
        """repo_url with disallowed host → error dict returned, no unhandled exception."""
        result = sync_repository("https://evil.com/user/repo")

        assert isinstance(result, dict)
        assert "error" in result
        assert result["upsert_results"] == []

    def test_non_default_branch_forwarded_to_graph(self) -> None:
        """branch kwarg is forwarded to graph.invoke()."""
        captured: list[dict] = []

        def capturing_invoke(state: dict, **kwargs) -> dict:
            captured.append(state)
            return {"upsert_results": [], "trace": []}

        graph = MagicMock()
        graph.invoke.side_effect = capturing_invoke

        with patch.object(container_module, "_sync_graph", graph):
            sync_repository("https://github.com/user/repo", branch="develop")

        assert captured[0]["branch"] == "develop"


# ── query_memory() ────────────────────────────────────────────────────────────


class TestQueryMemory:
    """query_memory() invokes _query_graph and returns a string answer."""

    def test_final_answer_returned_as_string(self) -> None:
        """Mocked graph returns final_answer → tool returns that string."""
        state = {"final_answer": "Pydantic is used in models/.", "error": None, "trace": []}
        with patch.object(server_module, "_query_graph", _mock_graph(state)):
            result = query_memory("Where have I used Pydantic?")

        assert result == "Pydantic is used in models/."

    def test_none_final_answer_falls_back_to_error(self) -> None:
        """final_answer=None → tool returns the error string."""
        state = {"final_answer": None, "error": "Query blocked by safety guard.", "trace": []}
        with patch.object(server_module, "_query_graph", _mock_graph(state)):
            result = query_memory("DROP TABLE users")

        assert result == "Query blocked by safety guard."

    def test_none_final_answer_and_none_error_returns_no_results(self) -> None:
        """final_answer=None + error=None → fallback 'No results found.' string."""
        state = {"final_answer": None, "error": None, "trace": []}
        with patch.object(server_module, "_query_graph", _mock_graph(state)):
            result = query_memory("empty query")

        assert result == "No results found."

    def test_non_empty_result_is_string(self) -> None:
        """Return type is always str — not a dict."""
        state = {"final_answer": "Answer.", "error": None, "trace": []}
        with patch.object(server_module, "_query_graph", _mock_graph(state)):
            result = query_memory("test query")

        assert isinstance(result, str)


# ── get_dev_persona() ─────────────────────────────────────────────────────────


class TestGetDevPersona:
    """get_dev_persona() invokes _persona_graph with scope and recency_months."""

    def test_returns_dict(self) -> None:
        """Mocked persona graph → tool returns dict."""
        state = {"persona_profile": {"dominant_language": "Python"}, "trace": []}
        with patch.object(server_module, "_persona_graph", _mock_graph(state)):
            result = get_dev_persona("project")

        assert isinstance(result, dict)

    def test_scope_and_recency_months_forwarded(self) -> None:
        """scope and recency_months are forwarded to graph.invoke()."""
        captured: list[dict] = []

        def capturing_invoke(state: dict, **kwargs) -> dict:
            captured.append(state)
            return {"trace": []}

        graph = MagicMock()
        graph.invoke.side_effect = capturing_invoke

        with patch.object(server_module, "_persona_graph", graph):
            get_dev_persona(scope="file", recency_months=3)

        assert captured[0]["scope"] == "file"
        assert captured[0]["recency_months"] == 3


# ── analyze_diff() ────────────────────────────────────────────────────────────


class TestAnalyzeDiff:
    """analyze_diff() invokes _diff_graph with the unified diff text."""

    def test_returns_dict(self) -> None:
        """Mocked diff graph → tool returns dict."""
        state = {"analysis_result": {"deviation_score": 0.3}, "coaching_alert": None, "trace": []}
        with patch.object(server_module, "_diff_graph", _mock_graph(state)):
            result = analyze_diff("--- a\n+++ b\n@@ -1,1 +1,1 @@\n-old\n+new\n")

        assert isinstance(result, dict)

    def test_diff_text_forwarded_to_graph(self) -> None:
        """diff_text is forwarded to graph.invoke()."""
        captured: list[dict] = []

        def capturing_invoke(state: dict, **kwargs) -> dict:
            captured.append(state)
            return {"trace": []}

        graph = MagicMock()
        graph.invoke.side_effect = capturing_invoke

        diff = "--- a\n+++ b\n"
        with patch.object(server_module, "_diff_graph", graph):
            analyze_diff(diff)

        assert captured[0]["diff_text"] == diff


# ── generate_skills_pkg() ─────────────────────────────────────────────────────


class TestGenerateSkillsPkg:
    """generate_skills_pkg() validates path then invokes _skills_graph."""

    def test_valid_path_returns_dict_with_export_result(self, tmp_path) -> None:
        """Mocked skills graph → tool returns dict with export_result."""
        target = str(tmp_path / "skills.md")
        state = {"export_result": {"path": target, "bytes_written": 128}, "trace": []}

        with (
            patch.object(server_module, "_skills_graph", _mock_graph(state)),
            patch("src.mcp_server.server.validate_target_path", return_value=target),
        ):
            result = generate_skills_pkg(target)

        assert isinstance(result, dict)
        assert "export_result" in result

    def test_invalid_path_returns_error_dict(self) -> None:
        """Path traversal attempt → error dict returned, no unhandled exception."""
        result = generate_skills_pkg("../../../etc/passwd")

        assert isinstance(result, dict)
        assert "error" in result
        assert result["export_result"] is None


# ── _sanitize_result() ────────────────────────────────────────────────────────


class TestSanitizeResult:
    """_sanitize_result() recursively converts Pydantic models for JSON serialization."""

    def test_primitive_values_returned_unchanged(self) -> None:
        """Strings, ints, None pass through without modification."""
        from src.mcp_server.server import _sanitize_result

        assert _sanitize_result("hello") == "hello"
        assert _sanitize_result(42) == 42
        assert _sanitize_result(None) is None

    def test_dict_values_recursed(self) -> None:
        """Nested dict with string values passes through."""
        from src.mcp_server.server import _sanitize_result

        data = {"key": "value", "nested": {"inner": 1}}
        assert _sanitize_result(data) == {"key": "value", "nested": {"inner": 1}}

    def test_list_items_recursed(self) -> None:
        """List items are recursively sanitized."""
        from src.mcp_server.server import _sanitize_result

        data = ["a", 2, {"k": "v"}]
        result = _sanitize_result(data)
        assert result == ["a", 2, {"k": "v"}]

    def test_pydantic_model_converted_to_dict(self) -> None:
        """Pydantic BaseModel instances are converted via .model_dump()."""
        from pydantic import BaseModel  # noqa: PLC0415

        from src.mcp_server.server import _sanitize_result  # noqa: PLC0415

        class Sample(BaseModel):
            name: str
            score: float

        result = _sanitize_result(Sample(name="test", score=1.5))
        assert isinstance(result, dict)
        assert result == {"name": "test", "score": 1.5}


# ── Progress callbacks ────────────────────────────────────────────────────────


class TestProgressCallbacks:
    """Progress callbacks update live SyncJob fields from sync_jobs module."""

    def test_sync_progress_callback_updates_job(self) -> None:
        """_sync_progress_callback updates chunks_done, chunks_total, stage."""
        job = _register_job("https://github.com/u/r", "main")
        _sync_progress_callback(job.job_id, 5, 20)

        assert job.chunks_done == 5
        assert job.chunks_total == 20
        assert job.stage == "summarizing"

    def test_pii_progress_callback_updates_job(self) -> None:
        """_pii_progress_callback updates pii_files_done, pii_files_total, stage."""
        job = _register_job("https://github.com/u/r", "main")
        _pii_progress_callback(job.job_id, 3, 10)

        assert job.pii_files_done == 3
        assert job.pii_files_total == 10
        assert job.stage == "pii_scanning"

    def test_callbacks_with_unknown_job_id_do_not_crash(self) -> None:
        """Callbacks with unknown job_id should silently no-op."""
        unknown = "00000000-0000-0000-0000-000000000001"
        _sync_progress_callback(unknown, 1, 5)
        _pii_progress_callback(unknown, 1, 5)


# ── HTTP routes (POST /sync, GET /sync/status/{job_id}) ───────────────────────


class TestHTTPRoutes:
    """Tests for POST /sync and GET /sync/status using Starlette TestClient.

    run_sync_job is patched at src.mcp_server.sync_jobs.run_sync_job — the canonical
    source location — so threads never actually run the graph.
    """

    def _get_client(self):
        from starlette.testclient import TestClient

        app = container_module.mcp.http_app()
        return TestClient(app, raise_server_exceptions=False)

    def test_post_sync_valid_url_returns_202_with_job_id(self) -> None:
        """POST /sync with valid URL → 202, job_id and status in response."""
        client = self._get_client()

        # Patch the name bound in server.py's namespace — patching sync_jobs.run_sync_job
        # has no effect because server.py imports it as _run_sync_job at module load time.
        with patch("src.mcp_server.server._run_sync_job"):
            resp = client.post(
                "/sync",
                json={"repo_url": "https://github.com/user/repo", "branch": "main"},
            )

        assert resp.status_code == 202
        body = resp.json()
        assert "job_id" in body
        assert body["status"] == "queued"

    def test_post_sync_invalid_url_returns_422(self) -> None:
        """POST /sync with disallowed host → 422 Unprocessable Entity."""
        client = self._get_client()
        resp = client.post("/sync", json={"repo_url": "https://evil.com/repo"})

        assert resp.status_code == 422
        assert "error" in resp.json()

    def test_post_sync_bad_json_returns_400(self) -> None:
        """POST /sync with non-JSON body → 400 Bad Request."""
        client = self._get_client()
        resp = client.post("/sync", content=b"not json", headers={"Content-Type": "text/plain"})

        assert resp.status_code == 400
        assert "error" in resp.json()

    def test_get_sync_status_unknown_job_returns_404(self) -> None:
        """GET /sync/status/{unknown_id} → 404."""
        client = self._get_client()
        resp = client.get("/sync/status/00000000-0000-0000-0000-000000000000")

        assert resp.status_code == 404
        assert "error" in resp.json()

    def test_get_sync_status_known_job_returns_200(self) -> None:
        """GET /sync/status/{known_id} → 200 with job fields."""
        job = _register_job("https://github.com/u/r", "main")
        client = self._get_client()
        resp = client.get(f"/sync/status/{job.job_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["job_id"] == job.job_id
        assert "status" in body
        assert "stage" in body
        assert "chunks_total" in body
        assert "pii_files_done" in body

    def test_get_sync_status_done_includes_result(self) -> None:
        """GET /sync/status for a completed job → result field populated."""
        job = _register_job("https://github.com/u/r", "main")
        with _JOBS_LOCK:
            job.status = "done"
            job.stage = "done"
            job.result = {"upsert_results": [], "trace": ["done"]}

        client = self._get_client()
        resp = client.get(f"/sync/status/{job.job_id}")

        assert resp.status_code == 200
        body = resp.json()
        assert body["result"] is not None

    def test_get_health_returns_ok(self) -> None:
        """GET /health → 200 {'status': 'ok'}."""
        client = self._get_client()
        resp = client.get("/health")

        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ── GET /quarantine / POST /quarantine/{doc_id}/release ───────────────────────


class TestQuarantineRoutes:
    """Tests for the quarantine REST endpoints via Starlette TestClient.

    _chroma is patched at server_module level — the route handlers access _chroma via
    the name bound in mcp_server/server.py's namespace at import time. Patching
    container_module._chroma has no effect on server_module._chroma after import.
    """

    def _get_client(self):
        from starlette.testclient import TestClient

        app = container_module.mcp.http_app()
        return TestClient(app, raise_server_exceptions=False)

    def _make_quarantine_item(self) -> dict:
        """Return a realistic quarantined doc metadata dict."""
        return {
            "id": "abc123",
            "document": "email@example.com",
            "metadata": {
                "file_path": "projects/foo/README.md",
                "quarantine_reason": "email detected",
                "repo_url": "https://github.com/u/r",
                "indexed_at": "2026-05-06T10:00:00Z",
                "semantic_type": "quarantine",
                "tech_stack": ["Python"],
                "intent_summary": "",
                "key_identifiers": "[]",
                "author_identity": "",
                "branch": "main",
                "commit_sha": "deadbeef",
            },
        }

    def test_get_quarantine_empty_queue_returns_empty_list(self) -> None:
        """GET /quarantine with no quarantined docs → 200 {'items': [], 'total': 0}."""
        mock_chroma = MagicMock()
        mock_chroma.get_quarantined.return_value = []
        client = self._get_client()

        with patch.object(server_module, "_chroma", mock_chroma):
            resp = client.get("/quarantine")

        assert resp.status_code == 200
        body = resp.json()
        assert body["items"] == []
        assert body["total"] == 0

    def test_get_quarantine_returns_items_and_total(self) -> None:
        """GET /quarantine with one quarantined doc → 200 with item and total == 1."""
        item = self._make_quarantine_item()
        mock_chroma = MagicMock()
        mock_chroma.get_quarantined.return_value = [item]
        client = self._get_client()

        with patch.object(server_module, "_chroma", mock_chroma):
            resp = client.get("/quarantine")

        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] == 1
        assert len(body["items"]) == 1
        result_item = body["items"][0]
        assert result_item["doc_id"] == "abc123"
        assert result_item["file_path"] == "projects/foo/README.md"
        assert result_item["quarantine_reason"] == "email detected"
        assert result_item["repo_url"] == "https://github.com/u/r"
        assert result_item["indexed_at"] == "2026-05-06T10:00:00Z"

    def test_get_quarantine_chroma_error_returns_500(self) -> None:
        """GET /quarantine when ChromaDB raises → 500 with error key."""
        mock_chroma = MagicMock()
        mock_chroma.get_quarantined.side_effect = RuntimeError("connection refused")
        client = self._get_client()

        with patch.object(server_module, "_chroma", mock_chroma):
            resp = client.get("/quarantine")

        assert resp.status_code == 500
        assert "error" in resp.json()

    def test_post_release_known_doc_returns_released(self) -> None:
        """POST /quarantine/{doc_id}/release with known doc → 200 {'status': 'released'}."""
        mock_chroma = MagicMock()
        mock_chroma.release_quarantine_item.return_value = True
        client = self._get_client()

        with patch.object(server_module, "_chroma", mock_chroma):
            resp = client.post("/quarantine/abc123/release")

        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_id"] == "abc123"
        assert body["status"] == "released"

    def test_post_release_unknown_doc_returns_404(self) -> None:
        """POST /quarantine/{doc_id}/release when doc not found → 404."""
        mock_chroma = MagicMock()
        mock_chroma.release_quarantine_item.return_value = False
        client = self._get_client()

        with patch.object(server_module, "_chroma", mock_chroma):
            resp = client.post("/quarantine/does-not-exist/release")

        assert resp.status_code == 404
        body = resp.json()
        assert "error" in body

    def test_post_release_chroma_error_returns_500(self) -> None:
        """POST /quarantine/{doc_id}/release when ChromaDB raises → 500 with error key."""
        mock_chroma = MagicMock()
        mock_chroma.release_quarantine_item.side_effect = RuntimeError("db error")
        client = self._get_client()

        with patch.object(server_module, "_chroma", mock_chroma):
            resp = client.post("/quarantine/abc123/release")

        assert resp.status_code == 500
        assert "error" in resp.json()
