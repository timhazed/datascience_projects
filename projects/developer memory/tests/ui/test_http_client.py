"""Tests for ui.http_client — HTTP transport layer for MCP server calls.

All network I/O is mocked: httpx calls via patch("httpx.post") / patch("httpx.get"),
fastmcp.Client via AsyncMock. No live network calls are made.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import streamlit as st

from ui.http_client import (
    _call_tool,
    _get_quarantine,
    _mcp_error,
    _poll_sync_status,
    _post_sync,
    _release_quarantine,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _mock_response(status_code: int, body: dict) -> MagicMock:
    """Build a minimal httpx.Response mock with .status_code, .json(), and .raise_for_status()."""
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json.return_value = body
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=MagicMock(),
            response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ── _post_sync ────────────────────────────────────────────────────────────────


class TestPostSync:
    """Verify _post_sync parses the job_id and surfaces errors correctly."""

    def test_returns_job_id_on_success(self) -> None:
        """202 response → job_id string returned."""
        resp = _mock_response(202, {"job_id": "abc-123"})
        with patch("httpx.post", return_value=resp):
            result = _post_sync("https://github.com/owner/repo", "main")
        assert result == "abc-123"

    def test_raises_value_error_on_422(self) -> None:
        """422 response → ValueError with server-provided message."""
        resp = _mock_response(422, {"error": "Invalid repo URL"})
        resp.raise_for_status.side_effect = None  # 422 handled before raise_for_status
        resp.status_code = 422
        with patch("httpx.post", return_value=resp), pytest.raises(ValueError, match="Invalid repo URL"):
            _post_sync("bad-url", "main")

    def test_raises_value_error_fallback_on_422_no_error_key(self) -> None:
        """422 with no 'error' key → ValueError with fallback message."""
        resp = _mock_response(422, {})
        resp.raise_for_status.side_effect = None
        resp.status_code = 422
        with patch("httpx.post", return_value=resp), pytest.raises(ValueError, match="Validation failed"):
            _post_sync("bad-url", "main")

    def test_raises_http_error_on_500(self) -> None:
        """500 response → HTTPStatusError propagated."""
        resp = _mock_response(500, {"detail": "Internal server error"})
        with patch("httpx.post", return_value=resp), pytest.raises(httpx.HTTPStatusError):
            _post_sync("https://github.com/owner/repo", "main")


# ── _poll_sync_status ─────────────────────────────────────────────────────────


class TestPollSyncStatus:
    """Verify _poll_sync_status returns the parsed payload and handles errors."""

    def test_returns_parsed_dict_on_200(self) -> None:
        """200 response → full status dict returned."""
        payload = {"status": "running", "stage": "summarizing", "chunks_done": 5}
        resp = _mock_response(200, payload)
        with patch("httpx.get", return_value=resp):
            result = _poll_sync_status("job-uuid")
        assert result == payload

    def test_raises_on_404(self) -> None:
        """404 response → HTTPStatusError propagated."""
        resp = _mock_response(404, {"detail": "Job not found"})
        with patch("httpx.get", return_value=resp), pytest.raises(httpx.HTTPStatusError):
            _poll_sync_status("unknown-job")


# ── _call_tool ────────────────────────────────────────────────────────────────


class TestCallTool:
    """Verify _call_tool handles both fastmcp response formats."""

    def test_returns_result_data_attribute(self) -> None:
        """FastMCP 3.x path: CallToolResult.data returned directly."""
        expected = {"answer": "42"}
        mock_result = SimpleNamespace(data=expected)

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("ui.http_client.Client", return_value=mock_client):
            result = _call_tool("query_memory", query="test")

        assert result == expected

    def test_falls_back_to_text_content_json(self) -> None:
        """Older fastmcp path: list of TextContent blocks with JSON text."""
        expected = {"result": "some text"}
        text_block = SimpleNamespace(text=json.dumps(expected))
        mock_result = [text_block]
        # No .data attribute on list

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("ui.http_client.Client", return_value=mock_client):
            result = _call_tool("query_memory", query="test")

        assert result == expected

    def test_falls_back_to_raw_text_on_invalid_json(self) -> None:
        """Older fastmcp path: non-JSON text content returned as-is."""
        raw_text = "plain answer string"
        text_block = SimpleNamespace(text=raw_text)
        mock_result = [text_block]

        mock_client = AsyncMock()
        mock_client.call_tool = AsyncMock(return_value=mock_result)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        with patch("ui.http_client.Client", return_value=mock_client):
            result = _call_tool("query_memory", query="test")

        assert result == raw_text


# ── _mcp_error ────────────────────────────────────────────────────────────────


class TestMcpError:
    """Verify _mcp_error renders safe UI and logs at ERROR level."""

    def test_calls_st_error(self) -> None:
        """_mcp_error calls st.error with the safe message."""
        exc = ConnectionError("refused")
        with patch.object(st, "error") as mock_error:
            _mcp_error(exc)
        mock_error.assert_called_once()
        call_arg = mock_error.call_args[0][0]
        assert "make up" in call_arg  # safe user-facing guidance present

    def test_does_not_expose_exception_message_to_user(self) -> None:
        """The raw exception message is NOT surfaced in the st.error call."""
        exc = RuntimeError("super secret internal state")
        with patch.object(st, "error") as mock_error:
            _mcp_error(exc)
        call_arg = mock_error.call_args[0][0]
        assert "super secret internal state" not in call_arg

    def test_logs_at_error_level(self, caplog: pytest.LogCaptureFixture) -> None:
        """_mcp_error logs the exception type and truncated message at ERROR."""
        exc = TimeoutError("connection timed out")
        with patch.object(st, "error"), caplog.at_level("ERROR", logger="ui.http_client"):
            _mcp_error(exc)
        assert any("TimeoutError" in r.message for r in caplog.records)


# ── _get_quarantine ───────────────────────────────────────────────────────────


class TestGetQuarantine:
    """Verify _get_quarantine returns item list and handles errors."""

    def test_returns_items_list_on_200(self) -> None:
        """200 response with items → list of item dicts returned."""
        items = [{"doc_id": "d1", "file_path": "src/main.py"}]
        resp = _mock_response(200, {"items": items})
        with patch("httpx.get", return_value=resp):
            result = _get_quarantine()
        assert result == items

    def test_returns_empty_list_when_items_absent(self) -> None:
        """200 response with no 'items' key → empty list."""
        resp = _mock_response(200, {})
        with patch("httpx.get", return_value=resp):
            result = _get_quarantine()
        assert result == []

    def test_raises_on_500(self) -> None:
        """500 response → HTTPStatusError propagated."""
        resp = _mock_response(500, {})
        with patch("httpx.get", return_value=resp), pytest.raises(httpx.HTTPStatusError):
            _get_quarantine()


# ── _release_quarantine ───────────────────────────────────────────────────────


class TestReleaseQuarantine:
    """Verify _release_quarantine return value and error handling."""

    def test_returns_true_on_200(self) -> None:
        """200 response → True returned."""
        resp = _mock_response(200, {"released": True})
        with patch("httpx.post", return_value=resp):
            assert _release_quarantine("doc-abc") is True

    def test_returns_false_on_404(self) -> None:
        """404 response → False returned (doc already gone or never existed)."""
        resp = _mock_response(404, {"detail": "Not found"})
        # 404 must NOT trigger raise_for_status — our function handles it explicitly
        resp.raise_for_status.side_effect = None
        resp.status_code = 404
        with patch("httpx.post", return_value=resp):
            assert _release_quarantine("missing-doc") is False

    def test_raises_on_500(self) -> None:
        """500 response → HTTPStatusError propagated."""
        resp = _mock_response(500, {})
        with patch("httpx.post", return_value=resp), pytest.raises(httpx.HTTPStatusError):
            _release_quarantine("doc-abc")
