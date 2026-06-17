"""Developer Memory UI — HTTP transport layer for all MCP server calls.

Owns all HTTP communication with the MCP server: sync job submission and polling
(plain httpx), MCP tool invocation (fastmcp.Client), and quarantine queue management.
No Streamlit page rendering — only network I/O and result parsing.
"""

import asyncio
import json
import logging

import httpx
import streamlit as st
from fastmcp import Client

from ui.config import _MCP_URL, _SERVER_BASE

logger = logging.getLogger(__name__)


# ── Sync job HTTP helpers ─────────────────────────────────────────────────────


def _post_sync(repo_url: str, branch: str) -> str:
    """POST /sync and return the job_id string.

    Args:
        repo_url: Full HTTPS URL of the git repository to sync.
        branch: Branch name to check out (e.g. "main").

    Returns:
        UUID4 job_id string returned by the server.

    Raises:
        ValueError: When the server returns 422 (validation error).
        httpx.HTTPStatusError: On any other non-2xx response.
    """
    resp = httpx.post(
        f"{_SERVER_BASE}/sync",
        json={"repo_url": repo_url, "branch": branch},
        timeout=10.0,
    )
    if resp.status_code == 422:
        raise ValueError(resp.json().get("error", "Validation failed"))
    resp.raise_for_status()
    return resp.json()["job_id"]


def _poll_sync_status(job_id: str) -> dict:
    """GET /sync/status/{job_id} and return the full status payload dict.

    Args:
        job_id: UUID4 returned by _post_sync.

    Returns:
        Dict with keys: status, stage, chunks_done, chunks_total, pii_files_done,
        pii_files_total, started_at, result (on done), error (on error).

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
    """
    resp = httpx.get(f"{_SERVER_BASE}/sync/status/{job_id}", timeout=10.0)
    resp.raise_for_status()
    return resp.json()


# ── MCP tool call helper ──────────────────────────────────────────────────────


def _call_tool(tool_name: str, **kwargs: object) -> object:
    """Call an MCP tool on the running MCP server and return the result content.

    Uses fastmcp.Client over HTTP — no in-process LLM or graph startup.
    All tool calls are synchronous from Streamlit's perspective (asyncio.run wraps
    the async Client API).

    Args:
        tool_name: Name of the registered @mcp.tool() to invoke.
        **kwargs: Tool arguments forwarded as the call payload.

    Returns:
        Parsed tool result (dict or str). FastMCP 3.x exposes this via
        CallToolResult.data which is already JSON-decoded by the client.

    Raises:
        Exception: On network failure, MCP protocol error, or tool-level error.
    """

    async def _invoke() -> object:
        async with Client(_MCP_URL) as client:
            result = await client.call_tool(tool_name, kwargs)
            # FastMCP 3.x returns a CallToolResult with .data (already parsed) and
            # .content (list of TextContent blocks).  Prefer .data — it is the
            # structured payload the MCP tool returned, already JSON-decoded.
            if hasattr(result, "data"):
                return result.data
            # Fallback: older fastmcp versions returned a list of content blocks.
            if result and hasattr(result[0], "text"):
                try:
                    return json.loads(result[0].text)
                except (ValueError, TypeError):
                    return result[0].text
            return result

    return asyncio.run(_invoke())


def _mcp_error(exc: Exception) -> None:
    """Render a safe error banner when the MCP server is unreachable or errors.

    Logs the exception type and a truncated message for debugging; shows a
    sanitized, user-safe message via st.error (no stack traces, no internals).

    Args:
        exc: The exception raised during the MCP tool call.
    """
    st.error(
        "MCP server call failed. Ensure the stack is running (`make up`) "
        "and all services are healthy (`make health`)."
    )
    logger.error("MCP call failed [%s]: %s", type(exc).__name__, str(exc)[:200])


# ── Quarantine queue HTTP helpers ─────────────────────────────────────────────


def _get_quarantine() -> list[dict]:
    """GET /quarantine and return the list of quarantined item dicts.

    Returns:
        List of item dicts, each containing doc_id, file_path, quarantine_reason,
        repo_url, and indexed_at. Empty list when the queue is empty.

    Raises:
        httpx.HTTPStatusError: On non-2xx response.
    """
    resp = httpx.get(f"{_SERVER_BASE}/quarantine", timeout=10.0)
    resp.raise_for_status()
    return resp.json().get("items", [])


def _release_quarantine(doc_id: str) -> bool:
    """POST /quarantine/{doc_id}/release and return True on success.

    Args:
        doc_id: The document ID of the quarantined file to release.

    Returns:
        True when the server returns 200.
        False when the server returns 404 (doc_id not found — already released
        or never existed).

    Raises:
        httpx.HTTPStatusError: On 5xx or other unexpected non-2xx responses.
    """
    resp = httpx.post(f"{_SERVER_BASE}/quarantine/{doc_id}/release", timeout=10.0)
    if resp.status_code == 404:
        return False
    resp.raise_for_status()
    return True
