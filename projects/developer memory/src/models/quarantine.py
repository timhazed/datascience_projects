"""Pydantic models for the quarantine REST API (F2 rework).

GET  /quarantine          → QuarantineListResponse
POST /quarantine/{id}/release → {doc_id, status: "released"} or 404

These models are the wire format between the MCP server and Streamlit UI.
No raw ChromaDB dicts leak beyond the server route handler — the handler
maps _unpack_get_results() output into QuarantineItem before serialising.
"""

from pydantic import BaseModel


class QuarantineItem(BaseModel):
    """A single document held in the PII quarantine queue.

    Fields map directly to the metadata keys stored by chroma_upsert_quarantined
    via ChromaLibrarianClient.upsert_chunk(). The server route handler constructs
    these from the raw dicts returned by chroma.get_quarantined().

    Args:
        doc_id: ChromaDB document ID (SHA-256 content_hash of the quarantined content).
        file_path: Relative file path within the repository.
        quarantine_reason: Human-readable reason the file was quarantined
            (e.g. "email detected", "File read error [PermissionError]").
        repo_url: Git repository URL the file was cloned from.
        indexed_at: ISO-8601 timestamp when the document was indexed.
    """

    doc_id: str
    file_path: str
    quarantine_reason: str
    repo_url: str
    indexed_at: str


class QuarantineListResponse(BaseModel):
    """Response body for GET /quarantine.

    Args:
        items: List of quarantined documents. Empty list when the queue is clear.
        total: Total count of quarantined documents (== len(items)).
    """

    items: list[QuarantineItem]
    total: int
