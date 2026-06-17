"""Developer Memory UI — PII Review Queue page renderer.

Owns the PII Review Queue page, which surfaces quarantined files from the MCP server
and allows manual review and release for ingestion. All HTTP calls are delegated to
ui.http_client — no raw httpx calls in this module.
"""

import logging

import streamlit as st

from ui.http_client import _get_quarantine, _release_quarantine

logger = logging.getLogger(__name__)


def page() -> None:
    """Render the PII Review Queue page.

    Fetches quarantined items from GET /quarantine via _get_quarantine(), then renders
    each item in an expander with a release button. Releasing calls _release_quarantine()
    and refreshes the page via st.rerun().
    """
    st.header("PII Review Queue")
    st.markdown(
        "Files held here could not be safely masked by the PII filter. "
        "Review each file manually before approving ingestion."
    )
    st.warning(
        "⚠️ Files in this queue may contain sensitive information. "
        "Do not share this page or its contents externally."
    )

    try:
        items = _get_quarantine()
    except Exception as exc:  # noqa: BLE001
        logger.error("GET /quarantine failed [%s]: %s", type(exc).__name__, str(exc)[:200])
        st.error("Could not load PII review queue — MCP server may be unavailable.")
        return

    if not items:
        st.success("PII Review Queue is empty — no files require manual review.")
        return

    st.markdown(f"**{len(items)} file(s) pending review:**")

    for item in items:
        doc_id: str = item.get("doc_id", "")
        path: str = item.get("file_path", "unknown")
        reason: str = item.get("quarantine_reason", "Unknown reason")

        with st.expander(f"`{path}` — {reason}"):
            st.markdown(f"**Path**: `{path}`")
            st.markdown(f"**Quarantine Reason**: {reason}")
            st.markdown(f"**Repo**: {item.get('repo_url', 'unknown')}")
            st.markdown(f"**Indexed At**: {item.get('indexed_at', 'unknown')}")

            if st.button("Release for ingestion", key=f"release_{doc_id}"):
                try:
                    released = _release_quarantine(doc_id)
                except Exception as rel_exc:  # noqa: BLE001
                    logger.error(
                        "POST /quarantine/%s/release failed [%s]: %s",
                        doc_id, type(rel_exc).__name__, str(rel_exc)[:200],
                    )
                    st.error(f"Failed to release `{path}` — check server logs.")
                else:
                    if released:
                        st.success(f"Released `{path}` — it will be indexed on next sync.")
                        st.rerun()
                    else:
                        st.warning(f"`{path}` was not found in the queue — it may have already been released.")
