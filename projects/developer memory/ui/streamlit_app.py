"""Developer Memory — Streamlit entry point.

All business logic lives in ui/pages/, ui/http_client.py, ui/state.py, and ui/config.py.
This file exists so that `streamlit run ui/streamlit_app.py` continues to work unchanged.
"""

import os

import streamlit as st

from ui.pages.analyze_diff import page as page_diff
from ui.pages.developer_persona import page as page_persona
from ui.pages.generate_skills import page as page_skills
from ui.pages.pii_review_queue import page as page_pii
from ui.pages.query_memory import page as page_query
from ui.pages.sync_repository import _build_stage_label
from ui.pages.sync_repository import page as page_sync
from ui.state import _sync_is_active, init_session_state

st.set_page_config(
    page_title="Developer Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_session_state()

PAGES = [
    "Sync Repository",
    "Query Memory",
    "Developer Persona",
    "Analyze Diff",
    "Generate Skills Pkg",
    "PII Review Queue",
]

with st.sidebar:
    st.title("🧠 Developer Memory")
    st.caption("Sovereign Developer Memory & Coaching System")
    st.divider()
    _active = _sync_is_active()
    if _active:
        _stage = st.session_state["_sync_stage"]
        # Delegate label construction to sync_repository — single source of truth.
        _label, _ = _build_stage_label(
            _stage,
            st.session_state["_sync_chunks_done"],
            st.session_state["_sync_chunks_total"],
            st.session_state["_sync_pii_done"],
            st.session_state["_sync_pii_total"],
        )
        st.warning(f"Sync in progress — {_label}")
        st.caption("Other pages are disabled until the sync completes.")
        page = "Sync Repository"
        st.radio("Navigate", PAGES, index=0, disabled=True, label_visibility="collapsed")
    else:
        page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption(f"MCP Server: {os.environ.get('MCP_SERVER_URL', 'http://localhost:9000')}")

_DISPATCH = {
    "Sync Repository": page_sync,
    "Query Memory": page_query,
    "Developer Persona": page_persona,
    "Analyze Diff": page_diff,
    "Generate Skills Pkg": page_skills,
    "PII Review Queue": page_pii,
}
_DISPATCH[page]()
