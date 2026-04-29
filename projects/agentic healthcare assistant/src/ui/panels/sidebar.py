"""Streamlit sidebar — version, provider, session status, clear session."""

from __future__ import annotations

import streamlit as st

from src.ui.constants import _APP_VERSION
from src.ui.runtime import settings
from src.ui.session_state import _reset_session_for_sidebar_clear


def _render_sidebar() -> None:
    """Render the application sidebar with session status and app version."""
    with st.sidebar:
        st.header("Healthcare Assistant")
        st.caption(f"Version {_APP_VERSION}")

        # Compact provider line — administrators don't need the full widget pair
        provider_name = settings.provider_name
        cfg = settings.provider_config
        st.caption(f"Powered by {provider_name} · `{cfg.model}`")

        st.divider()
        st.subheader("Session")
        resolved_name = st.session_state.get("resolved_patient_name")
        if resolved_name:
            st.success("Patient context active")
            st.caption(resolved_name)
        else:
            st.caption("No patient resolved yet.")

        if st.button("Clear session", key="sidebar_clear"):
            # Reset all mutable session keys; cache_resource keeps graph/DB alive
            _reset_session_for_sidebar_clear()
            st.rerun()
