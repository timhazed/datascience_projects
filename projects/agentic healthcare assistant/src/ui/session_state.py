"""Streamlit session state defaults and sidebar clear contract for the healthcare UI."""

from __future__ import annotations

import copy

import streamlit as st

# Single source for session keys — see docs/UI_Rework.md §5; must stay aligned with sidebar clear.
_SESSION_STATE_DEFAULTS: dict[str, object] = {
    "resolved_patient_id": None,
    "resolved_patient_name": None,  # display name; slug (P-xxxx) is never shown in UI
    "chat_history": [],             # list of (role, message) tuples
    "last_planner_output": None,
    "last_trace": [],               # HealthcareState.trace from last graph invocation
    "completed_tasks": [],          # accumulated TaskResult objects
    "has_chatted": False,           # onboarding banner shown until first message sent
    "_left_panel_matches": [],      # disambiguation list from left-panel Resolve button
    "session_metrics": [],          # live per-operation metrics; appended by _invoke_and_update
    "last_query_for_eval": "",      # last Chat query; consumed by Metrics RAGAS eval button
    "last_answer_for_eval": "",     # last Chat answer; consumed by Metrics RAGAS eval button
    "_doctor_field_update_flash": None,  # one-shot success after Doctor tab record update + rerun
}


def _init_session_state() -> None:
    """Initialise all required session state keys to their defaults.

    Called once per browser session before any tab renders.
    """
    for key, value in _SESSION_STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(value)


def _reset_session_for_sidebar_clear() -> None:
    """Reset session keys cleared by the sidebar 'Clear session' button (see UI_Rework.md)."""
    for key, value in _SESSION_STATE_DEFAULTS.items():
        st.session_state[key] = copy.deepcopy(value)
