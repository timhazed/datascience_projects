"""Streamlit UI entry point — Agentic Healthcare Assistant (Part 2).

Layout
------
Sidebar         : ``src.ui.panels.sidebar``
col_left (1)    : ``src.ui.panels.left_panel``
col_right (3)   : ``st.tabs`` — Patient | Doctor | Chat | Metrics | Memory & Logs

Tab implementations live under ``src/ui/tabs/`` (``patient_tab``, ``doctor_tab``,
``chat_tab``, ``metrics_tab``, ``memory_tab``). Shared layout: ``panels/left_panel``,
``panels/sidebar``. Shared record rendering: ``components/patient_record``. Wiring:
``runtime``, ``resources``, ``session_state``, ``graph_runtime``, ``metrics_session``,
``constants``.

Session state keys (shared across tabs)
----------------------------------------
resolved_patient_id   : str | None   — set by resolver; shared across tabs
resolved_patient_name : str | None   — display name; slug (P-xxxx) never shown
chat_history          : list[tuple]  — (role, message) tuples
last_planner_output   : dict | None  — PlannerOutput.model_dump() for Memory tab
last_trace            : list[str]    — HealthcareState.trace from last invocation
completed_tasks       : list         — accumulated TaskResult objects
has_chatted           : bool         — False until first chat message sent
_left_panel_matches   : list         — disambiguation rows from left-panel Resolve
session_metrics       : list[dict]   — live per-operation metrics for Metrics tab
last_query_for_eval   : str          — last Chat query for RAGAS on-demand eval
last_answer_for_eval  : str          — last Chat answer for RAGAS on-demand eval
"""

# NOTE: ``main()`` must run under Streamlit (``streamlit run`` / ``poetry run streamlit-app``).
# Invoking ``main()`` from a bare Python process lacks ScriptRunContext and triggers warnings.

from __future__ import annotations

import logging

import streamlit as st

from src.ui.panels.left_panel import _render_left_panel
from src.ui.panels.sidebar import _render_sidebar
from src.ui.resources import _load_resources
from src.ui.session_state import _init_session_state
from src.ui.tabs.chat_tab import _render_tab_chat
from src.ui.tabs.doctor_tab import _render_tab_doctor
from src.ui.tabs.memory_tab import _render_tab_memory
from src.ui.tabs.metrics_tab import _render_tab_metrics
from src.ui.tabs.patient_tab import _render_tab_patient

logging.basicConfig(level=logging.WARNING)

# Streamlit's default main block uses generous top padding; tighten for denser layout.
_COMPACT_MAIN_CSS = """
<style>
    section[data-testid="stMain"] .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    section[data-testid="stMain"] h1 {
        margin-top: 0;
        margin-bottom: 0.35rem;
    }
</style>
"""


def main() -> None:
    """Streamlit entry point — render the five-tab healthcare assistant UI.

    Tab order: Patient → Doctor → Chat → Metrics → Memory & Logs.
    Patient tab is first — the Medical Assistant's primary workflow is finding
    a patient, reviewing their record, and booking appointments.
    Doctor tab provides clinical write actions: encounter notes, record updates,
    and schedule views.
    """
    # Browser tab title (bookmark / tab bar)
    st.set_page_config(layout="wide", page_title="Healthcare Assistant")
    st.markdown(_COMPACT_MAIN_CSS, unsafe_allow_html=True)
    _init_session_state()
    _render_sidebar()

    # Load cached infrastructure (builds graph + DB on first call)
    (
        graph,
        patient_db,
        appt_db,
        metrics_db,
        vector_store,
        _checkpointer,
    ) = _load_resources()

    st.title("Healthcare Assistant")

    # Two-column layout: narrow left for patient context, wide right for tabs
    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("Active Patient")
        _render_left_panel(
            patient_db,
            graph,
            metrics_db=metrics_db,
        )

    with col_right:
        tab_patient, tab_doctor, tab_chat, tab_metrics, tab_memory = st.tabs(
            [
                "👤 Patient",
                "🩺 Doctor",
                "💬 Chat",
                "📊 Metrics",
                "🧠 Memory & Logs",
            ]
        )

        # Run Doctor before Patient so DB writes in Doctor (e.g. record update) happen
        # before Patient calls get_patient() — same-run freshness when switching tabs.
        with tab_doctor:
            _render_tab_doctor(
                graph,
                patient_db,
                metrics_db=metrics_db,
            )

        with tab_patient:
            _render_tab_patient(
                graph,
                patient_db,
                appt_db,
                vector_store,
                metrics_db=metrics_db,
            )

        with tab_chat:
            # Pass resolved name as hint — set by left panel Resolve button or Patient tab Select
            _render_tab_chat(
                graph,
                st.session_state.get("resolved_patient_name") or "",
                metrics_db=metrics_db,
            )

        with tab_metrics:
            _render_tab_metrics(metrics_db=metrics_db)

        with tab_memory:
            _render_tab_memory(graph, vector_store, metrics_db=metrics_db)


if __name__ == "__main__":
    main()
