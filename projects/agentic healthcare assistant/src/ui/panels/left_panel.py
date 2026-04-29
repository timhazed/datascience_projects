"""Streamlit left column — Active Patient resolve / disambiguation."""

from __future__ import annotations

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from src.db.metrics_db import MetricsDB
from src.db.patient_db import PatientDB
from src.ui.graph_runtime import _hydrate_chat_from_checkpoint
from src.ui.metrics_session import _record_metric
from src.ui.runtime import settings


def _clear_active_patient_conversation_history(
    graph: CompiledStateGraph,
    patient_id: str,
) -> None:
    """Delete checkpoint thread and reset chat UI session keys for patient_id.

    Calls graph.checkpointer.delete_thread(pid) when a checkpointer is attached,
    then clears session keys so the chat UI reflects the empty history immediately.

    Args:
        graph: Compiled graph — checkpointer.delete_thread() called when present.
        patient_id: Patient identifier whose stored conversation is removed.
    """
    pid = str(patient_id).strip()
    if getattr(graph, "checkpointer", None):
        graph.checkpointer.delete_thread(pid)  # str, not config dict (Decision 7)
    st.session_state["chat_history"] = []
    st.session_state["has_chatted"] = False
    st.session_state["last_planner_output"] = None


def _render_left_panel(
    patient_db: PatientDB,
    graph: CompiledStateGraph,
    metrics_db: MetricsDB | None = None,
) -> None:
    """Render the Active Patient panel in the left column.

    Provides a name input with an inline Resolve button that performs a direct
    DB fuzzy-search — no graph invocation required.  When exactly one patient
    matches, session state is set immediately and the UI reflects the resolved
    patient.  When multiple matches are found, each is shown with a Select
    button.  This replaces the earlier dead-end caption that told users to
    "send a message in Chat" with no visible path forward.

    Args:
        patient_db: PatientDB instance for direct name lookup.
        graph: Compiled LangGraph graph — used for checkpoint-based hydration and clear.
        metrics_db: Optional MetricsDB for lookup metrics.
    """
    db_path = str(settings.db.sqlite_path)

    patient_name_input = st.text_input(
        "Patient name",
        key="global_patient_name",
        placeholder="e.g. Ramesh Kulkarni",
        help="Type a name and click Resolve to set the active patient.",
    )

    resolved_name = st.session_state.get("resolved_patient_name")

    if resolved_name:
        # Patient already set — show status and an option to clear it
        st.success(f"✓ {resolved_name}")
        if st.button("Clear patient", key="left_clear_patient", use_container_width=True):
            st.session_state["resolved_patient_id"] = None
            st.session_state["resolved_patient_name"] = None
            st.session_state["_left_panel_matches"] = []
            _hydrate_chat_from_checkpoint(graph, None)
            st.rerun()
        if st.button(
            "Clear conversation history",
            key="left_clear_history",
            use_container_width=True,
        ):
            pid = str(st.session_state["resolved_patient_id"]).strip()
            _clear_active_patient_conversation_history(graph, pid)
            st.rerun()
    else:
        # Resolve button — direct DB lookup, no LLM needed
        resolve_disabled = not patient_name_input
        if st.button(
            "Resolve",
            disabled=resolve_disabled,
            key="left_resolve",
            use_container_width=True,
            help="Find this patient in the database.",
        ):
            matches = patient_db.fuzzy_search(db_path, patient_name_input, limit=10)
            if not matches:
                st.session_state["_left_panel_matches"] = []
                st.warning("No patient found. Check spelling or use History → Find a Patient.")
            elif len(matches) == 1:
                # Unambiguous match — set immediately
                if matches[0].patient_id != st.session_state.get("resolved_patient_id"):
                    _record_metric("Patient Lookup", success=True, metrics_db=metrics_db)
                st.session_state["resolved_patient_id"] = matches[0].patient_id
                st.session_state["resolved_patient_name"] = matches[0].name
                _hydrate_chat_from_checkpoint(
                    graph,
                    str(matches[0].patient_id).strip(),
                )
                st.rerun()
            else:
                # Multiple matches — store them so the selectbox below can render
                st.session_state["_left_panel_matches"] = [
                    (r.patient_id, r.name, r.age, r.gender) for r in matches
                ]

        # Show disambiguation list if multiple matches were found on last click
        pending_matches = st.session_state.get("_left_panel_matches", [])
        if pending_matches:
            st.caption(f"{len(pending_matches)} patients found — select one:")
            for pid, name, age, gender in pending_matches:
                col_info, col_btn = st.columns([3, 1])
                col_info.write(f"{name} · {age} · {gender}")
                if col_btn.button("Set", key=f"left_sel_{pid}"):
                    if pid != st.session_state.get("resolved_patient_id"):
                        _record_metric("Patient Lookup", success=True, metrics_db=metrics_db)
                    st.session_state["resolved_patient_id"] = pid
                    st.session_state["resolved_patient_name"] = name
                    st.session_state["_left_panel_matches"] = []
                    _hydrate_chat_from_checkpoint(graph, str(pid).strip())
                    st.rerun()

        if not patient_name_input:
            st.caption("No patient set — disease/search queries work without one.")

    st.divider()
    st.caption(
        "**Tip:** Resolve a patient here, then use **Chat** to ask questions "
        "or the **Patient** tab to book or view appointments. "
        "Leave blank for general medical questions."
    )
