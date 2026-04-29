"""Streamlit Memory & Logs tab — scenarios, planner trace, FAISS debug search."""

from __future__ import annotations

import pandas as pd
import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from src.db.metrics_db import MetricsDB
from src.db.patient_vector_store import PatientVectorStore
from src.ui.constants import _SCENARIOS
from src.ui.graph_runtime import _invoke_and_update


def _render_tab_memory(
    graph: CompiledStateGraph,
    vector_store: PatientVectorStore,
    metrics_db: MetricsDB | None = None,
) -> None:
    """Render Tab 5 — Memory & Logs.

    Sections:
      - Try a sample scenario: preset queries for quick end-to-end testing
      - Developer Details (collapsed): Planning Breakdown, Agent Trace, Patient Memory search

    Scenario runs show a **this run** latency strip below the answer; the Metrics tab
    holds session-wide activity and historical means.

    Args:
        graph: Compiled LangGraph graph (cached singleton).
        vector_store: PatientVectorStore for FAISS similarity search.
        metrics_db: Optional MetricsDB so scenario runs log latency like Chat (cross-session means).
    """
    # --- Try a sample scenario (user-facing — renamed from "Interactive Test Scenarios") ---
    st.subheader("Try a Sample Scenario")
    st.caption(
        "Run a preset multi-step query end-to-end. After each run you will see **this run** "
        "latency and a short summary of tool steps here; session-wide history and averages "
        "stay on the Metrics tab."
    )
    scenario_select = st.selectbox(
        "Select a scenario", list(_SCENARIOS.keys()), key="scenario_select"
    )
    scenario_query = _SCENARIOS[scenario_select]
    st.info(scenario_query)

    if st.button("Run Scenario", key="scenario_run"):
        with st.spinner("Running scenario…"):
            result = _invoke_and_update(
                graph,
                scenario_query,
                None,
                metrics_db=metrics_db,
            )

        summary = result.get("final_summary")
        if summary:
            st.success(summary)
        else:
            st.error(result.get("error", "Scenario failed."))

        lat = result.get("_ui_this_run_latency_ms")
        op = result.get("_ui_this_run_operation")
        if lat is not None and op is not None:
            st.markdown(
                f"**This run:** {lat} ms  \n**Recorded steps:** {op}",
            )
            st.caption(
                "Local timing for this scenario only. For the full session log and "
                "historical latency means, use the Metrics tab."
            )

    # --- Developer Details — collapsed by default so clinical users are not distracted ---
    st.divider()
    with st.expander("🔧 Developer Details", expanded=False):
        # Planning Breakdown: raw JSON of the last PlannerOutput
        st.subheader("Planning Breakdown")
        planner_json = st.session_state.get("last_planner_output")
        if planner_json:
            st.json(planner_json)
        else:
            st.caption("No plan yet — send a message in the Chat tab.")

        st.divider()
        # Agent Trace: node visit sequence stored by _invoke_and_update
        st.subheader("Agent Trace")
        last_trace = st.session_state.get("last_trace", [])
        if last_trace:
            st.code("\n".join(last_trace))
        else:
            st.caption("No trace yet — send a message in the Chat tab.")

        st.divider()
        # Patient Memory search: direct FAISS similarity query for debugging embeddings
        st.subheader("Patient Memory Search")
        st.caption("Search the vector index directly — useful for verifying embedded records.")
        memory_query = st.text_input("Search query", key="mem_query")
        if st.button("Search", key="mem_search"):
            if memory_query:
                with st.spinner("Searching…"):
                    hits = vector_store.search(memory_query, k=5)
                if hits:
                    rows = [
                        {
                            "patient_id": h.get("patient_id", ""),
                            "name": h.get("name", ""),
                            "conditions": h.get("conditions", ""),
                        }
                        for h in hits
                    ]
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No matches found.")
            else:
                st.warning("Enter a query first.")
