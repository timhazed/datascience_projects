"""Streamlit Chat tab — graph-backed Q&A and onboarding sample queries."""

from __future__ import annotations

import streamlit as st
from langgraph.graph.state import CompiledStateGraph

from src.db.metrics_db import MetricsDB
from src.ui.constants import _SAMPLE_QUERIES
from src.ui.graph_runtime import _invoke_and_update, user_facing_assistant_text


def _process_chat_message(
    graph: CompiledStateGraph,
    query: str,
    patient_hint: str | None,
    metrics_db: MetricsDB | None = None,
) -> None:
    """Append a user message to chat history, invoke the graph, append the reply.

    Shared by the onboarding sample-query buttons and the st.chat_input handler
    so the processing logic is not duplicated.

    Args:
        graph: Compiled LangGraph graph.
        query: Natural language query to send.
        patient_hint: Optional patient name context for the planner.
        metrics_db: MetricsDB for cross-session persistence.
    """
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state["chat_history"].append(("user", query))
    st.session_state["has_chatted"] = True

    with st.spinner("Thinking…"):
        result = _invoke_and_update(
            graph,
            query,
            patient_hint,
            metrics_db=metrics_db,
        )

    reply_text = user_facing_assistant_text(result)
    with st.chat_message("assistant"):
        st.markdown(reply_text)

        # Inline trust badge — only after the guard allowed the request to proceed.
        # UNSAFE short-circuit must not imply "general knowledge" (no planner ran).
        tasks = result.get("completed_tasks", [])
        retrieved = any(t.task == "retrieve_history" and t.success for t in tasks)
        if result.get("intent_safe") is True:
            if retrieved:
                st.caption("\u2713 Response draws on this patient's record.")
            elif st.session_state.get("resolved_patient_id"):
                st.caption(
                    "\u2139\ufe0f Response based on general knowledge — "
                    "patient record was not retrieved."
                )

        # Guard-only UNSAFE: trace is just ``intent_guard: UNSAFE`` — keep that out of
        # the UI so clinicians are not drawn to a technical line under "Agent Trace"
        # when the main reply already explains the block.
        trace = result.get("trace", [])
        if trace and result.get("intent_safe") is not False:
            with st.expander("Agent Trace", expanded=False):
                st.code("\n".join(trace))
    st.session_state["chat_history"].append(("assistant", reply_text))


def _render_tab_chat(
    graph: CompiledStateGraph,
    patient_name_input: str,
    metrics_db: MetricsDB | None = None,
) -> None:
    """Render Tab 3 — Chat: full natural-language Q&A via the graph.

    Displays an onboarding banner with example queries until the first message
    is sent. After that, shows the full chat history and accepts new input.
    An active patient banner is always shown when a patient is resolved so the
    user knows which record the assistant will reference.

    Args:
        graph: Compiled LangGraph graph (cached singleton).
        patient_name_input: Resolved patient name from session state (may be empty
            if no patient has been resolved via the left panel or History tab).
        metrics_db: Optional MetricsDB for cross-session metrics logging (Metrics tab).
    """
    # Active patient context banner — persistent so the user always knows who
    # is in context without having to look at the left panel while chatting.
    # Use the parameter rather than re-reading session state; both hold the same value.
    if patient_name_input:
        st.info(
            f"💊 Active patient: **{patient_name_input}** — "
            "questions will reference this patient's record. "
            "Clear the patient in the left panel for general questions."
        )

    # Onboarding banner — shown only before the first message is sent.
    # Message tail is patient-context-aware to avoid contradicting the active patient banner.
    if not st.session_state["has_chatted"]:
        if patient_name_input:
            welcome_tail = (
                f"**{patient_name_input}** is your active patient — "
                "try asking about their history or booking an appointment."
            )
        else:
            welcome_tail = (
                "For patient-specific questions, enter a name in the "
                "**Active Patient** panel on the left."
            )
        st.info(
            "\U0001f44b **Welcome to the Healthcare Assistant.**  \n"
            "Ask a clinical question below, or click one of the examples to get started.  \n"
            + welcome_tail
        )
        col1, col2, col3 = st.columns(3)
        clicked_query: str | None = None
        for col, (label, query) in zip([col1, col2, col3], _SAMPLE_QUERIES):
            if col.button(label, use_container_width=True, key=f"hint_{label[:8]}"):
                clicked_query = query

        if clicked_query:
            _process_chat_message(
                graph,
                clicked_query,
                patient_name_input or None,
                metrics_db=metrics_db,
            )
            st.rerun()

    # Replay chat history above the input box
    for role, message in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(message)

    user_query = st.chat_input(
        "Ask about a patient's history, a condition, or request a booking…",
        max_chars=1000,
    )
    if not user_query:
        return

    _process_chat_message(
        graph,
        user_query,
        patient_name_input or None,
        metrics_db=metrics_db,
    )
