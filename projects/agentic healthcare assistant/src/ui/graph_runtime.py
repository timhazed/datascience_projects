"""LangGraph invoke helpers: build initial state and sync results into Streamlit session."""

from __future__ import annotations

import logging
import time
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from src.db.metrics_db import MetricsDB
from src.ui.constants import _TASK_LABELS
from src.ui.runtime import _SESSION_ID, settings
from src.utils.graph_config import build_invoke_config

logger = logging.getLogger(__name__)

# Keys merged into invoke results for UI only (not HealthcareState / graph schema).
_UI_THIS_RUN_LATENCY_MS = "_ui_this_run_latency_ms"
_UI_THIS_RUN_OPERATION = "_ui_this_run_operation"


def _dedupe_task_operation_line(new_tasks: list) -> str:
    """Build the Metrics-tab style operation string from TaskResult objects."""
    task_labels = [_TASK_LABELS.get(t.task, t.task) for t in new_tasks]
    seen: set[str] = set()
    unique = [lb for lb in task_labels if not (lb in seen or seen.add(lb))]
    return " + ".join(unique)


def _this_run_operation_caption(new_tasks: list | None) -> str:
    """One-line description of tool work for Memory & Logs (this invocation only)."""
    if not new_tasks:
        return "No tool tasks in this run (summarizer only)."
    return _dedupe_task_operation_line(new_tasks)


def _hydrate_chat_from_checkpoint(
    graph: CompiledStateGraph,
    patient_id: str | None,
) -> None:
    """Clear chat state and reload from the LangGraph checkpoint for patient_id.

    Always clears chat_history first so switching patients cannot leak prior messages.
    No-ops gracefully when the graph has no checkpointer or patient_id is absent.

    Args:
        graph: Compiled LangGraph graph.
        patient_id: Canonical patient slug, or None when clearing active patient.
    """
    st.session_state["chat_history"] = []
    st.session_state["has_chatted"] = False

    if not getattr(graph, "checkpointer", None):
        return
    if not patient_id:
        return

    config = build_invoke_config(patient_id, settings)
    snapshot = graph.get_state(config)
    # get_state returns a StateSnapshot with values={} for non-existent threads (not None).
    if not snapshot or not snapshot.values:
        return

    messages = snapshot.values.get("messages", [])
    pairs = [
        ("user" if isinstance(m, HumanMessage) else "assistant", m.content)
        for m in messages
        if isinstance(m, HumanMessage | AIMessage)
    ][-settings.db.max_recent_turns:]
    if pairs:
        st.session_state["chat_history"] = list(pairs)
        st.session_state["has_chatted"] = True


def set_resolved_patient_with_hydration(
    graph: CompiledStateGraph,
    patient_id: str,
    patient_display_name: str,
) -> None:
    """Set resolved patient session keys and reload chat from checkpoint.

    Used by Patient and Doctor tabs when the user selects or registers a patient.
    Centralizes session updates + ``_hydrate_chat_from_checkpoint`` for testability.

    Args:
        graph: Compiled graph with checkpointer attached.
        patient_id: Canonical patient slug.
        patient_display_name: Human-readable name (slug never shown in UI).
    """
    st.session_state["resolved_patient_id"] = patient_id
    st.session_state["resolved_patient_name"] = patient_display_name
    _hydrate_chat_from_checkpoint(graph, str(patient_id).strip())


def _build_initial_state(query: str, patient_hint: str | None) -> dict:
    """Build the HealthcareState dict for graph.invoke().

    If a patient is already resolved in the session, the patient_id is injected
    so the planner can skip re-resolution for queries that need it.

    Args:
        query: Natural language query.
        patient_hint: Optional patient name to prepend as planner context.

    Returns:
        HealthcareState-compatible dict.
    """
    effective_query = query
    if patient_hint:
        effective_query = f"[Patient context: {patient_hint}]\n{query}"

    return {
        "user_query": effective_query,
        "patient_id": st.session_state.get("resolved_patient_id"),
        # Inject HumanMessage so add_messages reducer captures the turn from the start.
        # The summarizer_node will append the corresponding AIMessage on completion.
        "messages": [HumanMessage(content=effective_query)],
        "planner_output": None,
        "pending_tasks": [],
        "completed_tasks": [],
        "intent_safe": False,
        "final_summary": None,
        "error": None,
        "trace": [],
    }


def user_facing_assistant_text(result: dict) -> str:
    """Return the assistant message to show in Chat and store in ``chat_history``.

    ``final_summary`` and ``error`` may both be ``None`` (e.g. intent guard
    routed UNSAFE to END). Using ``dict.get("error", fallback)`` is wrong when
    the key exists with value ``None``. UNSAFE without ``error`` must still
    produce a clear user message.

    Args:
        result: Dict returned from ``graph.invoke()`` (or the synthetic error dict).

    Returns:
        Non-empty string suitable for ``st.markdown`` and session ``chat_history``.
    """
    fs = result.get("final_summary")
    if isinstance(fs, str) and fs.strip():
        return fs.strip()
    err = result.get("error")
    if isinstance(err, str) and err.strip():
        return err.strip()
    if result.get("intent_safe") is False:
        return (
            "This request could not be processed because it was classified as "
            "off-topic or unsafe. Please ask a healthcare-related question."
        )
    return "No response was generated. Please try rephrasing your question."


def _invoke_and_update(
    graph: CompiledStateGraph,
    query: str,
    patient_hint: str | None,
    metrics_db: MetricsDB | None = None,
) -> dict:
    """Invoke the graph and update shared session state from the result.

    Sets resolved_patient_id, resolved_patient_name, last_planner_output,
    last_trace, and accumulates completed_tasks so Tab 5 reflects the full
    session history.

    On any unhandled exception from the graph, returns a synthetic failure dict
    so callers can surface st.error() rather than crashing the Streamlit session.

    Args:
        graph: Compiled LangGraph CompiledStateGraph.
        query: Natural language query string.
        patient_hint: Optional patient name hint — used as display name when a
            new patient is resolved so the slug is never shown in the UI.
        metrics_db: MetricsDB for cross-session persistence; pass ``None`` to
            skip ``session_log`` writes (rare — e.g. tests).

    Returns:
        The graph result dict (shallow-copied) with UI-only keys
        ``_ui_this_run_latency_ms`` and ``_ui_this_run_operation`` for local
        display (e.g. Memory & Logs), or a synthetic error dict with the same keys.
    """
    initial_state = _build_initial_state(query, patient_hint)
    # Build config with thread_id=patient_id for checkpointing continuity;
    # falls back to _SESSION_ID when no patient is resolved yet.
    invoke_cfg = build_invoke_config(
        initial_state.get("patient_id"), settings, session_id=_SESSION_ID
    )
    _invoke_start = time.perf_counter()
    try:
        result = graph.invoke(initial_state, invoke_cfg)
    except GraphRecursionError:
        limit = settings.graph.recursion_limit
        logger.error(
            "[graph_runtime] GraphRecursionError — LangGraph stopped after %s supersteps "
            "(config: graph.recursion_limit). Often caused by a stuck task queue or an "
            "overly long plan. See GRAPH_RECURSION_LIMIT troubleshooting.",
            limit,
            exc_info=True,
        )
        _fail_ms = round((time.perf_counter() - _invoke_start) * 1000, 1)
        return {
            "final_summary": None,
            "error": (
                f"The workflow hit its execution step limit (recursion_limit={limit}). "
                "If your request is complex, try raising graph.recursion_limit in config.yaml; "
                "otherwise this may indicate an internal routing issue."
            ),
            "planner_output": None,
            "completed_tasks": [],
            "patient_id": None,
            "trace": [],
            _UI_THIS_RUN_LATENCY_MS: _fail_ms,
            _UI_THIS_RUN_OPERATION: "Graph step limit reached",
        }
    except Exception as exc:  # noqa: BLE001
        logger.error("[graph_runtime] graph.invoke failed: %s — %s", type(exc).__name__, exc)
        _fail_ms = round((time.perf_counter() - _invoke_start) * 1000, 1)
        return {
            "final_summary": None,
            "error": f"An internal error occurred ({type(exc).__name__}). Please try again.",
            "planner_output": None,
            "completed_tasks": [],
            "patient_id": None,
            "trace": [],
            _UI_THIS_RUN_LATENCY_MS: _fail_ms,
            _UI_THIS_RUN_OPERATION: "Graph invocation failed",
        }
    _invoke_ms = round((time.perf_counter() - _invoke_start) * 1000, 1)

    # Propagate resolved patient_id — resolver sets patient_id on state
    new_pid = result.get("patient_id")
    if new_pid and new_pid != st.session_state.get("resolved_patient_id"):
        st.session_state["resolved_patient_id"] = new_pid
        # Use the patient_hint (what the user typed) as the display name — the
        # slug itself is never surfaced in the UI per spec.
        if patient_hint:
            st.session_state["resolved_patient_name"] = patient_hint
        # Preserve in-flight chat rows: _process_chat_message appends the user
        # turn before invoke; _hydrate_chat_from_checkpoint wipes chat_history.
        in_flight = list(st.session_state.get("chat_history", []))
        _hydrate_chat_from_checkpoint(graph, str(new_pid).strip())
        st.session_state["chat_history"] = (
            list(st.session_state.get("chat_history", [])) + in_flight
        )
        if in_flight:
            st.session_state["has_chatted"] = True

    # Update Tab 5 planner breakdown
    if result.get("planner_output"):
        st.session_state["last_planner_output"] = result["planner_output"].model_dump()

    # Store node visit sequence for Tab 5 Agent Trace
    st.session_state["last_trace"] = result.get("trace", [])

    # Accumulate tool log (completed_tasks is a flat list across the session)
    _raw_completed = result.get("completed_tasks")
    new_tasks = _raw_completed if isinstance(_raw_completed, list) else []
    st.session_state["completed_tasks"].extend(new_tasks)

    # Record ONE summary entry per graph invocation so the Metrics tab shows one row
    # per chat prompt — not one row per internal task (which splits latency misleadingly).
    # The operation label lists all tasks that ran so the user can still see what happened.
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if new_tasks:
        operation_summary = _dedupe_task_operation_line(new_tasks)
        overall_success = all(t.success for t in new_tasks)
        errors = [t.error for t in new_tasks if t.error]
        error_str = "; ".join(errors) if errors else ""
        latency = round(_invoke_ms, 1)
        st.session_state["session_metrics"].append({
            "datetime": now_str,
            "operation": operation_summary,
            "success": overall_success,
            "latency_ms": latency,
            "error": error_str,
        })
        if metrics_db is not None:
            try:
                metrics_db.log_entry(
                    str(settings.db.sqlite_path),
                    _SESSION_ID,
                    now_str,
                    operation_summary,
                    overall_success,
                    latency,
                    error_str,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[metrics] log_entry failed: %s — %s", type(exc).__name__, exc
                )

    # Capture last query/answer for the Metrics tab RAGAS on-demand evaluation button.
    st.session_state["last_query_for_eval"] = query
    st.session_state["last_answer_for_eval"] = result.get("final_summary") or ""

    merged = dict(result)
    merged[_UI_THIS_RUN_LATENCY_MS] = round(_invoke_ms, 1)
    merged[_UI_THIS_RUN_OPERATION] = _this_run_operation_caption(new_tasks)
    return merged
