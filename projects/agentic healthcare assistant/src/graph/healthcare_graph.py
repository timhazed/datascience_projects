"""healthcare_graph — assembles and compiles the LangGraph StateGraph.

Public API
----------
build_graph(settings) -> BuildGraphResult
    Constructs LLM singletons, chain singletons, DB instances, and node
    functions from the provided Settings, then compiles the graph.
    Returns BuildGraphResult(graph, checkpointer) where checkpointer is a
    SqliteSaver instance when checkpointing_enabled is True, otherwise None.

_compile_healthcare_graph(**kwargs) -> CompiledGraph
    Internal wiring function — accepts pre-built chains and DB instances.
    Called by build_graph in production; called directly by integration tests
    with mock chains and real in-memory SQLite.

Graph topology
--------------
START
  ↓
[intent_guard] ──UNSAFE──→ END
  │ SAFE
  ↓
[planner]
  ↓
 ╔═══════════════╗
 ║  _route_next  ║  ←──────────────────────────────────────────┐
 ╚═══════════════╝                                             │
  │ resolve_patient / retrieve_history /                       │
  │ update_history / book_appointment / search_disease         │
  ↓                                                           │
[task node] ───────────────────────────────────────────────────┘
  │ (pending_tasks empty)
  ↓
[summarizer]
  ↓
END
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, NamedTuple

from langchain_openai import OpenAIEmbeddings
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.appointment_node import make_appointment_node
from src.agents.disease_search_node import make_disease_search_node
from src.agents.history_retriever_node import make_history_retriever_node
from src.agents.history_writer_node import make_history_writer_node
from src.agents.intent_guard_node import make_intent_guard_node
from src.agents.patient_resolver_node import make_patient_resolver_node
from src.agents.planner_node import make_planner_node
from src.agents.summarizer_node import make_summarizer_node
from src.chains.history_chain import build_history_chain
from src.chains.intent_guard_chain import build_intent_guard_chain
from src.chains.memory_summary_chain import build_memory_summary_chain
from src.chains.planner_chain import build_planner_chain
from src.chains.search_chain import build_search_chain
from src.chains.summary_chain import build_summary_chain
from src.config.settings import Settings
from src.db.appointment_db import AppointmentDB
from src.db.patient_db import PatientDB
from src.db.patient_vector_store import PatientVectorStore
from src.llm.llm_factory import get_llm
from src.models.graph_state import HealthcareState
from src.models.search_result_item import SearchResultItem
from src.models.task_result import TaskResult
from src.utils.medline_search import search_medline

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public result type — replaces the bare (graph, chain) 2-tuple
# ---------------------------------------------------------------------------


class BuildGraphResult(NamedTuple):
    """Named return type for build_graph().

    Attributes:
        graph: Compiled LangGraph StateGraph ready to invoke.
        checkpointer: SqliteSaver instance when checkpointing_enabled is True,
            otherwise None.  Stored here so callers (resources.py, tests) can
            thread it through to graph.get_state() calls without re-importing.
    """

    graph: CompiledStateGraph
    checkpointer: Any  # SqliteSaver | None — lazy import avoids hard dep


# ---------------------------------------------------------------------------
# Node name constants — single source of truth for graph wiring
# ---------------------------------------------------------------------------

_NODE_INTENT_GUARD = "intent_guard"
_NODE_PLANNER = "planner"
_NODE_RESOLVE_PATIENT = "resolve_patient"
_NODE_RETRIEVE_HISTORY = "retrieve_history"
_NODE_UPDATE_HISTORY = "update_history"
_NODE_BOOK_APPOINTMENT = "book_appointment"
_NODE_SEARCH_DISEASE = "search_disease"
_NODE_SUMMARIZER = "summarizer"
_NODE_DRAIN_UNKNOWN = "drain_unknown_task"

# Maps SubGoal.task values → graph node names (must stay in sync with SubGoal.task literals)
_TASK_TO_NODE: dict[str, str] = {
    "resolve_patient": _NODE_RESOLVE_PATIENT,
    "retrieve_history": _NODE_RETRIEVE_HISTORY,
    "update_history": _NODE_UPDATE_HISTORY,
    "book_appointment": _NODE_BOOK_APPOINTMENT,
    "search_disease": _NODE_SEARCH_DISEASE,
}

# All task node names — used to attach the same conditional edge to each
_TASK_NODES: list[str] = list(_TASK_TO_NODE.values())

# Fallback used when no search_fn is passed to build_graph().
# Medline results are still fetched via _combined_search; only web results are absent.
def _noop_search_fn(query: str, max_results: int) -> list[SearchResultItem]:  # noqa: ARG001
    """Return no web results. Used when build_graph() is called without a search_fn.

    Medline results are still included via the _combined_search wrapper in build_graph().
    """
    return []


# ---------------------------------------------------------------------------
# Routing functions (conditional edges — not graph nodes)
# ---------------------------------------------------------------------------


def _route_intent(state: HealthcareState) -> str:
    """Route after intent_guard: SAFE → planner, UNSAFE → END.

    Args:
        state: Current HealthcareState after intent_guard has run.

    Returns:
        Node name "planner" or the END sentinel.
    """
    return _NODE_PLANNER if state.get("intent_safe", False) else END


def _route_next_task(state: HealthcareState) -> str:
    """Route after planner or any task node: next task → matching node, empty → summarizer.

    Reads pending_tasks[0].task to determine which node to execute next.
    Unknown task values route to the drain node, which records a failure TaskResult
    and removes the bad task before routing continues — preventing silent partial execution.

    Args:
        state: Current HealthcareState after a planner or task node has run.

    Returns:
        Node name of the next task node, "drain_unknown_task" for unrecognised task
        types, or "summarizer" if the queue is empty.
    """
    pending = state.get("pending_tasks", [])
    if not pending:
        return _NODE_SUMMARIZER
    task_name = pending[0].task
    if task_name not in _TASK_TO_NODE:
        logger.warning(
            "[_route_next_task] Unknown task %r — routing to drain node. "
            "Valid tasks: %s",
            task_name,
            sorted(_TASK_TO_NODE),
        )
        return _NODE_DRAIN_UNKNOWN
    return _TASK_TO_NODE[task_name]


def _drain_unknown_task(state: HealthcareState) -> dict:
    """Record a failure for an unrecognized task type and drain it from the queue.

    Called when _route_next_task encounters a task name not present in _TASK_TO_NODE.
    Pops the offending task, appends a failure TaskResult, and sets the error field so
    the issue is visible in the Metrics tab and in the final summary. Routing continues
    via _route_next_task after this node, so any remaining valid tasks still execute.

    Args:
        state: Current HealthcareState with an unrecognised task at pending_tasks[0].

    Returns:
        State delta with updated pending_tasks, completed_tasks, and error.
        Returns empty dict if pending_tasks is unexpectedly empty (defensive guard).
    """
    pending = list(state.get("pending_tasks", []))
    if not pending:
        return {}
    unknown = pending.pop(0)
    error_msg = (
        f"Planner emitted unknown task {unknown.task!r}. "
        f"Valid tasks: {sorted(_TASK_TO_NODE)}. Task skipped."
    )
    logger.error("[drain_unknown_task] %s", error_msg)
    completed = list(state.get("completed_tasks", []))
    completed.append(TaskResult(task=unknown.task, success=False, error=error_msg))
    return {
        "pending_tasks": pending,
        "completed_tasks": completed,
        "error": error_msg,
    }


# ---------------------------------------------------------------------------
# Internal graph wiring — used directly by integration tests
# ---------------------------------------------------------------------------


def _compile_healthcare_graph(
    *,
    intent_guard_chain: Any,
    planner_chain: Any,
    history_chain: Any,
    search_chain: Any,
    summary_chain: Any,
    patient_db: PatientDB,
    appointment_db: AppointmentDB,
    vector_store: PatientVectorStore,
    db_path: str,
    search_fn: Callable[[str, int], list[SearchResultItem]],
    faiss_k: int = 3,
    max_recent_turns: int = 10,
    memory_summary_chain: Any | None = None,
    checkpointer: Any = None,
    max_summary_chars: int = 1500,
    cadence: int = 3,
) -> CompiledStateGraph:
    """Wire and compile the healthcare StateGraph from pre-built dependencies.

    Accepts chains and DB instances as explicit kwargs so integration tests can
    inject mock chains and real in-memory SQLite without touching the LLM factory.

    Args:
        intent_guard_chain: Built intent guard chain (prompt | llm | parser | normalizer).
        planner_chain: Built planner chain (prompt | llm.with_structured_output).
        history_chain: Built history chain (prompt | llm | parser).
        search_chain: Built search synthesis chain (prompt | llm | parser).
        summary_chain: Built summary chain (prompt | llm | parser).
        patient_db: PatientDB instance for resolver and history nodes.
        appointment_db: AppointmentDB instance for appointment node.
        vector_store: PatientVectorStore for FAISS retrieval and upsert.
        db_path: SQLite database file path (string, not Path).
        search_fn: Callable(query, max_results) → list[SearchResultItem].
        faiss_k: Number of FAISS nearest-neighbour chunks to retrieve. Default 3.
        max_recent_turns: Bound on recent chat turns read from checkpoint messages channel.
        memory_summary_chain: Rolling summary rollup chain for long-term memory (Phase 2+).
            Pass None in tests — summarizer degrades gracefully.
        checkpointer: SqliteSaver instance for persistent checkpointing, or None to compile
            without a checkpointer (in-memory only, no cross-session persistence).
        max_summary_chars: Character cap for rolling summary stored in checkpoint state.
        cadence: Refresh rolling summary every K successful turns.

    Returns:
        Compiled LangGraph StateGraph ready to invoke.
    """
    # Build node functions via closure factories — one construction per graph compile,
    # not per request. Each factory closes over its dependencies.
    intent_guard_node = make_intent_guard_node(intent_guard_chain)
    planner_node = make_planner_node(planner_chain, max_recent_turns)
    patient_resolver_node = make_patient_resolver_node(patient_db, db_path)
    history_retriever_node = make_history_retriever_node(
        patient_db, db_path, vector_store, history_chain, faiss_k=faiss_k
    )
    history_writer_node = make_history_writer_node(patient_db, db_path, vector_store)
    appointment_node = make_appointment_node(appointment_db, db_path)
    disease_search_node = make_disease_search_node(search_fn, search_chain)
    summarizer_node = make_summarizer_node(
        summary_chain,
        memory_summary_chain,
        max_recent_turns=max_recent_turns,
        max_summary_chars=max_summary_chars,
        cadence=cadence,
    )

    # Wire the StateGraph
    graph: StateGraph = StateGraph(HealthcareState)

    # Register nodes
    graph.add_node(_NODE_INTENT_GUARD, intent_guard_node)
    graph.add_node(_NODE_PLANNER, planner_node)
    graph.add_node(_NODE_RESOLVE_PATIENT, patient_resolver_node)
    graph.add_node(_NODE_RETRIEVE_HISTORY, history_retriever_node)
    graph.add_node(_NODE_UPDATE_HISTORY, history_writer_node)
    graph.add_node(_NODE_BOOK_APPOINTMENT, appointment_node)
    graph.add_node(_NODE_SEARCH_DISEASE, disease_search_node)
    graph.add_node(_NODE_SUMMARIZER, summarizer_node)
    graph.add_node(_NODE_DRAIN_UNKNOWN, _drain_unknown_task)

    # Entry point
    graph.add_edge(START, _NODE_INTENT_GUARD)

    # After intent_guard: route on intent_safe flag
    _intent_path_map = {_NODE_PLANNER: _NODE_PLANNER, END: END}
    graph.add_conditional_edges(_NODE_INTENT_GUARD, _route_intent, _intent_path_map)

    # After planner or any task node: route to next task, drain unknown, or summarizer.
    # _NODE_DRAIN_UNKNOWN must be in the map so LangGraph validates the edge target.
    _task_path_map = {
        **_TASK_TO_NODE,
        _NODE_SUMMARIZER: _NODE_SUMMARIZER,
        _NODE_DRAIN_UNKNOWN: _NODE_DRAIN_UNKNOWN,
    }
    graph.add_conditional_edges(_NODE_PLANNER, _route_next_task, _task_path_map)

    # After each task node: route to next task or summarizer (shared routing function)
    for task_node_name in _TASK_NODES:
        graph.add_conditional_edges(task_node_name, _route_next_task, _task_path_map)

    # After drain: re-enter routing so remaining valid tasks still execute
    graph.add_conditional_edges(_NODE_DRAIN_UNKNOWN, _route_next_task, _task_path_map)

    # Summarizer is terminal
    graph.add_edge(_NODE_SUMMARIZER, END)

    # checkpointer=None compiles without persistence (in-memory); SqliteSaver enables
    # cross-session state snapshots keyed on configurable.thread_id (Phase 1+).
    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_graph(
    settings: Settings,
    *,
    search_fn: Callable[[str, int], list[SearchResultItem]] | None = None,
) -> BuildGraphResult:
    """Build and compile the healthcare StateGraph from application settings.

    Constructs LLM singletons (one per temperature variant), chain singletons,
    DB instances, and node functions, then compiles the graph. Call once at
    application startup — not per request.

    Args:
        settings: Loaded and validated Settings instance (from load_settings()).
        search_fn: Optional callable(query, max_results) → list[SearchResultItem].
            If omitted, only Medline results are included; web search is disabled.

    Returns:
        ``BuildGraphResult(graph, checkpointer)`` — graph for invoke; checkpointer is
        a ``SqliteSaver`` instance when ``settings.db.checkpointing_enabled`` is True,
        otherwise ``None``.
    """
    cfg = settings.provider_config
    provider = settings.provider_name

    logger.info("[build_graph] provider=%s model=%s", provider, cfg.model)

    # One LLM instance per temperature/token-budget variant — never rebuilt per request.
    # reasoning_effort and reasoning_format are forwarded for Groq gpt-oss models only.
    llm_guard = get_llm(
        provider, cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens_guard,
        reasoning_effort=cfg.reasoning_effort,
        reasoning_format=cfg.reasoning_format,
    )
    llm_planner = get_llm(
        provider, cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens_planner,
        reasoning_effort=cfg.reasoning_effort,
        reasoning_format=cfg.reasoning_format,
    )
    llm_history = get_llm(
        provider, cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens_history,
        reasoning_effort=cfg.reasoning_effort,
        reasoning_format=cfg.reasoning_format,
    )
    llm_search = get_llm(
        provider, cfg.model,
        temperature=cfg.temperature_search,
        max_tokens=cfg.max_tokens_search,
        reasoning_effort=cfg.reasoning_effort,
        reasoning_format=cfg.reasoning_format,
    )
    llm_summarizer = get_llm(
        provider, cfg.model,
        temperature=cfg.temperature_summarizer,
        max_tokens=cfg.max_tokens_summarizer,
        reasoning_effort=cfg.reasoning_effort,
        reasoning_format=cfg.reasoning_format,
    )
    llm_memory_summary = get_llm(
        provider, cfg.model,
        temperature=cfg.temperature_summarizer,
        max_tokens=cfg.max_tokens_memory_summary,
        reasoning_effort=cfg.reasoning_effort,
        reasoning_format=cfg.reasoning_format,
    )

    # Chain singletons — built once, injected into node factories
    intent_guard_chain = build_intent_guard_chain(llm_guard)
    planner_chain = build_planner_chain(llm_planner)
    history_chain = build_history_chain(llm_history)
    search_chain = build_search_chain(llm_search)
    summary_chain = build_summary_chain(llm_summarizer)
    memory_summary_chain = build_memory_summary_chain(llm_memory_summary)

    # DB singletons — one instance per graph (not per request)
    db_path = str(settings.db.sqlite_path)
    patient_db = PatientDB()
    appointment_db = AppointmentDB()

    # Embeddings and vector store — OpenAI embeddings require OPENAI_API_KEY at runtime
    embeddings = OpenAIEmbeddings(model=settings.embeddings.model)
    vector_store = PatientVectorStore(str(settings.db.faiss_path), embeddings)

    # Combine web search with Medline — Medline results placed first so the LLM
    # sees peer-reviewed PubMed abstracts before consumer web content.
    # Per Experiment 2 findings rec. 2: always fetch both regardless of result count;
    # rec. 3: Medline (higher authority) precedes web results in the prompt.
    web_fn = search_fn or _noop_search_fn

    def _combined_search(query: str, max_results: int) -> list[SearchResultItem]:
        medline_results = search_medline(query, max_results)
        web_results = web_fn(query, max_results)
        return medline_results + web_results

    # Lazy import: SqliteSaver is an optional dependency (langgraph-checkpoint-sqlite).
    # Only imported when checkpointing_enabled=True to avoid a hard dep at startup.
    # from_conn_string is a context manager; construct directly with a persistent connection
    # so the saver outlives the build_graph call.
    checkpointer = None
    if settings.db.checkpointing_enabled:
        import sqlite3  # stdlib — safe to import lazily here with SqliteSaver

        from langgraph.checkpoint.sqlite import SqliteSaver  # lazy import — optional dep

        # Intentionally persistent connection — SqliteSaver holds it for the process
        # lifetime so the checkpointer survives across graph invocations.  No explicit
        # close() is needed for a single-process CLI/Streamlit deployment.
        _conn = sqlite3.connect(settings.db.checkpoint_db_path, check_same_thread=False)
        checkpointer = SqliteSaver(_conn)
        checkpointer.setup()  # creates checkpoints + writes tables if not present

    compiled = _compile_healthcare_graph(
        intent_guard_chain=intent_guard_chain,
        planner_chain=planner_chain,
        history_chain=history_chain,
        search_chain=search_chain,
        summary_chain=summary_chain,
        patient_db=patient_db,
        appointment_db=appointment_db,
        vector_store=vector_store,
        db_path=db_path,
        search_fn=_combined_search,
        max_recent_turns=settings.db.max_recent_turns,
        memory_summary_chain=memory_summary_chain,
        checkpointer=checkpointer,
        max_summary_chars=settings.db.max_summary_chars,
        cadence=settings.db.memory_summary_cadence_turns,
    )
    return BuildGraphResult(graph=compiled, checkpointer=checkpointer)
