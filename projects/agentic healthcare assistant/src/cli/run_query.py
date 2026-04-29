"""run_query — healthcare-cli entry point (Part 1 acceptance gate).

Parses arguments, initialises the database, builds the graph, invokes it with
the user's query, and prints structured output to stdout.

Usage:
    poetry run healthcare-cli --query "What medications is Ramesh on?" --show-plan
    poetry run healthcare-cli --scenario ckd --show-trace
    poetry run healthcare-cli --scenario all --strict
    poetry run healthcare-cli --health
    poetry run healthcare-cli --list-db
    poetry run healthcare-cli --init-db

Exit codes:
    0 — success (including --init-db after DB/FAISS setup)
    1 — any scenario produced final_summary=None (fatal graph failure)
        OR --strict and any TaskResult.success is False
        OR --health detected a configuration problem
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings
from langgraph.errors import GraphRecursionError

from src.config.settings import Settings, load_settings
from src.db.initializer import _ensure_db_initialized
from src.graph.healthcare_graph import build_graph
from src.utils.graph_config import build_invoke_config
from src.utils.search_provider import SearchProvider

# Built-in scenario presets — same queries as Tab 5 interactive scenarios
_SCENARIOS: dict[str, str] = {
    "ckd": (
        "My 70-year-old father has chronic kidney disease. "
        "Book a nephrologist and summarize treatment options."
    ),
    "hypertension": (
        "Retrieve Ramesh Kulkarni's history and update his medication to Telmisartan 80mg."
    ),
    "diabetes": (
        "Find the latest diabetes management guidelines and book a follow-up for David Thompson."
    ),
}

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# --init-db
# ---------------------------------------------------------------------------
def _init_db(settings, embeddings) -> None:
    """Run full initializer: schema, seeds, patient/FAISS load when needed."""
    _ensure_db_initialized(settings, embeddings)


# ---------------------------------------------------------------------------
# --health
# ---------------------------------------------------------------------------


def _health_check(settings) -> bool:
    """Check environment variables, config, and DB/FAISS state without invoking the graph.

    Args:
        settings: Loaded Settings instance.

    Returns:
        True if all checks pass, False if any check fails.
    """
    all_ok = True

    def _check(label: str, passed: bool, detail: str = "") -> None:
        nonlocal all_ok
        icon = "✓" if passed else "✗"
        suffix = f"  ({detail})" if detail else ""
        print(f"  {icon}  {label}{suffix}")
        if not passed:
            all_ok = False

    # --- API keys ---
    print("\n[HEALTH] API keys")
    groq_key = bool(os.getenv("GROQ_API_KEY"))
    openai_key = bool(os.getenv("OPENAI_API_KEY"))

    if settings.provider_name == "groq":
        _check("GROQ_API_KEY", groq_key)
    else:
        _check("OPENAI_API_KEY (LLM)", openai_key)
    # Embeddings always use OpenAI regardless of LLM provider
    _check("OPENAI_API_KEY (embeddings — always required)", openai_key)

    search_p = settings.search.provider
    if search_p == "serper":
        _check("SERPER_API_KEY", bool(os.getenv("SERPER_API_KEY")))
    else:
        _check("SERPAPI_API_KEY", bool(os.getenv("SERPAPI_API_KEY")))

    # --- Config ---
    print("\n[HEALTH] Configuration")
    cfg = settings.provider_config
    _check(f"Provider: {settings.provider_name}", True, cfg.model)
    _check(f"Search provider: {search_p}", True)

    # Warn when token budgets look too low for reasoning models
    reasoning_model = bool(cfg.reasoning_effort or cfg.reasoning_format)
    if reasoning_model:
        guard_ok = cfg.max_tokens_guard >= 256
        _check(
            f"max_tokens_guard: {cfg.max_tokens_guard}",
            guard_ok,
            "WARNING: < 256 causes empty output on reasoning models" if not guard_ok else "ok",
        )
        planner_ok = cfg.max_tokens_planner >= 1024
        _check(
            f"max_tokens_planner: {cfg.max_tokens_planner}",
            planner_ok,
            "WARNING: < 1024 may truncate planner JSON on reasoning models"
            if not planner_ok
            else "ok",
        )

    # --- DB / FAISS ---
    print("\n[HEALTH] Storage")
    db_path = str(settings.db.sqlite_path)
    db_exists = os.path.exists(db_path) and os.path.getsize(db_path) > 0
    _check("SQLite DB exists and non-empty", db_exists, db_path)

    faiss_path = str(settings.db.faiss_path)
    faiss_exists = os.path.exists(faiss_path)
    _check("FAISS index directory exists", faiss_exists, faiss_path)

    # --- Summary ---
    print()
    if all_ok:
        print("[HEALTH] All checks passed — ready to run.")
    else:
        print("[HEALTH] Issues found — fix the above before running --scenario all.")
    return all_ok


# ---------------------------------------------------------------------------
# --list-db
# ---------------------------------------------------------------------------


def _list_db(settings) -> None:
    """Print patients, specialties, and available appointment slot counts from SQLite.

    Args:
        settings: Loaded Settings instance.
    """
    db_path = str(settings.db.sqlite_path)

    if not os.path.exists(db_path):
        print(f"[LIST-DB] DB not found at {db_path} — run a scenario first to seed it.")
        return

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Patient count
        print("\n[LIST-DB] Patients")
        cur.execute("SELECT COUNT(*) FROM patients")
        print(f"  Total patients: {cur.fetchone()[0]}")

        # Specialty breakdown — total slots and available slots
        print("\n[LIST-DB] Appointment slots by specialty")
        cur.execute("""
            SELECT
                specialty,
                COUNT(slot_id)                                       AS total_slots,
                SUM(CASE WHEN booked = 0 THEN 1 ELSE 0 END)         AS available
            FROM slots
            GROUP BY specialty
            ORDER BY specialty
        """)
        rows = cur.fetchall()
        if rows:
            print(f"  {'Specialty':<28} {'Total':>7} {'Available':>11}")
            print(f"  {'-' * 28} {'-' * 7} {'-' * 11}")
            for specialty, total, available in rows:
                flag = "  ← no available slots" if available == 0 else ""
                print(f"  {specialty:<28} {total:>7} {available:>11}{flag}")
        else:
            print("  No doctors/slots found — DB may not be seeded yet.")

        # FAISS directory status
        faiss_path = str(settings.db.faiss_path)
        faiss_exists = os.path.exists(faiss_path)
        faiss_status = "present" if faiss_exists else "NOT FOUND"
        print(f"\n[LIST-DB] FAISS index: {faiss_status} ({faiss_path})")

        conn.close()

    except sqlite3.Error as exc:
        print(f"[LIST-DB] SQLite error: {exc}")


# ---------------------------------------------------------------------------
# Graph invocation helpers
# ---------------------------------------------------------------------------


def _build_initial_state(query: str, patient_hint: str | None) -> dict:
    """Build the initial HealthcareState dict for graph.invoke().

    Args:
        query: Natural language query string.
        patient_hint: Optional patient name to inject as context into the planner.

    Returns:
        HealthcareState-compatible dict with all required keys populated.
    """
    # patient_hint is prepended to the query so the planner sees it as context.
    # The resolver still runs to confirm the match — the hint is not trusted as authoritative.
    effective_query = query
    if patient_hint:
        effective_query = f"[Patient context: {patient_hint}]\n{query}"

    return {
        "user_query": effective_query,
        "patient_id": None,
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


def _print_result(
    query: str,
    result: dict,
    *,
    show_plan: bool,
    show_trace: bool,
    latency_s: float,
) -> None:
    """Print structured CLI output for one graph invocation.

    Always prints: [QUERY], [PLAN], [RESULT], [TRACE], [STATUS], task breakdown.
    Full PlannerOutput JSON printed only when show_plan=True.
    Full trace entries printed only when show_trace=True.

    Args:
        query: The user query that was run.
        result: The HealthcareState dict returned by graph.invoke().
        show_plan: Whether to print the full PlannerOutput JSON.
        show_trace: Whether to print the full trace list.
        latency_s: Wall-clock time in seconds for the graph.invoke() call.
    """
    print(f"\n[QUERY]  {query}")

    # Plan — always show task sequence; full JSON only with --show-plan
    planner_output = result.get("planner_output")
    if planner_output:
        task_seq = " → ".join(sg.task for sg in planner_output.sub_goals)
        print(f"[PLAN]   {task_seq}")
        if show_plan:
            print(json.dumps(planner_output.model_dump(), indent=2, default=str))
    else:
        print("[PLAN]   (no plan — intent may have been UNSAFE or planning failed)")

    # Result
    final_summary = result.get("final_summary")
    if final_summary:
        print(f"[RESULT] {final_summary}")
    else:
        error = result.get("error", "unknown error")
        print(f"[RESULT] (no summary — {error})")

    # Trace — compact node sequence always; full entries only with --show-trace
    trace = result.get("trace", [])
    if show_trace and trace:
        print(f"[TRACE]  {' → '.join(trace)}")
    elif trace:
        nodes = [entry.split(":")[0] for entry in trace]
        print(f"[TRACE]  {' → '.join(nodes)}")

    # Status line
    completed = result.get("completed_tasks", [])
    error_count = sum(1 for t in completed if not t.success)
    status_icon = "✓" if final_summary else "✗"
    print(
        f"[STATUS] {status_icon} {len(completed)} tasks completed"
        f"  |  {error_count} errors"
        f"  |  latency: {latency_s:.1f} s"
    )

    # Task breakdown — always printed so each node's outcome is visible
    for t in completed:
        icon = "✓" if t.success else "✗"
        detail = f": {t.error}" if not t.success and t.error else ""
        print(f"  {icon}  {t.task}{detail}")


def _run_single(
    graph,
    query: str,
    patient_hint: str | None,
    *,
    show_plan: bool,
    show_trace: bool,
    settings: Settings,
) -> tuple[bool, bool]:
    """Run one query through the graph and print results.

    Args:
        graph: Compiled LangGraph StateGraph.
        query: Natural language query string.
        patient_hint: Optional patient name context hint.
        show_plan: Whether to print full PlannerOutput JSON.
        show_trace: Whether to print full trace list.
        settings: Loaded application settings — provides recursion_limit and thread_id.
    Returns:
        Tuple of (has_final_summary, has_task_errors).
        has_final_summary — True when final_summary is non-empty (graph completed).
        has_task_errors   — True when any TaskResult.success is False.
    """
    initial_state = _build_initial_state(query, patient_hint)
    invoke_cfg = build_invoke_config(patient_hint, settings)

    start = time.perf_counter()
    try:
        result = graph.invoke(initial_state, invoke_cfg)
    except GraphRecursionError:
        recursion_limit = settings.graph.recursion_limit
        logger.error(
            "[run_query] GraphRecursionError — recursion_limit=%s exceeded (supersteps). "
            "Often a stuck task queue or an overly long plan. See GRAPH_RECURSION_LIMIT.",
            recursion_limit,
            exc_info=True,
        )
        latency_s = time.perf_counter() - start
        result = {
            "final_summary": None,
            "error": (
                f"Graph execution stopped: step limit reached (recursion_limit={recursion_limit}). "
                "Increase graph.recursion_limit in config.yaml for long plans, or investigate "
                "infinite routing."
            ),
            "planner_output": None,
            "completed_tasks": [],
            "trace": [],
        }
        _print_result(query, result, show_plan=show_plan, show_trace=show_trace, 
                    latency_s=latency_s, graph=graph)
        return False, True

    latency_s = time.perf_counter() - start

    _print_result(query, result, show_plan=show_plan, show_trace=show_trace, latency_s=latency_s)

    has_summary = result.get("final_summary") is not None
    has_errors = any(not t.success for t in result.get("completed_tasks", []))
    return has_summary, has_errors


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point — parse arguments, build graph, invoke, print results.

    Exit code 0: all scenarios completed (final_summary non-empty); no strict violations.
    Exit code 1: any scenario missing final_summary; or --strict with task errors;
                 or --health detected a problem.
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="healthcare-cli",
        description="Agentic Healthcare Assistant — Part 1 CLI",
    )

    # Mutually exclusive group: exactly one input mode required
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--query",
        metavar="TEXT",
        help="Natural language query to send to the graph",
    )
    input_group.add_argument(
        "--scenario",
        choices=[*_SCENARIOS.keys(), "all"],
        metavar="{ckd,hypertension,diabetes,all}",
        help="Run a built-in test scenario (or 'all' to run all three sequentially)",
    )
    input_group.add_argument(
        "--health",
        action="store_true",
        help="Check API keys, config, and DB/FAISS status without invoking the graph",
    )
    input_group.add_argument(
        "--list-db",
        action="store_true",
        dest="list_db",
        help="Show patient count, available specialties, and slot counts from SQLite",
    )
    input_group.add_argument(
        "--init-db",
        action="store_true",
        default=False,
        help="Ensure the db is initialized and exit",
    )

    parser.add_argument(
        "--patient",
        metavar="TEXT",
        default=None,
        help="Optional patient name hint (maps to patient_context in planner prompt)",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "openai"],
        default=None,
        help="LLM provider override (default: from config.yaml)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit 1 if any TaskResult.success is False (not just missing final_summary)",
    )
    parser.add_argument(
        "--show-trace",
        action="store_true",
        default=False,
        help="Print the full HealthcareState.trace after execution",
    )
    parser.add_argument(
        "--show-plan",
        action="store_true",
        default=False,
        help="Print the full PlannerOutput JSON after planning",
    )

    args = parser.parse_args()

    settings = load_settings()

    # --health / --list-db: no embeddings or graph. --init-db: run initializer then exit.
    if args.health:
        ok = _health_check(settings)
        sys.exit(0 if ok else 1)

    if args.list_db:
        _list_db(settings)
        sys.exit(0)

    # Validate --provider against the configured provider.
    # Settings enforces exactly one provider; switching requires updating config.yaml.
    if args.provider and args.provider != settings.provider_name:
        print(
            f"Error: --provider '{args.provider}' requested, but config.yaml configures "
            f"'{settings.provider_name}'. Update config.yaml to switch providers.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build embeddings once — required by _ensure_db_initialized for FAISS
    embeddings = OpenAIEmbeddings(model=settings.embeddings.model)

    # Seed DB and FAISS on first run (idempotent)
    _init_db(settings, embeddings)

    if args.init_db:
        db_path = str(settings.db.sqlite_path)
        faiss_path = str(settings.db.faiss_path)
        print(f"DB initialized — SQLite: {db_path}")
        print(f"FAISS index directory: {faiss_path}")
        sys.exit(0)

    # Wire SearchProvider from settings; passed into build_graph as search_fn
    search_provider = SearchProvider(settings.search.provider)

    result = build_graph(settings, search_fn=search_provider.search)
    graph = result.graph

    # Determine which queries to run
    if args.query:
        queries = [(args.query, args.patient)]
    elif args.scenario == "all":
        queries = [(_SCENARIOS[key], None) for key in _SCENARIOS]
    else:
        queries = [(_SCENARIOS[args.scenario], None)]

    # Run all queries and collect results
    scenario_mode = args.scenario == "all"
    passed = 0
    failed = 0
    error_free = 0  # passed AND no task-level errors (for strict reporting)

    for query, patient_hint in queries:
        has_summary, has_errors = _run_single(
            graph,
            query,
            patient_hint,
            show_plan=args.show_plan,
            show_trace=args.show_trace,
            settings=settings,
        )
        if has_summary:
            passed += 1
            if not has_errors:
                error_free += 1
            if scenario_mode:
                if has_errors:
                    print("  [PASS] scenario succeeded (with task errors — run --strict to fail)")
                else:
                    print("  [PASS] scenario succeeded — error-free")
        else:
            failed += 1
            if scenario_mode:
                print(f"  [FAIL] final_summary was None for: {query[:60]}...")

    # Scenario gate summary for --scenario all
    if scenario_mode:
        total = len(queries)
        strict_line = (
            f"  |  strict (error-free): {error_free}/{total}" if args.strict else ""
        )
        print(f"\n=== Scenario gate: {passed}/{total} passed{strict_line} ===")

    # Exit code: 1 if any scenario missing summary, or --strict with any task errors
    any_fatal = failed > 0
    any_strict_fail = args.strict and (error_free < passed or failed > 0)
    sys.exit(1 if (any_fatal or any_strict_fail) else 0)


if __name__ == "__main__":
    main()
