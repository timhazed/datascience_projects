"""Sync Pipeline LangGraph StateGraph — Spec §4.1, §6, §12.5.

build_sync_graph() compiles the StateGraph for the sync_repository MCP tool.
All node factories are called once here and the compiled graph is stored as a module-level
singleton in src/server.py — never recompiled per request.

Graph topology (updated for Spec §6A–D):
  START → delta_extractor
  delta_extractor → END (error or already_up_to_date)
  delta_extractor → cache_filter
  cache_filter → pii_sanitizer (new/changed files remain)
  cache_filter → chroma_upsert_quarantined (all files already cached)
  pii_sanitizer → multimodal_parser → chunk_dispatcher
  chunk_dispatcher → [Send("summarize_and_upsert", {chunk, ...}) × N] → chroma_upsert_quarantined
  chunk_dispatcher → chroma_upsert_quarantined (empty-chunks fast path)
  chroma_upsert_quarantined → sha_store_update → END

Each Send target (summarize_and_upsert) summarizes and upserts its chunk immediately,
eliminating the old fan-in bottleneck where no writes occurred until ALL LLM calls
finished. Quarantined files are upserted by chroma_upsert_quarantined after the fan-out
converges (or on the empty-chunks path). sha_store_update writes commit_sha only on
clean completion — never on partial failures.
"""

import logging
from collections.abc import Callable

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from src.agents.cache_filter import make_cache_filter_node
from src.agents.chroma_upsert_quarantined import make_chroma_upsert_quarantined_node
from src.agents.chunk_dispatcher import make_chunk_dispatcher_node
from src.agents.delta_extractor import make_delta_extractor_node
from src.agents.multimodal_parser import make_multimodal_parser_node
from src.agents.pii_sanitizer import make_pii_sanitizer_node
from src.agents.sha_store_update import make_sha_store_update_node
from src.agents.summarize_and_upsert import ProgressCallback, make_summarize_and_upsert_node
from src.config.settings import Settings
from src.db.chroma_client import ChromaLibrarianClient
from src.middleware.pii_filter import PIIFilter
from src.models.sync_state import SyncState

logger = logging.getLogger(__name__)


def build_sync_graph(
    llm: BaseChatModel,
    chroma: ChromaLibrarianClient,
    pii: PIIFilter,
    git_client: Callable[[str, str], list[str]],
    *,
    ollama_concurrency: int = 1,
    interrupt_before: list[str] | None = None,
    progress_callback: ProgressCallback | None = None,
    sha_store=None,
    pii_progress_callback: Callable[[str, int, int], None] | None = None,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Build and compile the Sync Pipeline StateGraph.

    All node factories are called once — dependencies injected at build time,
    never reconstructed per request. The compiled graph is safe to invoke concurrently.

    Args:
        llm: BaseChatModel instance (temperature=0.0, num_ctx=8192) — summarize_and_upsert.
        chroma: ChromaLibrarianClient singleton.
        pii: PIIFilter singleton.
        git_client: Callable(repo_url, branch) → list[str] returning changed file paths.
            Wraps compute_delta() + Repo clone/fetch in production; inject a lambda in tests.
        ollama_concurrency: Maximum simultaneous LLM calls — must match OLLAMA_NUM_PARALLEL
            on the Ollama server. Passed to make_summarize_and_upsert_node(). Default 1 (serial).
            Use 2 for M4 Max 128GB per §12.3.
        interrupt_before: Node names to pause at for human-in-the-loop approval (spec §6.6).
            Defaults to [] (disabled).
        progress_callback: Optional callable forwarded to make_summarize_and_upsert_node so
            the server can update live job progress during streaming upserts. See
            summarize_and_upsert.ProgressCallback for the signature.
        sha_store: Optional SHAStore instance. If provided, wires into delta_extractor for
            the SHA freshness gate and into sha_store_update to write commit_sha on success.
        pii_progress_callback: Optional callable forwarded to make_pii_sanitizer_node so
            the server can report per-file PII scan progress. Signature:
            (job_id: str, pii_files_done: int, pii_files_total: int) -> None.
        settings: Optional Settings instance. If None, a default Settings() is constructed
            for cache_filter_node and delta_extractor's sha_freshness_enabled flag.

    Returns:
        Compiled LangGraph StateGraph ready for .invoke() calls.
        Callers must pass recursion_limit in the config:
            graph.invoke(state, config=RunnableConfig(recursion_limit=10_000))
    """
    if interrupt_before is None:
        interrupt_before = []

    # Use injected settings or construct a default; Settings() reads from env/.env file.
    _settings: Settings = settings or Settings()

    # ── Build node functions from factories ───────────────────────────────────
    delta_extractor = make_delta_extractor_node(
        git_client,
        sha_store=sha_store,
        settings=_settings,
    )
    cache_filter = make_cache_filter_node(chroma, _settings)
    pii_sanitizer_node = make_pii_sanitizer_node(pii, pii_progress_callback=pii_progress_callback)
    multimodal_parser = make_multimodal_parser_node()
    chunk_dispatcher = make_chunk_dispatcher_node()
    summarize_and_upsert = make_summarize_and_upsert_node(
        llm,
        chroma,
        ollama_concurrency=ollama_concurrency,
        progress_callback=progress_callback,
    )
    chroma_upsert_quarantined = make_chroma_upsert_quarantined_node(chroma)
    sha_store_update = make_sha_store_update_node(sha_store)

    # ── Routing functions ─────────────────────────────────────────────────────

    def route_after_delta(state: SyncState) -> str:
        """Route to END on error or already_up_to_date; else proceed to cache_filter."""
        if state.get("error"):
            return END
        if state.get("already_up_to_date"):
            return END
        return "cache_filter"

    def route_after_cache_filter(state: SyncState) -> str:
        """Route to pii_sanitizer if files remain; else directly to chroma_upsert_quarantined."""
        if not state.get("changed_files"):
            return "chroma_upsert_quarantined"
        return "pii_sanitizer"

    def route_after_pii(state: SyncState) -> str:
        """Route to END on error (pii_sanitizer failed), else continue."""
        return END if state.get("error") else "multimodal_parser"

    def route_after_parse(state: SyncState) -> str:
        """Route to END on error (multimodal_parser failed), else continue."""
        return END if state.get("error") else "chunk_dispatcher"

    def route_after_dispatch(state: SyncState) -> str | list[Send]:
        """Fan out to summarize_and_upsert via Send, or route to quarantined-only path.

        Emits one Send per parsed_chunk — each carries the full metadata payload so the
        Send target does not need to read from SyncState directly (§12.5 Send payload spec):
            chunk: ParsedChunk
            repo_url, branch, commit_sha: forwarded from SyncState for ChromaDB metadata
            _job_id: forwarded for progress callback
            chunk_index: 1-based position of this chunk in the batch
            chunks_total: total Send targets dispatched (for progress reporting denominator)
            file_shas: forwarded so summarize_and_upsert can write source_file_sha metadata

        When parsed_chunks is empty (all files quarantined or repo is empty), routes
        directly to chroma_upsert_quarantined to upsert the quarantine records.
        """
        chunks = state.get("parsed_chunks", [])
        if not chunks:
            return "chroma_upsert_quarantined"

        repo_url = state.get("repo_url", "")
        branch = state.get("branch", "main")
        commit_sha = state.get("commit_sha", "")
        job_id = state.get("_job_id")
        chunks_total = len(chunks)
        file_shas = state.get("file_shas", {})

        return [
            Send(
                "summarize_and_upsert",
                {
                    "chunk": chunk,
                    "repo_url": repo_url,
                    "branch": branch,
                    "commit_sha": commit_sha,
                    "_job_id": job_id,
                    "chunk_index": idx + 1,
                    "chunks_total": chunks_total,
                    "file_shas": file_shas,
                    # Required: trace_events must be present in every Send payload because
                    # SyncState.trace_events uses operator.add — LangGraph raises KeyError
                    # when reducing a required field absent from the sub-state dict.
                    "trace_events": [],
                },
            )
            for idx, chunk in enumerate(chunks)
        ]

    # ── Assemble graph ────────────────────────────────────────────────────────
    builder = StateGraph(SyncState)

    builder.add_node("delta_extractor", delta_extractor)
    builder.add_node("cache_filter", cache_filter)
    builder.add_node("pii_sanitizer", pii_sanitizer_node)
    builder.add_node("multimodal_parser", multimodal_parser)
    builder.add_node("chunk_dispatcher", chunk_dispatcher)
    builder.add_node("summarize_and_upsert", summarize_and_upsert)
    builder.add_node("chroma_upsert_quarantined", chroma_upsert_quarantined)
    builder.add_node("sha_store_update", sha_store_update)

    # Entry point
    builder.add_edge(START, "delta_extractor")

    # SHA freshness gate: END on error or already_up_to_date, else cache_filter
    builder.add_conditional_edges("delta_extractor", route_after_delta)

    # Pre-PII cache filter: skip files already indexed
    builder.add_conditional_edges("cache_filter", route_after_cache_filter)

    # PII → parse → dispatch chain
    builder.add_conditional_edges("pii_sanitizer", route_after_pii)
    builder.add_conditional_edges("multimodal_parser", route_after_parse)

    # chunk_dispatcher: fan-out or direct route to quarantined node
    builder.add_conditional_edges(
        "chunk_dispatcher",
        route_after_dispatch,
        {"chroma_upsert_quarantined": "chroma_upsert_quarantined"},
    )

    # All summarize_and_upsert Send targets converge into chroma_upsert_quarantined
    # via operator.add on upsert_results; quarantined files are always upserted last.
    builder.add_edge("summarize_and_upsert", "chroma_upsert_quarantined")

    # SHA write guard — only writes on clean completion
    builder.add_edge("chroma_upsert_quarantined", "sha_store_update")
    builder.add_edge("sha_store_update", END)

    return builder.compile(interrupt_before=interrupt_before)
