"""Singleton construction for the Developer Memory MCP server.

All long-lived, process-global objects are constructed here — once at import time —
and then referenced by name throughout the application. Nothing is rebuilt per request.

Construction sequence (matches spec §6.1 startup order):
  1. _check_memory() — preflight guard; raises RuntimeError if RAM is insufficient.
  2. LLM singletons — four profiles covering the five graph pipelines.
  3. ChromaLibrarianClient — the only ChromaDB client in the process.
  4. PIIFilter — spaCy model loaded here; largest single allocation.
  5. SHAStore — thread-safe JSON file store for last-synced commit SHAs.
  6. All five LangGraph graphs — compiled once with injected dependencies.
  7. _GRAPH_CONFIG — shared RunnableConfig with recursion_limit=10_000.
  8. mcp = FastMCP(...) — the MCP server instance decorated by routes and tools.

Circular-import note:
  container → sync_jobs (for _sync_progress_callback, _pii_progress_callback)
  sync_jobs → container (lazy import inside run_sync_job body)
  This is intentional and spec-prescribed. The lazy import in sync_jobs breaks
  the cycle — container is imported first at server startup, then sync_jobs is
  already in sys.modules when run_sync_job executes.
"""

import os

from fastmcp import FastMCP
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig

from src.config.settings import Settings
from src.db.chroma_client import ChromaLibrarianClient
from src.db.sha_store import SHAStore
from src.graphs.diff_graph import build_diff_graph
from src.graphs.persona_graph import build_persona_graph
from src.graphs.query_graph import build_query_graph
from src.graphs.skills_graph import build_skills_graph
from src.graphs.sync_graph import build_sync_graph
from src.llm.factory import get_llm
from src.mcp_server.git_client import make_git_client
from src.mcp_server.startup import _check_memory, _log_memory, logger
from src.mcp_server.sync_jobs import _pii_progress_callback, _sync_progress_callback
from src.middleware.pii_filter import PIIFilter

# ── Preflight ─────────────────────────────────────────────────────────────────

# Raises RuntimeError if available RAM is below the minimum threshold.
_model: str = _check_memory()

# Skills synthesis model — independently configurable via OLLAMA_SKILLS_MODEL.
# Falls back to _model if the env var is unset or empty.
# gpt-oss:20b is recommended for skills export jobs (see docs/model_evaluation_report.md).
_skills_model: str = Settings().ollama_skills_model or _model
logger.info("Skills synthesis model: %s", _skills_model)

# ── LLM singletons ────────────────────────────────────────────────────────────
# Named singletons — one per (temperature, num_ctx) profile required across the five graphs.
# skills_synthesizer requires num_ctx=16384: 4096-token output budget + skills input + system
# prompt easily exceeds the default 8192 window.

_llm_exact: BaseChatModel = get_llm(model=_model, temperature=0.0, num_ctx=8192)  # summarize_and_upsert, coaching_analyzer
_llm_light: BaseChatModel = get_llm(model=_model, temperature=0.1, num_ctx=8192)  # snippet_summarizer
_llm_creative: BaseChatModel = get_llm(model=_model, temperature=0.3, num_ctx=8192)  # persona_synthesizer
_llm_balanced: BaseChatModel = get_llm(model=_skills_model, temperature=0.2, num_ctx=16384, num_predict=4096)  # skills_synthesizer

# ── Infrastructure singletons ─────────────────────────────────────────────────

_chroma: ChromaLibrarianClient = ChromaLibrarianClient()
_pii: PIIFilter = PIIFilter()
_log_memory("after PIIFilter init")  # spaCy model loaded here — first big allocation

# SHA store singleton — thread-safe JSON file store for last-synced commit SHAs.
# Lives at {CHROMA_DATA_PATH}/sha_store.json (same Docker volume as ChromaDB).
_sha_store: SHAStore = SHAStore()

# ── Graph singletons ──────────────────────────────────────────────────────────
# All graphs compiled once at startup with injected singletons.

_ollama_concurrency: int = int(os.environ.get("OLLAMA_NUM_PARALLEL", "1"))
logger.info("OLLAMA_NUM_PARALLEL=%d — semaphore will gate simultaneous LLM calls", _ollama_concurrency)

_sync_graph = build_sync_graph(
    llm=_llm_exact,
    chroma=_chroma,
    pii=_pii,
    git_client=make_git_client(),
    ollama_concurrency=_ollama_concurrency,
    progress_callback=_sync_progress_callback,
    sha_store=_sha_store,
    pii_progress_callback=_pii_progress_callback,
)
_log_memory("after graph compilation")
_query_graph = build_query_graph(llm=_llm_light, chroma=_chroma)
_persona_graph = build_persona_graph(llm=_llm_creative, chroma=_chroma)
_diff_graph = build_diff_graph(llm=_llm_exact, chroma=_chroma)
_skills_graph = build_skills_graph(llm=_llm_balanced, chroma=_chroma, sha_store=_sha_store)

# Shared RunnableConfig — recursion_limit enforced at invocation, not compile time (spec §6.1).
# LangGraph counts each Send invocation against recursion_limit; default 50 causes RecursionError
# on repos with > ~40 chunks. 10_000 accommodates large repos without risk of infinite loops
# (the graph terminates naturally at chroma_upsert_quarantined → END).
_GRAPH_CONFIG = RunnableConfig(recursion_limit=10_000)

# ── MCP server instance ───────────────────────────────────────────────────────

mcp = FastMCP("developer-memory")
