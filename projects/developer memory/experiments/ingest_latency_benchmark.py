"""Ingest latency benchmark — Experiment §E-LATENCY-01.

Research Question:
    What is the per-stage and end-to-end latency to ingest, PII-sanitize, parse,
    and LLM-summarize up to 250 files from a real GitHub repository?

Approach:
    Drive the production sync pipeline (src.graphs.sync_graph) with instrumented
    wrappers around each LangGraph node. Stage timings are injected into the graph
    state and extracted from the final state after the graph completes.

    Measured stages:
      - git_clone          : clone / fetch repo via _make_git_client()
      - delta_extractor    : compute changed-file list from HEAD tree
      - pii_sanitizer      : Presidio NER scan (parallel, PII_WORKERS threads)
      - multimodal_parser  : file chunking (pure string ops, no I/O)
      - summarize_and_upsert : LLM inference + ChromaDB upsert per chunk
      - chroma_upsert_quarantined : quarantine-only upserts (no LLM)
      - total_pipeline     : wall-clock from graph.invoke() start → end

    Each stage also reports:
      - items_processed : files or chunks handled
      - items_skipped   : pre-existing chunks (skip-before-LLM fast path)
      - items_error     : error results

Run (target repo, up to 250 files):
    poetry run python experiments/ingest_latency_benchmark.py \\
        https://github.com/owner/repo

Run with branch and file cap:
    poetry run python experiments/ingest_latency_benchmark.py \\
        https://github.com/owner/repo@develop --max-files 100

Options:
    --max-files N         Cap files processed (0 = up to 250, default: 250)
    --ollama-concurrency N  Parallel Ollama calls (default: from OLLAMA_NUM_PARALLEL or 1)
    --fresh               Drop and recreate the ChromaDB collection before running
                          (forces all chunks through LLM — no skip-before-LLM)
    --collection NAME     Override ChromaDB collection name (default: from env or benchmark-specific)
    --no-pii              Skip Presidio NER (pass through all content) — faster cold runs
    --output DIR          Override results output dir (default: experiments/results)

Environment (read from .env):
    GITHUB_TOKEN          Personal access token for private repos
    OLLAMA_HOST           Ollama endpoint (default: http://localhost:11434)
    OLLAMA_MODEL          Gemma variant (default: gemma4:26b)
    OLLAMA_EMBED_MODEL    Embedding model (default: nomic-embed-text)
    OLLAMA_NUM_PARALLEL   Parallel LLM calls — matched to Ollama server setting
    CHROMA_HOST           ChromaDB endpoint (default: in-memory ephemeral for this benchmark)

Promotion Decision (fill in after running):
    [ ] Promote — latency data is sufficient to set SLOs and tune concurrency
    [ ] Keep as reference — useful baseline but not blocking production
    [ ] Dead end — document blocker
"""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ── Sys-path: resolve src.* imports when running directly (not via poetry run) ──
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

# Ollama runs natively on macOS (Mode A hybrid — no Docker container).
# .env sets OLLAMA_HOST=http://host.docker.internal:11434 so the MCP container
# can reach native Ollama from inside Docker. That hostname doesn't resolve on
# the macOS host itself. Hard-override to localhost so experiments always reach
# the native Ollama process directly.
os.environ["OLLAMA_HOST"] = "http://localhost:11434"

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest_latency")

# Suppress noisy library loggers that would obscure progress output.
for _noisy in ("httpx", "httpcore", "chromadb", "langchain_core", "langgraph"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# ── Lazy src imports (after sys.path and dotenv) ───────────────────────────────
from src.db.chroma_client import ChromaLibrarianClient  # noqa: E402
from src.graphs.sync_graph import build_sync_graph  # noqa: E402
from src.ingest.git_delta import compute_delta  # noqa: E402
from src.llm.factory import get_llm  # noqa: E402
from src.middleware.pii_filter import PIIFilter  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────

_RESULTS_DIR = _PROJECT_ROOT / "experiments" / "results"
_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

_MAX_FILES_DEFAULT = 250
_BENCH_COLLECTION = "devmem_latency_bench"  # isolated from production collection


# ── Progress printer ──────────────────────────────────────────────────────────

class _Progress:
    """Thread-safe single-line progress reporter.

    Prints stage start/end banners and optional per-chunk counters to stderr so
    that the JSON result stream on stdout stays clean if the caller pipes it.
    """

    _lock = threading.Lock()

    @staticmethod
    def stage(name: str, msg: str = "") -> None:
        suffix = f"  {msg}" if msg else ""
        with _Progress._lock:
            print(f"\n{'─' * 60}", flush=True)
            print(f"  ▶  {name}{suffix}", flush=True)
            print(f"{'─' * 60}", flush=True)

    @staticmethod
    def done(name: str, elapsed_s: float, msg: str = "") -> None:
        suffix = f"  {msg}" if msg else ""
        with _Progress._lock:
            print(f"  ✓  {name}  [{elapsed_s:.2f}s]{suffix}", flush=True)

    @staticmethod
    def tick(done: int, total: int, label: str = "chunks") -> None:
        bar_width = 30
        filled = int(bar_width * done / max(total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)
        pct = int(100 * done / max(total, 1))
        with _Progress._lock:
            print(
                f"\r  [{bar}] {pct:3d}%  {done}/{total} {label}",
                end="",
                flush=True,
            )

    @staticmethod
    def nl() -> None:
        print(flush=True)

    @staticmethod
    def info(msg: str) -> None:
        with _Progress._lock:
            print(f"  ℹ  {msg}", flush=True)

    @staticmethod
    def warn(msg: str) -> None:
        with _Progress._lock:
            print(f"  ⚠  {msg}", flush=True)

    @staticmethod
    def error(msg: str) -> None:
        with _Progress._lock:
            print(f"  ✗  {msg}", flush=True)


# ── Git client (reused from server._make_git_client logic) ────────────────────

def _make_timed_git_client(
    max_files: int,
    timings: dict,
) -> Callable[[str, str], tuple[list[str], str]]:
    """Return a git client that clones the repo and records clone + delta timing.

    Reuses the same clone-to-tmpdir pattern as src.server._make_git_client().
    Caps returned file list to max_files (0 = up to _MAX_FILES_DEFAULT).

    Args:
        max_files: Maximum files to return. 0 = use _MAX_FILES_DEFAULT.
        timings: Mutable dict; writes "git_clone_s" and "delta_extractor_s".

    Returns:
        git_client callable compatible with make_delta_extractor_node().
    """
    cap = max_files if max_files > 0 else _MAX_FILES_DEFAULT

    def _auth_url(repo_url: str) -> str:
        """Inject GITHUB_TOKEN into an HTTPS GitHub URL for private repo access.

        Transforms https://github.com/owner/repo
                → https://<token>@github.com/owner/repo

        If GITHUB_TOKEN is unset the URL is returned unchanged — public repos
        work without a token; private repos will fail with a 128 exit code.
        """
        token = os.environ.get("GITHUB_TOKEN", "")
        if not token:
            return repo_url
        if "github.com" in repo_url and "://" in repo_url:
            scheme, rest = repo_url.split("://", 1)
            # Don't double-inject if token is already present.
            if "@" not in rest:
                return f"{scheme}://{token}@{rest}"
        return repo_url

    def git_client(repo_url: str, branch: str) -> tuple[list[str], str]:
        from git import GitCommandError, Repo

        auth_url = _auth_url(repo_url)
        _Progress.stage("git clone", f"{repo_url}@{branch}")
        if os.environ.get("GITHUB_TOKEN"):
            _Progress.info("Using GITHUB_TOKEN for authentication")
        else:
            _Progress.warn("GITHUB_TOKEN not set — private repos will fail")

        t0 = time.perf_counter()
        try:
            tmpdir = tempfile.mkdtemp(prefix="devmem_bench_")
            atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)
            repo_dir = Path(tmpdir) / "repo"
            repo = Repo.clone_from(auth_url, repo_dir, branch=branch)
            clone_s = time.perf_counter() - t0
            timings["git_clone_s"] = round(clone_s, 3)
            _Progress.done("git clone", clone_s)

            _Progress.stage("delta_extractor", "computing file list from HEAD tree")
            t1 = time.perf_counter()
            all_paths = compute_delta(repo, branch)
            delta_s = time.perf_counter() - t1
            timings["delta_extractor_s"] = round(delta_s, 3)
            timings["total_files_in_repo"] = len(all_paths)

            capped = all_paths[:cap]
            timings["files_submitted"] = len(capped)

            if len(all_paths) > cap:
                _Progress.warn(
                    f"Repo has {len(all_paths)} files — capped at {cap} for this benchmark"
                )
            _Progress.done(
                "delta_extractor",
                delta_s,
                f"{len(capped)} files (repo total: {len(all_paths)})",
            )
            return (capped, str(repo_dir))
        except GitCommandError as exc:
            stderr = exc.stderr.strip()[:300] if exc.stderr else str(exc)
            _Progress.error(f"git clone failed: {stderr}")
            raise RuntimeError(f"git operation failed: {stderr}") from exc

    return git_client


# ── Instrumented node wrappers ─────────────────────────────────────────────────

def _wrap_node(
    node_fn: Callable[[dict], dict],
    stage_name: str,
    timings: dict,
    on_done: Callable[[dict, dict, float], None] | None = None,
) -> Callable[[dict], dict]:
    """Wrap a LangGraph node function to record wall-clock elapsed time.

    Args:
        node_fn: The original node callable.
        stage_name: Key written into timings (e.g. "pii_sanitizer_s").
        timings: Mutable dict to accumulate stage results.
        on_done: Optional callback(state_in, result, elapsed_s) for richer reporting.

    Returns:
        Wrapped node function with identical signature.
    """
    def _wrapped(state: dict) -> dict:
        _Progress.stage(stage_name)
        t0 = time.perf_counter()
        result = node_fn(state)
        elapsed = time.perf_counter() - t0
        timings[f"{stage_name}_s"] = round(elapsed, 3)
        if on_done:
            on_done(state, result, elapsed)
        return result

    return _wrapped


def _pii_done(state: dict, result: dict, elapsed_s: float) -> None:
    san = len(result.get("sanitized_files", []))
    qua = len(result.get("quarantined_files", []))
    _Progress.done(
        "pii_sanitizer",
        elapsed_s,
        f"{san} sanitized, {qua} quarantined",
    )
    if qua:
        _Progress.info(f"Quarantined files: {[f.path for f in result['quarantined_files']]}")


def _parser_done(state: dict, result: dict, elapsed_s: float) -> None:
    n = len(result.get("parsed_chunks", []))
    _Progress.done("multimodal_parser", elapsed_s, f"{n} chunks produced")


# ── Chunk-level progress callback for summarize_and_upsert ────────────────────

class _ChunkProgress:
    """Accumulates per-chunk UpsertResult timing via the progress_callback protocol."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.inserted = 0
        self.skipped = 0
        self.errors = 0
        self.total = 0
        self._t0: float = 0.0

    def start(self, total: int) -> None:
        self.total = total
        self._t0 = time.perf_counter()

    def callback(self, job_id: str | None, done: int, total: int) -> None:
        # progress_callback is called after each chunk; job_id is None here (no async job)
        with self._lock:
            _Progress.tick(done, total, "chunks")

    def elapsed_s(self) -> float:
        return time.perf_counter() - self._t0


# ── Result serialisation ───────────────────────────────────────────────────────

def _build_result(
    repo_url: str,
    branch: str,
    model: str,
    embed_model: str,
    timings: dict,
    final_state: dict,
    chunk_progress: _ChunkProgress,
    args: argparse.Namespace,
) -> dict:
    """Assemble the final result dict from all collected measurements."""
    upsert_results = final_state.get("upsert_results", [])
    inserted = sum(1 for r in upsert_results if r.action == "inserted")
    skipped  = sum(1 for r in upsert_results if r.action == "skipped")
    errors   = sum(1 for r in upsert_results if r.action == "error")
    total    = len(upsert_results)

    quarantined = len(final_state.get("quarantined_files", []))
    parsed      = len(final_state.get("parsed_chunks", []))  # may be [] after dispatcher
    changed     = len(final_state.get("changed_files", []))
    error_msg   = final_state.get("error")

    # Derived throughput metrics
    pipeline_s  = timings.get("total_pipeline_s", 0.0)
    llm_s       = timings.get("summarize_and_upsert_s", 0.0)  # set by wrapper; includes semaphore wait
    throughput_files_per_s  = round(changed / pipeline_s, 2) if pipeline_s else None
    throughput_chunks_per_s = round(total / pipeline_s, 2)   if pipeline_s else None
    throughput_llm_per_s    = round(inserted / llm_s, 2)     if (llm_s and inserted) else None

    return {
        "experiment": "ingest_latency_benchmark",
        "version": "1.0",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "repo_url": repo_url,
            "branch": branch,
            "model": model,
            "embed_model": embed_model,
            "max_files": args.max_files,
            "ollama_concurrency": args.ollama_concurrency,
            "fresh_collection": args.fresh,
            "no_pii": args.no_pii,
            "collection": args.collection or _BENCH_COLLECTION,
        },
        "summary": {
            "total_files_in_repo": timings.get("total_files_in_repo"),
            "files_submitted": timings.get("files_submitted"),
            "files_sanitized": changed - quarantined,
            "files_quarantined": quarantined,
            "chunks_produced": parsed,
            "chunks_inserted": inserted,
            "chunks_skipped": skipped,
            "chunks_error": errors,
            "pipeline_error": error_msg,
            "total_pipeline_s": pipeline_s,
            "throughput_files_per_s": throughput_files_per_s,
            "throughput_chunks_per_s": throughput_chunks_per_s,
            "throughput_llm_inserts_per_s": throughput_llm_per_s,
        },
        "stage_timings_s": {
            "git_clone":               timings.get("git_clone_s"),
            "delta_extractor":         timings.get("delta_extractor_s"),
            "pii_sanitizer":           timings.get("pii_sanitizer_s"),
            "multimodal_parser":       timings.get("multimodal_parser_s"),
            "summarize_and_upsert":    timings.get("summarize_and_upsert_s"),
            "chroma_upsert_quarantined": timings.get("chroma_upsert_quarantined_s"),
            "total_pipeline":          pipeline_s,
        },
        "trace_tail": final_state.get("trace", [])[-20:],  # last 20 trace entries
    }


def _print_summary(result: dict) -> None:
    """Print a human-readable benchmark summary to stdout."""
    s = result["summary"]
    t = result["stage_timings_s"]
    cfg = result["config"]

    print("\n" + "═" * 62)
    print("  INGEST LATENCY BENCHMARK — RESULTS")
    print("═" * 62)
    print(f"  Repo     : {cfg['repo_url']}@{cfg['branch']}")
    print(f"  Model    : {cfg['model']}  (embed: {cfg['embed_model']})")
    print(f"  Concurrency: {cfg['ollama_concurrency']}  |  Fresh: {cfg['fresh_collection']}")
    print()
    print("  ── File Counts ────────────────────────────────────────")
    print(f"  Files in repo          : {s['total_files_in_repo']}")
    print(f"  Files submitted        : {s['files_submitted']}")
    print(f"  Files sanitized        : {s['files_sanitized']}")
    print(f"  Files quarantined      : {s['files_quarantined']}")
    print(f"  Chunks produced        : {s['chunks_produced']}")
    print(f"  Chunks inserted (LLM)  : {s['chunks_inserted']}")
    print(f"  Chunks skipped (cached): {s['chunks_skipped']}")
    print(f"  Chunks errored         : {s['chunks_error']}")
    if s["pipeline_error"]:
        print(f"  ⚠ Pipeline error       : {s['pipeline_error']}")
    print()
    print("  ── Stage Timings ──────────────────────────────────────")
    stages = [
        ("git_clone",               "Git clone"),
        ("delta_extractor",         "Delta extractor"),
        ("pii_sanitizer",           "PII sanitizer"),
        ("multimodal_parser",       "Multimodal parser"),
        ("summarize_and_upsert",    "Summarize + upsert (LLM)"),
        ("chroma_upsert_quarantined", "Quarantine upsert"),
        ("total_pipeline",          "TOTAL pipeline"),
    ]
    for key, label in stages:
        val = t.get(key)
        if val is not None:
            bar_pct = val / max(t.get("total_pipeline") or 1, 0.001)
            bar = "█" * int(bar_pct * 20)
            print(f"  {label:<28} {val:7.2f}s  {bar}")
    print()
    print("  ── Throughput ─────────────────────────────────────────")
    if s["throughput_files_per_s"] is not None:
        print(f"  Files/sec (end-to-end) : {s['throughput_files_per_s']:.2f}")
    if s["throughput_chunks_per_s"] is not None:
        print(f"  Chunks/sec (pipeline)  : {s['throughput_chunks_per_s']:.2f}")
    if s["throughput_llm_inserts_per_s"] is not None:
        print(f"  LLM inserts/sec        : {s['throughput_llm_inserts_per_s']:.2f}")
    print("═" * 62)


# ── GitHub helpers ─────────────────────────────────────────────────────────────

def _default_branch(repo_url: str) -> str:
    """Fetch the default branch for a GitHub repo via the GitHub API.

    Uses GITHUB_TOKEN if set. Falls back to "main" on any error so the
    benchmark degrades gracefully without a token or network access.

    Args:
        repo_url: Full GitHub repo URL, e.g. https://github.com/owner/repo

    Returns:
        Default branch name string, e.g. "main", "master", "trunk".
    """
    import httpx  # noqa: PLC0415

    try:
        parts = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
        headers: dict[str, str] = {}
        token = os.environ.get("GITHUB_TOKEN", "")
        if token:
            headers["Authorization"] = f"token {token}"
        resp = httpx.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        branch = resp.json().get("default_branch", "main")
        logger.info("Auto-detected default branch for %s/%s: %s", owner, repo, branch)
        return branch
    except Exception as exc:
        logger.warning(
            "Could not detect default branch for %s (%s) — using 'main'",
            repo_url,
            exc,
        )
        return "main"


# ── Chroma setup ───────────────────────────────────────────────────────────────

def _make_raw_chroma_client():
    """Return a local PersistentClient in experiments/.chroma_bench/.

    Experiments run directly on the macOS host — ChromaDB is not accessible
    at the Docker-internal hostname (chromadb:8000) stored in .env. Always
    use a local PersistentClient so no Docker stack is required to run this
    benchmark. Vectors are stored in a benchmark-specific directory, separate
    from the production chroma_data/ volume.
    """
    import chromadb

    data_dir = _PROJECT_ROOT / "experiments" / ".chroma_bench"
    data_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(data_dir))


def _make_chroma(collection_name: str, fresh: bool) -> ChromaLibrarianClient:
    """Construct a ChromaLibrarianClient for the benchmark collection.

    Always uses a benchmark-specific collection name and data directory, never
    the production collection. If --fresh, drops the collection before constructing
    so every chunk goes through the LLM (no skip-before-LLM fast path).
    """
    raw_client = _make_raw_chroma_client()
    data_dir = _PROJECT_ROOT / "experiments" / ".chroma_bench"
    _Progress.info(f"ChromaDB: {data_dir}  collection={collection_name!r}")

    if fresh:
        try:
            raw_client.delete_collection(collection_name)
            _Progress.info(f"Dropped existing collection {collection_name!r} (--fresh)")
        except Exception:
            pass  # collection may not exist yet — harmless

    return ChromaLibrarianClient(client=raw_client, collection_name=collection_name)


# ── PIIFilter factory ─────────────────────────────────────────────────────────

def _make_pii(no_pii: bool) -> PIIFilter:
    """Return a PIIFilter — or a passthrough mock when --no-pii is set."""
    if no_pii:
        from unittest.mock import MagicMock
        pii = MagicMock(spec=PIIFilter)
        pii.sanitize.side_effect = lambda path, content: (content, None)
        _Progress.info("PII sanitizer: passthrough mode (--no-pii)")
        return pii
    _Progress.info("PII sanitizer: Presidio (loading spaCy model — may take ~10s first run)")
    return PIIFilter()


# ── Main ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Measure per-stage ingest latency for up to 250 files from a GitHub repo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "repo",
        metavar="REPO_URL[@BRANCH]",
        help="GitHub repo URL, optionally with @branch suffix (default branch: main)",
    )
    p.add_argument(
        "--max-files",
        type=int,
        default=_MAX_FILES_DEFAULT,
        metavar="N",
        help=f"Maximum files to process (0 = up to {_MAX_FILES_DEFAULT}, default: {_MAX_FILES_DEFAULT})",
    )
    p.add_argument(
        "--ollama-concurrency",
        type=int,
        default=int(os.environ.get("OLLAMA_NUM_PARALLEL", "1")),
        metavar="N",
        help="Parallel Ollama LLM calls (default: OLLAMA_NUM_PARALLEL env or 1)",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Drop and recreate the benchmark ChromaDB collection (forces all chunks through LLM)",
    )
    p.add_argument(
        "--collection",
        default="",
        metavar="NAME",
        help=f"Override ChromaDB collection name (default: {_BENCH_COLLECTION})",
    )
    p.add_argument(
        "--no-pii",
        action="store_true",
        help="Skip Presidio NER scan — content passes through unsanitized (faster cold runs)",
    )
    p.add_argument(
        "--output",
        default=str(_RESULTS_DIR),
        metavar="DIR",
        help=f"Output directory for result JSON (default: {_RESULTS_DIR})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the full Ollama prompt for the first chunk then exit — no LLM call made.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Parse repo URL and optional branch suffix.
    _url_branch = args.repo.rsplit("@", 1)
    repo_url = _url_branch[0]
    if len(_url_branch) == 2:
        branch = _url_branch[1]
    else:
        _Progress.info("No branch specified — querying GitHub API for default branch...")
        branch = _default_branch(repo_url)
        _Progress.info(f"Default branch: {branch}")

    model      = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    collection = args.collection or _BENCH_COLLECTION

    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       Developer Memory — Ingest Latency Benchmark        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    _Progress.info(f"Repo       : {repo_url}@{branch}")
    _Progress.info(f"LLM model  : {model}")
    _Progress.info(f"Embed model: {embed_model}")
    _Progress.info(f"Max files  : {args.max_files or _MAX_FILES_DEFAULT}")
    _Progress.info(f"Concurrency: {args.ollama_concurrency}")
    _Progress.info(f"Collection : {collection}")
    _Progress.info(f"Fresh run  : {args.fresh}")

    timings: dict[str, Any] = {}

    # ── Construct dependencies (reuse src singletons) ──────────────────────────
    _Progress.stage("startup", "constructing LLM / embed / chroma singletons")
    t_start = time.perf_counter()

    llm   = get_llm(model=model, temperature=0.0, num_ctx=8192)
    chroma = _make_chroma(collection, args.fresh)
    pii    = _make_pii(args.no_pii)
    timings["startup_s"] = round(time.perf_counter() - t_start, 3)
    _Progress.done("startup", timings["startup_s"])

    # ── Chunk progress tracker ─────────────────────────────────────────────────
    chunk_prog = _ChunkProgress()

    # ── Timed git client ───────────────────────────────────────────────────────
    git_client = _make_timed_git_client(args.max_files, timings)

    # ── Phase 1: run git → PII → parse through the graph ─────────────────────
    # interrupt_before=["chunk_dispatcher"] stops the graph after multimodal_parser
    # so we get parsed_chunks in state without triggering the Send fan-out.
    _Progress.stage("build graph", "wiring instrumented pre-LLM graph")
    pre_llm_graph = _InstrumentedSyncGraph(
        llm=llm,
        chroma=chroma,
        pii=pii,
        git_client=git_client,
        ollama_concurrency=args.ollama_concurrency,
        progress_callback=None,
        timings=timings,
        interrupt_before=["chunk_dispatcher"],
    )

    _Progress.stage("pipeline phase 1", f"git → PII → parse for {repo_url}@{branch}")
    t_pipeline = time.perf_counter()
    pre_state = pre_llm_graph.invoke(repo_url=repo_url, branch=branch)

    if pre_state.get("error"):
        timings["total_pipeline_s"] = round(time.perf_counter() - t_pipeline, 3)
        final_state = pre_state
    else:
        # ── Phase 2: serial LLM summarization loop ─────────────────────────────
        # Drive summarize_and_upsert one chunk at a time — no Send fan-out, no
        # thread storm. This gives accurate per-chunk timing and keeps Ollama stable.
        from src.agents.summarize_and_upsert import make_summarize_and_upsert_node

        sau_node = make_summarize_and_upsert_node(
            llm, chroma,
            ollama_concurrency=1,   # serial — one at a time
            progress_callback=None,
        )

        chunks = pre_state.get("parsed_chunks", [])
        chunks_total = len(chunks)
        chunk_prog.start(chunks_total)
        job_id = str(uuid.uuid4())

        _Progress.stage(
            "pipeline phase 2",
            f"LLM summarize+upsert {chunks_total} chunks (serial)",
        )
        t_llm_phase = time.perf_counter()
        all_upsert_results = []

        for idx, chunk in enumerate(chunks):
            sub_state = {
                "chunk": chunk,
                "repo_url": repo_url,
                "branch": branch,
                "commit_sha": pre_state.get("commit_sha", ""),
                "_job_id": job_id,
                "chunk_index": idx + 1,
                "chunks_total": chunks_total,
            }

            if args.dry_run:
                from src.chains.intent_summary_chain import _HUMAN, _SYSTEM
                _Progress.stage("dry-run", f"chunk [1/{chunks_total}]  {chunk.path}")
                print()
                print("── SYSTEM ──────────────────────────────────────────────────")
                print(_SYSTEM)
                print()
                print("── HUMAN ───────────────────────────────────────────────────")
                print(_HUMAN.format(path=chunk.path, content=chunk.content))
                print("────────────────────────────────────────────────────────────")
                print(f"\n  chunk size : {len(chunk.content)} chars")
                print(f"  model      : {model}")
                print("  num_ctx    : 8192")
                print("  num_predict: 2048")
                sys.exit(0)

            result = sau_node(sub_state)
            all_upsert_results.extend(result.get("upsert_results", []))
            _Progress.tick(idx + 1, chunks_total, "chunks")

        timings["summarize_and_upsert_s"] = round(time.perf_counter() - t_llm_phase, 3)
        _Progress.nl()
        _Progress.done("summarize_and_upsert", timings["summarize_and_upsert_s"],
                       f"{chunks_total} chunks")

        # Merge phase 2 results back into a final_state dict for _build_result
        final_state = {
            **pre_state,
            "upsert_results": all_upsert_results,
        }

    total_pipeline_s = time.perf_counter() - t_pipeline
    timings["total_pipeline_s"] = round(total_pipeline_s, 3)
    _Progress.done("pipeline", total_pipeline_s, "complete")

    # ── Compile and save results ───────────────────────────────────────────────
    result = _build_result(
        repo_url=repo_url,
        branch=branch,
        model=model,
        embed_model=embed_model,
        timings=timings,
        final_state=final_state,
        chunk_progress=chunk_prog,
        args=args,
    )

    _print_summary(result)

    # Persist to experiments/results/
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "_", repo_url.lower().split("github.com/")[-1])[:40]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"latency_{slug}_{ts}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"\n  Results saved → {out_path.relative_to(_PROJECT_ROOT)}")

    if result["summary"]["pipeline_error"]:
        sys.exit(1)


# ── Instrumented graph wrapper ────────────────────────────────────────────────

class _InstrumentedSyncGraph:
    """Thin wrapper around build_sync_graph that injects timing hooks.

    Wraps pii_sanitizer and multimodal_parser nodes with timing decorators.
    summarize_and_upsert timing is collected via the progress_callback.
    git_clone and delta_extractor timing comes from _make_timed_git_client.
    chroma_upsert_quarantined is timed via a post-state comparison.
    """

    def __init__(
        self,
        llm,
        chroma: ChromaLibrarianClient,
        pii: PIIFilter,
        git_client: Callable,
        ollama_concurrency: int,
        progress_callback: Callable | None,
        timings: dict,
        interrupt_before: list[str] | None = None,
    ) -> None:
        self._timings = timings
        self._chroma = chroma
        self._pii = pii
        self._git_client = git_client
        self._interrupt_before = interrupt_before or []

        # Build the graph using the production builder.
        # The instrumented wrappers for pii_sanitizer and multimodal_parser
        # are applied by monkey-patching the node factories BEFORE graph compile.
        self._graph = self._build(llm, chroma, pii, git_client, ollama_concurrency, progress_callback)

    def _build(self, llm, chroma, pii, git_client, ollama_concurrency, progress_callback):
        """Build the sync graph with instrumented node factories."""
        from langchain_core.runnables import RunnableConfig

        import src.agents.chroma_upsert_quarantined as _quar_mod
        import src.agents.multimodal_parser as _parser_mod
        import src.agents.pii_sanitizer as _pii_mod

        timings = self._timings

        # Wrap make_pii_sanitizer_node to instrument its returned node.
        _orig_pii = _pii_mod.make_pii_sanitizer_node

        def _timed_pii_factory(sanitizer):
            node = _orig_pii(sanitizer)
            def _timed(state):
                _Progress.stage("pii_sanitizer", f"scanning {len(state.get('changed_files', []))} files")
                t0 = time.perf_counter()
                result = node(state)
                elapsed = time.perf_counter() - t0
                timings["pii_sanitizer_s"] = round(elapsed, 3)
                _pii_done(state, result, elapsed)
                return result
            return _timed

        # Wrap make_multimodal_parser_node similarly.
        _orig_parser = _parser_mod.make_multimodal_parser_node

        def _timed_parser_factory():
            node = _orig_parser()
            def _timed(state):
                n_files = len(state.get("sanitized_files", []))
                _Progress.stage("multimodal_parser", f"chunking {n_files} sanitized files")
                t0 = time.perf_counter()
                result = node(state)
                elapsed = time.perf_counter() - t0
                timings["multimodal_parser_s"] = round(elapsed, 3)
                _parser_done(state, result, elapsed)
                return result
            return _timed

        # Wrap make_chroma_upsert_quarantined_node.
        _orig_quar = _quar_mod.make_chroma_upsert_quarantined_node

        def _timed_quar_factory(chroma_client):
            node = _orig_quar(chroma_client)
            def _timed(state):
                n_qua = len(state.get("quarantined_files", []))
                if n_qua:
                    _Progress.stage(
                        "chroma_upsert_quarantined",
                        f"upserting {n_qua} quarantined files",
                    )
                t0 = time.perf_counter()
                result = node(state)
                elapsed = time.perf_counter() - t0
                timings["chroma_upsert_quarantined_s"] = round(elapsed, 3)
                if n_qua:
                    _Progress.done("chroma_upsert_quarantined", elapsed)
                return result
            return _timed

        # Wrap summarize_and_upsert to time the whole fan-out wall clock.
        # Individual chunk timing comes from the progress_callback.
        import src.agents.summarize_and_upsert as _sau_mod
        _orig_sau = _sau_mod.make_summarize_and_upsert_node
        _sau_t0: list[float] = [0.0]
        _sau_started: list[bool] = [False]

        def _timed_sau_factory(llm_, chroma_, ollama_concurrency_=1, progress_callback_=None):
            node = _orig_sau(llm_, chroma_, ollama_concurrency_, progress_callback_)
            def _timed(state):
                if not _sau_started[0]:
                    _sau_started[0] = True
                    _sau_t0[0] = time.perf_counter()
                    total = state.get("chunks_total", 0)
                    _Progress.stage(
                        "summarize_and_upsert",
                        f"LLM + upsert for {total} chunks  (concurrency={ollama_concurrency_})",
                    )
                result = node(state)
                # Always update the running total so the final value is accurate.
                timings["summarize_and_upsert_s"] = round(
                    time.perf_counter() - _sau_t0[0], 3
                )
                return result
            return _timed

        # Apply patches, build graph, restore originals.
        _pii_mod.make_pii_sanitizer_node = _timed_pii_factory
        _parser_mod.make_multimodal_parser_node = _timed_parser_factory
        _quar_mod.make_chroma_upsert_quarantined_node = _timed_quar_factory
        _sau_mod.make_summarize_and_upsert_node = _timed_sau_factory
        try:
            graph = build_sync_graph(
                llm,
                chroma,
                pii,
                git_client,
                ollama_concurrency=ollama_concurrency,
                progress_callback=progress_callback,
                interrupt_before=self._interrupt_before,
            )
        finally:
            _pii_mod.make_pii_sanitizer_node = _orig_pii
            _parser_mod.make_multimodal_parser_node = _orig_parser
            _quar_mod.make_chroma_upsert_quarantined_node = _orig_quar
            _sau_mod.make_summarize_and_upsert_node = _orig_sau

        self._config = RunnableConfig(recursion_limit=10_000)
        return graph

    def invoke(self, repo_url: str, branch: str) -> dict:
        """Invoke the compiled graph and return the final state."""
        initial: dict = {
            "repo_url": repo_url,
            "branch": branch,
            "commit_sha": "",
            "changed_files": [],
            "sanitized_files": [],
            "quarantined_files": [],
            "parsed_chunks": [],
            "upsert_results": [],
            "error": None,
            "trace": [],
            # A non-None _job_id is required for the progress_callback to fire
            # inside summarize_and_upsert._report_progress(). The benchmark uses
            # a synthetic UUID — no real job registry is involved.
            "_job_id": str(uuid.uuid4()),
        }
        return self._graph.invoke(initial, config=self._config)


if __name__ == "__main__":
    main()
