"""Ingestion smoke test — Experiment §E-INGEST-01.

Research Question:
    Can a real GitHub repository be fetched via the API with a personal access token,
    have its changed files identified, parsed, and upserted into ChromaDB — end-to-end
    using the production stack?

Approach:
    Call src.server.sync_repository() directly — the same function the MCP server
    exposes. This exercises the full production path:
      delta_extractor → pii_sanitizer → multimodal_parser
      → intent_summarizer → chroma_upsert

    No graph wiring, no dependency injection — if this works, the MCP tool works.

Run (default public repos):
    poetry run python experiments/ingest_smoke_test.py

Run with your own repos:
    poetry run python experiments/ingest_smoke_test.py \\
        https://github.com/owner/repo1 \\
        https://github.com/owner/repo2@develop

    URL format: https://github.com/owner/repo[@branch]
    Branch defaults to "main" if omitted.

Options:
    --max-files N   Cap files processed per repo (0 = unlimited, default: 10)

Environment:
    GITHUB_TOKEN     — required for private repos; optional for public repos
    OLLAMA_HOST      — Ollama base URL (default: http://localhost:11434)
    OLLAMA_MODEL     — model tag (default: gemma4:26b)

Prerequisites:
    ollama serve (or running as LaunchAgent)
    ollama pull gemma4:26b   (or whichever OLLAMA_MODEL points to)
    ollama pull nomic-embed-text

Promotion Decision (fill in after running):
    [ ] Promote — ingester is ready for Phase 12 field testing
    [ ] Keep as reference — works but needs tuning before field use
    [ ] Dead end — document blocker
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

# ── Environment ────────────────────────────────────────────────────────────────
# Add project root to sys.path so src.* imports resolve when running directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(_PROJECT_ROOT / ".env")

# Experiments run directly on the macOS host, never inside Docker.
# OLLAMA_HOST in .env is set to host.docker.internal for container use — override
# it here so get_llm() and get_embeddings() always reach the native Ollama process.
os.environ["OLLAMA_HOST"] = "http://localhost:11434"
# CHROMA_HOST in .env points to Docker-internal chromadb:8000 — override so
# experiments always use a local PersistentClient instead of the Docker network.
os.environ["CHROMA_HOST"] = ""
# Direct PersistentClient to a smoke-test-specific directory to avoid version
# mismatch errors with the production chroma_data/ volume (written by Docker).
os.environ["CHROMA_DATA_PATH"] = str(_PROJECT_ROOT / "experiments" / ".chroma_smoke")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("ingest_smoke_test")

# ── Configuration ─────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _SCRIPT_DIR / "results"

_DEFAULT_REPOS = [
    {"repo_url": "https://github.com/anthropics/anthropic-sdk-python", "branch": "main"},
    {"repo_url": "https://github.com/langchain-ai/langchain", "branch": "master"},
]

# Default file cap — keeps a full Ollama run under ~5 minutes.
# Override with --max-files N (0 = unlimited).
MAX_FILES_PER_REPO: int = 10


# ── Ollama preflight ──────────────────────────────────────────────────────────


def _check_ollama() -> None:
    """Verify Ollama is reachable and required models are pulled.

    Raises:
        SystemExit: If Ollama is unreachable or a required model is missing.
    """
    import httpx

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    llm_model = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
    required = {llm_model, "nomic-embed-text"}

    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        available = {m["name"].split(":")[0] for m in resp.json().get("models", [])}
        missing = {m.split(":")[0] for m in required} - available
        if missing:
            print(f"\nERROR: Models not pulled: {missing}")
            for m in missing:
                print(f"  ollama pull {m}")
            sys.exit(1)
        logger.info("Ollama OK — models available: %s", required)
    except httpx.HTTPError as exc:
        print(f"\nERROR: Ollama not reachable at {host}: {exc}")
        print("  Start it with: ollama serve")
        sys.exit(1)


# ── File cap patch ────────────────────────────────────────────────────────────


def _cap_patch(max_files: int | None):
    """Return a context manager that caps compute_delta() output to max_files.

    Wraps the production compute_delta function so the experiment processes only
    the first N changed files without modifying any production code.

    Args:
        max_files: Maximum files to return. None = no cap (passthrough).

    Returns:
        unittest.mock.patch context manager, or a no-op context manager if
        max_files is None.
    """
    from contextlib import nullcontext

    if max_files is None:
        return nullcontext()

    from src.ingest.git_delta import compute_delta as _real_compute_delta

    def _capped(repo, branch="main"):
        return _real_compute_delta(repo, branch)[:max_files]

    return patch("src.server.compute_delta", side_effect=_capped)


# ── Main experiment ───────────────────────────────────────────────────────────


def run_ingest_smoke_test(repo_url: str, branch: str, max_files: int | None) -> dict:
    """Invoke the production sync_repository MCP tool against a single repo.

    Imports src.server at call time (not module level) so the MCP server startup
    sequence — _check_memory(), LLM construction, graph compilation — runs once
    on first call and is cached for subsequent calls.

    Args:
        repo_url: GitHub repository URL (public or private with GITHUB_TOKEN).
        branch: Branch to sync.
        max_files: Maximum changed files to process. None = unlimited.

    Returns:
        Result dict with fields: repo_url, branch, success, upsert_count,
        quarantine_count, error, elapsed_seconds, trace_tail.
    """
    from src.server import sync_repository  # noqa: PLC0415 — lazy: triggers server startup once

    label = repo_url.rstrip("/").split("/")[-1]
    logger.info("=" * 60)
    logger.info("Testing: %s (branch=%s)", label, branch)
    logger.info("=" * 60)

    start = time.perf_counter()
    with _cap_patch(max_files):
        result = sync_repository(repo_url=repo_url, branch=branch)
    elapsed = time.perf_counter() - start

    upsert_count = len(result.get("upsert_results", []))
    quarantine_count = len(result.get("quarantined_files", []))
    error = result.get("error")
    trace = result.get("trace", [])
    success = error is None and upsert_count > 0

    summary = {
        "label": label,
        "repo_url": repo_url,
        "branch": branch,
        "success": success,
        "upsert_count": upsert_count,
        "quarantine_count": quarantine_count,
        "error": error,
        "elapsed_seconds": round(elapsed, 2),
        "trace_tail": trace[-5:],
    }

    print(f"\n{'=' * 60}")
    print(f"RESULT: {label} ({branch})")
    print(f"  Status     : {'PASS' if success else 'FAIL'}")
    print(f"  Upserted   : {upsert_count} chunks")
    print(f"  Quarantined: {quarantine_count} files")
    print(f"  Elapsed    : {elapsed:.1f}s")
    if error:
        print(f"  Error      : {error}")
    print("  Trace tail :")
    for entry in trace[-5:]:
        print(f"    {entry}")
    print(f"{'=' * 60}\n")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────


def _default_branch(repo_url: str) -> str:
    """Fetch the default branch for a GitHub repo via the API.

    Uses GITHUB_TOKEN if set. Falls back to "main" on any error so the
    experiment degrades gracefully without a token or network access.

    Args:
        repo_url: Full GitHub repo URL, e.g. https://github.com/owner/repo

    Returns:
        Default branch name string, e.g. "main", "master", "trunk".
    """
    import httpx  # noqa: PLC0415

    try:
        parts = repo_url.rstrip("/").split("/")
        owner, repo = parts[-2], parts[-1]
        headers = {}
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
        logger.warning("Could not detect default branch for %s (%s) — using 'main'", repo_url, exc)
        return "main"


def _parse_repo_arg(raw: str) -> tuple[str, str]:
    """Parse URL[@branch] into (repo_url, branch).

    If no branch is specified, queries the GitHub API to detect the default branch
    rather than assuming 'main' — repos commonly use 'master', 'trunk', or 'develop'.

    Args:
        raw: Repository URL, optionally suffixed with @branch.

    Returns:
        Tuple of (repo_url, branch).
    """
    if "@" in raw.split("/")[-1]:
        repo_url, branch = raw.rsplit("@", 1)
    else:
        repo_url = raw
        branch = _default_branch(repo_url)
    return repo_url, branch


def save_findings(results: list[dict], model: str, max_files: int | None) -> Path:
    """Write run results to experiments/results/ingest_<timestamp>.json.

    Follows the same structure as portfolio_experiment.py and eval_output_quality.py
    so all experiment outputs are queryable with the same tooling.

    Args:
        results: List of per-repo result dicts from run_ingest_smoke_test().
        model: LLM model tag used for this run.
        max_files: File cap applied during this run (None = unlimited).

    Returns:
        Path to the written JSON file.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    model_slug = re.sub(r"[^a-zA-Z0-9]", "_", model)
    out = RESULTS_DIR / f"ingest_{model_slug}_{ts}.json"

    passed = sum(1 for r in results if r["success"])
    embed_model = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    payload = {
        "run_id": f"ingest_{model_slug}_{ts}",
        "experiment": "E-INGEST-01",
        "timestamp": datetime.now(UTC).isoformat(),
        "config": {
            "llm_model": model,
            "embed_model": embed_model,
            "max_files_per_repo": max_files,
            "path": "src.server.sync_repository() [production]",
        },
        "summary": {
            "total_repos": len(results),
            "passed": passed,
            "failed": len(results) - passed,
            "pass_rate": round(passed / len(results), 3) if results else 0.0,
            "total_chunks_upserted": sum(r["upsert_count"] for r in results),
            "total_files_quarantined": sum(r["quarantine_count"] for r in results),
            "total_elapsed_seconds": round(sum(r["elapsed_seconds"] for r in results), 2),
        },
        "repo_results": results,
    }

    out.write_text(json.dumps(payload, indent=2))
    logger.info("Findings written to %s", out)
    return out


def main() -> None:
    """Parse CLI args, run preflight checks, and execute the smoke test."""
    parser = argparse.ArgumentParser(
        description="Developer Memory — ingestion smoke test via production sync_repository()",
        epilog=(
            "Examples:\n"
            "  poetry run python experiments/ingest_smoke_test.py\n"
            "  poetry run python experiments/ingest_smoke_test.py \\\n"
            "      https://github.com/owner/repo@develop\n"
            "  poetry run python experiments/ingest_smoke_test.py --max-files 5 \\\n"
            "      https://github.com/owner/repo"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "repos",
        nargs="*",
        metavar="URL[@branch]",
        help="Repo URLs to test. Defaults to two public repos if omitted.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=MAX_FILES_PER_REPO,
        metavar="N",
        help=f"Max files per repo (0 = unlimited, default: {MAX_FILES_PER_REPO}).",
    )
    args = parser.parse_args()

    repos = (
        [_parse_repo_arg(r) for r in args.repos]
        if args.repos
        else [(r["repo_url"], r["branch"]) for r in _DEFAULT_REPOS]
    )
    max_files: int | None = args.max_files if args.max_files > 0 else None

    _check_ollama()

    model = os.environ.get("OLLAMA_MODEL", "gemma4:26b")
    print("\nDeveloper Memory — Ingestion Smoke Test")
    print(f"Run at        : {datetime.now(UTC).isoformat()}")
    print(f"Repos         : {len(repos)}")
    print(f"Max files/repo: {max_files or 'unlimited'}")
    print(f"LLM           : {model}")
    print("Embeddings    : nomic-embed-text")
    print("Path          : src.server.sync_repository() [production]")
    token = os.environ.get("GITHUB_TOKEN", "")
    print(f"GITHUB_TOKEN  : {'set' if token else 'not set (public repos only)'}\n")

    results = []
    for repo_url, branch in repos:
        try:
            summary = run_ingest_smoke_test(repo_url, branch, max_files)
            results.append(summary)
        except Exception as exc:
            logger.exception("Crashed on %s", repo_url)
            results.append({
                "label": repo_url.rstrip("/").split("/")[-1],
                "success": False,
                "upsert_count": 0,
                "quarantine_count": 0,
                "error": f"{type(exc).__name__}: {str(exc)[:200]}",
                "elapsed_seconds": 0,
                "trace_tail": [],
            })

    # ── Summary ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SMOKE TEST SUMMARY")
    print("=" * 70)
    passed = sum(1 for r in results if r["success"])
    print(f"{'Repo':<35} {'Status':<8} {'Chunks':>6} {'Quarantine':>10} {'Time':>8}")
    print("-" * 70)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"{r['label']:<35} {status:<8} {r['upsert_count']:>6} "
            f"{r['quarantine_count']:>10} {r['elapsed_seconds']:>7.1f}s"
        )
    print("=" * 70)
    print(f"Passed: {passed}/{len(results)}\n")

    findings_path = save_findings(results, model=model, max_files=max_files)
    print(f"Findings saved : {findings_path}\n")

    print("PROMOTION DECISION")
    print("-" * 70)
    if passed == len(results):
        print("[x] Ready to promote — production sync_repository() validated end-to-end.")
        print("    Next: run --max-files 0 against your target repos for full ingestion.")
    elif passed > 0:
        print("[ ] Partial — review FAIL entries. Check token, URL, Ollama model.")
    else:
        print("[ ] Dead end — all repos failed. Check logs above.")
    print()

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
