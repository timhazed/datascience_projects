"""CLI ingest tool for Developer Memory evaluation pipeline.

Runs the full ingest pipeline (git delta → PII sanitize → LLM summarize → ChromaDB upsert)
against one or more GitHub repos, writing results to experiments/results/.

The eval ChromaDB is isolated from production: set via --chroma-path (default:
experiments/.chroma_eval). Re-running is idempotent — the skip-before-LLM fast path
means already-indexed chunks are skipped in ~1ms each.

Usage:
    poetry run python -m src.cli.ingest https://github.com/owner/repo
    poetry run python -m src.cli.ingest https://github.com/owner/repo@branch --max-files 50
    poetry run python -m src.cli.ingest https://github.com/owner/repo --chroma-path /tmp/eval_db
    poetry run python -m src.cli.ingest https://github.com/owner/repo --collection my_eval_v1

Environment (read from experiments/.env, then root .env):
    GITHUB_TOKEN    — required for private repos
    OLLAMA_MODEL    — model tag (default: gemma4:e4b)
    OLLAMA_HOST     — Ollama base URL (overridden to http://localhost:11434 for local runs)
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

# ── Path bootstrap — must happen before any src.* imports ────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# Load experiments/.env first (eval-specific settings), then root .env as fallback
_EXPERIMENTS_DIR = _PROJECT_ROOT / "experiments"
load_dotenv(_EXPERIMENTS_DIR / ".env")
load_dotenv(_PROJECT_ROOT / ".env")

# Force local Ollama and local ChromaDB — never route through Docker network
os.environ["OLLAMA_HOST"] = "http://localhost:11434"
os.environ["CHROMA_HOST"] = ""

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("devmem.cli.ingest")

_RESULTS_DIR = _PROJECT_ROOT / "src" / "cli" / "results"
_DEFAULT_CHROMA_PATH = str(_PROJECT_ROOT / "src" / "cli" / "data" / "chroma")
_DEFAULT_COLLECTION = "eval_collection_v1"
_DEFAULT_MAX_FILES = 0  # 0 = unlimited


# ── Preflight ─────────────────────────────────────────────────────────────────


def _check_ollama() -> None:
    """Verify Ollama is reachable and required models are available.

    Raises:
        SystemExit: If Ollama is unreachable or a required model is not pulled.
    """
    import httpx

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    llm_model = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
    required = {llm_model, "nomic-embed-text"}

    try:
        resp = httpx.get(f"{host}/api/tags", timeout=5)
        resp.raise_for_status()
        available = {m["name"] for m in resp.json().get("models", [])}
        # Match on base name (without tag) for flexibility
        available_bases = {m.split(":")[0] for m in available}
        missing = [m for m in required if m.split(":")[0] not in available_bases]
        if missing:
            print(f"\nERROR: Required models not pulled: {missing}")
            for m in missing:
                print(f"  ollama pull {m}")
            sys.exit(1)
        logger.info("Ollama OK — models available: %s", required)
    except httpx.HTTPError as exc:
        print(f"\nERROR: Ollama not reachable at {host}: {exc}")
        print("  Start with: ollama serve")
        sys.exit(1)


# ── Repo URL parsing ──────────────────────────────────────────────────────────


def _parse_repo_arg(raw: str) -> tuple[str, str]:
    """Parse a repo URL with optional @branch suffix.

    Args:
        raw: URL string, e.g. "https://github.com/owner/repo" or
             "https://github.com/owner/repo@develop"

    Returns:
        (repo_url, branch) tuple.
    """
    match = re.match(r"^(.+?)(?:@([^@]+))?$", raw.strip())
    if not match:
        raise ValueError(f"Cannot parse repo argument: {raw!r}")
    repo_url = match.group(1)
    branch = match.group(2) or "main"
    return repo_url, branch


# ── File cap helper ───────────────────────────────────────────────────────────


def _cap_patch(max_files: int):
    """Return a context manager that caps compute_delta output to max_files.

    When max_files is 0, returns a no-op context (unlimited). Otherwise wraps
    the production compute_delta function to return only the first N files.

    Args:
        max_files: Cap on number of files returned. 0 means unlimited.

    Returns:
        Context manager suitable for use in a with statement.
    """
    if max_files == 0:
        return nullcontext()

    from src.ingest.git_delta import compute_delta as _real

    def _capped(repo, branch="main"):
        return _real(repo, branch)[:max_files]

    return patch("src.mcp_server.git_client.compute_delta", side_effect=_capped)


# ── Core ingest ───────────────────────────────────────────────────────────────


def run_ingest(
    repo_url: str,
    branch: str,
    max_files: int,
    chroma_path: str,
    collection: str,
    model: str,
) -> dict:
    """Run the production ingest pipeline against a single repo.

    Sets CHROMA_DATA_PATH, CHROMA_COLLECTION, and OLLAMA_MODEL before importing
    src.server so all three are resolved at construction time, not production defaults.

    Args:
        repo_url: Full GitHub repo URL.
        branch: Branch to ingest.
        max_files: Maximum files to process (0 = unlimited).
        chroma_path: Local directory for the eval ChromaDB PersistentClient.
        collection: ChromaDB collection name.
        model: Ollama model tag for LLM summarization (e.g. "phi4:14b").

    Returns:
        Dict with keys: repo_url, branch, elapsed_seconds, result (server response dict).
    """
    os.environ["CHROMA_DATA_PATH"] = chroma_path
    os.environ["CHROMA_COLLECTION"] = collection
    os.environ["OLLAMA_MODEL"] = model

    # Import after env vars are set — server module reads them at import time
    import src.server as server

    logger.info("Ingesting %s @ %s → %s [%s]", repo_url, branch, chroma_path, collection)
    t0 = time.monotonic()

    with _cap_patch(max_files):
        result = server.sync_repository(repo_url=repo_url, branch=branch)

    elapsed = time.monotonic() - t0
    logger.info("Done in %.1fs — %s", elapsed, result)
    return {"repo_url": repo_url, "branch": branch, "elapsed_seconds": round(elapsed, 2), "result": result}


# ── Results persistence ───────────────────────────────────────────────────────


def save_results(repo_results: list[dict], chroma_path: str, collection: str) -> Path:
    """Write ingest results to a timestamped JSON file in experiments/results/.

    Args:
        repo_results: List of per-repo result dicts from run_ingest().
        chroma_path: ChromaDB path used for this run (recorded in output).
        collection: ChromaDB collection name.

    Returns:
        Path to the written JSON file.
    """
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    llm_model = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
    model_slug = llm_model.replace(":", "_").replace("/", "_")
    run_id = f"ingest_eval_{model_slug}_{timestamp}"

    total_elapsed = sum(r["elapsed_seconds"] for r in repo_results)
    total_inserted = sum(
        r["result"].get("inserted", 0) if isinstance(r["result"], dict) else 0
        for r in repo_results
    )
    total_skipped = sum(
        r["result"].get("skipped", 0) if isinstance(r["result"], dict) else 0
        for r in repo_results
    )

    payload = {
        "run_id": run_id,
        "experiment": "E-RETRIEVAL-INGEST",
        "timestamp": datetime.now(UTC).isoformat(),
        "chroma_path": chroma_path,
        "collection": collection,
        "config": {
            "llm_model": llm_model,
            "embed_model": "nomic-embed-text",
            "ollama_host": os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        },
        "summary": {
            "total_repos": len(repo_results),
            "total_inserted": total_inserted,
            "total_skipped": total_skipped,
            "total_elapsed_seconds": round(total_elapsed, 2),
        },
        "repo_results": repo_results,
    }

    out_path = _RESULTS_DIR / f"{run_id}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Results written → %s", out_path)
    return out_path


# ── CLI entry point ───────────────────────────────────────────────────────────


def main() -> None:
    """Parse arguments and run the ingest pipeline."""
    parser = argparse.ArgumentParser(
        prog="devmem-ingest",
        description="Ingest GitHub repos into the Developer Memory eval ChromaDB.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python -m src.cli.ingest https://github.com/timhazed/datascience_projects
  poetry run python -m src.cli.ingest https://github.com/timhazed/datascience_projects@trunk --max-files 100
  poetry run python -m src.cli.ingest https://github.com/owner/repo --collection my_eval_v2
        """,
    )
    parser.add_argument(
        "repos",
        nargs="+",
        metavar="REPO_URL[@BRANCH]",
        help="One or more GitHub repo URLs. Append @branch to specify a branch (default: main).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=_DEFAULT_MAX_FILES,
        metavar="N",
        help="Cap files processed per repo. 0 = unlimited (default: 0).",
    )
    parser.add_argument(
        "--chroma-path",
        default=_DEFAULT_CHROMA_PATH,
        metavar="PATH",
        help=f"Local ChromaDB directory (default: {_DEFAULT_CHROMA_PATH}).",
    )
    parser.add_argument(
        "--collection",
        default=_DEFAULT_COLLECTION,
        metavar="NAME",
        help=f"ChromaDB collection name (default: {_DEFAULT_COLLECTION}).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OLLAMA_MODEL", "gemma4:e4b"),
        metavar="TAG",
        help="Ollama model tag for LLM summarization (default: $OLLAMA_MODEL or gemma4:e4b).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print repos that would be ingested without running.",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip Ollama reachability check (useful when models are confirmed available).",
    )

    args = parser.parse_args()
    os.environ["OLLAMA_MODEL"] = args.model

    if not args.skip_preflight:
        _check_ollama()

    repos = [_parse_repo_arg(r) for r in args.repos]

    if args.dry_run:
        print("Dry run — would ingest:")
        for repo_url, branch in repos:
            print(f"  {repo_url} @ {branch}")
        print(f"  model:       {args.model}")
        print(f"  chroma-path: {args.chroma_path}")
        print(f"  collection:  {args.collection}")
        print(f"  max-files:   {args.max_files or 'unlimited'}")
        return

    repo_results: list[dict] = []
    for repo_url, branch in repos:
        try:
            result = run_ingest(
                repo_url=repo_url,
                branch=branch,
                max_files=args.max_files,
                chroma_path=args.chroma_path,
                collection=args.collection,
                model=args.model,
            )
        except Exception as exc:
            logger.error("Ingest failed for %s: [%s] %s", repo_url, type(exc).__name__, exc)
            result = {
                "repo_url": repo_url,
                "branch": branch,
                "elapsed_seconds": 0,
                "result": {"error": str(exc)},
            }
        repo_results.append(result)

    out_path = save_results(repo_results, args.chroma_path, args.collection)

    # Print summary to stdout
    summary = {r["repo_url"]: r["elapsed_seconds"] for r in repo_results}
    print(f"\nIngest complete. Results: {out_path}")
    for url, elapsed in summary.items():
        print(f"  {url}: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
