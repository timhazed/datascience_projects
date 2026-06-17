#!/usr/bin/env python3
"""Backfill source_file_sha metadata on existing ChromaDB documents.

Spec §10, Step 0 — mandatory pre-deploy migration.

All documents indexed before cache_filter was deployed have no source_file_sha
metadata field. Without it, cache_filter.cached_sha_set is always empty on the
first post-deploy sync, so all files pass through PII scanning unchanged — identical
to the pre-cache-filter behavior. This script adds source_file_sha to each document
so that subsequent re-syncs skip already-indexed files correctly.

Algorithm:
  1. Connect to ChromaDB (reads CHROMA_HOST / CHROMA_DATA_PATH / CHROMA_COLLECTION).
  2. Fetch all documents: collection.get(include=["metadatas"]).
  3. Group by (repo_url, branch) to minimize clone operations.
  4. For each unique (repo_url, branch): clone repo into a temp dir.
  5. For each document: read raw bytes of metadata["file_path"] from the clone,
     compute sha256, call collection.update() to add source_file_sha.
  6. Skip documents that already have source_file_sha (idempotent).
  7. Skip documents whose file no longer exists at HEAD (deleted/renamed).
  8. Log per-repo progress and final summary.

Usage:
    poetry run python scripts/backfill_source_file_sha.py

    # Dry run (shows what would be updated, writes nothing):
    DRY_RUN=1 poetry run python scripts/backfill_source_file_sha.py

    # Against a non-default collection:
    CHROMA_COLLECTION=developer_memory_v1 poetry run python scripts/backfill_source_file_sha.py

Estimated runtime: 2–5 minutes for ~2,900 documents across one medium repo
(dominated by the git clone step).
"""

import atexit
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import chromadb
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DRY_RUN: bool = os.environ.get("DRY_RUN", "").lower() in ("1", "true", "yes")


def _connect_chroma() -> chromadb.Collection:
    """Connect to ChromaDB and return the target collection.

    Reads CHROMA_HOST (HTTP client) or CHROMA_DATA_PATH (persistent local client)
    and CHROMA_COLLECTION from environment.

    Returns:
        chromadb.Collection instance.

    Raises:
        RuntimeError: If the collection cannot be accessed.
    """
    chroma_host = os.environ.get("CHROMA_HOST", "")
    collection_name = os.environ.get("CHROMA_COLLECTION", "developer_memory_v1")

    if chroma_host:
        parsed = urlparse(chroma_host)
        client = chromadb.HttpClient(
            host=parsed.hostname or "localhost",
            port=parsed.port or 8000,
        )
        logger.info("Connected to ChromaDB at %s", chroma_host)
    else:
        data_path = os.environ.get("CHROMA_DATA_PATH", "chroma_data")
        client = chromadb.PersistentClient(path=data_path)
        logger.info("Connected to local ChromaDB at %s", data_path)

    try:
        collection = client.get_collection(collection_name)
    except Exception as exc:
        raise RuntimeError(
            f"Could not access collection '{collection_name}': {exc}"
        ) from exc

    logger.info("Collection '%s' — %d documents", collection_name, collection.count())
    return collection


def _clone_repo(repo_url: str, branch: str, tmpdir: str) -> str | None:
    """Clone repo_url@branch into tmpdir/repo. Returns repo_dir path or None on failure.

    Args:
        repo_url: Git repository URL.
        branch: Branch to clone.
        tmpdir: Parent temp directory.

    Returns:
        Absolute path to cloned repo directory, or None if clone failed.
    """
    repo_dir = str(Path(tmpdir) / "repo")
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"}
    try:
        proc = subprocess.run(
            ["git", "clone", "--branch", branch, "--depth", "1", repo_url, repo_dir],
            capture_output=True,
            text=True,
            timeout=180,
            env=env,
        )
        if proc.returncode != 0:
            logger.error(
                "git clone failed for %s@%s: %s",
                repo_url, branch, proc.stderr.strip()[:200],
            )
            return None
        logger.info("Cloned %s@%s → %s", repo_url, branch, repo_dir)
        return repo_dir
    except subprocess.TimeoutExpired:
        logger.error("git clone timed out for %s@%s", repo_url, branch)
        return None
    except Exception as exc:  # noqa: BLE001
        logger.error("git clone error for %s@%s: %s", repo_url, branch, exc)
        return None


def _compute_file_sha(repo_dir: str, file_path: str) -> str | None:
    """Compute SHA-256 of raw file bytes for the given relative file_path.

    Validates that the resolved path stays within repo_dir to prevent path
    traversal attacks from malicious file_path metadata values (e.g. "../../etc/passwd").

    Args:
        repo_dir: Absolute path to cloned repo root.
        file_path: Relative file path from repo root (as stored in ChromaDB metadata).

    Returns:
        Hex SHA-256 string, or None if the file cannot be read or escapes repo_dir.
    """
    repo_root = Path(repo_dir).resolve()
    abs_path = (repo_root / file_path).resolve()
    # Reject any path that resolves outside the cloned repo directory.
    if not str(abs_path).startswith(str(repo_root) + os.sep) and abs_path != repo_root:
        logger.warning("Path traversal rejected for file_path=%r — skipping", file_path)
        return None
    try:
        return hashlib.sha256(abs_path.read_bytes()).hexdigest()
    except OSError:
        return None


def main() -> None:
    """Run the backfill migration."""
    if DRY_RUN:
        logger.info("DRY RUN mode — no writes will be made")

    collection = _connect_chroma()

    # Fetch all documents with their metadata
    logger.info("Fetching all documents…")
    results = collection.get(include=["metadatas"])
    ids = results.get("ids") or []
    metadatas = results.get("metadatas") or []

    if not ids:
        logger.info("No documents found — nothing to backfill.")
        return

    # Partition: already-have-sha (skip) vs need-sha
    need_sha: list[tuple[str, dict]] = []  # (doc_id, metadata)
    already_done = 0
    for doc_id, meta in zip(ids, metadatas, strict=False):
        if meta and meta.get("source_file_sha"):
            already_done += 1
        else:
            need_sha.append((doc_id, meta or {}))

    logger.info(
        "%d documents total: %d already have source_file_sha, %d need backfill",
        len(ids), already_done, len(need_sha),
    )

    if not need_sha:
        logger.info("All documents already have source_file_sha — nothing to do.")
        return

    # Group documents by (repo_url, branch) to share clone operations
    groups: dict[tuple[str, str], list[tuple[str, dict]]] = defaultdict(list)
    no_repo: list[tuple[str, dict]] = []

    for doc_id, meta in need_sha:
        repo_url = meta.get("repo_url", "")
        branch = meta.get("branch", "main")
        if repo_url:
            groups[(repo_url, branch)].append((doc_id, meta))
        else:
            no_repo.append((doc_id, meta))

    if no_repo:
        logger.warning("%d documents have no repo_url — skipping them", len(no_repo))

    total_updated = 0
    total_skipped = 0
    total_missing = 0

    for (repo_url, branch), docs in groups.items():
        logger.info(
            "Processing %s@%s — %d documents", repo_url, branch, len(docs)
        )
        tmpdir = tempfile.mkdtemp(prefix="devmem_backfill_")
        atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)

        repo_dir = _clone_repo(repo_url, branch, tmpdir)
        if repo_dir is None:
            logger.warning("Skipping %d docs for %s@%s (clone failed)", len(docs), repo_url, branch)
            total_skipped += len(docs)
            continue

        repo_updated = 0
        repo_missing = 0

        for doc_id, meta in docs:
            file_path = meta.get("file_path", "")
            if not file_path:
                total_skipped += 1
                continue

            sha = _compute_file_sha(repo_dir, file_path)
            if sha is None:
                logger.debug("File not found at HEAD: %s — skipping", file_path)
                repo_missing += 1
                continue

            if not DRY_RUN:
                try:
                    collection.update(
                        ids=[doc_id],
                        metadatas=[{**meta, "source_file_sha": sha}],
                    )
                    repo_updated += 1
                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "Failed to update doc %s (%s): %s", doc_id[:8], file_path, exc
                    )
                    total_skipped += 1
                    continue
            else:
                repo_updated += 1  # count as "would update" in dry run

        logger.info(
            "  %s@%s: %d updated, %d missing at HEAD",
            repo_url, branch, repo_updated, repo_missing,
        )
        total_updated += repo_updated
        total_missing += repo_missing

        # Clean up clone immediately to free disk space
        shutil.rmtree(tmpdir, ignore_errors=True)

    action = "Would update" if DRY_RUN else "Updated"
    logger.info(
        "Backfill complete: %s %d documents, %d missing at HEAD, %d skipped",
        action, total_updated, total_missing, total_skipped,
    )

    if DRY_RUN:
        logger.info("Re-run without DRY_RUN=1 to apply changes.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        logger.error("Backfill failed: %s", exc)
        sys.exit(1)
