"""skills_cache_guard LangGraph node — Spec §3b (SkillsSHAGate).

SHA gate for generate_skills_pkg: probes ChromaDB for the current indexed commit SHA,
compares it to the stored skills SHA in sha_store.json, and short-circuits synthesis
when the corpus is unchanged.

On cache hit: copies the cached PROJECT_SKILLS.md to target_path, sets cache_hit=True,
and returns export_result so the graph routes directly to END.

On cache miss or force=True: sets cache_hit=False and writes repo_url, branch, synced_sha
to state so file_exporter can call set_skills_sha without re-probing ChromaDB.
"""

import logging
import shutil
from collections.abc import Callable
from pathlib import Path

from src.db.chroma_client import ChromaLibrarianClient
from src.db.sha_store import SHAStore

logger = logging.getLogger(__name__)

_OUTPUT_FILENAME = "PROJECT_SKILLS.md"


def make_skills_cache_guard_node(
    chroma: ChromaLibrarianClient,
    sha_store: SHAStore,
) -> Callable[[dict], dict]:
    """Return a skills_cache_guard node with injected ChromaLibrarianClient and SHAStore.

    Single concern: SHA gate — compare current indexed commit SHA to the stored skills
    SHA and decide whether synthesis can be skipped.

    Args:
        chroma: ChromaLibrarianClient singleton — used for the metadata probe.
        sha_store: SHAStore singleton — used to read and write skills SHA records.

    Returns:
        LangGraph node function implementing the SHA gate logic.
    """

    def _resolve_target(target_path: str) -> Path:
        """Resolve the write destination path, appending filename if target is a directory.

        Args:
            target_path: Raw target_path string from state.

        Returns:
            Resolved Path for the output file.
        """
        resolved = Path(target_path)
        if resolved.suffix.lower() != ".md":
            resolved = resolved / _OUTPUT_FILENAME
        return resolved

    def _miss_state(
        repo_url: str,
        branch: str,
        commit_sha: str,
        trace: list[str],
        reason: str,
    ) -> dict:
        """Build the state dict for a cache miss, routing to skills_aggregator.

        Args:
            repo_url: Git repository URL from ChromaDB metadata.
            branch: Branch name from ChromaDB metadata.
            commit_sha: Current commit SHA from ChromaDB metadata.
            trace: Running trace list.
            reason: Short description for the trace entry.

        Returns:
            State dict with cache_hit=False and SHA fields for file_exporter.
        """
        return {
            "cache_hit": False,
            "repo_url": repo_url,
            "branch": branch,
            "synced_sha": commit_sha,
            "trace": trace + [f"skills_cache_guard: miss ({reason})"],
        }

    def skills_cache_guard(state: dict) -> dict:
        """Execute the SHA gate: probe ChromaDB, compare SHAs, copy file or route to aggregator.

        Guard clauses handle: empty collection, empty SHA fields, force=True, SHA mismatch,
        missing cached file, and OSError on copy. All guard exits route to skills_aggregator
        (cache_hit=False). The happy path copies the file and sets cache_hit=True.

        Args:
            state: LangGraph state dict — reads target_path, force, trace.

        Returns:
            Dict with cache_hit, repo_url, branch, synced_sha, optional export_result/error,
            and trace.
        """
        target_path: str = state.get("target_path", "")
        force: bool = state.get("force", False)
        trace: list[str] = list(state.get("trace", []))

        # Probe ChromaDB for repo metadata — no embedding cost (collection.get limit=1)
        meta = chroma.get_indexed_repo_metadata()

        repo_url: str = meta.get("repo_url", "")
        branch: str = meta.get("branch", "")
        commit_sha: str = meta.get("commit_sha", "")

        # Guard: empty collection or missing metadata fields → miss
        if not repo_url or not branch or not commit_sha:
            return _miss_state(repo_url, branch, commit_sha, trace, "no indexed metadata")

        # Guard: force=True → always synthesize, but still populate SHA fields for file_exporter
        if force:
            return _miss_state(repo_url, branch, commit_sha, trace, "force=True")

        # Read skills cache entry — single lock acquisition for both sha and path
        entry = sha_store.get_skills_entry(repo_url, branch)
        stored_sha: str = entry.get("skills_sha", "")
        skills_path: str = entry.get("skills_path", "")

        # Guard: SHA mismatch or no stored SHA → miss
        if not stored_sha or stored_sha != commit_sha:
            return _miss_state(repo_url, branch, commit_sha, trace, "SHA mismatch or miss")

        # Guard: cached file absent on disk → fall through to synthesis (non-fatal)
        cached = Path(skills_path)
        if not cached.exists():
            logger.warning(
                "skills_cache_guard: cache hit but skills_path not on disk (%s) — synthesizing",
                skills_path,
            )
            return _miss_state(repo_url, branch, commit_sha, trace, "cached file missing")

        # Happy path: SHA matches and file exists — copy to target_path
        target = _resolve_target(target_path)
        try:
            # Only copy when paths differ — avoids a no-op write to the same file
            if cached.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached, target)
        except OSError as exc:
            logger.warning(
                "skills_cache_guard: copy failed [%s]: %s — falling through to synthesis",
                type(exc).__name__,
                str(exc)[:200],
            )
            return {
                **_miss_state(repo_url, branch, commit_sha, trace, "copy OSError"),
                "error": "Cache copy failed; regenerating.",
            }

        logger.info("skills_cache_guard: cache hit — skipping synthesis (%s)", commit_sha[:12])
        return {
            "cache_hit": True,
            "repo_url": repo_url,
            "branch": branch,
            "synced_sha": commit_sha,
            "export_result": {"path": str(target), "bytes_written": cached.stat().st_size},
            "trace": trace + [f"skills_cache_guard: hit (sha={commit_sha[:12]})"],
        }

    return skills_cache_guard
