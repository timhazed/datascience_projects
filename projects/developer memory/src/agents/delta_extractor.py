"""delta_extractor LangGraph node — Spec §2.1.

Computes the set of files that changed between HEAD and its parent commit. Always the
first node in the Sync Pipeline — if it fails, the pipeline terminates early.

The git_client injected here is a callable `(repo_url: str, branch: str) -> list[str]`
that encapsulates repository access (clone/fetch + compute_delta). This abstraction
makes the node trivially testable: inject a lambda in tests, wire compute_delta in prod.

Production git_client contract:
    Returns a 3-tuple (relative_paths: list[str], repo_root: str, commit_sha: str) so
    that pii_sanitizer and multimodal_parser can resolve absolute file paths for reading.
    The cloned repo directory (repo_root) must remain on disk until the pipeline
    completes — server.py's _make_git_client() uses tempfile.mkdtemp() (not a context
    manager) and registers atexit cleanup so the directory is never deleted mid-pipeline.

Test lambda contract:
    May return a plain list[str] or a 2-tuple (list[str], str) — all forms are handled
    by the node. 3-tuple is the new canonical production form.

SHA freshness gate (spec §6A):
    If sha_store is provided, get_last_sha() is called before cloning. If sha_freshness
    is enabled in settings, _ls_remote_head() is called to check remote HEAD SHA without
    cloning. If head_sha matches cached_sha → already_up_to_date=True, skip clone.
"""

import logging
from collections.abc import Callable

from src.models.trace_event import TraceEvent, TraceEventType
from src.utils.git_utils import _BRANCH_NOT_FOUND, _ls_remote_head

logger = logging.getLogger(__name__)


def make_delta_extractor_node(
    git_client: Callable[[str, str], list[str]],
    sha_store=None,
    settings=None,
) -> Callable[[dict], dict]:
    """Return a delta_extractor node with injected git client.

    Args:
        git_client: Callable(repo_url, branch) → list[str] | tuple[list[str], str] |
            tuple[list[str], str, str].
            Production client returns a 3-tuple (relative_paths, repo_root, commit_sha).
            Test lambdas may return a plain list[str] or 2-tuple — both forms handled.
        sha_store: Optional SHAStore instance for SHA freshness gate. If provided,
            get_last_sha() is called before cloning to check if repo is up-to-date.
        settings: Optional Settings instance. Used for sha_freshness_enabled flag.
            If None, SHA freshness check is skipped.

    Returns:
        LangGraph node function that reads repo_url and branch from SyncState and
        writes changed_files, repo_root, commit_sha back to state.
    """

    def delta_extractor(state: dict) -> dict:
        """Compute changed files for the repository and branch in state.

        On SHA match: sets already_up_to_date=True and returns (no clone).
        On success: sets changed_files to relative file paths, repo_root to the
            absolute path of the cloned repo directory, commit_sha to HEAD SHA.
        On failure: sets error to a safe message, changed_files to [], trace updated.
        """
        repo_url: str = state.get("repo_url", "")
        branch: str = state.get("branch", "main")

        # ── SHA freshness gate ────────────────────────────────────────────────
        # Step 1: Get cached SHA from sha_store (if provided)
        cached_sha: str | None = None
        if sha_store is not None:
            try:
                cached_sha = sha_store.get_last_sha(repo_url, branch)
            except Exception:  # noqa: BLE001
                cached_sha = None

        # Step 2: Get remote HEAD SHA via git ls-remote (if settings enable it)
        head_sha: str | None = None
        if settings is not None and getattr(settings, "sha_freshness_enabled", False):
            head_sha = _ls_remote_head(repo_url, branch)

            # Branch not found on remote — set error, no clone attempt
            if head_sha == _BRANCH_NOT_FOUND:
                logger.error(
                    "delta_extractor: branch '%s' not found in %s", branch, repo_url
                )
                return {
                    "changed_files": [],
                    "repo_root": "",
                    "commit_sha": "",
                    "error": f"Branch '{branch}' not found in remote repository.",
                    "head_sha": _BRANCH_NOT_FOUND,
                    "trace": [f"delta_extractor: branch '{branch}' not found"],
                    "trace_events": [
                        TraceEvent(
                            node="delta_extractor",
                            event=TraceEventType.ERROR,
                            detail=f"branch '{branch}' not found",
                        )
                    ],
                }

        # Step 3: SHA match → already up to date, skip clone
        if head_sha is not None and cached_sha is not None and head_sha == cached_sha:
            logger.info(
                "delta_extractor: already_up_to_date for %s@%s (sha=%s)",
                repo_url, branch, head_sha[:8],
            )
            return {
                "changed_files": [],
                "repo_root": "",
                "commit_sha": head_sha,
                "already_up_to_date": True,
                "head_sha": head_sha,
                "cached_sha": cached_sha,
                "trace": [f"delta_extractor: already_up_to_date (sha={head_sha[:8]})"],
                "trace_events": [
                    TraceEvent(
                        node="delta_extractor",
                        event=TraceEventType.SKIP,
                        detail=f"already_up_to_date (sha={head_sha[:8]})",
                    )
                ],
            }

        # ── Clone and compute delta ───────────────────────────────────────────
        try:
            result = git_client(repo_url, branch)
            # Handle all return forms from git_client:
            # 3-tuple (new production): (relative_paths, repo_root, commit_sha)
            # 2-tuple (legacy):         (relative_paths, repo_root)
            # plain list (test lambda): relative_paths
            if isinstance(result, tuple) and len(result) == 3:
                changed, repo_root, commit_sha = result
            elif isinstance(result, tuple):
                changed, repo_root = result
                commit_sha = ""
            else:
                changed = result
                repo_root = ""
                commit_sha = ""

            logger.info(
                "delta_extractor: %d changed files in %s@%s", len(changed), repo_url, branch
            )

            out: dict = {
                "changed_files": changed,
                "repo_root": repo_root,
                "commit_sha": commit_sha,
                "trace": [f"delta_extractor: {len(changed)} changed files"],
                "trace_events": [
                    TraceEvent(
                        node="delta_extractor",
                        event=TraceEventType.OK,
                        count=len(changed),
                    )
                ],
            }
            if head_sha is not None:
                out["head_sha"] = head_sha
            if cached_sha is not None:
                out["cached_sha"] = cached_sha
            return out

        except Exception as exc:
            logger.error(
                "delta_extractor failed [%s]: %s", type(exc).__name__, str(exc)[:200]
            )
            return {
                "changed_files": [],
                "repo_root": "",
                "commit_sha": "",
                "error": "Git delta computation failed. Please retry.",
                "trace": ["delta_extractor: error"],
                "trace_events": [
                    TraceEvent(
                        node="delta_extractor",
                        event=TraceEventType.ERROR,
                        detail=type(exc).__name__,
                    )
                ],
            }

    return delta_extractor
