"""Git clone orchestration for the Developer Memory MCP server.

Owns: the production git client callable injected into build_sync_graph().

make_git_client() returns a closure that clones a repository into a managed
temp directory and returns the delta file paths required by the sync pipeline.

Lifetime note: the cloned repo lives in a tempfile.mkdtemp() directory (not a
context-manager TemporaryDirectory). This is intentional — the pipeline runs
synchronously, so pii_sanitizer must be able to read file content after
delta_extractor returns. Using a context-manager would destroy the tmpdir
before pii_sanitizer executes. The directory is registered with atexit for
cleanup so it is always removed when the process exits, even on error.

gitpython, atexit, shutil, and tempfile are lazy-imported inside make_git_client()
because gitpython is a heavy dependency unused in all non-sync paths (e.g. query,
persona, diff). Lazy import keeps import-time cost near zero for those paths.
"""

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

from src.config.settings import Settings
from src.ingest.git_delta import compute_delta
from src.mcp_server.startup import logger


def make_git_client() -> Callable[[str, str], tuple[list[str], str, str]]:
    """Return the production git client callable.

    The returned callable accepts (repo_url, branch) and returns a 3-tuple
    (relative_paths, repo_root, commit_sha) consumed by delta_extractor to
    write repo_root and commit_sha into SyncState.

    gitpython (heavy) is lazy-imported inside this function so import-time cost
    is zero for non-sync paths (query, persona, diff, skills).

    Returns:
        Callable[[str, str], tuple[list[str], str, str]] — the git client.
    """
    # Lazy imports: gitpython is heavy and unused in all non-sync code paths.
    # Placing them here rather than at module level keeps startup time fast.
    import atexit  # noqa: PLC0415
    import shutil  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    try:
        from git import GitCommandError, Repo  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "gitpython is required for git clone operations. "
            "Install it with: pip install gitpython"
        ) from exc

    def git_client(repo_url: str, branch: str) -> tuple[list[str], str, str]:
        """Clone repo_url into a managed temp dir and return sync delta info.

        Args:
            repo_url: Validated git repository URL.
            branch: Branch name to check out and diff.

        Returns:
            3-tuple: (relative_paths, repo_root, commit_sha) so delta_extractor
            can write repo_root and commit_sha to SyncState.

        Raises:
            RuntimeError: On git operation failure or clone timeout.
        """
        try:
            # mkdtemp — NOT a context manager — so the directory outlives this function.
            # pii_sanitizer reads files in a later LangGraph node; the dir must still exist.
            # The atexit handler guarantees cleanup even if the pipeline errors.
            tmpdir = tempfile.mkdtemp(prefix="devmem_repo_")
            atexit.register(shutil.rmtree, tmpdir, ignore_errors=True)

            repo_dir = Path(tmpdir) / "repo"

            clone_settings = Settings()
            timeout_secs = clone_settings.git_clone_timeout_secs
            # Disable interactive prompts — required for non-interactive Docker/CI runs.
            clone_env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"}

            # Embed PAT in HTTPS URL for private repo access. Token is never logged —
            # only the sanitized repo_url (without credentials) appears in log lines.
            clone_url = repo_url
            if clone_settings.github_token and repo_url.startswith("https://"):
                clone_url = repo_url.replace(
                    "https://", f"https://{clone_settings.github_token}@", 1
                )

            if timeout_secs > 0:
                try:
                    proc = subprocess.run(
                        ["git", "clone", "--branch", branch, clone_url, str(repo_dir)],
                        capture_output=True,
                        text=True,
                        timeout=timeout_secs,
                        env=clone_env,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"git clone failed: {proc.stderr.strip()[:200]}"
                        )
                    repo = Repo(str(repo_dir))
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f"git clone timed out after {timeout_secs}s — "
                        "check network connectivity and repository accessibility."
                    ) from exc
            else:
                # timeout_secs == 0 means no timeout — use gitpython directly.
                repo = Repo.clone_from(clone_url, repo_dir, branch=branch, env=clone_env)

            relative_paths = compute_delta(repo, branch)
            logger.debug(
                "git_client: cloned %s@%s → %d changed files",
                repo_url,
                branch,
                len(relative_paths),
            )
            # Return 3-tuple: delta_extractor unpacks and writes these to SyncState.
            return (relative_paths, str(repo_dir), repo.head.commit.hexsha)

        except GitCommandError as exc:
            raise RuntimeError(f"git operation failed: {exc.stderr.strip()[:200]}") from exc

    return git_client
