"""Git utility helpers shared between server.py and delta_extractor.py.

Extracted here to avoid a circular import: delta_extractor imports _ls_remote_head
which was originally only in server.py. Moving both the sentinel and the function
to this module lets both server.py and delta_extractor.py import without cycles.
"""

import logging
import os
import subprocess

logger = logging.getLogger(__name__)

# Sentinel returned by _ls_remote_head when the remote is reachable but the
# requested branch does not exist.
_BRANCH_NOT_FOUND = "__BRANCH_NOT_FOUND__"


def _ls_remote_head(repo_url: str, branch: str, timeout: int = 15) -> str | None:
    """Call git ls-remote to get HEAD SHA without cloning.

    Uses GIT_TERMINAL_PROMPT=0 and GIT_ASKPASS=echo to prevent blocking on
    interactive credential prompts.

    Returns:
        SHA string on success.
        _BRANCH_NOT_FOUND if remote is reachable but branch is absent.
        None on any network/auth/timeout error (caller falls through to full clone).
    """
    try:
        result = subprocess.run(
            ["git", "ls-remote", repo_url, f"refs/heads/{branch}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_ASKPASS": "echo"},
        )
        if result.returncode == 0:
            lines = result.stdout.strip().splitlines()
            if not lines:
                return _BRANCH_NOT_FOUND
            return lines[0].split("\t")[0]
    except Exception:  # noqa: BLE001
        logger.warning(
            "_ls_remote_head: failed for %s@%s — falling through to full clone",
            repo_url,
            branch,
        )
    return None
