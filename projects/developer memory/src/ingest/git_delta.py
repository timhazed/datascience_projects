"""Git delta computation — Spec §4 delta_extractor support.

compute_delta() returns all file paths present in the HEAD tree of the given branch.
This performs a full-tree ingest: every file on the branch tip is returned so that
the sync pipeline indexes the complete state of the repository, not just the last
commit's diff.

Deleted files (type != "blob") are excluded — they have no content to embed.

This is a pure function with no LangGraph coupling — it is called by
make_delta_extractor_node() with a GitPython Repo object (or mock) injected via the
factory closure.
"""

import logging

from git import Repo

logger = logging.getLogger(__name__)


def compute_delta(repo: Repo, branch: str = "main") -> list[str]:
    """Return all file paths present in the HEAD tree of the cloned branch.

    Walks the entire HEAD commit tree and returns one path per blob (file).
    Directories (tree objects) are excluded. This produces a full-repo ingest —
    every file on the branch tip is returned regardless of when it was last modified.

    Args:
        repo: GitPython Repo instance pointing to the target repository.
        branch: Branch name — informational only; the caller clones the correct
            branch before calling this function.

    Returns:
        List of file paths relative to the repository root. Empty list only if the
        repository has no files (e.g., empty initial commit with no tree).

    Raises:
        git.GitCommandError: Re-raised on git operation failure so make_delta_extractor_node
            can catch it and set the error state.
    """
    head = repo.head.commit
    blobs = [item for item in head.tree.traverse() if item.type == "blob"]
    logger.debug("compute_delta: branch=%s, %d files in HEAD tree", branch, len(blobs))
    return [blob.path for blob in blobs]
