"""Tests for compute_delta (src/ingest/git_delta.py).

No live git operations — GitPython Repo is mocked in all tests.

Strategy: compute_delta always walks the HEAD tree and returns all blobs.
There is no parent-diff path — the function performs a full-tree ingest.

Cases:
  - HEAD tree with N blobs → N paths returned
  - HEAD tree with no blobs → empty list
  - Tree items that are directories (type="tree") are excluded
  - git.GitCommandError re-raised to caller (delta_extractor handles it)
"""

from unittest.mock import MagicMock

import pytest

from src.ingest.git_delta import compute_delta


def _make_blob(path: str) -> MagicMock:
    """Create a fake tree blob."""
    blob = MagicMock()
    blob.path = path
    blob.type = "blob"
    return blob


def _make_repo(blob_paths: list[str]) -> MagicMock:
    """Build a mock GitPython Repo whose HEAD tree contains the given blobs."""
    repo = MagicMock()
    blobs = [_make_blob(p) for p in blob_paths]
    repo.head.commit.tree.traverse.return_value = blobs
    return repo


class TestComputeDelta:
    def test_two_files_in_tree_returned(self) -> None:
        """HEAD tree with 2 blobs → list of 2 file paths."""
        repo = _make_repo(["src/main.py", "src/utils.py"])
        result = compute_delta(repo, branch="main")
        assert result == ["src/main.py", "src/utils.py"]

    def test_empty_tree_returns_empty_list(self) -> None:
        """HEAD tree with no blobs → empty list."""
        repo = _make_repo([])
        result = compute_delta(repo)
        assert result == []

    def test_directory_tree_items_excluded(self) -> None:
        """Non-blob tree items (directories) are excluded from result."""
        repo = MagicMock()
        blob = _make_blob("src/app.py")
        tree_item = MagicMock()
        tree_item.type = "tree"
        tree_item.path = "src"
        repo.head.commit.tree.traverse.return_value = [blob, tree_item]

        result = compute_delta(repo)
        assert result == ["src/app.py"]

    def test_large_tree_returns_all_blobs(self) -> None:
        """HEAD tree with many blobs → all paths returned."""
        paths = [f"project_{i}/src/main.py" for i in range(30)]
        repo = _make_repo(paths)
        result = compute_delta(repo, branch="trunk")
        assert len(result) == 30
        assert set(result) == set(paths)

    def test_git_error_re_raised(self) -> None:
        """Git exceptions propagate to the caller (delta_extractor handles them)."""
        from git import GitCommandError

        repo = MagicMock()
        repo.head.commit.tree.traverse.side_effect = GitCommandError("git ls-tree", 128)

        with pytest.raises(GitCommandError):
            compute_delta(repo)
