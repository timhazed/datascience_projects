"""Tests for src/mcp_server/git_client.py.

make_git_client() returns a closure — the inner git_client() callable is tested
by mocking subprocess.run, gitpython Repo, Settings, and compute_delta so no
real network calls or disk clones are made.

Cases:
  - make_git_client() returns a callable
  - callable returns (relative_paths, repo_root, commit_sha) tuple on success
  - callable raises RuntimeError on non-zero subprocess.run returncode
  - callable raises RuntimeError on subprocess.TimeoutExpired
  - timeout_secs == 0 path uses Repo.clone_from (no subprocess)
  - GitCommandError from gitpython is re-raised as RuntimeError
"""

import subprocess
from unittest.mock import MagicMock, patch

import pytest


class TestMakeGitClient:
    def test_returns_callable(self) -> None:
        """make_git_client() must return a callable, not raise at construction time."""
        from src.mcp_server.git_client import make_git_client

        client = make_git_client()
        assert callable(client)

    def _make_mock_repo(self, hexsha: str = "abc123def456") -> MagicMock:
        """Build a minimal fake gitpython Repo object."""
        repo = MagicMock()
        repo.head.commit.hexsha = hexsha
        return repo

    def _make_mock_settings(self, timeout: int = 60) -> MagicMock:
        """Build a fake Settings object with git_clone_timeout_secs set."""
        settings = MagicMock()
        settings.git_clone_timeout_secs = timeout
        return settings

    def test_success_with_timeout_returns_tuple(self, tmp_path) -> None:
        """Successful clone with timeout → (relative_paths, repo_root, commit_sha)."""
        from src.mcp_server.git_client import make_git_client

        mock_repo = self._make_mock_repo("deadbeef")
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        with (
            patch("src.mcp_server.git_client.Settings", return_value=self._make_mock_settings(60)),
            patch("src.mcp_server.git_client.subprocess.run", return_value=mock_proc),
            patch("src.mcp_server.git_client.compute_delta", return_value=["src/foo.py"]),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("atexit.register"),
            # Patch at git module level — Repo is imported inside make_git_client() body,
            # so it's a local name there, not a module-level attribute of git_client.
            patch("git.Repo", return_value=mock_repo),
        ):
            client = make_git_client()
            result = client("https://github.com/user/repo", "main")

        paths, repo_root, sha = result
        assert paths == ["src/foo.py"]
        assert sha == "deadbeef"
        assert isinstance(repo_root, str)

    def test_nonzero_returncode_raises_runtime_error(self, tmp_path) -> None:
        """subprocess.run returning non-zero exit code → RuntimeError raised."""
        from src.mcp_server.git_client import make_git_client

        mock_proc = MagicMock()
        mock_proc.returncode = 128
        mock_proc.stderr = "fatal: repository not found"

        with (
            patch("src.mcp_server.git_client.Settings", return_value=self._make_mock_settings(60)),
            patch("src.mcp_server.git_client.subprocess.run", return_value=mock_proc),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("atexit.register"),
        ):
            client = make_git_client()
            with pytest.raises(RuntimeError, match="git clone failed"):
                client("https://github.com/user/repo", "main")

    def test_timeout_expired_raises_runtime_error(self, tmp_path) -> None:
        """subprocess.TimeoutExpired → RuntimeError with timeout message."""
        from src.mcp_server.git_client import make_git_client

        with (
            patch("src.mcp_server.git_client.Settings", return_value=self._make_mock_settings(30)),
            patch(
                "src.mcp_server.git_client.subprocess.run",
                side_effect=subprocess.TimeoutExpired(cmd="git clone", timeout=30),
            ),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("atexit.register"),
        ):
            client = make_git_client()
            with pytest.raises(RuntimeError, match="timed out after 30s"):
                client("https://github.com/user/repo", "main")

    def test_zero_timeout_uses_repo_clone_from(self, tmp_path) -> None:
        """timeout_secs == 0 → Repo.clone_from used, no subprocess.run call."""
        from src.mcp_server.git_client import make_git_client

        mock_repo = self._make_mock_repo("cafebabe")

        with (
            patch("src.mcp_server.git_client.Settings", return_value=self._make_mock_settings(0)),
            # Patch at git module level — Repo is a local name inside make_git_client body.
            patch("git.Repo") as mock_repo_cls,
            patch("src.mcp_server.git_client.subprocess.run") as mock_subprocess,
            patch("src.mcp_server.git_client.compute_delta", return_value=["README.md"]),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("atexit.register"),
        ):
            mock_repo_cls.clone_from.return_value = mock_repo
            client = make_git_client()
            result = client("https://github.com/user/repo", "main")

        # subprocess.run must NOT be called when timeout_secs == 0
        mock_subprocess.assert_not_called()
        mock_repo_cls.clone_from.assert_called_once()
        paths, _, sha = result
        assert sha == "cafebabe"
        assert paths == ["README.md"]

    def test_git_command_error_raised_as_runtime_error(self, tmp_path) -> None:
        """GitCommandError from gitpython is caught and re-raised as RuntimeError."""
        from git import GitCommandError  # noqa: PLC0415

        from src.mcp_server.git_client import make_git_client

        mock_proc = MagicMock()
        mock_proc.returncode = 0

        # GitCommandError is raised after clone succeeds but Repo() constructor fails.
        # Patch at git module level — Repo is a local name inside make_git_client body.
        with (
            patch("src.mcp_server.git_client.Settings", return_value=self._make_mock_settings(60)),
            patch("src.mcp_server.git_client.subprocess.run", return_value=mock_proc),
            patch("tempfile.mkdtemp", return_value=str(tmp_path)),
            patch("atexit.register"),
            patch(
                "git.Repo",
                side_effect=GitCommandError(command="git clone", status=128, stderr="fatal: error"),
            ),
        ):
            client = make_git_client()
            with pytest.raises(RuntimeError, match="git operation failed"):
                client("https://github.com/user/repo", "main")
