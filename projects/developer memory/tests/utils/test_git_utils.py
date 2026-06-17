"""Tests for git utility helpers (src/utils/git_utils.py).

Cases:
  - _ls_remote_head: success returns SHA string
  - _ls_remote_head: returncode != 0 → returns None
  - _ls_remote_head: empty stdout → returns _BRANCH_NOT_FOUND sentinel
  - _ls_remote_head: TimeoutExpired → returns None
  - _ls_remote_head: other exception → returns None
  - _BRANCH_NOT_FOUND sentinel value is the expected string
"""

import subprocess
from unittest.mock import MagicMock, patch

from src.utils.git_utils import _BRANCH_NOT_FOUND, _ls_remote_head


class TestBranchNotFoundSentinel:
    def test_sentinel_value(self) -> None:
        """_BRANCH_NOT_FOUND is the expected sentinel string."""
        assert _BRANCH_NOT_FOUND == "__BRANCH_NOT_FOUND__"


class TestLsRemoteHead:
    def _mock_proc(self, returncode: int = 0, stdout: str = "") -> MagicMock:
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = stdout
        return proc

    def test_success_returns_sha(self) -> None:
        """Valid ls-remote output → SHA string returned."""
        sha = "abc123def456" * 3
        stdout = f"{sha}\trefs/heads/main\n"
        proc = self._mock_proc(returncode=0, stdout=stdout)

        with patch("subprocess.run", return_value=proc):
            result = _ls_remote_head("https://github.com/u/r", "main")

        assert result == sha

    def test_nonzero_returncode_returns_none(self) -> None:
        """Non-zero returncode (auth failure etc.) → None."""
        proc = self._mock_proc(returncode=128, stdout="")

        with patch("subprocess.run", return_value=proc):
            result = _ls_remote_head("https://github.com/u/r", "main")

        assert result is None

    def test_empty_stdout_returns_branch_not_found(self) -> None:
        """returncode=0 but empty stdout → branch absent → _BRANCH_NOT_FOUND."""
        proc = self._mock_proc(returncode=0, stdout="")

        with patch("subprocess.run", return_value=proc):
            result = _ls_remote_head("https://github.com/u/r", "nonexistent")

        assert result == _BRANCH_NOT_FOUND

    def test_timeout_returns_none(self) -> None:
        """subprocess.TimeoutExpired → None (caller falls through to full clone)."""
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="git", timeout=15)):
            result = _ls_remote_head("https://github.com/u/r", "main", timeout=15)

        assert result is None

    def test_generic_exception_returns_none(self) -> None:
        """Any unexpected exception → None."""
        with patch("subprocess.run", side_effect=OSError("network unreachable")):
            result = _ls_remote_head("https://github.com/u/r", "main")

        assert result is None

    def test_uses_git_terminal_prompt_env(self) -> None:
        """GIT_TERMINAL_PROMPT=0 and GIT_ASKPASS=echo must be in subprocess env."""
        proc = self._mock_proc(returncode=0, stdout="sha123\trefs/heads/main")
        call_env: list[dict] = []

        def capture_run(*args, **kwargs):
            call_env.append(kwargs.get("env", {}))
            return proc

        with patch("subprocess.run", side_effect=capture_run):
            _ls_remote_head("https://github.com/u/r", "main")

        assert call_env, "subprocess.run was not called"
        env = call_env[0]
        assert env.get("GIT_TERMINAL_PROMPT") == "0"
        assert env.get("GIT_ASKPASS") == "echo"
