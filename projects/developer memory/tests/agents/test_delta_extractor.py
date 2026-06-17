"""Tests for make_delta_extractor_node (src/agents/delta_extractor.py).

Phase 4 exit gate — no live git operations. git_client is a lambda injected
into the factory.

Cases:
  - 2 changed files → changed_files has 2 entries, trace appended
  - 0 changed files → changed_files is empty list, no error
  - git_client raises → error populated, changed_files is []
  - error message is sanitized (no stack trace, no internal paths)
  - 3-tuple return → commit_sha written to state
  - 2-tuple return → commit_sha defaults to ""
  - SHA match → already_up_to_date=True, no clone
  - ls-remote returns None → falls through to full clone
  - __BRANCH_NOT_FOUND__ → error set, no clone attempt
"""

from unittest.mock import MagicMock, patch

from src.agents.delta_extractor import make_delta_extractor_node
from src.models.trace_event import TraceEventType
from src.utils.git_utils import _BRANCH_NOT_FOUND


class TestDeltaExtractorNode:
    def _state(self, repo_url: str = "https://github.com/user/repo", branch: str = "main") -> dict:
        return {
            "repo_url": repo_url,
            "branch": branch,
            "changed_files": [],
            "sanitized_files": [],
            "quarantined_files": [],
            "parsed_chunks": [],
            "summarized_chunks": [],
            "upsert_results": [],
            "error": None,
            "trace": [],
        }

    def test_two_changed_files_returned(self) -> None:
        """Mock git_client returning 2 files → changed_files has 2 entries."""
        git_client = lambda url, branch: ["src/main.py", "src/utils.py"]  # noqa: E731
        node = make_delta_extractor_node(git_client)

        result = node(self._state())

        assert result["changed_files"] == ["src/main.py", "src/utils.py"]
        assert len(result["trace"]) == 1
        assert "2 changed files" in result["trace"][0]

    def test_zero_changed_files_returned(self) -> None:
        """git_client returning [] → changed_files is empty list, no error."""
        git_client = lambda url, branch: []  # noqa: E731
        node = make_delta_extractor_node(git_client)

        result = node(self._state())

        assert result["changed_files"] == []
        assert "error" not in result or result.get("error") is None

    def test_git_error_sets_error_state(self) -> None:
        """git_client raising → error populated, changed_files is []."""

        def failing_client(url: str, branch: str) -> list[str]:
            raise RuntimeError("Repository not accessible")

        node = make_delta_extractor_node(failing_client)
        result = node(self._state())

        assert result["changed_files"] == []
        assert result["error"] is not None
        assert "trace" in result
        assert "error" in result["trace"][0]

    def test_error_message_is_sanitized(self) -> None:
        """Error returned to state does not contain stack traces or internal details."""

        def failing_client(url: str, branch: str) -> list[str]:
            raise OSError("Permission denied: /private/var/folders/secret")

        node = make_delta_extractor_node(failing_client)
        result = node(self._state())

        # Safe user-facing message, not raw exception message
        assert "Git delta computation failed" in result["error"]
        assert "Traceback" not in result["error"]

    def test_repo_url_and_branch_passed_to_client(self) -> None:
        """git_client receives the repo_url and branch from state."""
        received: list[tuple[str, str]] = []

        def recording_client(url: str, branch: str) -> list[str]:
            received.append((url, branch))
            return []

        node = make_delta_extractor_node(recording_client)
        node(self._state(repo_url="https://github.com/user/repo", branch="develop"))

        assert received == [("https://github.com/user/repo", "develop")]


class TestDeltaExtractorTupleReturn:
    def _state(self, repo_url: str = "https://github.com/user/repo", branch: str = "main") -> dict:
        return {
            "repo_url": repo_url,
            "branch": branch,
            "changed_files": [],
            "sanitized_files": [],
            "quarantined_files": [],
            "parsed_chunks": [],
            "upsert_results": [],
            "error": None,
            "trace": [],
        }

    def test_three_tuple_sets_commit_sha(self) -> None:
        """git_client returning 3-tuple → commit_sha written to state."""
        git_client = lambda url, branch: (["src/a.py"], "/tmp/repo", "abc123def456")  # noqa: E731
        node = make_delta_extractor_node(git_client)
        result = node(self._state())

        assert result["commit_sha"] == "abc123def456"
        assert result["repo_root"] == "/tmp/repo"
        assert result["changed_files"] == ["src/a.py"]

    def test_two_tuple_defaults_commit_sha_to_empty(self) -> None:
        """git_client returning 2-tuple → commit_sha defaults to empty string."""
        git_client = lambda url, branch: (["src/a.py"], "/tmp/repo")  # noqa: E731
        node = make_delta_extractor_node(git_client)
        result = node(self._state())

        assert result["commit_sha"] == ""
        assert result["repo_root"] == "/tmp/repo"

    def test_plain_list_defaults_commit_sha_and_repo_root_to_empty(self) -> None:
        """git_client returning plain list → commit_sha and repo_root default to ''."""
        git_client = lambda url, branch: ["src/a.py"]  # noqa: E731
        node = make_delta_extractor_node(git_client)
        result = node(self._state())

        assert result["commit_sha"] == ""
        assert result["repo_root"] == ""


class TestDeltaExtractorSHAFreshness:
    def _state(self, repo_url: str = "https://github.com/user/repo", branch: str = "main") -> dict:
        return {
            "repo_url": repo_url,
            "branch": branch,
            "changed_files": [],
            "sanitized_files": [],
            "quarantined_files": [],
            "parsed_chunks": [],
            "upsert_results": [],
            "error": None,
            "trace": [],
        }

    def _make_settings(self, sha_freshness_enabled: bool = True) -> MagicMock:
        s = MagicMock()
        s.sha_freshness_enabled = sha_freshness_enabled
        return s

    def test_sha_match_sets_already_up_to_date(self) -> None:
        """head_sha == cached_sha → already_up_to_date=True, git_client not called."""
        sha = "deadbeef" * 8
        mock_store = MagicMock()
        mock_store.get_last_sha.return_value = sha

        clones: list = []
        git_client = lambda url, branch: (clones.append(1), [])[-1] or []  # noqa: E731

        with patch("src.agents.delta_extractor._ls_remote_head", return_value=sha):
            node = make_delta_extractor_node(
                git_client,
                sha_store=mock_store,
                settings=self._make_settings(sha_freshness_enabled=True),
            )
            result = node(self._state())

        assert result.get("already_up_to_date") is True
        assert clones == []  # git_client was never called
        assert "already_up_to_date" in result["trace"][0]

    def test_sha_mismatch_proceeds_to_clone(self) -> None:
        """head_sha != cached_sha → clone runs, already_up_to_date not set."""
        mock_store = MagicMock()
        mock_store.get_last_sha.return_value = "old_sha"

        git_client = lambda url, branch: (["src/a.py"], "/tmp/repo", "new_sha")  # noqa: E731

        with patch("src.agents.delta_extractor._ls_remote_head", return_value="new_sha"):
            node = make_delta_extractor_node(
                git_client,
                sha_store=mock_store,
                settings=self._make_settings(sha_freshness_enabled=True),
            )
            result = node(self._state())

        assert not result.get("already_up_to_date")
        assert result["changed_files"] == ["src/a.py"]
        assert result["commit_sha"] == "new_sha"

    def test_ls_remote_returns_none_falls_through_to_clone(self) -> None:
        """_ls_remote_head returns None (network error) → clone proceeds normally."""
        mock_store = MagicMock()
        mock_store.get_last_sha.return_value = "cached_sha"

        git_client = lambda url, branch: (["src/a.py"], "/tmp/repo", "new_sha")  # noqa: E731

        with patch("src.agents.delta_extractor._ls_remote_head", return_value=None):
            node = make_delta_extractor_node(
                git_client,
                sha_store=mock_store,
                settings=self._make_settings(sha_freshness_enabled=True),
            )
            result = node(self._state())

        assert not result.get("already_up_to_date")
        assert result["changed_files"] == ["src/a.py"]

    def test_branch_not_found_sets_error_no_clone(self) -> None:
        """_ls_remote_head returns _BRANCH_NOT_FOUND → error set, no clone."""
        mock_store = MagicMock()
        mock_store.get_last_sha.return_value = None

        clones: list = []
        git_client = lambda url, branch: (clones.append(1), [])[-1] or []  # noqa: E731

        with patch("src.agents.delta_extractor._ls_remote_head", return_value=_BRANCH_NOT_FOUND):
            node = make_delta_extractor_node(
                git_client,
                sha_store=mock_store,
                settings=self._make_settings(sha_freshness_enabled=True),
            )
            result = node(self._state(branch="nonexistent"))

        assert result["error"] is not None
        assert "not found" in result["error"]
        assert clones == []  # no clone

    def test_sha_freshness_disabled_skips_ls_remote(self) -> None:
        """sha_freshness_enabled=False → _ls_remote_head not called, clone always runs."""
        mock_store = MagicMock()
        mock_store.get_last_sha.return_value = "some_sha"

        git_client = lambda url, branch: (["src/a.py"], "/tmp/repo", "some_sha")  # noqa: E731

        with patch("src.agents.delta_extractor._ls_remote_head") as mock_ls:
            node = make_delta_extractor_node(
                git_client,
                sha_store=mock_store,
                settings=self._make_settings(sha_freshness_enabled=False),
            )
            result = node(self._state())
            mock_ls.assert_not_called()

        # Clone ran since ls-remote was skipped → no already_up_to_date
        assert not result.get("already_up_to_date")


class TestDeltaExtractorTraceEvents:
    """Assert trace_events are emitted on every code path."""

    def _state(self) -> dict:
        return {"repo_url": "https://github.com/u/r", "branch": "main", "trace": [], "trace_events": []}

    def test_ok_path_emits_ok_event(self) -> None:
        """Successful clone emits TraceEventType.OK with count of changed files."""
        node = make_delta_extractor_node(lambda url, branch: ["a.py", "b.py"])
        result = node(self._state())
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].node == "delta_extractor"
        assert events[0].event == TraceEventType.OK
        assert events[0].count == 2

    def test_error_path_emits_error_event(self) -> None:
        """git_client raising emits TraceEventType.ERROR."""
        def boom(url, branch):
            raise RuntimeError("clone failed")
        node = make_delta_extractor_node(boom)
        result = node(self._state())
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].event == TraceEventType.ERROR

    def test_already_up_to_date_emits_skip_event(self) -> None:
        """SHA match path emits TraceEventType.SKIP."""
        store = MagicMock()
        store.get_last_sha.return_value = "abc123"
        with patch("src.agents.delta_extractor._ls_remote_head", return_value="abc123"):
            from src.config.settings import Settings
            settings = Settings(sha_freshness_enabled=True)
            node = make_delta_extractor_node(lambda u, b: [], sha_store=store, settings=settings)
            result = node(self._state())
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].event == TraceEventType.SKIP
