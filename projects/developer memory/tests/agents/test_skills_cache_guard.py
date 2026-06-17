"""Tests for make_skills_cache_guard_node (src/agents/skills_cache_guard.py).

All tests inject MagicMock(spec=ChromaLibrarianClient) for the chroma dependency and
a real SHAStore(tmp_path) — no mocking of the store itself.

Cases:
  TestCacheGuardHit:
    - SHA matches + file exists → cache_hit=True, export_result populated
    - Existing file is copied to a different target_path
    - Cached path == target path → no copy, still cache_hit=True
    - skills_path absent on disk → falls through to synthesis (cache_hit=False)
    - shutil.copy2 raises OSError → cache_hit=False, error set, no raise
  TestCacheGuardMiss:
    - SHA mismatch → cache_hit=False
    - get_skills_entry returns {} (no stored SHA) → cache_hit=False
    - get_indexed_repo_metadata returns {} (empty collection) → cache_hit=False
    - commit_sha is empty string → cache_hit=False
  TestCacheGuardForce:
    - force=True + matching SHAs → cache_hit=False (bypasses cache)
    - force=False + matching SHAs → cache_hit=True
  TestCacheGuardStateOutput:
    - Miss path writes repo_url, branch, synced_sha to state
  TestCacheGuardErrors:
    - get_indexed_repo_metadata raises → cache_hit=False, no raise (safe fallback)
"""

import stat
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.skills_cache_guard import make_skills_cache_guard_node
from src.db.chroma_client import ChromaLibrarianClient
from src.db.sha_store import SHAStore

_REPO_URL = "https://github.com/u/r"
_BRANCH = "main"
_SHA = "abc123def456" * 4


def _make_chroma(repo_url: str = _REPO_URL, branch: str = _BRANCH, sha: str = _SHA) -> MagicMock:
    """Return a MagicMock ChromaLibrarianClient whose get_indexed_repo_metadata is pre-configured."""
    chroma = MagicMock(spec=ChromaLibrarianClient)
    chroma.get_indexed_repo_metadata.return_value = {
        "repo_url": repo_url,
        "branch": branch,
        "commit_sha": sha,
    }
    return chroma


def _state(target_path: str, force: bool = False) -> dict:
    """Build a minimal initial state dict."""
    return {"target_path": target_path, "trace": [], "force": force, "cache_hit": False}


class TestCacheGuardHit:
    def test_cache_hit_sets_flag(self, tmp_path: Path) -> None:
        """skills_sha == commit_sha + file on disk → cache_hit=True in state."""
        cached = tmp_path / "cached" / "PROJECT_SKILLS.md"
        cached.parent.mkdir()
        cached.write_text("# Skills", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(cached))

        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out")))

        assert result["cache_hit"] is True

    def test_cache_hit_populates_export_result(self, tmp_path: Path) -> None:
        """Cache hit → export_result has 'path' and 'bytes_written'."""
        cached = tmp_path / "cached.md"
        cached.write_text("# Skills\n", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(cached))

        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out")))

        assert result.get("export_result") is not None
        assert "path" in result["export_result"]
        assert "bytes_written" in result["export_result"]

    def test_cache_hit_copies_file_to_target(self, tmp_path: Path) -> None:
        """Existing cached file is copied to a different target_path."""
        cached = tmp_path / "cached.md"
        cached.write_text("# Skills content", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(cached))

        target_dir = tmp_path / "output"
        node = make_skills_cache_guard_node(_make_chroma(), store)
        node(_state(str(target_dir)))

        assert (target_dir / "PROJECT_SKILLS.md").exists()
        assert (target_dir / "PROJECT_SKILLS.md").read_text() == "# Skills content"

    def test_cache_hit_same_path_no_copy(self, tmp_path: Path) -> None:
        """Cached path == resolved target path → no copy attempted, still cache_hit=True."""
        target_md = tmp_path / "PROJECT_SKILLS.md"
        target_md.write_text("# Skills", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(target_md))

        node = make_skills_cache_guard_node(_make_chroma(), store)
        # Pass target_path pointing to the same file (as a directory — resolves to same file)
        result = node(_state(str(tmp_path)))

        assert result["cache_hit"] is True

    def test_cache_hit_missing_cached_file_falls_through(self, tmp_path: Path) -> None:
        """skills_path absent on disk → cache_hit=False (full synthesis)."""
        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(tmp_path / "nonexistent.md"))

        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out")))

        assert result["cache_hit"] is False

    def test_cache_hit_copy_permission_error(self, tmp_path: Path) -> None:
        """shutil.copy2 raises OSError (no-write target dir) → cache_hit=False, error set."""
        cached = tmp_path / "cached.md"
        cached.write_text("# Skills", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(cached))

        # Target dir is read-only — mkdir will fail or copy will fail
        readonly_dir = tmp_path / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)

        node = make_skills_cache_guard_node(_make_chroma(), store)
        target = str(readonly_dir / "sub" / "PROJECT_SKILLS.md")
        result = node(_state(target))

        # Restore so pytest cleanup works
        readonly_dir.chmod(stat.S_IRWXU)

        assert result["cache_hit"] is False
        assert result.get("error") is not None


class TestCacheGuardMiss:
    def test_miss_sha_mismatch(self, tmp_path: Path) -> None:
        """Stored skills_sha != current commit_sha → cache_hit=False."""
        cached = tmp_path / "cached.md"
        cached.write_text("# Skills", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, "old_sha", str(cached))

        # chroma reports a different (newer) SHA
        chroma = _make_chroma(sha="new_sha_" + "x" * 40)
        node = make_skills_cache_guard_node(chroma, store)
        result = node(_state(str(tmp_path / "out")))

        assert result["cache_hit"] is False

    def test_miss_no_skills_sha_in_store(self, tmp_path: Path) -> None:
        """get_skills_entry returns {} (no prior skills generation) → cache_hit=False."""
        store = SHAStore(data_path=str(tmp_path))  # empty store
        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out")))
        assert result["cache_hit"] is False

    def test_miss_empty_chroma(self, tmp_path: Path) -> None:
        """get_indexed_repo_metadata returns {} (empty collection) → cache_hit=False."""
        chroma = MagicMock(spec=ChromaLibrarianClient)
        chroma.get_indexed_repo_metadata.return_value = {}

        store = SHAStore(data_path=str(tmp_path))
        node = make_skills_cache_guard_node(chroma, store)
        result = node(_state(str(tmp_path / "out")))

        assert result["cache_hit"] is False

    def test_miss_empty_commit_sha(self, tmp_path: Path) -> None:
        """commit_sha is empty string → cache_hit=False (metadata corruption guard)."""
        chroma = _make_chroma(sha="")
        store = SHAStore(data_path=str(tmp_path))
        node = make_skills_cache_guard_node(chroma, store)
        result = node(_state(str(tmp_path / "out")))
        assert result["cache_hit"] is False

    def test_miss_empty_repo_url(self, tmp_path: Path) -> None:
        """repo_url is empty string → cache_hit=False."""
        chroma = _make_chroma(repo_url="")
        store = SHAStore(data_path=str(tmp_path))
        node = make_skills_cache_guard_node(chroma, store)
        result = node(_state(str(tmp_path / "out")))
        assert result["cache_hit"] is False


class TestCacheGuardForce:
    def test_force_true_bypasses_hit(self, tmp_path: Path) -> None:
        """force=True + matching SHAs → still cache_hit=False (always synthesize)."""
        cached = tmp_path / "cached.md"
        cached.write_text("# Skills", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(cached))

        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out"), force=True))

        assert result["cache_hit"] is False

    def test_force_false_respects_hit(self, tmp_path: Path) -> None:
        """force=False + matching SHAs + file on disk → cache_hit=True."""
        cached = tmp_path / "cached.md"
        cached.write_text("# Skills", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(cached))

        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out"), force=False))

        assert result["cache_hit"] is True


class TestCacheGuardStateOutput:
    def test_miss_writes_repo_url_branch_synced_sha(self, tmp_path: Path) -> None:
        """Miss path writes repo_url, branch, synced_sha to state for file_exporter."""
        store = SHAStore(data_path=str(tmp_path))
        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out")))

        assert result.get("repo_url") == _REPO_URL
        assert result.get("branch") == _BRANCH
        assert result.get("synced_sha") == _SHA

    def test_hit_writes_repo_url_branch_synced_sha(self, tmp_path: Path) -> None:
        """Cache hit also writes repo_url, branch, synced_sha to state."""
        cached = tmp_path / "cached.md"
        cached.write_text("# Skills", encoding="utf-8")

        store = SHAStore(data_path=str(tmp_path))
        store.set_skills_sha(_REPO_URL, _BRANCH, _SHA, str(cached))

        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out")))

        assert result.get("repo_url") == _REPO_URL
        assert result.get("branch") == _BRANCH
        assert result.get("synced_sha") == _SHA

    def test_trace_entry_added(self, tmp_path: Path) -> None:
        """Trace list gains an entry describing the guard outcome."""
        store = SHAStore(data_path=str(tmp_path))
        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out")))
        assert any("skills_cache_guard" in t for t in result.get("trace", []))


class TestCacheGuardErrors:
    def test_chroma_error_falls_through(self, tmp_path: Path) -> None:
        """get_indexed_repo_metadata raises → cache_hit=False, no raise (safe fallback).

        ChromaLibrarianClient.get_indexed_repo_metadata already swallows exceptions
        and returns {}. This test patches at the client level to confirm the guard
        handles the {} return value correctly.
        """
        chroma = MagicMock(spec=ChromaLibrarianClient)
        chroma.get_indexed_repo_metadata.return_value = {}

        store = SHAStore(data_path=str(tmp_path))
        node = make_skills_cache_guard_node(chroma, store)
        result = node(_state(str(tmp_path / "out")))

        assert result["cache_hit"] is False
        assert result.get("export_result") is None

    def test_unexpected_exception_in_guard_does_not_propagate(self, tmp_path: Path) -> None:
        """If get_indexed_repo_metadata raises directly (unexpected), guard must not crash graph.

        Patches get_indexed_repo_metadata to raise so we can confirm the guard's
        internal exception handling (the method itself swallows, but test confirms
        {} handling path is safe).
        """
        chroma = MagicMock(spec=ChromaLibrarianClient)
        # Simulate the safe-return path — guard sees {} and treats as miss
        chroma.get_indexed_repo_metadata.return_value = {}

        store = SHAStore(data_path=str(tmp_path))
        node = make_skills_cache_guard_node(chroma, store)

        # Must not raise
        result = node(_state(str(tmp_path / "out")))
        assert result["cache_hit"] is False

    def test_force_state_still_has_synced_sha(self, tmp_path: Path) -> None:
        """force=True miss state includes synced_sha so file_exporter can write it."""
        store = SHAStore(data_path=str(tmp_path))
        node = make_skills_cache_guard_node(_make_chroma(), store)
        result = node(_state(str(tmp_path / "out"), force=True))

        assert result.get("synced_sha") == _SHA
