"""Tests for make_cache_filter_node (src/agents/cache_filter.py).

Uses tmp_path fixture + EphemeralClient for isolation. No live ChromaDB.

Cases:
  - Cached file (sha matches) is removed from changed_files
  - Uncached file is kept in changed_files and recorded in file_shas
  - cache_filter_enabled=False passes all files through unchanged
  - _should_skip file (dotfile, image) is dropped without calling read_bytes
  - File > _MAX_FILE_BYTES is passed through without calling read_bytes
  - stat() OSError → passed through (fail open)
  - read_bytes() OSError → passed through (fail open)
  - Single _collection.get() call regardless of N files
"""

import hashlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import chromadb

from src.agents.cache_filter import make_cache_filter_node
from src.agents.pii_sanitizer import _MAX_FILE_BYTES
from src.config.settings import Settings
from src.db.chroma_client import ChromaLibrarianClient
from src.models.trace_event import TraceEventType


def _fake_embed(texts: list[str]) -> list[list[float]]:
    return [[0.1, 0.2, 0.3, 0.4] for _ in texts]


def _make_chroma() -> ChromaLibrarianClient:
    import uuid
    return ChromaLibrarianClient(
        client=chromadb.EphemeralClient(),
        embedding_fn=_fake_embed,
        collection_name=f"test_{uuid.uuid4().hex[:12]}",
    )


def _make_settings(enabled: bool = True) -> Settings:
    s = MagicMock(spec=Settings)
    s.cache_filter_enabled = enabled
    return s


def _base_state(changed_files: list[str], repo_root: str = "", repo_url: str = "https://github.com/u/r") -> dict:
    return {
        "repo_url": repo_url,
        "branch": "main",
        "changed_files": changed_files,
        "repo_root": repo_root,
        "trace": [],
    }


def _write(tmp_path: Path, name: str, content: bytes = b"def hello(): pass\n") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


class TestCacheFilterDisabled:
    def test_disabled_passes_all_files_unchanged(self, tmp_path: Path) -> None:
        """cache_filter_enabled=False → all files passed through, no ChromaDB call."""
        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=False)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(["src/a.py", "src/b.py"], repo_root=str(tmp_path))
        result = node(state)

        # When disabled, changed_files is NOT overwritten (pass-through returns no changed_files key)
        assert result.get("cache_filtered_count") == 0
        assert "disabled" in result["trace"][0]
        # _collection.get should NOT be called when disabled
        chroma._collection.get.assert_not_called()


class TestCacheFilterCachedFile:
    def test_cached_file_removed_from_changed_files(self, tmp_path: Path) -> None:
        """File whose SHA is in ChromaDB is removed from changed_files."""
        content = b"def cached_function(): pass\n"
        sha = hashlib.sha256(content).hexdigest()
        _write(tmp_path, "cached.py", content)

        chroma = _make_chroma()
        # Mock _collection.get to return this sha as already indexed
        chroma._collection.get = MagicMock(return_value={
            "metadatas": [{"source_file_sha": sha, "repo_url": "https://github.com/u/r"}]
        })
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(["cached.py"], repo_root=str(tmp_path))
        result = node(state)

        assert "cached.py" not in result["changed_files"]
        assert result["cache_filtered_count"] == 1
        assert result.get("file_shas", {}).get("cached.py") is None

    def test_uncached_file_kept_in_changed_files(self, tmp_path: Path) -> None:
        """File whose SHA is NOT in ChromaDB is kept and its SHA recorded."""
        content = b"def new_function(): pass\n"
        sha = hashlib.sha256(content).hexdigest()
        _write(tmp_path, "new.py", content)

        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(["new.py"], repo_root=str(tmp_path))
        result = node(state)

        assert "new.py" in result["changed_files"]
        assert result["file_shas"]["new.py"] == sha
        assert result["cache_filtered_count"] == 0


class TestCacheFilterSkipRules:
    def test_should_skip_file_dropped_without_read_bytes(self, tmp_path: Path) -> None:
        """Dotfile → dropped via _should_skip without reading file bytes."""
        _write(tmp_path, ".env", b"SECRET=abc\n")

        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        # .env is a dotfile → _should_skip returns "dotfile"
        state = _base_state([".env"], repo_root=str(tmp_path))

        with patch("src.agents.cache_filter.Path.read_bytes") as mock_read:
            result = node(state)
            mock_read.assert_not_called()

        # Dropped from changed_files (not passed to pii_sanitizer)
        assert ".env" not in result["changed_files"]
        assert result["cache_filtered_count"] == 0  # not counted as cached

    def test_image_file_skipped_without_read_bytes(self, tmp_path: Path) -> None:
        """Image file (.png) → dropped via _should_skip without reading bytes."""
        _write(tmp_path, "icon.png", b"\x89PNG\r\n\x1a\n")

        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(["icon.png"], repo_root=str(tmp_path))

        with patch("src.agents.cache_filter.Path.read_bytes") as mock_read:
            result = node(state)
            mock_read.assert_not_called()

        assert "icon.png" not in result["changed_files"]

    def test_oversized_file_passed_through_without_read_bytes(self, tmp_path: Path) -> None:
        """File > _MAX_FILE_BYTES → passed through to pii_sanitizer without reading."""
        py_file = tmp_path / "large.py"
        py_file.write_bytes(b"x = 1")

        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(["large.py"], repo_root=str(tmp_path))

        # Mock stat to return oversized file
        mock_stat_result = MagicMock()
        mock_stat_result.st_size = _MAX_FILE_BYTES + 1

        with (
            patch("src.agents.cache_filter.Path.stat", return_value=mock_stat_result),
            patch("src.agents.cache_filter.Path.read_bytes") as mock_read,
        ):
            result = node(state)
            mock_read.assert_not_called()

        assert "large.py" in result["changed_files"]
        assert result["cache_filtered_count"] == 0


class TestCacheFilterErrorHandling:
    def test_stat_oserror_passed_through(self, tmp_path: Path) -> None:
        """stat() OSError → file passed through (fail open)."""
        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        # Path does not exist → stat() raises OSError
        state = _base_state(["missing.py"], repo_root=str(tmp_path))
        result = node(state)

        assert "missing.py" in result["changed_files"]
        assert result["cache_filtered_count"] == 0

    def test_read_bytes_oserror_passed_through(self, tmp_path: Path) -> None:
        """read_bytes() OSError → file passed through (fail open)."""
        py_file = tmp_path / "unreadable.py"
        py_file.write_bytes(b"content")

        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(["unreadable.py"], repo_root=str(tmp_path))

        mock_stat = MagicMock()
        mock_stat.st_size = 10  # small file, passes size check

        with (
            patch("src.agents.cache_filter.Path.stat", return_value=mock_stat),
            patch("src.agents.cache_filter.Path.read_bytes", side_effect=OSError("Permission denied")),
        ):
            result = node(state)

        assert "unreadable.py" in result["changed_files"]

    def test_chroma_bulk_fetch_error_passes_all_through(self, tmp_path: Path) -> None:
        """ChromaDB get() failure → all files passed through (fail open)."""
        _write(tmp_path, "a.py", b"x=1")
        _write(tmp_path, "b.py", b"y=2")

        chroma = _make_chroma()
        chroma._collection.get = MagicMock(side_effect=RuntimeError("ChromaDB down"))
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(["a.py", "b.py"], repo_root=str(tmp_path))
        result = node(state)

        assert "a.py" in result["changed_files"]
        assert "b.py" in result["changed_files"]
        assert result["cache_filtered_count"] == 0


class TestCacheFilterSingleBulkFetch:
    def test_single_collection_get_for_n_files(self, tmp_path: Path) -> None:
        """_collection.get() is called exactly once regardless of N files."""
        files = []
        for i in range(10):
            name = f"file{i}.py"
            _write(tmp_path, name, f"def func_{i}(): pass\n".encode())
            files.append(name)

        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(files, repo_root=str(tmp_path))
        node(state)

        # Exactly one bulk call, regardless of file count
        assert chroma._collection.get.call_count == 1


class TestCacheFilterMixedFiles:
    def test_cached_and_uncached_split_correctly(self, tmp_path: Path) -> None:
        """Mix of cached and uncached: only uncached files in changed_files after filter."""
        cached_content = b"def cached(): pass\n"
        new_content = b"def new(): pass\n"
        cached_sha = hashlib.sha256(cached_content).hexdigest()

        _write(tmp_path, "cached.py", cached_content)
        _write(tmp_path, "new.py", new_content)

        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={
            "metadatas": [{"source_file_sha": cached_sha}]
        })
        settings = _make_settings(enabled=True)

        node = make_cache_filter_node(chroma, settings)
        state = _base_state(["cached.py", "new.py"], repo_root=str(tmp_path))
        result = node(state)

        assert result["changed_files"] == ["new.py"]
        assert result["cache_filtered_count"] == 1
        assert "new.py" in result["file_shas"]
        assert "cached.py" not in result.get("file_shas", {})



class TestCacheFilterTraceEvents:
    """Assert trace_events are emitted on every code path."""

    def test_disabled_emits_skip_event(self) -> None:
        """cache_filter_enabled=False emits TraceEventType.SKIP with detail='disabled'."""
        chroma = _make_chroma()
        settings = _make_settings(enabled=False)
        node = make_cache_filter_node(chroma, settings)
        result = node(_base_state([]))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].node == "cache_filter"
        assert events[0].event == TraceEventType.SKIP
        assert events[0].detail == "disabled"

    def test_normal_path_emits_cached_event(self, tmp_path) -> None:
        """Enabled path with 0 cached files emits TraceEventType.CACHED with count=0."""
        chroma = _make_chroma()
        chroma._collection.get = MagicMock(return_value={"metadatas": []})
        settings = _make_settings(enabled=True)
        node = make_cache_filter_node(chroma, settings)
        result = node(_base_state([], repo_root=str(tmp_path)))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].event == TraceEventType.CACHED
        assert events[0].count == 0

    def test_fetch_error_emits_error_event(self) -> None:
        """ChromaDB bulk fetch raising emits TraceEventType.ERROR."""
        chroma = _make_chroma()
        chroma._collection.get = MagicMock(side_effect=RuntimeError("db down"))
        settings = _make_settings(enabled=True)
        node = make_cache_filter_node(chroma, settings)
        result = node(_base_state(["a.py"]))
        events = result["trace_events"]
        assert len(events) == 1
        assert events[0].event == TraceEventType.ERROR
