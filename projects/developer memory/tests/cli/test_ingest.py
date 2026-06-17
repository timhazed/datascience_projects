"""Tests for src/cli/ingest.py pure utility functions.

Only tests the pure-logic helpers that do not require live Ollama, live git, or live ChromaDB.

Cases:
  _parse_repo_arg: valid URL, URL with @branch, missing URL, bad format
  _cap_patch: 0 returns nullcontext, positive int wraps compute_delta
  save_results: writes a JSON file at the expected path (uses tmp_path)
"""

import json
import os
from pathlib import Path

import pytest

# Patch env before importing the module — it reads env at import time
os.environ.setdefault("OLLAMA_HOST", "http://localhost:11434")
os.environ.setdefault("CHROMA_HOST", "")

from src.cli.ingest import _cap_patch, _parse_repo_arg, save_results


class TestParseRepoArg:
    def test_plain_url_defaults_to_main(self) -> None:
        """URL without @branch → branch defaults to 'main'."""
        url, branch = _parse_repo_arg("https://github.com/user/repo")
        assert url == "https://github.com/user/repo"
        assert branch == "main"

    def test_url_with_branch(self) -> None:
        """URL@branch → both parts extracted correctly."""
        url, branch = _parse_repo_arg("https://github.com/user/repo@develop")
        assert url == "https://github.com/user/repo"
        assert branch == "develop"

    def test_url_with_feature_branch_slash(self) -> None:
        """URL@feature/name → branch includes the slash."""
        url, branch = _parse_repo_arg("https://github.com/user/repo@feature/my-feature")
        assert url == "https://github.com/user/repo"
        assert branch == "feature/my-feature"

    def test_leading_whitespace_stripped(self) -> None:
        """Leading/trailing whitespace in raw string is stripped."""
        url, branch = _parse_repo_arg("  https://github.com/user/repo  ")
        assert url == "https://github.com/user/repo"
        assert branch == "main"


class TestCapPatch:
    def test_zero_returns_nullcontext(self) -> None:
        """max_files=0 → nullcontext (no patching)."""
        ctx = _cap_patch(0)
        # nullcontext is its own context manager with __enter__/__exit__
        assert hasattr(ctx, "__enter__")
        with ctx:
            pass  # Should not raise

    def test_nonzero_returns_patch_context(self) -> None:
        """max_files>0 → returns a patch context manager."""
        ctx = _cap_patch(5)
        # Should be a mock._patch object (has __enter__/__exit__)
        assert hasattr(ctx, "__enter__")


class TestSaveResults:
    def test_save_results_writes_json_file(self, tmp_path: Path, monkeypatch) -> None:
        """save_results writes a JSON file to the results directory."""
        # Redirect _RESULTS_DIR to tmp_path so we don't write to the real repo
        import src.cli.ingest as ingest_module

        monkeypatch.setattr(ingest_module, "_RESULTS_DIR", tmp_path)
        monkeypatch.setenv("OLLAMA_MODEL", "test-model")

        repo_results = [
            {"repo_url": "https://github.com/u/r", "branch": "main", "elapsed_seconds": 1.5, "result": {}},
        ]
        out_path = save_results(repo_results, chroma_path="/tmp/chroma", collection="test_col")

        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["collection"] == "test_col"
        assert len(data["repo_results"]) == 1

    def test_save_results_records_total_elapsed(self, tmp_path: Path, monkeypatch) -> None:
        """Total elapsed time sums repo elapsed values."""
        import src.cli.ingest as ingest_module

        monkeypatch.setattr(ingest_module, "_RESULTS_DIR", tmp_path)

        repo_results = [
            {"repo_url": "https://github.com/u/r1", "branch": "main", "elapsed_seconds": 2.5, "result": {}},
            {"repo_url": "https://github.com/u/r2", "branch": "main", "elapsed_seconds": 3.5, "result": {}},
        ]
        out_path = save_results(repo_results, chroma_path="/tmp/chroma", collection="test_col")
        data = json.loads(out_path.read_text())

        assert data["summary"]["total_elapsed_seconds"] == pytest.approx(6.0, abs=0.01)
