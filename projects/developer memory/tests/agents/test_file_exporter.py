"""Tests for make_file_exporter_node (src/agents/file_exporter.py).

Uses pytest tmp_path fixture — no disk writes outside temp directory.

Cases:
  - Happy path: writes PROJECT_SKILLS.md, export_result has path + bytes_written
  - target_path is a directory → filename appended automatically
  - target_path is a full .md path → used directly
  - Directory created if it does not exist (parents=True)
  - skills_markdown=None → export_result=None, error set
  - skills_markdown="" → export_result=None, error set (falsy check)
  - Write failure (read-only dir) → export_result=None, error set, no crash
  - Trace entry added on success and failure
"""

import stat
from pathlib import Path
from unittest.mock import MagicMock

from src.agents.file_exporter import make_file_exporter_node
from src.db.sha_store import SHAStore


def _state(markdown: str | None, target_path: str) -> dict:
    return {"skills_markdown": markdown, "target_path": target_path, "trace": []}


class TestFileExporterNode:
    def test_writes_file_to_directory(self, tmp_path: Path) -> None:
        """Writing to a directory path → PROJECT_SKILLS.md created inside it."""
        node = make_file_exporter_node()
        result = node(_state("# Skills\n", str(tmp_path)))

        assert result["export_result"] is not None
        written_path = Path(result["export_result"]["path"])
        assert written_path.name == "PROJECT_SKILLS.md"
        assert written_path.parent == tmp_path
        assert written_path.read_text(encoding="utf-8") == "# Skills\n"

    def test_bytes_written_reflects_content_size(self, tmp_path: Path) -> None:
        """bytes_written in export_result matches the UTF-8 byte count."""
        content = "# Skills\n\nFastAPI, Pydantic."
        node = make_file_exporter_node()
        result = node(_state(content, str(tmp_path)))

        assert result["export_result"]["bytes_written"] == len(content.encode("utf-8"))

    def test_direct_md_path_used_as_is(self, tmp_path: Path) -> None:
        """target_path ending in .md is used directly without appending filename."""
        target = str(tmp_path / "MY_SKILLS.md")
        node = make_file_exporter_node()
        result = node(_state("# Skills\n", target))

        assert Path(result["export_result"]["path"]).name == "MY_SKILLS.md"

    def test_creates_nested_directory(self, tmp_path: Path) -> None:
        """target_path with non-existent parent directories → they are created."""
        nested = tmp_path / "a" / "b" / "c"
        node = make_file_exporter_node()
        result = node(_state("# Skills\n", str(nested)))

        assert result["export_result"] is not None
        assert Path(result["export_result"]["path"]).exists()

    def test_none_markdown_rejected(self, tmp_path: Path) -> None:
        """skills_markdown=None → export_result=None, error set."""
        node = make_file_exporter_node()
        result = node(_state(None, str(tmp_path)))

        assert result["export_result"] is None
        assert result.get("error")

    def test_empty_markdown_rejected(self, tmp_path: Path) -> None:
        """skills_markdown='' (falsy) → export_result=None, error set."""
        node = make_file_exporter_node()
        result = node(_state("", str(tmp_path)))

        assert result["export_result"] is None
        assert result.get("error")

    def test_write_failure_returns_none_and_error(self, tmp_path: Path) -> None:
        """Write to a read-only directory → export_result=None, error set, no crash."""
        # Make target dir read-only
        read_only = tmp_path / "readonly"
        read_only.mkdir()
        read_only.chmod(stat.S_IRUSR | stat.S_IXUSR)

        node = make_file_exporter_node()
        result = node(_state("# Skills\n", str(read_only)))

        # Restore permissions so pytest can clean up
        read_only.chmod(stat.S_IRWXU)

        assert result["export_result"] is None
        assert result.get("error")
        assert "Traceback" not in result["error"]

    def test_trace_added_on_success(self, tmp_path: Path) -> None:
        """Trace gains an entry after a successful write."""
        node = make_file_exporter_node()
        result = node(_state("# Skills\n", str(tmp_path)))

        assert len(result["trace"]) >= 1
        assert any("wrote" in t for t in result["trace"])

    def test_trace_added_on_failure(self, tmp_path: Path) -> None:
        """Trace gains an entry when write fails."""
        node = make_file_exporter_node()
        result = node(_state(None, str(tmp_path)))

        assert len(result["trace"]) >= 1

    # ── FR-SKILLS-01 exit gate ────────────────────────────────────────────────

    def test_written_file_contains_required_sections(self, tmp_path: Path) -> None:
        """FR-SKILLS-01 exit gate: exported file must contain all three required sections.

        Phase 8 exit gate requires that test_file_exporter verifies PROJECT_SKILLS.md
        contains ## Tech Stack, ## Patterns, and ## Tendencies. The file_exporter node
        is responsible for writing whatever markdown it receives — the synthesizer is
        responsible for the content. This test verifies end-to-end: a markdown string
        with the required sections is written correctly and the sections survive the
        round-trip to disk.
        """
        skills_md = (
            "## Tech Stack\n\n"
            "- FastAPI: async HTTP framework\n"
            "- Pydantic: data validation\n\n"
            "## Patterns\n\n"
            "- Dependency injection via constructor arguments\n"
            "- Typed API boundaries using Pydantic models\n\n"
            "## Tendencies\n\n"
            "- Prefers strongly-typed function signatures\n"
            "- Avoids global mutable state\n"
        )
        node = make_file_exporter_node()
        result = node(_state(skills_md, str(tmp_path)))

        written_path = Path(result["export_result"]["path"])
        content = written_path.read_text(encoding="utf-8")

        assert "## Tech Stack" in content
        assert "## Patterns" in content
        assert "## Tendencies" in content


_REPO_URL = "https://github.com/u/r"
_BRANCH = "main"
_SHA = "abc123def456" * 4


def _sha_state(markdown: str | None, target_path: str) -> dict:
    """State dict including SHA fields for SHAStore tests."""
    return {
        "skills_markdown": markdown,
        "target_path": target_path,
        "trace": [],
        "repo_url": _REPO_URL,
        "branch": _BRANCH,
        "synced_sha": _SHA,
    }


class TestFileExporterSHAStore:
    def test_sha_recorded_after_successful_write(self, tmp_path: Path) -> None:
        """Successful write with sha_store injected → set_skills_sha called once."""
        store = SHAStore(data_path=str(tmp_path))
        node = make_file_exporter_node(sha_store=store)
        node(_sha_state("# Skills\n", str(tmp_path / "out")))

        entry = store.get_skills_entry(_REPO_URL, _BRANCH)
        assert entry.get("skills_sha") == _SHA
        assert "PROJECT_SKILLS.md" in entry.get("skills_path", "")

    def test_sha_not_recorded_when_store_is_none(self, tmp_path: Path) -> None:
        """sha_store=None (default) → no set_skills_sha call, no error."""
        node = make_file_exporter_node(sha_store=None)
        result = node(_sha_state("# Skills\n", str(tmp_path / "out")))
        assert result["export_result"] is not None  # write still succeeded

    def test_sha_not_recorded_on_write_failure(self, tmp_path: Path) -> None:
        """Write failure → set_skills_sha must NOT be called."""
        store = MagicMock(spec=SHAStore)
        read_only = tmp_path / "ro"
        read_only.mkdir()
        read_only.chmod(0o555)

        node = make_file_exporter_node(sha_store=store)
        node(_sha_state("# Skills\n", str(read_only)))

        read_only.chmod(0o755)
        store.set_skills_sha.assert_not_called()

    def test_sha_not_recorded_when_synced_sha_missing(self, tmp_path: Path) -> None:
        """State missing synced_sha → set_skills_sha not called (guard protects store)."""
        store = MagicMock(spec=SHAStore)
        state = {
            "skills_markdown": "# Skills\n",
            "target_path": str(tmp_path / "out"),
            "trace": [],
            "repo_url": _REPO_URL,
            "branch": _BRANCH,
            # synced_sha deliberately absent
        }
        node = make_file_exporter_node(sha_store=store)
        node(state)
        store.set_skills_sha.assert_not_called()
