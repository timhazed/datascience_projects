"""Tests for validate_target_path (src/middleware/path_guard.py).

Covers all guard branches including the sibling-directory attack that defeated the
old str.startswith() check: /tmp/export_evil starts with /tmp/export as a string
but is NOT under that directory. is_relative_to() catches this correctly.
"""

from pathlib import Path

import pytest

from src.middleware.path_guard import validate_target_path


class TestValidateTargetPath:
    def test_valid_relative_path(self, tmp_path, monkeypatch) -> None:
        """A simple filename resolves under the export root — accepted."""
        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        result = validate_target_path("PROJECT_SKILLS.md")
        assert result == str(tmp_path / "PROJECT_SKILLS.md")

    def test_valid_nested_relative_path(self, tmp_path, monkeypatch) -> None:
        """A subdirectory path that stays under the export root is accepted."""
        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        result = validate_target_path("subdir/PROJECT_SKILLS.md")
        assert result == str(tmp_path / "subdir" / "PROJECT_SKILLS.md")

    def test_path_traversal_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="illegal characters"):
            validate_target_path("../outside.md")

    def test_null_byte_raises(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="illegal characters"):
            validate_target_path("file\x00.md")

    def test_sibling_name_attack_rejected(self, tmp_path, monkeypatch) -> None:
        """Sibling directory whose name is a prefix of the export root is rejected.

        This is the attack that str.startswith() misses:
          export_root = /tmp/pytest-123/test_sibling0/export
          evil_path   = /tmp/pytest-123/test_sibling0/export_evil/file.md
          startswith("/tmp/.../export") → True  ← old guard BYPASSED
          is_relative_to(...)           → False ← new guard BLOCKS
        """
        # Create a sibling dir whose name starts with the export root's directory name
        export_dir = tmp_path / "export"
        sibling_dir = tmp_path / "export_evil"
        export_dir.mkdir()
        sibling_dir.mkdir()

        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(export_dir))

        evil_target = str(sibling_dir / "malware.sh")
        with pytest.raises(ValueError, match="resolves outside the allowed export directory"):
            validate_target_path(evil_target)

    def test_absolute_path_outside_root_raises(self, tmp_path, monkeypatch) -> None:
        """An absolute path resolving to a completely different subtree is rejected."""
        import tempfile

        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        # tempfile.gettempdir() is a parent of tmp_path — never a child
        outside = tempfile.gettempdir()
        with pytest.raises(ValueError, match="resolves outside the allowed export directory"):
            validate_target_path(outside)

    def test_export_root_itself_is_accepted(self, tmp_path, monkeypatch) -> None:
        """Resolving exactly to the export root (empty sub-path) is valid."""
        monkeypatch.setenv("SKILLS_EXPORT_DIR", str(tmp_path))
        # Passing "." resolves to export_root / "." → export_root itself
        result = validate_target_path(".")
        assert result == str(tmp_path)

    def test_defaults_to_cwd_when_env_unset(self, monkeypatch) -> None:
        """Without SKILLS_EXPORT_DIR, cwd is used as export root."""
        monkeypatch.delenv("SKILLS_EXPORT_DIR", raising=False)
        result = validate_target_path("subdir/file.md")
        assert Path(result).is_absolute()
        assert "subdir" in result
