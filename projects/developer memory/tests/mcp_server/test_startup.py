"""Tests for src/mcp_server/startup.py.

Covers _check_memory() and _log_memory(). All psutil and pathlib calls are mocked
so no live system state is read during the test run.

Patch targets use src.mcp_server.startup.* — not src.server.* — because _check_memory
and _log_memory now live in startup.py and that is where the names are resolved.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.mcp_server.startup import _check_memory, _log_memory


class TestCheckMemory:
    def _mock_memory(self, available_gb: float) -> MagicMock:
        """Return a mock virtual_memory object with .available set in bytes."""
        mock_vm = MagicMock()
        mock_vm.available = available_gb * (1024**3)
        return mock_vm

    def test_below_8gb_raises_runtime_error(self) -> None:
        """< 8 GB available → RuntimeError raised; server cannot start."""
        with (
            patch(
                "src.mcp_server.startup.psutil.virtual_memory",
                return_value=self._mock_memory(4.0),
            ),
            pytest.raises(RuntimeError, match="Insufficient Unified Memory"),
        ):
            _check_memory()

    def test_between_8_and_16gb_halves_workers_and_sets_e4b(self, monkeypatch) -> None:
        """12 GB available → PARSER_WORKERS halved; model switched to gemma4:e4b."""
        monkeypatch.setenv("PARSER_WORKERS", "6")
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
        with patch(
            "src.mcp_server.startup.psutil.virtual_memory",
            return_value=self._mock_memory(12.0),
        ):
            model = _check_memory()

        assert model == "gemma4:e4b"
        # 6 // 2 = 3
        assert os.environ["PARSER_WORKERS"] == "3"

    def test_above_16gb_returns_configured_model_unchanged(self, monkeypatch) -> None:
        """32 GB available → no env changes; returns OLLAMA_MODEL."""
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
        with patch(
            "src.mcp_server.startup.psutil.virtual_memory",
            return_value=self._mock_memory(32.0),
        ):
            model = _check_memory()

        assert model == "gemma4:26b"

    def test_exactly_8gb_does_not_raise(self, monkeypatch) -> None:
        """Exactly 8 GB is the minimum — should not raise; returns degraded model."""
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
        with patch(
            "src.mcp_server.startup.psutil.virtual_memory",
            return_value=self._mock_memory(8.0),
        ):
            # 8.0 < 16 → degraded path (not raise); expect e4b returned
            model = _check_memory()

        assert model == "gemma4:e4b"

    def test_worker_floor_is_2_when_workers_below_4(self, monkeypatch) -> None:
        """max(2, workers // 2) floor prevents PARSER_WORKERS dropping below 2."""
        monkeypatch.setenv("PARSER_WORKERS", "2")
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:26b")
        with patch(
            "src.mcp_server.startup.psutil.virtual_memory",
            return_value=self._mock_memory(10.0),
        ):
            _check_memory()

        # 2 // 2 = 1 → floored to 2
        assert os.environ["PARSER_WORKERS"] == "2"

    def test_check_memory_inside_docker_skips_ram_check(self, monkeypatch) -> None:
        """Inside Docker (/.dockerenv exists) → no RAM check, configured model returned."""
        monkeypatch.setenv("OLLAMA_MODEL", "phi4-mini")
        with patch("src.mcp_server.startup.pathlib.Path.exists", return_value=True):
            model = _check_memory()

        assert model == "phi4-mini"

    def test_small_model_skips_degradation_under_low_memory(self, monkeypatch) -> None:
        """gemma4:e4b (small model) at 12 GB → no worker halving; model returned unchanged."""
        monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
        monkeypatch.setenv("PARSER_WORKERS", "6")
        with patch(
            "src.mcp_server.startup.psutil.virtual_memory",
            return_value=self._mock_memory(12.0),
        ):
            model = _check_memory()

        assert model == "gemma4:e4b"
        # Workers must not be halved for small models
        assert os.environ["PARSER_WORKERS"] == "6"


class TestLogMemory:
    def test_log_memory_exception_does_not_crash(self) -> None:
        """If psutil.Process() raises in _log_memory, the function logs a warning and returns."""
        with patch(
            "src.mcp_server.startup.psutil.Process",
            side_effect=RuntimeError("psutil error"),
        ):
            # Must not raise — exceptions in _log_memory are swallowed with a warning.
            _log_memory("test label")

    def test_log_memory_no_label(self) -> None:
        """_log_memory() with no label runs without error."""
        mock_proc = MagicMock()
        mock_proc.memory_info.return_value.rss = 200 * 1024 * 1024  # 200 MB
        mock_vm = MagicMock()
        mock_vm.available = 16 * (1024**3)
        mock_vm.percent = 50.0

        with (
            patch("src.mcp_server.startup.psutil.Process", return_value=mock_proc),
            patch("src.mcp_server.startup.psutil.virtual_memory", return_value=mock_vm),
        ):
            _log_memory()  # no label — must not raise
