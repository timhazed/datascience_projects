"""Tests for ui.config — server URL constants and stage label definitions."""

import importlib

import pytest


class TestServerUrls:
    """Verify _SERVER_BASE and _MCP_URL resolve correctly from env."""

    @pytest.fixture(autouse=True)
    def _restore_config_module(self) -> pytest.FixtureRequest:
        """Reload ui.config after each test to flush any env-driven module-level state.

        importlib.reload() mutates the module object in sys.modules. Without this
        fixture, a test that reloads with a custom MCP_SERVER_URL leaves _SERVER_BASE
        and _MCP_URL at that value for any subsequent import in the same session.
        """
        yield
        import importlib

        import ui.config as cfg

        importlib.reload(cfg)

    def test_server_base_defaults_to_localhost(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_SERVER_BASE defaults to localhost:9000 when env var is absent."""
        monkeypatch.delenv("MCP_SERVER_URL", raising=False)
        import ui.config as cfg
        importlib.reload(cfg)
        assert cfg._SERVER_BASE == "http://localhost:9000"

    def test_server_base_reads_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_SERVER_BASE reflects MCP_SERVER_URL when set."""
        monkeypatch.setenv("MCP_SERVER_URL", "http://mcp-server:9000")
        import ui.config as cfg
        importlib.reload(cfg)
        assert cfg._SERVER_BASE == "http://mcp-server:9000"

    def test_mcp_url_ends_with_mcp(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_MCP_URL is always _SERVER_BASE + '/mcp'."""
        monkeypatch.setenv("MCP_SERVER_URL", "http://example.com:9000")
        import ui.config as cfg
        importlib.reload(cfg)
        assert cfg._MCP_URL == "http://example.com:9000/mcp"

    def test_mcp_url_suffix_with_default_base(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """_MCP_URL ends with '/mcp' even with the default base."""
        monkeypatch.delenv("MCP_SERVER_URL", raising=False)
        import ui.config as cfg
        importlib.reload(cfg)
        assert cfg._MCP_URL.endswith("/mcp")


class TestStageLabels:
    """Verify _STAGE_LABELS structure and completeness."""

    EXPECTED_KEYS = {
        "queued",
        "cloning",
        "cache_filtering",
        "pii_scanning",
        "parsing",
        "summarizing",
        "done",
        "error",
    }

    def test_all_expected_keys_present(self) -> None:
        """_STAGE_LABELS contains all required pipeline stage keys."""
        from ui.config import _STAGE_LABELS
        assert set(_STAGE_LABELS.keys()) == self.EXPECTED_KEYS

    def test_each_value_is_two_tuple(self) -> None:
        """Each _STAGE_LABELS value is a 2-tuple of (str, float)."""
        from ui.config import _STAGE_LABELS
        for stage, value in _STAGE_LABELS.items():
            assert isinstance(value, tuple), f"Stage '{stage}' value is not a tuple"
            assert len(value) == 2, f"Stage '{stage}' tuple has {len(value)} elements, expected 2"
            label, fraction = value
            assert isinstance(label, str), f"Stage '{stage}' label is not a str"
            assert isinstance(fraction, float), f"Stage '{stage}' fraction is not a float"

    def test_fractions_in_valid_range(self) -> None:
        """All progress fractions are in [0.0, 1.0]."""
        from ui.config import _STAGE_LABELS
        for stage, (_, fraction) in _STAGE_LABELS.items():
            assert 0.0 <= fraction <= 1.0, (
                f"Stage '{stage}' fraction {fraction} is outside [0.0, 1.0]"
            )

    def test_done_fraction_is_one(self) -> None:
        """'done' stage has fraction 1.0 — full progress bar."""
        from ui.config import _STAGE_LABELS
        _, done_fraction = _STAGE_LABELS["done"]
        assert done_fraction == 1.0

    def test_queued_fraction_is_zero(self) -> None:
        """'queued' stage has fraction 0.0 — empty progress bar."""
        from ui.config import _STAGE_LABELS
        _, queued_fraction = _STAGE_LABELS["queued"]
        assert queued_fraction == 0.0
