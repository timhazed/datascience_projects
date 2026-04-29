"""Tests for Streamlit UI pure-logic helpers (graph_runtime, metrics_tab).

Covers three functions identified in code review as untested:
  - _quality_label          : pure scoring formatter, no I/O
  - _build_initial_state    : builds HealthcareState dict (in graph_runtime)
  - _load_all_experiment_results : scans experiment output directories for JSON

Streamlit rendering functions (which require a live server) are intentionally
excluded from this file.  The coverage config in pyproject.toml reflects this:
  omit = ["src/ui/*", ...]

Import notes:
  Importing ``src.ui.tabs.metrics_tab`` loads ``runtime`` (``load_settings()``);
  config.yaml must exist in the working directory.  pytest runs from project root
  where config.yaml is committed.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from src.ui.graph_runtime import _build_initial_state, user_facing_assistant_text
from src.ui.tabs.metrics_tab import _load_all_experiment_results, _quality_label

# ---------------------------------------------------------------------------
# _quality_label
# ---------------------------------------------------------------------------


class TestQualityLabel:
    """_quality_label returns a display string for the quality column."""

    def test_appointment_booking_always_na(self) -> None:
        """Appointment booking is deterministic — quality is not applicable."""
        assert _quality_label(0.99, "appointment_booking") == "N/A"

    def test_none_score_returns_na(self) -> None:
        assert _quality_label(None, "disease_search") == "N/A"

    def test_nan_score_returns_na(self) -> None:
        assert _quality_label(float("nan"), "disease_search") == "N/A"

    def test_score_above_threshold_shows_pass(self) -> None:
        result = _quality_label(0.85, "disease_search")
        assert "0.85" in result
        assert "✓ Pass" in result

    def test_score_below_threshold_shows_fail(self) -> None:
        result = _quality_label(0.75, "disease_search")
        assert "0.75" in result
        assert "✗ Below threshold" in result

    def test_threshold_boundary_passes(self) -> None:
        """Score exactly at 0.80 should pass (≥ threshold)."""
        result = _quality_label(0.80, "disease_search")
        assert "✓ Pass" in result

    def test_score_just_below_threshold_fails(self) -> None:
        result = _quality_label(0.799, "disease_search")
        assert "✗ Below threshold" in result

    def test_medical_history_experiment(self) -> None:
        """Experiment name other than appointment_booking uses numeric path."""
        result = _quality_label(0.90, "medical_history")
        assert "0.90" in result


# ---------------------------------------------------------------------------
# _build_initial_state
# ---------------------------------------------------------------------------


class TestBuildInitialState:
    """_build_initial_state constructs the dict passed to graph.invoke()."""

    def _fake_session(self, resolved_id: str | None = None) -> dict:
        """Return a minimal session-state dict for patching."""
        return {"resolved_patient_id": resolved_id}

    def test_query_passed_through_without_hint(self) -> None:
        session = self._fake_session()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state.get = lambda k, d=None: session.get(k, d)
            result = _build_initial_state("What is CKD?", None)
        assert result["user_query"] == "What is CKD?"

    def test_patient_hint_prepended_to_query(self) -> None:
        session = self._fake_session()
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state.get = lambda k, d=None: session.get(k, d)
            result = _build_initial_state("Retrieve history", "Robert Singh")
        assert result["user_query"].startswith("[Patient context: Robert Singh]")
        assert "Retrieve history" in result["user_query"]

    def test_resolved_patient_id_propagated(self) -> None:
        session = self._fake_session(resolved_id="P-abc123")
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state.get = lambda k, d=None: session.get(k, d)
            result = _build_initial_state("Book nephrologist", None)
        assert result["patient_id"] == "P-abc123"

    def test_no_resolved_patient_id_is_none(self) -> None:
        session = self._fake_session(resolved_id=None)
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state.get = lambda k, d=None: session.get(k, d)
            result = _build_initial_state("General query", None)
        assert result["patient_id"] is None

    def test_all_required_state_keys_present(self) -> None:
        """Every key required by HealthcareState must appear in the returned dict."""
        required_keys = (
            "user_query", "patient_id", "messages", "planner_output",
            "pending_tasks", "completed_tasks", "intent_safe",
            "final_summary", "error", "trace",
        )
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state.get = lambda k, d=None: None
            result = _build_initial_state("query", None)
        for key in required_keys:
            assert key in result, f"Missing HealthcareState key: {key}"

    def test_lists_initialised_correctly(self) -> None:
        """messages starts with the HumanMessage for the query; other lists are empty."""
        from langchain_core.messages import HumanMessage

        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state.get = lambda k, d=None: None
            result = _build_initial_state("query", None)
        # messages must contain exactly one HumanMessage with the query content
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], HumanMessage)
        assert result["messages"][0].content == "query"
        assert result["pending_tasks"] == []
        assert result["completed_tasks"] == []
        assert result["trace"] == []

    def test_intent_safe_initialised_false(self) -> None:
        with patch("src.ui.graph_runtime.st") as mock_st:
            mock_st.session_state.get = lambda k, d=None: None
            result = _build_initial_state("query", None)
        assert result["intent_safe"] is False


class TestUserFacingAssistantText:
    """Chat must never pass None to st.markdown when guard returns UNSAFE."""

    def test_prefers_final_summary(self) -> None:
        assert user_facing_assistant_text({
            "final_summary": "  Hello  ",
            "error": "ignored",
            "intent_safe": True,
        }) == "Hello"

    def test_falls_back_to_error_when_no_summary(self) -> None:
        assert user_facing_assistant_text({
            "final_summary": None,
            "error": "Planning failed",
            "intent_safe": True,
        }) == "Planning failed"

    def test_unsafe_without_error_message(self) -> None:
        out = user_facing_assistant_text({
            "final_summary": None,
            "error": None,
            "intent_safe": False,
            "trace": ["intent_guard: UNSAFE"],
        })
        assert "unsafe" in out.lower()
        assert "off-topic" in out.lower()

    def test_empty_error_and_summary_generic_fallback(self) -> None:
        assert user_facing_assistant_text({
            "final_summary": None,
            "error": None,
            "intent_safe": True,
        }) == "No response was generated. Please try rephrasing your question."


# ---------------------------------------------------------------------------
# _load_all_experiment_results
# ---------------------------------------------------------------------------


class TestLoadAllExperimentResults:
    """_load_all_experiment_results scans three experiment output directories."""

    def test_returns_empty_list_when_no_directories_exist(
        self, tmp_path: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        assert _load_all_experiment_results() == []

    def test_loads_single_json_file(
        self, tmp_path: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "experiments" / "appointment" / "data"
        data_dir.mkdir(parents=True)
        record = {
            "experiment_name": "appointment_booking",
            "run_id": "run-1",
            "latency_ms": 12.5,
            "success": True,
            "quality_score": None,
            "tool_calls_made": 2,
            "error": None,
        }
        (data_dir / "result_1.json").write_text(json.dumps(record))

        results = _load_all_experiment_results()

        assert len(results) == 1
        assert results[0]["experiment_name"] == "appointment_booking"

    def test_skips_malformed_json_without_raising(
        self, tmp_path: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "experiments" / "search" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "bad.json").write_text("{not valid json{{")

        results = _load_all_experiment_results()

        assert results == []

    def test_aggregates_across_all_three_directories(
        self, tmp_path: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        for subdir in ("appointment", "history", "search"):
            d = tmp_path / "experiments" / subdir / "data"
            d.mkdir(parents=True)
            (d / "r.json").write_text(json.dumps({"experiment_name": subdir}))

        results = _load_all_experiment_results()

        assert len(results) == 3
        names = {r["experiment_name"] for r in results}
        assert names == {"appointment", "history", "search"}

    def test_loads_multiple_files_from_one_directory(
        self, tmp_path: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "experiments" / "history" / "data"
        data_dir.mkdir(parents=True)
        for i in range(3):
            (data_dir / f"result_{i}.json").write_text(
                json.dumps({"experiment_name": "medical_history", "run": i})
            )

        results = _load_all_experiment_results()

        assert len(results) == 3

    def test_valid_and_invalid_files_in_same_dir(
        self, tmp_path: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Valid records are returned; malformed files are silently skipped."""
        monkeypatch.chdir(tmp_path)
        data_dir = tmp_path / "experiments" / "appointment" / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "good.json").write_text(json.dumps({"experiment_name": "ok"}))
        (data_dir / "bad.json").write_text("corrupted")

        results = _load_all_experiment_results()

        assert len(results) == 1
        assert results[0]["experiment_name"] == "ok"
