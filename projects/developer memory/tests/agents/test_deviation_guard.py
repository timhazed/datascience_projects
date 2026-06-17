"""Tests for make_deviation_guard_node (src/agents/deviation_guard.py).

Pure decision logic — no external dependencies.

Cases:
  - score ≤ 0.65 → coaching_alert=None
  - score > 0.65 and ≤ 0.75 → severity="info"
  - score > 0.75 and ≤ 0.85 → severity="warning"
  - score > 0.85 → severity="critical"
  - analysis_result=None → coaching_alert=None, no crash
  - Alert message contains score and rationale
  - Trace entry added in all paths
"""

from src.agents.deviation_guard import make_deviation_guard_node
from src.models.analysis import AnalysisResult


def _analysis(score: float, rationale: str = "Explanation.", patterns: list[str] | None = None) -> AnalysisResult:
    return AnalysisResult(
        deviation_score=score,
        rationale=rationale,
        patterns_matched=patterns or ["uses_type_annotations"],
    )


def _state(analysis: AnalysisResult | None) -> dict:
    return {"analysis_result": analysis, "trace": []}


class TestDeviationGuardNode:
    def test_score_below_threshold_no_alert(self) -> None:
        """Score ≤ 0.65 → coaching_alert=None."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.65)))

        assert result["coaching_alert"] is None

    def test_score_zero_no_alert(self) -> None:
        """Score of 0.0 (perfectly aligned) → no alert."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.0)))

        assert result["coaching_alert"] is None

    def test_score_info_range(self) -> None:
        """Score 0.66–0.75 → severity='info'."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.70)))

        assert result["coaching_alert"] is not None
        assert result["coaching_alert"].severity == "info"

    def test_score_warning_range(self) -> None:
        """Score 0.76–0.85 → severity='warning'."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.80)))

        assert result["coaching_alert"] is not None
        assert result["coaching_alert"].severity == "warning"

    def test_score_critical_range(self) -> None:
        """Score > 0.85 → severity='critical'."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.90)))

        assert result["coaching_alert"] is not None
        assert result["coaching_alert"].severity == "critical"

    def test_score_exactly_0_85_is_warning(self) -> None:
        """Score exactly at 0.85 → severity='warning' (boundary check)."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.85)))

        assert result["coaching_alert"] is not None
        assert result["coaching_alert"].severity == "warning"

    def test_no_analysis_result_returns_none_alert(self) -> None:
        """analysis_result=None → coaching_alert=None, no exception."""
        node = make_deviation_guard_node()
        result = node(_state(None))

        assert result["coaching_alert"] is None

    def test_alert_message_contains_score_and_rationale(self) -> None:
        """Alert message includes the deviation score and rationale."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.90, rationale="Avoids mutable state.")))

        message = result["coaching_alert"].message
        assert "0.90" in message
        assert "Avoids mutable state." in message

    def test_trace_added_on_no_alert(self) -> None:
        """Trace gains an entry even when no alert is raised."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.30)))

        assert len(result["trace"]) >= 1

    def test_trace_added_on_alert(self) -> None:
        """Trace gains an entry when an alert is raised."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.90)))

        assert len(result["trace"]) >= 1
        assert any("alert" in t for t in result["trace"])

    # ── Phase 8 exit gate: 0.65 boundary — both sides ────────────────────────

    def test_score_exactly_0_65_no_alert(self) -> None:
        """Score exactly at threshold (0.65) → no alert (≤ threshold means safe)."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.65)))

        assert result["coaching_alert"] is None

    def test_score_just_above_0_65_triggers_alert(self) -> None:
        """Score just above threshold (0.66) → alert raised (> threshold means deviation)."""
        node = make_deviation_guard_node()
        result = node(_state(_analysis(0.66)))

        assert result["coaching_alert"] is not None
