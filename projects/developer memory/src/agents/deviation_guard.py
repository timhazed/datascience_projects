"""deviation_guard LangGraph node — Spec §2.4, §7.

Reads analysis_result and populates coaching_alert when deviation_score > 0.65.
Always runs after coaching_analyzer — it is a pure decision node with no LLM call.
coaching_alert=None is a valid result (no coaching needed for aligned changes).
"""

import logging
from collections.abc import Callable

from src.models.analysis import AnalysisResult, CoachingAlert

logger = logging.getLogger(__name__)

_DEVIATION_THRESHOLD = 0.65
_CRITICAL_THRESHOLD = 0.85
_WARNING_THRESHOLD = 0.75  # scores > 0.75 and ≤ 0.85 are "warning"; ≤ 0.75 are "info"


def make_deviation_guard_node() -> Callable[[dict], dict]:
    """Return a deviation_guard node (no injected dependencies — pure decision logic).

    Returns:
        LangGraph node function that reads state["analysis_result"] and writes
        state["coaching_alert"]. No LLM call — threshold comparison only.
    """

    def deviation_guard(state: dict) -> dict:
        """Evaluate deviation_score and emit a coaching_alert when threshold is exceeded.

        Severity mapping (§2.4):
          info     : score > 0.65 and ≤ 0.75 — slight drift from patterns
          warning  : score > 0.75 and ≤ 0.85 — notable deviation
          critical : score > 0.85             — strong anti-pattern

        Returns coaching_alert=None if score ≤ 0.65 (aligned change) or if
        analysis_result is None (upstream error).
        """
        analysis: AnalysisResult | None = state.get("analysis_result")
        trace: list[str] = list(state.get("trace", []))

        if analysis is None:
            return {
                "coaching_alert": None,
                "trace": trace + ["deviation_guard: skipped (no analysis_result)"],
            }

        score = analysis.deviation_score

        if score <= _DEVIATION_THRESHOLD:
            logger.debug("deviation_guard: score=%.2f — within threshold, no alert", score)
            return {
                "coaching_alert": None,
                "trace": trace + [f"deviation_guard: ok (score={score:.2f})"],
            }

        # Determine severity from score
        if score > _CRITICAL_THRESHOLD:
            severity = "critical"
        elif score > _WARNING_THRESHOLD:
            severity = "warning"
        else:
            severity = "info"

        patterns_text = ", ".join(analysis.patterns_matched) or "none identified"
        alert = CoachingAlert(
            message=(
                f"Deviation score {score:.2f} — {analysis.rationale} "
                f"Patterns affected: {patterns_text}."
            ),
            severity=severity,
        )
        logger.info("deviation_guard: alert (severity=%s, score=%.2f)", severity, score)
        return {
            "coaching_alert": alert,
            "trace": trace + [f"deviation_guard: alert severity={severity}"],
        }

    return deviation_guard
