"""State TypedDict for the Diff Analysis Pipeline (analyze_diff MCP tool).

Spec §3 — DiffState flows through: diff_validator → persona_loader → coaching_analyzer
→ deviation_guard.
"""

import operator
from typing import Annotated, TypedDict

from src.models.analysis import AnalysisResult, CoachingAlert
from src.models.persona import PersonaProfile


class DiffState(TypedDict):
    """Pipeline state for analyze_diff.

    Fields:
        diff_text: Raw unified diff text from the MCP client.
        diff_safe: Set to True by diff_validator when the diff passes format/size checks.
            Conditional edge routes to END (with user-facing error) if False.
        persona_context: Developer persona loaded by persona_loader for coaching context.
            coaching_analyzer runs with a generic baseline if this is None.
        analysis_result: Structured coaching output from coaching_analyzer.
        coaching_alert: Populated by deviation_guard when deviation_score > 0.65.
        error: Non-fatal error message.
        trace: Append-only execution log.
    """

    diff_text: str
    diff_safe: bool
    persona_context: PersonaProfile | None
    analysis_result: AnalysisResult | None
    coaching_alert: CoachingAlert | None
    error: str | None
    trace: Annotated[list[str], operator.add]
