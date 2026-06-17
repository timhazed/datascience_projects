"""Analysis models for the diff coaching pipeline.

AnalysisResult and CoachingAlert are co-located because CoachingAlert is produced
directly from AnalysisResult by deviation_guard — they form a single output unit.
"""

from typing import Literal

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """Structured output from coaching_analyzer — validated by Pydantic before use.

    Args:
        deviation_score: 0.0 = perfectly aligned with persona; 1.0 = complete deviation.
            Threshold for coaching alert: > 0.65 (§6.4 deviation_guard).
        rationale: Gemma 4 explanation of the code change intent and its alignment (or
            deviation) from the developer's established patterns.
        patterns_matched: Named persona patterns this diff aligns with or deviates from,
            e.g. ["uses_type_annotations", "avoids_global_state"].
    """

    deviation_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "0.0 = perfectly aligned with persona; 1.0 = complete deviation. "
            "Threshold for coaching alert: > 0.65."
        ),
    )
    rationale: str = Field(
        description=(
            "Gemma 4 explanation of the code change intent and its alignment (or "
            "deviation) from the developer's established patterns."
        ),
    )
    patterns_matched: list[str] = Field(
        default_factory=list,
        description=(
            "List of named persona patterns this diff aligns with or deviates from, "
            "e.g. ['uses_type_annotations', 'avoids_global_state']."
        ),
    )


class CoachingAlert(BaseModel):
    """Populated by deviation_guard when deviation_score > 0.65.

    Args:
        message: Human-readable coaching message surfaced to the developer via Streamlit.
        severity: info = slight drift; warning = notable deviation; critical = strong anti-pattern.
    """

    message: str = Field(
        description="Human-readable coaching message surfaced to the developer via Streamlit.",
    )
    severity: Literal["info", "warning", "critical"] = Field(
        description="info: slight drift; warning: notable deviation; critical: strong anti-pattern.",
    )
