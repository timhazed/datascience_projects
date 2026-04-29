"""ExperimentMetrics — typed output schema for all three experiment runners."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ExperimentMetrics(BaseModel):
    """Typed metrics produced by each experiment run.

    Used by ExperimentRunner._save_results() to persist results as JSON.
    """

    experiment_name: Literal[
        "medical_history", "disease_search", "appointment_booking"
    ] = Field(description="Which experiment these metrics belong to")
    run_id: str = Field(description="UUID for this experiment run")
    timestamp: datetime = Field(default_factory=datetime.now)
    latency_ms: float = Field(
        description="End-to-end tool execution latency in milliseconds"
    )
    success: bool = Field(description="Whether the primary operation succeeded")
    quality_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Evaluator score (0.0–1.0): load_evaluator CORRECT/INCORRECT ratio for Exp 1; "
            "RAGAS faithfulness for Exp 2; None for Exp 3"
        ),
    )
    tool_calls_made: int = Field(
        description="Number of tool invocations in this run"
    )
    error: str | None = Field(
        default=None,
        description="Error message if success is False",
    )
