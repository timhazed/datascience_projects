"""PlannerOutput — structured decomposition produced by the planner node."""

from pydantic import BaseModel, Field

from src.models.sub_goal import SubGoal


class PlannerOutput(BaseModel):
    """Structured plan produced by the planner node from the user's query."""

    patient_id: str | None = Field(
        default=None,
        description=(
            "Resolved patient slug ID if already known; "
            "None if resolution is the first sub-goal"
        ),
    )
    sub_goals: list[SubGoal] = Field(
        description=(
            "Ordered list of atomic sub-tasks to execute; "
            "must include resolve_patient first if patient_id is None"
        )
    )
    reasoning: str = Field(
        description="One sentence explaining the decomposition decision"
    )
