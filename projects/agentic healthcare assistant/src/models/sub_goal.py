"""SubGoal — a single atomic task in the planner's decomposed plan."""

from typing import Literal

from pydantic import BaseModel, Field


class SubGoal(BaseModel):
    """A single atomic task in the planner's decomposed execution plan."""

    task: Literal[
        "resolve_patient",
        "retrieve_history",
        "update_history",
        "book_appointment",
        "search_disease",
    ] = Field(description="The atomic task type to execute")
    parameters: dict[str, str | int | None] = Field(
        default_factory=dict,
        description=(
            "Key-value parameters for this task; "
            "keys match the corresponding typed params model"
        ),
    )
    order: int = Field(ge=1, description="Execution order (1 = first)")
