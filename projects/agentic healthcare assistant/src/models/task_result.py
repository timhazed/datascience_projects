"""TaskResult — typed accumulator for completed_tasks in HealthcareState."""

from pydantic import BaseModel, Field


class TaskResult(BaseModel):
    """Result of a completed tool node; accumulated in HealthcareState.completed_tasks."""

    task: str = Field(
        description="Task type that produced this result, e.g. 'retrieve_history'"
    )
    success: bool = Field(description="Whether the task completed successfully")
    result: dict = Field(
        default_factory=dict,
        description="Serialized output of the task",
    )
    error: str | None = Field(
        default=None,
        description="Error message if success is False",
    )
