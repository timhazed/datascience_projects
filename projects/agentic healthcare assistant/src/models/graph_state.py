"""HealthcareState — LangGraph TypedDict with reducer annotations."""

from typing import Annotated, NotRequired, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from src.models.planner_output import PlannerOutput
from src.models.sub_goal import SubGoal
from src.models.task_result import TaskResult


class HealthcareState(TypedDict):
    """LangGraph state shared across all nodes in the healthcare graph.

    Reducer annotations:
      - messages: add_messages (LangGraph built-in — appends, deduplicates by id)

    REPLACE semantics (no annotation):
      - completed_tasks: nodes return the full accumulated list each time
      - trace: nodes return the full accumulated list each time
      - pending_tasks: each node returns pending_tasks[1:] to dequeue the current task
      - all other scalar fields: last-write-wins

    Optional fields (NotRequired):
      - conversation_summary: rolling summary string; absent until first cadence trigger
    """

    user_query: str
    patient_id: str | None
    messages: Annotated[list[AnyMessage], add_messages]
    planner_output: PlannerOutput | None
    pending_tasks: list[SubGoal]
    completed_tasks: list[TaskResult]
    intent_safe: bool
    final_summary: str | None
    error: str | None
    trace: list[str]
    conversation_summary: NotRequired[str]
