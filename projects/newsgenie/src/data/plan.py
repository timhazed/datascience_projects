from pydantic import BaseModel, Field

from src.data.enums import AgentName


class PlanStep(BaseModel):
    """One routing instruction produced by the supervisor node for one identified intent."""

    agent: AgentName = Field(description="Which agent handles this step")
    query: str = Field(
        description="Extracted sub-query for this agent, verbatim from LLM output"
    )


class RoutingPlan(BaseModel):
    """
    The full routing plan for a user message. Produced by supervisor_node via regex parsing.
    Up to 3 steps (matching the supervisor prompt's max-3-intent rule).
    """

    steps: list[PlanStep] = Field(
        min_length=1,
        max_length=3,
        description=(
            "Ordered list of agent dispatch instructions. "
            "Fan-out executes all steps in parallel."
        ),
    )
    original_query: str = Field(
        description="The original unmodified user query text"
    )
