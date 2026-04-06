from pydantic import BaseModel, Field

from src.data.enums import AgentName
from src.data.normalized_article import NormalizedArticle


class IntentSection(BaseModel):
    """
    One section of the final response, corresponding to one PlanStep.
    Constructed programmatically from agent_results articles for that agent.
    Multi-step plans produce multiple IntentSections in SupervisorResponse.
    """

    agent: AgentName = Field(
        description="Which agent produced this section's articles"
    )
    query: str = Field(
        description="The sub-query this section answers, from the routing plan"
    )
    articles: list[NormalizedArticle] = Field(
        default_factory=list,
        description=(
            "Normalized articles returned by this agent. "
            "Empty if agent returned no results."
        ),
    )


class SupervisorResponse(BaseModel):
    """Final structured response returned to the UI and CLI."""

    session_id: str
    sections: list[IntentSection] = Field(
        min_length=1,
        description="One section per plan step. Single-intent prompts have one section.",
    )
    sources_used: list[str] = Field(
        description="Deduplicated, sorted provider names across all sections"
    )
    fallback_used: bool = Field(
        description="True if any agent returned empty results"
    )
