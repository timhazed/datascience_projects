from __future__ import annotations

import operator
from typing import Annotated

from pydantic import BaseModel, Field

from src.data.agent_result import NewsAgentResult
from src.data.llm_config import LLMConfig
from src.data.plan import RoutingPlan
from src.data.response import SupervisorResponse
from src.data.user_query import UserQuery
from src.utils.constants import MAX_HISTORY_TURNS


class AgentState(BaseModel):
    """
    Mutable LangGraph graph state. Passed between all nodes.

    LangGraph note: nodes return dict patches (not mutated AgentState instances).
    The graph is constructed as StateGraph(AgentState) — LangGraph 0.3+ handles
    Pydantic v2 models as state natively.

    agent_results uses Annotated[..., operator.add] so LangGraph's parallel
    fan-out dispatch accumulates results across branches rather than overwriting.
    """

    query: UserQuery
    llm_config: LLMConfig | None = None
    plan: RoutingPlan | None = None
    agent_results: Annotated[list[NewsAgentResult], operator.add] = Field(
        default_factory=list
    )
    final_response: SupervisorResponse | None = None
    conversation_history: list[dict] = Field(
        default_factory=list,
        description=(
            "List of {'role': str, 'content': str} dicts. "
            f"Capped at {MAX_HISTORY_TURNS} turns (20 messages). Oldest turns dropped FIFO."
        ),
    )
    error: str | None = None
