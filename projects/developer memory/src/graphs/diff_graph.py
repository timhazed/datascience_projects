"""Diff Analysis Pipeline LangGraph StateGraph — Spec §4.4, §6.

build_diff_graph() compiles the StateGraph for the analyze_diff MCP tool.

Graph topology (§4.4):
  START → diff_validator → persona_loader → coaching_analyzer → deviation_guard → END
  diff_validator → END (diff_safe=False)

persona_loader is always followed by coaching_analyzer regardless of whether persona
context was loaded — a missing persona is a valid best-effort outcome, not an error path.
"""

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.coaching_analyzer import make_coaching_analyzer_node
from src.agents.deviation_guard import make_deviation_guard_node
from src.agents.diff_validator import make_diff_validator_node
from src.agents.persona_loader import make_persona_loader_node
from src.db.chroma_client import ChromaLibrarianClient
from src.models.diff_state import DiffState

logger = logging.getLogger(__name__)


def build_diff_graph(
    llm: BaseChatModel,
    chroma: ChromaLibrarianClient,
    *,
    recursion_limit: int = 50,
) -> CompiledStateGraph:
    """Build and compile the Diff Analysis Pipeline StateGraph.

    Args:
        llm: BaseChatModel instance (temperature=0.0, num_ctx=8192) — coaching_analyzer.
        chroma: ChromaLibrarianClient singleton — persona_loader.
        recursion_limit: Maximum graph steps before LangGraph raises RecursionError.
            Default 50 is generous for this linear 4-node graph. Callers must pass this
            value to .invoke(): graph.invoke(state, config={"recursion_limit": recursion_limit}).

    Returns:
        Compiled LangGraph StateGraph ready for .invoke() calls.
    """
    diff_validator = make_diff_validator_node()
    persona_loader = make_persona_loader_node(chroma)
    coaching_analyzer = make_coaching_analyzer_node(llm)
    deviation_guard = make_deviation_guard_node()

    def route_after_validator(state: DiffState) -> str:
        """Route to END if diff is invalid, else to persona_loader."""
        return "persona_loader" if state.get("diff_safe") else END

    builder = StateGraph(DiffState)

    builder.add_node("diff_validator", diff_validator)
    builder.add_node("persona_loader", persona_loader)
    builder.add_node("coaching_analyzer", coaching_analyzer)
    builder.add_node("deviation_guard", deviation_guard)

    builder.add_edge(START, "diff_validator")
    builder.add_conditional_edges("diff_validator", route_after_validator)
    # persona_loader always continues to coaching_analyzer — missing persona is not an error path
    builder.add_edge("persona_loader", "coaching_analyzer")
    builder.add_edge("coaching_analyzer", "deviation_guard")
    builder.add_edge("deviation_guard", END)

    return builder.compile()
