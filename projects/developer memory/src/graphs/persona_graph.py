"""Persona Pipeline LangGraph StateGraph — Spec §4.3, §6.

build_persona_graph() compiles the StateGraph for the get_dev_persona MCP tool.

Graph topology (§4.3):
  START → tendency_scanner → persona_synthesizer → END
  tendency_scanner → END (tendency_data=None, insufficient indexed data)
"""

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.persona_synthesizer import make_persona_synthesizer_node
from src.agents.tendency_scanner import make_tendency_scanner_node
from src.db.chroma_client import ChromaLibrarianClient
from src.models.persona_state import PersonaState

logger = logging.getLogger(__name__)


def build_persona_graph(
    llm: BaseChatModel,
    chroma: ChromaLibrarianClient,
    *,
    recursion_limit: int = 50,
) -> CompiledStateGraph:
    """Build and compile the Persona Pipeline StateGraph.

    Args:
        llm: BaseChatModel instance (temperature=0.3, num_ctx=8192) — persona_synthesizer.
        chroma: ChromaLibrarianClient singleton.
        recursion_limit: Maximum graph steps before LangGraph raises RecursionError.
            Default 50 is generous for this linear 2-node graph. Callers must pass this
            value to .invoke(): graph.invoke(state, config={"recursion_limit": recursion_limit}).

    Returns:
        Compiled LangGraph StateGraph ready for .invoke() calls.
    """
    tendency_scanner = make_tendency_scanner_node(chroma)
    persona_synthesizer = make_persona_synthesizer_node(llm)

    def route_after_scanner(state: PersonaState) -> str:
        """Route to END if insufficient indexed data, else to persona_synthesizer."""
        return "persona_synthesizer" if state.get("tendency_data") is not None else END

    builder = StateGraph(PersonaState)

    builder.add_node("tendency_scanner", tendency_scanner)
    builder.add_node("persona_synthesizer", persona_synthesizer)

    builder.add_edge(START, "tendency_scanner")
    builder.add_conditional_edges("tendency_scanner", route_after_scanner)
    builder.add_edge("persona_synthesizer", END)

    return builder.compile()
