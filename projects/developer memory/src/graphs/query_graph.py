"""Query Pipeline LangGraph StateGraph — Spec §4.2, §6.

build_query_graph() compiles the StateGraph for the query_memory MCP tool.

Graph topology (§4.2):
  START → query_guard → semantic_searcher → snippet_summarizer → END
  query_guard → END (query_safe=False)
"""

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.query_guard import make_query_guard_node
from src.agents.semantic_searcher import make_semantic_searcher_node
from src.agents.snippet_summarizer import make_snippet_summarizer_node
from src.db.chroma_client import ChromaLibrarianClient
from src.models.query_state import QueryState

logger = logging.getLogger(__name__)


def build_query_graph(
    llm: BaseChatModel,
    chroma: ChromaLibrarianClient,
    *,
    recursion_limit: int = 50,
) -> CompiledStateGraph:
    """Build and compile the Query Pipeline StateGraph.

    Args:
        llm: BaseChatModel instance (temperature=0.1, num_ctx=8192) — snippet_summarizer.
        chroma: ChromaLibrarianClient singleton.
        recursion_limit: Maximum graph steps before LangGraph raises RecursionError.
            Default 50 is generous for this linear 3-node graph. Callers must pass this
            value to .invoke(): graph.invoke(state, config={"recursion_limit": recursion_limit}).

    Returns:
        Compiled LangGraph StateGraph ready for .invoke() calls.
    """
    query_guard = make_query_guard_node()
    semantic_searcher = make_semantic_searcher_node(chroma)
    snippet_summarizer = make_snippet_summarizer_node(llm)

    def route_after_guard(state: QueryState) -> str:
        """Route to END if query is unsafe, else to semantic_searcher."""
        return "semantic_searcher" if state.get("query_safe") else END

    builder = StateGraph(QueryState)

    builder.add_node("query_guard", query_guard)
    builder.add_node("semantic_searcher", semantic_searcher)
    builder.add_node("snippet_summarizer", snippet_summarizer)

    builder.add_edge(START, "query_guard")
    builder.add_conditional_edges("query_guard", route_after_guard)
    builder.add_edge("semantic_searcher", "snippet_summarizer")
    builder.add_edge("snippet_summarizer", END)

    return builder.compile()
