"""Skills Export Pipeline LangGraph StateGraph — Spec §4.5, §6, SkillsSHAGate.

build_skills_graph() compiles the StateGraph for the generate_skills_pkg MCP tool.

Graph topology (updated — SkillsSHAGate):
  START → skills_cache_guard → (hit) → END
  START → skills_cache_guard → (miss) → skills_aggregator → skills_synthesizer → file_exporter → END
  skills_aggregator → END (skills_data=None, no indexed content)

Interrupt hook: file_exporter — spec §6.6 requires a human-in-the-loop checkpoint before
disk writes. To disable in CI/automated runs, pass interrupt_before=[].
"""

import logging

from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from src.agents.file_exporter import make_file_exporter_node
from src.agents.skills_aggregator import make_skills_aggregator_node
from src.agents.skills_cache_guard import make_skills_cache_guard_node
from src.agents.skills_synthesizer import make_skills_synthesizer_node
from src.db.chroma_client import ChromaLibrarianClient
from src.db.sha_store import SHAStore
from src.models.skills_state import SkillsState

logger = logging.getLogger(__name__)


def build_skills_graph(
    llm: BaseChatModel,
    chroma: ChromaLibrarianClient,
    *,
    sha_store: SHAStore | None = None,
    recursion_limit: int = 50,
    interrupt_before: list[str] | None = None,
) -> CompiledStateGraph:
    """Build and compile the Skills Export Pipeline StateGraph.

    Args:
        llm: BaseChatModel instance (temperature=0.2, num_ctx=16384) — skills_synthesizer.
            Larger context window required: output budget (4096) + skills input exceeds 8192.
        chroma: ChromaLibrarianClient singleton — skills_aggregator and skills_cache_guard.
        sha_store: SHAStore singleton — passed to skills_cache_guard and file_exporter to
            enable SHA-gated cache hits. When None, the SHA gate always misses (safe default
            that preserves backward compatibility with callers that do not inject sha_store).
        recursion_limit: Maximum graph steps before LangGraph raises RecursionError.
            Default 50 is generous for this linear 4-node graph. Callers must pass this
            value to .invoke(): graph.invoke(state, config={"recursion_limit": recursion_limit}).
        interrupt_before: Node names to pause at for human-in-the-loop approval (spec §6.6).
            Defaults to [] (disabled). Production callers should pass ["file_exporter"] and
            supply a checkpointer to .compile() to enable pre-write approval checkpoints.

    Returns:
        Compiled LangGraph StateGraph ready for .invoke() calls.
    """
    if interrupt_before is None:
        # Default: no interrupts — the graph runs to completion.
        # Production callers should pass interrupt_before=["file_exporter"] per spec §6.6
        # to enable human-in-the-loop approval before disk writes (requires a checkpointer).
        interrupt_before = []

    # When sha_store is None, skills_cache_guard always misses (safe backward-compatible default)
    skills_cache_guard = make_skills_cache_guard_node(chroma, sha_store) if sha_store else None
    skills_aggregator = make_skills_aggregator_node(chroma)
    skills_synthesizer = make_skills_synthesizer_node(llm)
    file_exporter = make_file_exporter_node(sha_store=sha_store)

    def route_after_cache_guard(state: SkillsState) -> str:
        """Route to END on cache hit, else to skills_aggregator for full synthesis."""
        return END if state.get("cache_hit") else "skills_aggregator"

    def route_after_aggregator(state: SkillsState) -> str:
        """Route to END if no indexed content found, else to skills_synthesizer."""
        return "skills_synthesizer" if state.get("skills_data") is not None else END

    builder = StateGraph(SkillsState)

    # SHA gate node is only wired when sha_store is provided; otherwise graph starts at aggregator
    if skills_cache_guard is not None:
        builder.add_node("skills_cache_guard", skills_cache_guard)
        builder.add_node("skills_aggregator", skills_aggregator)
        builder.add_node("skills_synthesizer", skills_synthesizer)
        builder.add_node("file_exporter", file_exporter)

        builder.add_edge(START, "skills_cache_guard")
        builder.add_conditional_edges("skills_cache_guard", route_after_cache_guard)
        builder.add_conditional_edges("skills_aggregator", route_after_aggregator)
        builder.add_edge("skills_synthesizer", "file_exporter")
        builder.add_edge("file_exporter", END)
    else:
        # Legacy topology: no SHA gate, starts directly at aggregator
        builder.add_node("skills_aggregator", skills_aggregator)
        builder.add_node("skills_synthesizer", skills_synthesizer)
        builder.add_node("file_exporter", file_exporter)

        builder.add_edge(START, "skills_aggregator")
        builder.add_conditional_edges("skills_aggregator", route_after_aggregator)
        builder.add_edge("skills_synthesizer", "file_exporter")
        builder.add_edge("file_exporter", END)

    return builder.compile(interrupt_before=interrupt_before)
