"""State TypedDict for the Query Pipeline (query_memory MCP tool).

Spec §3 — QueryState flows through: query_guard → semantic_searcher → snippet_summarizer.
"""

import operator
from typing import Annotated, TypedDict


class QueryState(TypedDict):
    """Pipeline state for query_memory.

    Fields:
        query: The raw query string from the MCP client.
        tech_filter: Optional tech stack labels to narrow ChromaDB search.
        query_safe: Set to True by query_guard when the query passes all safety checks.
            Conditional edge routes to END (with user-facing error) if False.
        raw_results: ChromaDB result dicts from semantic_searcher.
        final_answer: LLM-synthesized answer from snippet_summarizer, or None on error.
        error: Non-fatal error message surfaced to the MCP client.
        trace: Append-only execution log.
    """

    query: str
    tech_filter: list[str] | None
    query_safe: bool
    raw_results: list[dict]
    final_answer: str | None
    error: str | None
    trace: Annotated[list[str], operator.add]
