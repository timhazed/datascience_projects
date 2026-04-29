"""search_tool — LangChain @tool wrapping SearchProvider.search + search_chain.

This tool is used by the experiment runner and any LangChain agent that needs
to search for disease information directly. The graph pipeline uses
disease_search_node instead (which injects dependencies via factory).

Typed input: SearchDiseaseParams (validated by LangChain before invoke).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from langchain_core.tools import tool

from src.models.disease_search_result import DiseaseSearchResult
from src.models.search_disease_params import SearchDiseaseParams
from src.models.search_result_item import SearchResultItem
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)

_NO_RESULTS_SUMMARY = "No results found from trusted sources."


def make_search_tool(
    search_fn: Callable[[str, int], list[SearchResultItem]],
    chain: Any,
):
    """Return a LangChain @tool that searches for disease information and synthesises a summary.

    Calls search_fn (duck-typed: any callable(query, max_results) → list), applies
    the domain whitelist filter (already applied inside SearchProvider.search), then
    invokes the search chain to produce a clinical synthesis. Returns an empty
    DiseaseSearchResult when no results pass the domain filter.

    Args:
        search_fn: Callable(query: str, max_results: int) → list[SearchResultItem].
            In production this is SearchProvider.search; in tests it is a mock.
        chain: Compiled search chain — search_summary_prompt | llm | StrOutputParser.

    Returns:
        A LangChain tool callable that accepts SearchDiseaseParams fields
        and returns a DiseaseSearchResult.
    """

    @tool
    def search_tool(query: str, max_results: int = 5) -> DiseaseSearchResult:
        """Search medical sources and synthesise a clinical summary of disease information.

        Calls the configured search provider (SerpAPI or Serper), filters results
        to trusted medical domains, then invokes the LLM synthesis chain. Returns
        a DiseaseSearchResult with empty items and a "no results" summary when the
        domain filter removes all results.

        Args:
            query: Clinical search query string.
            max_results: Maximum number of results to request (1–10). Default 5.

        Returns:
            DiseaseSearchResult with items, summary, and citations.
        """
        # Validate via the existing Pydantic params model for consistent constraint enforcement
        params = SearchDiseaseParams(query=query, max_results=max_results)

        try:
            items = search_fn(params.query, params.max_results)
        except Exception as exc:  # noqa: BLE001
            logger.error("[search_tool] search_fn failed: %s — %s", type(exc).__name__, exc)
            return DiseaseSearchResult(
                query=params.query,
                items=[],
                summary=f"Search failed: {type(exc).__name__}",
                citations=[],
            )

        if not items:
            logger.warning("[search_tool] 0 results after domain filter for %r", params.query)
            return DiseaseSearchResult(
                query=params.query,
                items=[],
                summary=_NO_RESULTS_SUMMARY,
                citations=[],
            )

        # Format as numbered list for the synthesis chain; include source domain for citation
        formatted = "\n".join(
            f"[{i + 1}] ({item.source_domain}) {item.snippet}"
            for i, item in enumerate(items)
        )

        try:
            summary = invoke_with_retry(
                chain,
                {"search_results": formatted, "query": params.query},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[search_tool] chain failed: %s", exc)
            summary = f"Synthesis failed: {type(exc).__name__}"

        citations = [item.url for item in items]
        logger.info("[search_tool] synthesised %d results for %r", len(items), params.query)
        return DiseaseSearchResult(
            query=params.query,
            items=items,
            summary=summary,
            citations=citations,
        )

    return search_tool
