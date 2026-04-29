"""disease_search_node — web search + clinical synthesis.

Calls a search_fn (duck-typed: any callable(query, max_results) → list) then
formats results as a numbered list and invokes the search chain. In production
search_fn is SearchProvider.search; in tests it is a mock callable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from src.models.graph_state import HealthcareState
from src.models.search_disease_params import SearchDiseaseParams
from src.models.search_result_item import SearchResultItem
from src.models.task_result import TaskResult
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def make_disease_search_node(
    search_fn: Callable[[str, int], list[SearchResultItem]],
    chain: Any,
) -> Callable[[HealthcareState], dict]:
    """Return a disease_search_node with search function and chain injected via closure.

    Args:
        search_fn: Callable(query: str, max_results: int) → list[SearchResultItem].
            In production this is SearchProvider.search; in tests it is a mock.
        chain: Compiled search chain — search_summary_prompt | llm | StrOutputParser.

    Returns:
        LangGraph node function compatible with HealthcareState.
    """

    def disease_search_node(state: HealthcareState) -> dict:
        """Search for disease information and synthesise a clinical summary.

        Always dequeues pending_tasks[0]. Calls search_fn, formats results as
        a numbered list with source domains, then invokes the search chain.
        Zero results after search → failure TaskResult (non-fatal).

        Args:
            state: Current HealthcareState.

        Returns:
            Dict with completed_tasks, pending_tasks[1:], and trace.
        """
        # Accumulate under REPLACE semantics — read prior lists and extend them.
        completed_prior = list(state.get("completed_tasks", []))
        trace_prior = list(state.get("trace", []))

        pending = state["pending_tasks"]
        if not pending:
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(
                        task="search_disease",
                        success=False,
                        error="Node called with empty pending_tasks — graph routing error.",
                    )
                ],
                "pending_tasks": [],
                "trace": trace_prior + ["disease_search: called with empty queue"],
            }
        remaining = pending[1:]

        try:
            params = SearchDiseaseParams(**pending[0].parameters)
        except ValidationError as exc:
            logger.warning("[disease_search] invalid params: %s", exc)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="search_disease", success=False, error=str(exc))
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"disease_search: ValidationError — {exc}"],
            }

        try:
            items = search_fn(params.query, params.max_results)
        except Exception as exc:  # noqa: BLE001
            logger.error("[disease_search] search API error: %s", exc)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(
                        task="search_disease",
                        success=False,
                        error=f"Search API error: {type(exc).__name__} — {exc}",
                    )
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"disease_search: search API ERROR — {exc}"],
            }

        if not items:
            msg = f"No results found for query: {params.query}"
            logger.warning("[disease_search] 0 results for '%s'", params.query)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(task="search_disease", success=False, error=msg)
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"disease_search: 0 results — {params.query}"],
            }

        # Format as numbered list; use getattr so both Pydantic models and plain
        # dicts work during testing
        formatted = "\n".join(
            f"[{i + 1}] ({getattr(r, 'source_domain', 'web')}) "
            f"{getattr(r, 'snippet', str(r))}"
            for i, r in enumerate(items)
        )

        try:
            summary = invoke_with_retry(
                chain,
                {"search_results": formatted, "query": params.query},
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("[disease_search] chain failed: %s", exc)
            return {
                "completed_tasks": completed_prior + [
                    TaskResult(
                        task="search_disease",
                        success=False,
                        error=f"Synthesis chain failed: {type(exc).__name__}",
                    )
                ],
                "pending_tasks": remaining,
                "trace": trace_prior + [f"disease_search: chain ERROR — {exc}"],
            }

        logger.info(
            "[disease_search] synthesised %d results for '%s'", len(items), params.query
        )
        return {
            "completed_tasks": completed_prior + [
                TaskResult(
                    task="search_disease",
                    success=True,
                    result={
                        "query": params.query,
                        "summary": summary,
                        "result_count": len(items),
                    },
                )
            ],
            "pending_tasks": remaining,
            "trace": trace_prior + [
                f"disease_search: {len(items)} results synthesised for '{params.query}'"
            ],
        }

    return disease_search_node
