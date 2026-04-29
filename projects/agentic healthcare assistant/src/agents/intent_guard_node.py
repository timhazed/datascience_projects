"""intent_guard_node — classifies the user query and sets intent_safe on state.

Wraps build_intent_guard_chain output via invoke_with_retry. A chain failure
is treated as UNSAFE (fail-closed) so the graph never proceeds to the planner
on an ambiguous or errored classification.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from src.models.graph_state import HealthcareState
from src.utils.retry import invoke_with_retry

logger = logging.getLogger(__name__)


def make_intent_guard_node(chain: Any) -> Callable[[HealthcareState], dict]:
    """Return an intent_guard_node function with the guard chain injected via closure.

    The chain is expected to return exactly "SAFE" or "UNSAFE" (normalised by
    the chain's internal RunnableLambda step). Any other value or any exception
    is treated as UNSAFE — fail-closed behaviour is intentional.

    Args:
        chain: Compiled intent guard chain —
               guard_prompt | llm | StrOutputParser | normalizer.

    Returns:
        LangGraph node function compatible with HealthcareState.
    """

    def intent_guard_node(state: HealthcareState) -> dict:
        """Classify the user query and write intent_safe to state.

        Invokes the guard chain with the user_query. Sets intent_safe=True
        only when the chain returns exactly "SAFE". All other outcomes
        (including chain failures) set intent_safe=False.

        Args:
            state: Current HealthcareState.

        Returns:
            Dict with intent_safe (bool) and trace.
        """
        # Accumulate under REPLACE semantics — read prior trace and extend it.
        trace_prior = list(state.get("trace", []))
        try:
            result: str = invoke_with_retry(chain, {"user_query": state["user_query"]})
            safe = result == "SAFE"
            logger.info("[intent_guard] classified as %s", result)
            return {
                "intent_safe": safe,
                "trace": trace_prior + [f"intent_guard: {result}"],
            }
        except Exception as exc:  # noqa: BLE001
            # Fail-closed: any chain error → UNSAFE so the graph exits cleanly.
            logger.error("[intent_guard] chain failed: %s — defaulting to UNSAFE", exc)
            return {
                "intent_safe": False,
                "error": f"Intent guard failed: {type(exc).__name__}",
                "trace": trace_prior + [f"intent_guard: ERROR — {exc}"],
            }

    return intent_guard_node
