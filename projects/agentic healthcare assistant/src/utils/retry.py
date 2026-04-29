"""invoke_with_retry — retry wrapper for LangChain chain invocations.

Retries on transient LLM failures (OutputParserException, ValueError) with
exponential backoff: 1 s after attempt 1, 2 s after attempt 2.

Philosophy note: this function uses multiple return/raise paths by design.
The guard-clause pattern (early raise on non-retryable) is acceptable per
the architecture spec; the mid-loop break on success is the only alternative
to restructuring into a single-exit loop with a flag variable, which would
obscure the intent. This exception is documented here per Architecture.md Section 3.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.exceptions import OutputParserException

_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = [1, 2]  # sleep after attempt 0, after attempt 1


def invoke_with_retry(chain: Any, inputs: dict) -> Any:
    """Invoke a LangChain chain with up to 3 attempts and exponential backoff.

    Retryable exceptions: OutputParserException, ValueError.
    All other exceptions pass through immediately without retry.

    Args:
        chain: Any object with an .invoke(inputs) method (chain, runnable, tool).
        inputs: Dict of inputs to pass to chain.invoke().

    Returns:
        The result of chain.invoke(inputs) on the first successful attempt.

    Raises:
        OutputParserException: If all 3 attempts fail with OutputParserException.
        ValueError: If all 3 attempts fail with ValueError.
        Exception: Any non-retryable exception from the first attempt, re-raised immediately.
    """
    last_exc: Exception | None = None

    for attempt in range(_MAX_ATTEMPTS):
        try:
            return chain.invoke(inputs)
        except (OutputParserException, ValueError) as exc:
            last_exc = exc
            if attempt < len(_BACKOFF_SECONDS):
                time.sleep(_BACKOFF_SECONDS[attempt])
        except Exception:
            # Non-retryable — pass through immediately.
            raise

    raise last_exc  # type: ignore[misc]
