"""Retry utility for LangChain Runnable invocations.

Spec §5 — every chain.invoke() call in the system must go through invoke_with_retry.
Provides exponential backoff for transient Ollama connectivity errors while immediately
re-raising structured-output contract failures (ValidationError) and process shutdown
signals (KeyboardInterrupt) that should never be swallowed.
"""

import logging
import time
from typing import Any, TypeVar

from pydantic import ValidationError

T = TypeVar("T")
logger = logging.getLogger(__name__)


def invoke_with_retry(
    chain: Any,
    inputs: dict,
    max_attempts: int = 3,
    backoff_base: float = 1.5,
) -> Any:
    """Invoke a LangChain Runnable with exponential backoff retry.

    Retries on transient connectivity errors (OllamaConnectionError, httpx errors).
    Does NOT retry on ValidationError — a structured-output contract failure will
    produce the same malformed output on every attempt; fail fast instead.
    Does NOT retry on KeyboardInterrupt — process shutdown must not be suppressed.

    Sleep only occurs *between* attempts, never after the final failure. This avoids
    an unnecessary delay (up to backoff_base^max_attempts seconds) before the
    RuntimeError is raised to the caller.

    Args:
        chain: Any LangChain Runnable (chain, LLM, etc.) with an .invoke() method.
        inputs: Dict of inputs forwarded to chain.invoke().
        max_attempts: Total number of attempts before raising RuntimeError. Min 1.
        backoff_base: Base for exponential wait: wait = backoff_base ** attempt_number.
            Sleep happens only when attempt < max_attempts.

    Returns:
        The result of chain.invoke(inputs) on the first successful attempt.

    Raises:
        ValidationError: Re-raised immediately on the first attempt — not retried.
        KeyboardInterrupt: Re-raised immediately — process shutdown must not be suppressed.
        RuntimeError: After all max_attempts have been exhausted, with the last exception
            attached as the cause for debugging context.
    """
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return chain.invoke(inputs)
        except (ValidationError, KeyboardInterrupt):
            # Structured output failure or shutdown — never retry these
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < max_attempts:
                # Only sleep between retries — not after the final failed attempt
                wait = backoff_base**attempt
                logger.warning(
                    "invoke_with_retry: attempt %d/%d failed [%s]: %s — retrying in %.1fs",
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    str(exc)[:120],
                    wait,
                )
                time.sleep(wait)
            else:
                # Final attempt — log without sleeping before the raise
                logger.warning(
                    "invoke_with_retry: attempt %d/%d failed [%s]: %s — no more retries",
                    attempt,
                    max_attempts,
                    type(exc).__name__,
                    str(exc)[:120],
                )

    raise RuntimeError(f"invoke_with_retry: all {max_attempts} attempts failed") from last_exc
