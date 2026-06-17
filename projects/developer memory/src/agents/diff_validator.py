"""diff_validator LangGraph node — Spec §2.4, §6.3, §7.

Validates the diff_text before any LLM or ChromaDB interaction. Rejects diffs that:
  - Are empty or whitespace-only
  - Exceed 100 KB (context overflow risk for Gemma 4)
  - Lack unified diff format markers (--- / +++ headers)

User-facing error messages are sanitized — no internal state exposed.
"""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_MAX_DIFF_BYTES = 100_000  # 100 KB per §6.3


def make_diff_validator_node() -> Callable[[dict], dict]:
    """Return a diff_validator node (no injected dependencies — validation is pure logic).

    Returns:
        LangGraph node function that reads state["diff_text"] and writes diff_safe=True/False.
        On rejection, sets state["error"] with a safe user-facing message.
    """

    def diff_validator(state: dict) -> dict:
        """Validate diff_text against all §6.3 rejection criteria.

        Sets diff_safe=True if the diff passes all checks, False otherwise.
        Always appends a trace entry describing the outcome.
        """
        diff_text: str = state.get("diff_text", "")
        trace: list[str] = list(state.get("trace", []))

        # ── Guard: empty or whitespace-only ──────────────────────────────
        if not diff_text or not diff_text.strip():
            return {
                "diff_safe": False,
                "error": "Diff rejected: diff must not be empty.",
                "trace": trace + ["diff_validator: rejected (empty)"],
            }

        # ── Guard: size limit ─────────────────────────────────────────────
        if len(diff_text.encode()) > _MAX_DIFF_BYTES:
            logger.warning("diff_validator: rejected — diff exceeds %d bytes", _MAX_DIFF_BYTES)
            return {
                "diff_safe": False,
                "error": "Diff rejected: diff exceeds 100 KB size limit.",
                "trace": trace + ["diff_validator: rejected (too large)"],
            }

        # ── Guard: unified diff format ────────────────────────────────────
        # A valid unified diff must have at least one --- and +++ header pair.
        lines = diff_text.splitlines()
        has_old = any(line.startswith("---") for line in lines)
        has_new = any(line.startswith("+++") for line in lines)
        if not (has_old and has_new):
            return {
                "diff_safe": False,
                "error": "Diff rejected: invalid unified diff format.",
                "trace": trace + ["diff_validator: rejected (invalid format)"],
            }

        logger.debug("diff_validator: accepted diff (%d bytes)", len(diff_text.encode()))
        return {
            "diff_safe": True,
            "trace": trace + ["diff_validator: accepted"],
        }

    return diff_validator
